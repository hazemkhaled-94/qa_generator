"""The topic modelling flow: claim a run, read the corpus, fit, replace."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import ClassVar

from opentelemetry.trace import Span

from blob_store.seaweedfs import ExportBucket
from stages import StageService
from telemetry import tracer
from topic_modelling.labels import TopicLabeller
from topic_modelling.models import Fitting
from topic_modelling.repository import TopicQueue
from topic_modelling.topics import NoVocabulary, TopicFitter, carry_labels

log = logging.getLogger(__name__)
span = tracer(__name__)

#: How many of a topic's strongest passages the model is shown when naming it.
_EXCERPTS = 4


class TopicModellingService(StageService):
    """Fits the corpus topic model when a run asks for it.

    The unit of work is a whole language. Each fit replaces every topic and
    every membership, carrying human labels across by matching top terms. A
    fit that fails leaves the previous topics in place, with the reason on
    the request row beside them.
    """

    name: ClassVar[str] = "topic_modelling"
    unit: ClassVar[str] = "fit"

    def __init__(
        self,
        *,
        repository: TopicQueue,
        fitter: TopicFitter,
        export: ExportBucket,
        labeller: TopicLabeller | None = None,
    ) -> None:
        """Initialises the service with its collaborators.

        The labeller is optional; without one a topic keeps its terms and no
        name.
        """
        super().__init__(repository)
        self._repository: TopicQueue = repository
        self._fitter = fitter
        self._export = export
        self._labeller = labeller

    def request(self) -> int:
        """Asks for a fit, without doing it."""
        fit_id = self._repository.request()
        log.info("requested topic fit %d", fit_id)
        return fit_id

    def process_next(self) -> int | None:
        """Runs one requested fit, recording a failure against the request."""
        fit_id = self._repository.claim()
        if fit_id is None:
            return None

        with span.start_as_current_span("model_topics") as current:
            current.set_attribute("topic_fit.id", fit_id)
            try:
                self._fit(fit_id, current)
            except NoVocabulary as exc:
                self._fail(fit_id, str(exc), current)
            except Exception as exc:
                self._fail(fit_id, f"{type(exc).__name__}: {exc}", current)
                log.exception("failed topic fit %d", fit_id)
        return fit_id

    def _fit(self, fit_id: int, current: Span) -> None:
        """Fits every language the corpus holds and stores what they produced.

        A language whose vocabulary the frequency filter empties is logged
        and left out rather than failing the run.

        Raises:
            NoVocabulary: If no language produced a model at all.
        """
        languages = self._repository.languages()
        if not languages:
            raise NoVocabulary(
                "no passage carries a language, so there is nothing to fit. "
                "Chunk the corpus first: chunking is what detects it."
            )

        fittings: list[Fitting] = []
        for language in languages:
            try:
                fitting = self._fitter.fit(
                    lambda language=language: self._repository.passages(language),
                    language,
                )
            except NoVocabulary as exc:
                log.warning("no %s model: %s", language, exc)
                continue
            carried = carry_labels(
                fitting.topics, self._repository.labelled_topics(language)
            )
            fittings.append(
                replace(fitting, topics=self._named(carried, fitting, language))
            )

        if not fittings:
            raise NoVocabulary(
                f"none of {', '.join(languages)} had a vocabulary left after "
                f"filtering, so no model was fitted"
            )

        memberships = self._repository.replace(fit_id, fittings)
        self._draw(fittings)
        self._annotate(current, fittings, memberships)

        for fitting in fittings:
            log.info(
                "fit %d [%s]: %d topic(s) over %d passage(s), %d term(s), "
                "%d label(s) carried",
                fit_id,
                fitting.language,
                len(fitting.topics),
                fitting.passages,
                fitting.vocabulary,
                fitting.labels_carried,
            )
            if fitting.without_topics:
                log.warning(
                    "%d of %d %s passage(s) have no topic. They and everything "
                    "drawn from them are outside every topic-weighted report.",
                    fitting.without_topics,
                    fitting.passages,
                    fitting.language,
                )

    def _draw(self, fittings: list[Fitting]) -> None:
        """Stores each language's model as a page.

        The topics are already written when this runs, so a failure here is
        logged and left: a model with no picture is still a model. pyLDAvis is
        imported here for the same reason, so an image built before it was
        declared still fits.
        """
        try:
            from topic_modelling.visualisation import render
        except ImportError:
            log.exception("pyLDAvis is missing, so the topics were not drawn")
            return

        for fitting in fittings:
            # Taken away before the new one is drawn: these topics have just
            # replaced the ones the stored figure describes, and a figure of
            # topics that no longer exist is worse than none. The route
            # answers 404 for a language that has no figure.
            key = self._export.topic_visualisation_key(fitting.language)
            try:
                self._export.remove(key)
            except Exception:
                log.exception("could not take away the old %s figure", fitting.language)
                continue
            try:
                self._export.put(
                    key,
                    render(fitting.space, fitting.language),
                    content_type=self._export.TOPIC_VISUALISATION_TYPE,
                )
            except Exception:
                log.exception("could not draw the %s topics", fitting.language)

    def _named(self, topics: list, fitting: Fitting, language: str) -> list:
        """Names the topics no person has named, with the model.

        Only the unnamed ones: a name a person typed, or one carried from the
        last fit, is left alone.
        """
        if self._labeller is None:
            return topics

        strongest: dict[int, list[int]] = {}
        for weight in sorted(fitting.weights, key=lambda w: -w.weight):
            held = strongest.setdefault(weight.topic_index, [])
            if len(held) < _EXCERPTS:
                held.append(weight.passage_id)

        named = []
        for topic in topics:
            if topic.label:
                named.append(topic)
                continue
            label = self._labeller.label(
                topic,
                language,
                self._repository.excerpts(strongest.get(topic.topic_index, [])),
            )
            named.append(
                replace(topic, label=label, labelled_by=self._labeller.model)
                if label
                else topic
            )
        found = sum(1 for t in named if t.label)
        log.info("%s: %d of %d topic(s) named", language, found, len(named))
        return named

    def _annotate(
        self, current: Span, fittings: list[Fitting], memberships: int
    ) -> None:
        """Records the run's shape on the active span."""
        self._done(current)
        current.set_attribute("topic.languages", len(fittings))
        current.set_attribute("topic.topics", sum(len(f.topics) for f in fittings))
        current.set_attribute("topic.passages", sum(f.passages for f in fittings))
        current.set_attribute("topic.memberships", memberships)
        current.set_attribute(
            "topic.without_topics", sum(f.without_topics for f in fittings)
        )
