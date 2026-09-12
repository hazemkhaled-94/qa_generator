"""The topic modelling flow: claim a run, read the corpus, fit, replace."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import ClassVar

from opentelemetry.trace import Span

from stages import StageService
from telemetry import tracer
from topic_modelling.models import Fitting
from topic_modelling.repository import TopicQueue
from topic_modelling.topics import NoVocabulary, TopicFitter, carry_labels

log = logging.getLogger(__name__)
span = tracer(__name__)


class TopicModellingService(StageService):
    """Fits the corpus topic model when a run asks for it.

    The unit of work is a whole language, because the factorisation estimates
    every topic together over one vocabulary. That is also why work arrives as an explicit
    request: a new document does not make one topic stale, it makes all of
    them stale, and whether that is worth a refit is a person's decision.

    Each fit replaces every topic and every membership. Human labels are
    carried across by matching top terms. A fit that fails leaves the previous
    topics in place, with the reason on the request row beside them.
    """

    name: ClassVar[str] = "topic_modelling"
    unit: ClassVar[str] = "fit"

    def __init__(self, *, repository: TopicQueue, fitter: TopicFitter) -> None:
        """Initialises the service with its collaborators."""
        super().__init__(repository)
        self._repository: TopicQueue = repository
        self._fitter = fitter

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

        A language whose vocabulary the frequency filter empties is logged and
        left out rather than failing the run: one unusable language must not
        cost the others their topics.

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
            fittings.append(
                replace(
                    fitting,
                    topics=carry_labels(
                        fitting.topics, self._repository.labelled_topics(language)
                    ),
                )
            )

        if not fittings:
            raise NoVocabulary(
                f"none of {', '.join(languages)} had a vocabulary left after "
                f"filtering, so no model was fitted"
            )

        memberships = self._repository.replace(fit_id, fittings)
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
                # Loud on purpose: a passage with no topic is absent from
                # every topic-weighted report, and so are its facts.
                log.warning(
                    "%d of %d %s passage(s) have no topic. They and everything "
                    "drawn from them are outside every topic-weighted report.",
                    fitting.without_topics,
                    fitting.passages,
                    fitting.language,
                )

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
