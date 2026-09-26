"""The topic modelling flow: claim a run, read the corpus, fit, replace."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import ClassVar

from opentelemetry.trace import Span

from blob_store.s3 import ExportBucket, ModelsBucket
from stages import StageService
from telemetry import tracer, working
from telemetry.evaluations import current_ids
from topic_modelling.labels import TopicLabeller
from topic_modelling.models import FittedTopic, Fitting
from topic_modelling.repository import TopicQueue
from topic_modelling.space import dump
from topic_modelling.topics import NoVocabulary, TopicFitter, carry_labels

log = logging.getLogger(__name__)
span = tracer(__name__)

#: How many of a topic's strongest passages the model is shown when naming it.
_EXCERPTS = 4


class TopicModellingService(StageService):
    """Fits the corpus topic model when a run asks for it.

    The unit of work is one fit, over every language. Each fit replaces every
    topic and every membership, carrying labels across by matching top terms.
    A fit that fails leaves the previous topics in place, with the reason on
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
        models: ModelsBucket,
        labeller: TopicLabeller | None = None,
        min_fact_share: float = 0.0,
    ) -> None:
        """Initialises the service with its collaborators.

        Args:
            repository: The fit queue and the corpus reader.
            fitter: Fits one language at a time.
            export: Where each language's figure is stored.
            models: Where the factorisation the figure is drawn from is
                stored. The figure is a view of this and can be redrawn
                from it; this is what a re-fit would otherwise be needed
                to get back.
            labeller: Names the unnamed topics. Without one a topic keeps its
                terms and no name.
            min_fact_share: The smallest share of a topic's passages that must
                carry a fact before it is worth naming. 0 names every topic.
        """
        super().__init__(repository)
        self._repository: TopicQueue = repository
        self._fitter = fitter
        self._export = export
        self._models = models
        self._labeller = labeller
        self._min_fact_share = min_fact_share

    def request(self, trigger: str | None = None) -> int:
        """Asks for a fit, without doing it.

        Args:
            trigger: Which run is asking, adopted by the worker that takes
                the fit.

        Returns:
            The id of the request row.
        """
        fit_id = self._repository.request(trigger)
        log.info("requested topic fit %d", fit_id)
        return fit_id

    def process_next(self) -> int | None:
        """Runs one requested fit, recording a failure against the request.

        Returns:
            The id of the fit that was claimed, or None if none was queued.
        """
        fit_id = self._repository.claim()
        if fit_id is None:
            return None

        with working(
            span, "model_topics", {"stage": self.name, "topic_fit.id": fit_id}
        ) as current:
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

        A language the fitter refuses is logged and left out.

        Raises:
            NoVocabulary: If no passage carries a language, or if no language
                produced a model.
        """
        languages = self._repository.languages()
        if not languages:
            raise NoVocabulary(
                "no passage carries a language, so there is nothing to fit. "
                "Chunk the corpus first: chunking is what detects it."
            )

        fittings: list[Fitting] = []
        for language in languages:
            # A span per language, not one for the lot: each is a separate
            # factorisation over its own vocabulary and then one model call
            # per topic to name it, which is minutes. Under one span a fit
            # that was slow in German reads as a fit that was slow.
            with working(span, "fit_language", {"topic_fit.language": language}):
                trace_id, span_id = current_ids()
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
                    replace(
                        fitting,
                        topics=self._named(carried, fitting, language),
                        trace_id=trace_id,
                        span_id=span_id,
                    )
                )

        if not fittings:
            raise NoVocabulary(
                f"none of {', '.join(languages)} had a vocabulary left after "
                f"filtering, so no model was fitted"
            )

        memberships = self._repository.replace(fit_id, fittings)
        self._keep(fittings)
        self._draw(fittings)
        self._annotate(current, fittings, memberships)

        for fitting in fittings:
            log.info(
                "fit %d [%s]: %d topic(s) over %d passage(s), %d term(s), %d named",
                fit_id,
                fitting.language,
                len(fitting.topics),
                fitting.passages,
                fitting.vocabulary,
                fitting.labelled,
            )
            if fitting.without_topics:
                log.warning(
                    "%d of %d %s passage(s) have no topic. They and everything "
                    "drawn from them are outside every topic-weighted report.",
                    fitting.without_topics,
                    fitting.passages,
                    fitting.language,
                )

    def _keep(self, fittings: list[Fitting]) -> None:
        """Stores each language's factorisation.

        Before `_draw`, because the figure is a view of this: if only one
        of the two survives a bad run it should be the one the other can
        be rebuilt from.

        Every failure is logged and left rather than raised, as the
        drawing's are. The topics are already written by the time this
        runs, and a fit whose rows landed is not a failed fit - refusing it
        here would roll a good corpus back over an object store that was
        briefly unreachable.
        """
        for fitting in fittings:
            try:
                self._models.put(
                    self._models.topic_model_key(fitting.language),
                    dump(fitting.space),
                    content_type=self._models.TOPIC_MODEL_TYPE,
                )
            except Exception:
                log.exception("could not keep the %s model", fitting.language)

    def _draw(self, fittings: list[Fitting]) -> None:
        """Stores each language's model as a page.

        Runs after the topics are written, so every failure here is logged and
        left rather than raised. pyLDAvis is imported here, so an image built
        without it still fits.
        """
        try:
            from topic_modelling.visualisation import render
        except ImportError:
            log.exception("pyLDAvis is missing, so the topics were not drawn")
            return

        for fitting in fittings:
            # Taken away before the new one is drawn, so a language whose
            # figure fails to render has none rather than a stale one. The
            # route answers 404 for that.
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

    def _barren(self, fitting: Fitting) -> frozenset[int]:
        """Which topics hold almost no facts, and so name no subject.

        The deterministic half of "Mixed". Asking the model was answered
        `Mixed` for none of 38 topics, including one whose terms were
        `inhaltsverzeichnis, einführung, urheberschutzvermerk,
        änderungsübersicht, danksagung` and which it named `Testverfahren`
        off the one subject word among them. A model reliably names
        whatever it is shown, so what it is shown is decided here.

        Read off the facts rather than off the terms, which is what makes it
        language-neutral: a passage extraction found nothing in says nothing,
        whatever language it says it in, and a word list of front-matter
        terms would have to be written again for every language a deployment
        adds.

        Turned off when the corpus carries no facts at all - on a fit run
        before extraction, every topic looks barren and the signal is absent
        rather than unanimous.
        """
        floor = self._min_fact_share
        if floor <= 0:
            return frozenset()

        dominant: dict[int, int] = {}
        best: dict[int, float] = {}
        for weight in fitting.weights:
            if weight.weight > best.get(weight.passage_id, 0.0):
                best[weight.passage_id] = weight.weight
                dominant[weight.passage_id] = weight.topic_index

        bearing = self._repository.bearing_facts(list(dominant))
        if not bearing:
            return frozenset()

        held: dict[int, list[int]] = {}
        for passage_id, topic_index in dominant.items():
            held.setdefault(topic_index, []).append(passage_id)
        return frozenset(
            topic_index
            for topic_index, passages in held.items()
            if sum(1 for one in passages if one in bearing) / len(passages) < floor
        )

    def _named(
        self, topics: list[FittedTopic], fitting: Fitting, language: str
    ) -> list[FittedTopic]:
        """Names the topics that hold no label yet, with the model.

        A topic that already carries one - typed by a person, or carried from
        the last fit - is left alone. Returns `topics` unchanged when no
        labeller is configured.
        """
        if self._labeller is None:
            return topics

        strongest: dict[int, list[int]] = {}
        for weight in sorted(fitting.weights, key=lambda w: -w.weight):
            held = strongest.setdefault(weight.topic_index, [])
            if len(held) < _EXCERPTS:
                held.append(weight.passage_id)

        barren = self._barren(fitting)
        named: list[FittedTopic] = []
        # Every name given so far, including those a person typed: the model
        # is shown them so one fit cannot put two topics under one name.
        taken = [topic.label for topic in topics if topic.label]
        for topic in topics:
            if topic.label:
                named.append(topic)
                continue
            if topic.topic_index in barren:
                # Mixed, decided before the call rather than asked for in it.
                log.info(
                    "%s topic %d: its passages carry almost no facts; not named",
                    language,
                    topic.topic_index,
                )
                named.append(topic)
                continue
            label = self._labeller.label(
                topic,
                language,
                self._repository.excerpts(strongest.get(topic.topic_index, [])),
                taken,
            )
            if label:
                taken.append(label)
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
