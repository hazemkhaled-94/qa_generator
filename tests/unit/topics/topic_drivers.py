"""Drivers the topic unit tests act through.

A driver is the page object of a service test: one per thing under test,
exposing the operations a reader cares about and holding the wiring out of
the test body. The doubles below stand in for the database and the object
store; everything else is the real code.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator

from database.qa_generator import Status
from topic_modelling.labels import TopicLabeller
from topic_modelling.models import (
    FittedTopic,
    Fitting,
    LanguageFit,
    PassageVocabulary,
    TopicFit,
    TopicRemoval,
    TopicSpace,
)
from topic_modelling.service import TopicModellingService
from topic_modelling.topics import TopicFitter

#: Anything the browser would fetch from a page.
_FETCHED = re.compile(r'(?:src|href)\s*=\s*"([^"]*)"')


def corpus(*texts: str, first_id: int = 1) -> list[PassageVocabulary]:
    """Builds a corpus from whitespace-separated lemmas, one text per passage."""
    return [
        PassageVocabulary(id=index, lemmas=text.split())
        for index, text in enumerate(texts, start=first_id)
    ]


class FittingDriver:
    """One fitted model, read the way a test asks about it."""

    def __init__(self, fitting: Fitting) -> None:
        """Wraps what a fit produced."""
        self.fitting = fitting

    @property
    def space(self) -> TopicSpace:
        """The whole fitted space."""
        return self.fitting.space

    @property
    def unplaced(self) -> int:
        """How many passages ended up in no topic."""
        return self.fitting.without_topics

    @property
    def placed_ids(self) -> set[int]:
        """The passages holding at least one membership."""
        return {weight.passage_id for weight in self.fitting.weights}

    @property
    def topic_count(self) -> int:
        """How many topics the fit produced."""
        return len(self.fitting.topics)

    def weights_of(self, passage_id: int) -> dict[int, float]:
        """One passage's memberships, by topic index."""
        return {
            weight.topic_index: weight.weight
            for weight in self.fitting.weights
            if weight.passage_id == passage_id
        }

    def terms_of(self, topic_index: int) -> list[str]:
        """One topic's signature."""
        return next(
            topic.top_terms
            for topic in self.fitting.topics
            if topic.topic_index == topic_index
        )

    def dominant_topic(self, passage_id: int) -> int | None:
        """The topic holding one passage most strongly."""
        held = self.weights_of(passage_id)
        return max(held, key=lambda index: held[index]) if held else None

    def drawn(self) -> str:
        """Renders the figure and hands back the page."""
        from topic_modelling.visualisation import render

        return render(self.space, self.fitting.language).decode("utf-8")

    def fetched_by(self, page: str) -> list[str]:
        """The addresses a page would reach out to when opened."""
        return [
            url
            for url in _FETCHED.findall(page)
            if url.startswith(("http://", "https://", "//"))
        ]


class FitterDriver:
    """The fitter, over a corpus held in memory."""

    def __init__(
        self, settings: dict, passages: list[PassageVocabulary] | None = None
    ) -> None:
        """Builds the fitter with its settings and its corpus."""
        self.settings = dict(settings)
        self._fitter = TopicFitter(**self.settings)
        self._passages = list(passages or [])
        self.walks = 0

    def holding(self, *passages: PassageVocabulary) -> FitterDriver:
        """Adds passages to the corpus."""
        self._passages.extend(passages)
        return self

    def with_passage(self, passage_id: int, *lemmas: str) -> FitterDriver:
        """Adds one passage by its lemmas."""
        return self.holding(PassageVocabulary(id=passage_id, lemmas=list(lemmas)))

    def _walk(self) -> Iterator[PassageVocabulary]:
        """Hands back a fresh walk, counting it."""
        self.walks += 1
        return iter(list(self._passages))

    def fit(self, language: str = "de") -> FittingDriver:
        """Fits the corpus as it stands."""
        return FittingDriver(self._fitter.fit(self._walk, language))

    def fit_once_walkable(self, language: str = "de") -> FittingDriver:
        """Fits a corpus that can be walked only once."""
        spent = iter(list(self._passages))
        return FittingDriver(self._fitter.fit(lambda: spent, language))


class ScriptedModel:
    """A served model that answers whatever it was told to, and counts asks."""

    def __init__(self, *says: str, model: str = "test-model") -> None:
        """Takes one answer per call, repeating the last once they run out."""
        self.says = list(says) or [""]
        self.model = model
        self.asked: list[dict] = []
        self.raises: Exception | None = None

    @property
    def calls(self) -> int:
        """How many times the model was asked."""
        return len(self.asked)

    def answer(self, **kwargs: object):
        """Answers, and records what it was asked."""
        self.asked.append(kwargs)
        if self.raises is not None:
            raise self.raises
        said = self.says[min(self.calls - 1, len(self.says) - 1)]
        return type("Answer", (), {"label": said})()


class LabellerDriver:
    """The labeller, over a scripted model."""

    def __init__(self, *says: str, languages: dict[str, str] | None = None) -> None:
        """Builds the labeller with the answers the model will give."""
        self.model = ScriptedModel(*says)
        self.labeller = TopicLabeller(self.model, languages or {"en": "English"})

    def refusing(self, error: Exception) -> LabellerDriver:
        """Makes the model raise instead of answering."""
        self.model.raises = error
        return self

    def names(
        self, *terms: str, language: str = "en", excerpts: list[str] | None = None
    ) -> str | None:
        """Names one topic built from these terms."""
        return self.labeller.label(
            FittedTopic(0, list(terms)), language, excerpts or ["an excerpt"]
        )

    @property
    def prompt(self) -> str:
        """The user message of the last ask."""
        return str(self.model.asked[-1]["user"])

    @property
    def system(self) -> str:
        """The system message of the last ask."""
        return str(self.model.asked[-1]["system"])


class RecordingQueue:
    """The topic queue, held in memory."""

    done = Status.MODELLED

    def __init__(
        self,
        corpora: dict[str, list[PassageVocabulary]],
        texts: dict[int, str] | None = None,
    ) -> None:
        """Takes one corpus per language, and the text of any passage."""
        self.corpora = corpora
        self.texts = texts or {}
        self.previous: dict[str, list[FittedTopic]] = {}
        self.stored: list[Fitting] = []
        self.failures: dict[int, str] = {}
        self.pending: list[int] = []
        self.claims: list[int] = []
        self.abandoned = 0
        self.refuses: Exception | None = None
        self._next_id = 1

    def request(self) -> int:
        """Queues a fit, replacing any unclaimed one."""
        self.pending = [self._next_id]
        self._next_id += 1
        return self.pending[0]

    def claim(self) -> int | None:
        """Takes the next queued fit."""
        if not self.pending:
            return None
        claimed = self.pending.pop(0)
        self.claims.append(claimed)
        return claimed

    def languages(self) -> list[str]:
        """The languages the corpus holds."""
        return list(self.corpora)

    def passages(self, language: str) -> Iterator[PassageVocabulary]:
        """A fresh walk over one language's passages."""
        return iter(list(self.corpora.get(language, [])))

    def excerpts(self, passage_ids: list[int]) -> list[str]:
        """The text of the passages that have any."""
        return [self.texts[one] for one in passage_ids if one in self.texts]

    def labelled_topics(self, language: str) -> list[FittedTopic]:
        """What a previous fit or a person left on this language."""
        return self.previous.get(language, [])

    def replace(self, fit_id: int, fittings: list[Fitting]) -> int:
        """Stores what one fit produced."""
        if self.refuses is not None:
            raise self.refuses
        self.stored = list(fittings)
        return sum(len(one.weights) for one in fittings)

    def fail(self, key: int, error: str) -> None:
        """Records a fit that did not finish."""
        self.failures[key] = error

    def abandon(self) -> int:
        """Sweeps claims an earlier run left behind."""
        return self.abandoned


class RecordingExport:
    """The export bucket, held in memory."""

    TOPIC_VISUALISATION_TYPE = "text/html; charset=utf-8"

    def __init__(self) -> None:
        """Starts with no figure stored."""
        self.figures: dict[str, bytes] = {}
        self.removed: list[str] = []
        self.types: dict[str, str] = {}
        self.refuses_put: Exception | None = None
        self.refuses_remove: Exception | None = None

    @staticmethod
    def topic_visualisation_key(language: str) -> str:
        """The key one language's figure is stored under."""
        return f"topics/{language}.html"

    def put(self, key: str, body: bytes, content_type: str | None = None) -> None:
        """Stores one figure."""
        if self.refuses_put is not None:
            raise self.refuses_put
        self.figures[key] = body
        if content_type:
            self.types[key] = content_type

    def remove(self, key: str) -> None:
        """Takes one figure away."""
        if self.refuses_remove is not None:
            raise self.refuses_remove
        self.removed.append(key)
        self.figures.pop(key, None)


class CommandDriver:
    """The command line, with every collaborator it reaches for replaced.

    What a flag does is which collaborator it calls, so the doubles record
    the calls and the driver hands back the exit status beside them.
    """

    def __init__(self, monkeypatch, written) -> None:
        """Replaces the module's collaborators and its output directory."""
        from topic_modelling import run as module

        self.written = written
        self.calls: list[str] = []
        self.watched: list[float] = []
        self.figures: dict[str, bytes] = {}
        self.fits: list[str] = ["de"]
        self.queued: list[int] = []
        self.drains = 0

        driver = self

        class Queue:
            """The fit queue."""

            def counts(self) -> dict[str, int]:
                """Reports the queue and the topics held."""
                driver.calls.append("counts")
                return {"modelled": 12}

            def stop(self) -> int:
                """Withdraws a queued fit."""
                driver.calls.append("stop")
                return 1

            def retry(self) -> int:
                """Returns a failed fit to the queue."""
                driver.calls.append("retry")
                return 1

        class Catalog:
            """The stored topics."""

            def delete_all(self):
                """Removes every topic."""
                driver.calls.append("delete_all")
                return TopicRemoval(topics=12, memberships=40, labels=3)

            def fit_state(self):
                """Reports which languages hold topics."""
                driver.calls.append("fit_state")
                return TopicFit(
                    status=Status.MODELLED,
                    error=None,
                    requested_at=None,
                    topics=len(driver.fits),
                    languages=[
                        LanguageFit(
                            language=code,
                            topics=1,
                            corpus_passages=1,
                            corpus_vocabulary=1,
                            passages_without_topics=0,
                            fitted_at=None,
                        )
                        for code in driver.fits
                    ],
                )

        class Export:
            """The export bucket."""

            @staticmethod
            def topic_visualisation_key(language: str) -> str:
                """The key one language's figure is stored under."""
                return f"topics/{language}.html"

            def find(self, key: str) -> bytes | None:
                """Reads one figure, or nothing."""
                driver.calls.append(f"find {key}")
                return driver.figures.get(key)

        class Service:
            """The worker."""

            def request(self) -> int:
                """Queues a fit."""
                driver.queued.append(1)
                return len(driver.queued)

            def drain(self) -> int:
                """Runs whatever is queued."""
                driver.drains += 1
                return 0

        monkeypatch.setattr(module, "TopicQueue", Queue)
        monkeypatch.setattr(module, "TopicCatalog", Catalog)
        monkeypatch.setattr(module, "ExportBucket", Export)
        monkeypatch.setattr(module, "build_service", lambda _: Service())
        # The settings the command reads, without the database they are
        # stored in. The environment is what a deployment with nothing
        # configured resolves to, and Settings.load is still the real one.
        monkeypatch.setattr(module, "resolved", lambda: dict(os.environ))
        monkeypatch.setattr(module, "_DRAWN", written)
        monkeypatch.setattr(module.telemetry, "configure", lambda *_: None)
        monkeypatch.setattr(module.telemetry, "trace_engine", lambda *_: None)
        monkeypatch.setattr(module, "engine", lambda: None)
        monkeypatch.setattr(
            module, "watch", lambda _, seconds: driver.watched.append(seconds)
        )
        self._main = module.main

    def drew(self, language: str, page: bytes = b"<html>a figure</html>") -> None:
        """Puts one language's figure in the bucket."""
        self.figures[f"topics/{language}.html"] = page

    def run(self, *argv: str) -> int:
        """Runs the command line with these arguments."""
        return self._main(list(argv))

    def files(self) -> dict[str, bytes]:
        """Whatever --visualise wrote, by file name."""
        if not self.written.exists():
            return {}
        return {one.name: one.read_bytes() for one in self.written.iterdir()}


class ServiceDriver:
    """The whole flow, with the database and the object store in memory.

    The fitter, the labeller and the renderer are the real ones, so a fit run
    through this goes through gensim and pyLDAvis.
    """

    def __init__(
        self,
        *,
        settings: dict,
        corpora: dict[str, list[PassageVocabulary]] | None = None,
        texts: dict[int, str] | None = None,
        labeller: LabellerDriver | None = None,
    ) -> None:
        """Wires the service over the doubles."""
        self.queue = RecordingQueue(corpora or {}, texts)
        self.export = RecordingExport()
        self.labeller = labeller
        self.service = TopicModellingService(
            repository=self.queue,  # pyright: ignore[reportArgumentType]
            fitter=TopicFitter(**settings),
            export=self.export,  # pyright: ignore[reportArgumentType]
            labeller=labeller.labeller if labeller else None,
        )

    def request(self) -> int:
        """Asks for a fit."""
        return self.service.request()

    def run(self) -> int | None:
        """Runs one queued fit."""
        return self.service.process_next()

    def drain(self) -> int:
        """Runs every queued fit."""
        return self.service.drain()

    def request_and_run(self) -> int | None:
        """Asks for a fit and runs it."""
        self.request()
        return self.run()

    @property
    def stored(self) -> list[Fitting]:
        """What the last fit wrote, one entry per language."""
        return self.queue.stored

    @property
    def languages_stored(self) -> list[str]:
        """The languages the last fit wrote topics for."""
        return [one.language for one in self.stored]

    def topics_of(self, language: str) -> list[FittedTopic]:
        """One language's stored topics."""
        return next((one.topics for one in self.stored if one.language == language), [])

    def labels_of(self, language: str) -> list[str | None]:
        """One language's stored labels, in topic order."""
        return [topic.label for topic in self.topics_of(language)]

    @property
    def failure(self) -> str | None:
        """Why the last fit failed, if it did."""
        return next(iter(self.queue.failures.values()), None)

    @property
    def figures(self) -> dict[str, bytes]:
        """The figures the fit drew, by language."""
        return {
            key.removeprefix("topics/").removesuffix(".html"): body
            for key, body in self.export.figures.items()
        }
