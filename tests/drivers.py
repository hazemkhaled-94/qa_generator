"""The objects the facts tests drive the extraction service through.

One driver per unit under test, each exposing what a caller does with that
unit rather than how it is built: a test says `checker.summary(...)` and
never assembles a `CandidateFact`. The same idea as a page object, for a
service with no pages.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from factories import passage as build_passage

from database.qa_generator import FactKind
from extraction.models import CandidateFact, CheckedFact, Cited, PassageToExtract
from extraction.validation import FactChecker

#: The share a digest must come in under. Stated here rather than read from
#: the environment, so a test asserts against a number it can see.
DIGEST_SHARE = 0.6


def passage(text: str | None = None, **kwargs: Any) -> PassageToExtract:
    """Builds one passage, numbered as chunking stores it.

    Args:
        text: What it says. The two-sentence default when not given.
        **kwargs: Anything `factories.passage` takes, plus `id` and
            `doc_sha256`.

    Returns:
        The passage.
    """
    identifier = kwargs.pop("id", 1)
    document = kwargs.pop("doc_sha256", None)
    built = build_passage(**kwargs) if text is None else build_passage(text, **kwargs)
    return replace(built, id=identifier, doc_sha256=document)


def group(*texts: str, language: str = "en") -> list[PassageToExtract]:
    """Builds a group of passages, one per document and numbered from 11.

    Args:
        *texts: What each passage says.
        language: The language to read them in.

    Returns:
        The passages, in the order a bridge candidate names them.
    """
    return [
        passage(text, language=language, id=11 * (at + 1), doc_sha256=f"doc-{at}")
        for at, text in enumerate(texts)
    ]


#: The Markdown a table passage renders as. The Standard row is in the table
#: but not in this passage, so the chunker kept no line for it.
RENDERED = "Table 1: Models\n\n| Model | Mass |\n| --- | --- |\n| Compact | 4 |\n"


def cell(row: int, col: int, text: str, line: int | None, **flags: bool) -> dict:
    """Builds one stored table cell.

    Args:
        row: Its row in the grid, from 0.
        col: Its column, from 0.
        text: What it holds.
        line: The rendered Markdown row it sits in, or None when the passage
            does not render it.
        **flags: `column_header` or `row_header`.

    Returns:
        The cell as chunking stores it.
    """
    return {
        "row": row,
        "col": col,
        "row_span": 1,
        "col_span": 1,
        "column_header": flags.get("column_header", False),
        "row_header": flags.get("row_header", False),
        "text": text,
        "line": line,
    }


#: One two-column table, one value cell of which the passage renders.
GRID: dict[str, Any] = {
    "caption": "Table 1: Models",
    "num_rows": 3,
    "num_cols": 2,
    "cells": [
        cell(0, 0, "Model", None, column_header=True),
        cell(0, 1, "Mass", None, column_header=True),
        cell(2, 0, "Compact", 3, row_header=True),
        cell(2, 1, "4", 3),
    ],
}


def table_passage(grid: dict | None = None) -> PassageToExtract:
    """Builds a table passage, numbered by line as chunking stores it.

    Args:
        grid: The cell grid behind it. The one above when not given.

    Returns:
        The passage.
    """
    from nlp.models import Sentence
    from preprocessing.chunking.passages import lines_of

    return PassageToExtract(
        id=1,
        text=RENDERED,
        section_path="Models",
        block_type="table",
        language="en",
        sentences=[
            Sentence(
                index=line["i"],
                start=line["start"],
                end=line["end"],
                text=RENDERED[line["start"] : line["end"]],
                predicates=0,
            )
            for line in lines_of(RENDERED)
        ],
        table_cells=[grid if grid is not None else GRID],
    )


class Checker:
    """Puts statements to the fact checker without assembling candidates."""

    def __init__(
        self,
        under: PassageToExtract | None = None,
        *,
        share: float = DIGEST_SHARE,
    ) -> None:
        """Initialises the driver.

        Args:
            under: The passage single-passage kinds are checked against.
            share: The longest a digest may be, as a share of its passage.
        """
        self.passage = under if under is not None else passage()
        self.checker = FactChecker(share)

    def atomic(self, statement: str, cites: int | tuple[int, ...] = 0) -> CheckedFact:
        """Checks one claim drawn from the passage's numbered sentences."""
        return self._single(statement, cites, FactKind.ATOMIC, "llm")

    def composed(self, statement: str, cites: int | tuple[int, ...] = 0) -> CheckedFact:
        """Checks a statement a deterministic reader composed from a grid."""
        return self._single(statement, cites, FactKind.ATOMIC, "deterministic")

    def summary(self, statement: str) -> CheckedFact:
        """Checks prose standing in for the whole passage."""
        return self._digest(statement, FactKind.SUMMARY)

    def outline(self, *points: str) -> CheckedFact:
        """Checks bullet points standing in for the whole passage."""
        from extraction.extractors.digest import BULLET

        return self._digest(
            "\n".join(f"{BULLET}{point}" for point in points), FactKind.OUTLINE
        )

    def raw_outline(self, statement: str) -> CheckedFact:
        """Checks an outline given as one already-joined string."""
        return self._digest(statement, FactKind.OUTLINE)

    def bridge(
        self,
        statement: str,
        offered: list[PassageToExtract],
        rests_on: tuple[int, ...] = (0, 1),
        cites: dict[int, tuple[int, ...]] | None = None,
    ) -> CheckedFact:
        """Checks a claim drawn from a group of passages.

        Args:
            statement: What the model wrote.
            offered: The passages it was shown.
            rests_on: Which of them it named, by position.
            cites: Which sentences of each named position, when a test cares.
                Every position cites its sentence 0 otherwise.

        Returns:
            The checked fact.
        """
        named = cites or {}
        return self.checker.check_bridge(
            offered,
            CandidateFact(
                statement,
                (),
                kind=FactKind.BRIDGE,
                passages=tuple(
                    Cited(position=position, sentences=named.get(position, (0,)))
                    for position in rests_on
                ),
            ),
        )

    def _single(
        self, statement: str, cites: int | tuple[int, ...], kind: str, method: str
    ) -> CheckedFact:
        """Checks a candidate citing sentences of the one passage."""
        named = (cites,) if isinstance(cites, int) else cites
        return self.checker.check(
            self.passage, CandidateFact(statement, named, kind=kind), method
        )

    def _digest(self, statement: str, kind: str) -> CheckedFact:
        """Checks a candidate standing in for the whole passage."""
        return self.checker.check(
            self.passage,
            CandidateFact(
                statement,
                tuple(sentence.index for sentence in self.passage.sentences),
                kind=kind,
            ),
            "llm",
        )


class Model:
    """A served model that answers from a script and records the ask.

    Answers by building whatever shape the caller asked for out of the
    payload it was given, so a test that sends the wrong shape fails here
    rather than three layers down.
    """

    model = "ollama/test-model"
    temperature = 0.0

    def __init__(self, answer: dict | Exception | None = None) -> None:
        """Initialises the model.

        Args:
            answer: The keyword arguments to build the answer shape from, or
                an exception to raise instead. An empty answer when None.
        """
        self._answer = {} if answer is None else answer
        self.asked: list[dict] = []

    @property
    def sent(self) -> dict:
        """The last thing asked of it."""
        return self.asked[-1]

    @property
    def calls(self) -> int:
        """How many times it was asked."""
        return len(self.asked)

    def answer(self, *, system: str, user: str, shape):
        """Answers, and records what it was sent.

        Raises:
            Exception: Whatever it was built with, when that is one.
        """
        self.asked.append({"system": system, "user": user, "shape": shape})
        if isinstance(self._answer, Exception):
            raise self._answer
        return shape(**self._answer)


class Queue:
    """A passage queue held in memory, recording what was written to it."""

    done = "extracted"

    def __init__(self, *passages: PassageToExtract) -> None:
        """Initialises the queue with the passages waiting on it."""
        self._pending = list(passages)
        self.stored: dict[int, list[CheckedFact]] = {}
        self.failed: dict[int, str] = {}
        self.swept = 0

    def claim(self) -> PassageToExtract | None:
        """Takes the next passage off the queue."""
        return self._pending.pop(0) if self._pending else None

    def store(self, passage_id: int, facts: list[CheckedFact]) -> int:
        """Records what one passage yielded."""
        self.stored[passage_id] = facts
        return len(facts)

    def fail(self, passage_id: int, error: str) -> None:
        """Records a passage the service could not read."""
        self.failed[passage_id] = error

    def abandon(self) -> int:
        """Reports how many claims a previous run left behind."""
        return self.swept

    @property
    def facts(self) -> list[CheckedFact]:
        """Everything written, in the order the passages were read."""
        return [fact for facts in self.stored.values() for fact in facts]

    def kinds(self) -> list[str]:
        """The kind of every fact written."""
        return [fact.kind for fact in self.facts]


class Catalogue:
    """A fact catalogue held in memory, for the bridge pass to write to."""

    def __init__(self, **topics: list[PassageToExtract]) -> None:
        """Initialises the catalogue.

        Args:
            **topics: The passages of each topic, keyed by anything; the
                topic ids are assigned from 1 in the order given.
        """
        self._topics = list(topics.values())
        self.written: list[CheckedFact] = []
        self.cleared = 0
        self.narrowed: list[Any] = []
        self.stored: list[tuple[int, list[PassageToExtract], CandidateFact, str]] = []
        self.rejudged: list[list[tuple[int, CheckedFact]]] = []

    def holding(self, *facts: tuple[int, list[PassageToExtract], CandidateFact, str]):
        """Fills the catalogue with facts a re-judgement will read back.

        Args:
            *facts: Each fact's id, its passages, what was proposed and how.

        Returns:
            This catalogue, so a test reads as one expression.
        """
        self.stored += facts
        return self

    def judged(self, within=None):
        """Streams every stored fact beside the passages it was drawn from."""
        self.narrowed.append(within)
        yield from self.stored

    def rejudge(self, verdicts: list[tuple[int, CheckedFact]]) -> int:
        """Records one batch of re-judged facts."""
        self.rejudged.append(list(verdicts))
        return len(verdicts)

    @property
    def verdicts(self) -> list[tuple[int, CheckedFact]]:
        """Every re-judged fact, across the batches it was written in."""
        return [one for batch in self.rejudged for one in batch]

    def by_topic(self, within=None):
        """Streams each topic's passages."""
        self.narrowed.append(within)
        yield from enumerate(self._topics, start=1)

    def clear_bridges(self, within=None) -> int:
        """Records that the previous pass's bridges were replaced."""
        self.cleared += 1
        return 0

    def add_bridges(self, facts: list[CheckedFact]) -> int:
        """Records the bridges one group produced."""
        self.written += facts
        return len(facts)

    @property
    def kinds(self) -> list[str]:
        """The kind of every fact written."""
        return [fact.kind for fact in self.written]


class Extraction:
    """The per-passage service, wired around scripted models."""

    def __init__(
        self,
        *passages: PassageToExtract,
        facts: dict | Exception | None = None,
        digest: dict | Exception | None = None,
        kinds: tuple[str, ...] = (),
    ) -> None:
        """Wires the service.

        Args:
            *passages: What the queue holds.
            facts: What the atomic model answers with.
            digest: What the digest model answers with.
            kinds: Which digest kinds to keep. No digest reader at all when
                empty.
        """
        from extraction.extractors import (
            DigestExtractor,
            ExtractorRegistry,
            LlmExtractor,
            TableExtractor,
        )
        from extraction.service import ExtractionService

        self.queue = Queue(*passages)
        self.model = Model(facts if facts is not None else {"facts": []})
        self.digest_model = Model(digest)
        self.service = ExtractionService(
            repository=self.queue,  # type: ignore[arg-type]
            extractors=ExtractorRegistry(
                extractors=(TableExtractor(),),
                default=LlmExtractor(self.model),  # type: ignore[arg-type]
            ),
            checker=FactChecker(DIGEST_SHARE),
            digest=DigestExtractor(self.digest_model, kinds)  # type: ignore[arg-type]
            if kinds
            else None,
        )

    def run(self) -> int:
        """Drains the queue and returns how many passages were read."""
        return self.service.drain()

    def next(self) -> int | None:
        """Reads one passage and returns its id."""
        return self.service.process_next()
