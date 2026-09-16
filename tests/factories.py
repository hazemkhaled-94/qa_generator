"""Builders the tests share."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from database.qa_generator import Difficulty, FactKind, QuestionType
from extraction.models import PassageToExtract
from preprocessing.chunking.models import Chunking
from preprocessing.chunking.passages import chunking_of
from question_generation.models import (
    Candidate,
    FactGroup,
    SourceFact,
    SourcePassage,
)
from question_generation.planning import Plan
from question_generation.selection import Shape
from question_generation.types import SPECS

#: Two sentences, the first carrying two claims and the second a reference.
TEXT = (
    "The device weighs 4 kg and runs for 12 hours. "
    "It arrives in March 2026 from the Hamburg plant."
)


def passage(text: str = TEXT, language: str = "en", **kwargs: Any) -> PassageToExtract:
    """Builds a passage with its sentences numbered, as chunking stores them."""
    from nlp.analysis import sentences as split

    given = kwargs.pop("sentences", ...)
    return PassageToExtract(
        id=1,
        text=text,
        section_path=kwargs.pop("section_path", None),
        block_type=kwargs.pop("block_type", None),
        language=language,
        sentences=split(text, language) if given is ... else given,
        table_cells=kwargs.pop("table_cells", []),
    )


def source(
    fact_id: int = 1,
    document: str = "doc-a",
    passage_id: int = 1,
    statement: str = "The device weighs 4 kg.",
    language: str = "en",
    passage_text: str | None = None,
    section_path: str | None = None,
    topic_id: int | None = None,
    document_title: str | None = None,
    ordinal: int = 0,
    lemmas: tuple[str, ...] = (),
    units: tuple[str, ...] = (),
    kind: str = FactKind.ATOMIC,
    rests_on: tuple[SourcePassage, ...] = (),
) -> SourceFact:
    """One validated fact, as question generation reads it off a topic.

    The passage and its heading are what the writer is shown, so a test
    about phrasing sets them and one about sampling does not. `lemmas` and
    `ordinal` are what pairs two passages, so a test about the deal sets
    those. `rests_on` names the passages besides the anchor; use `bridge()`
    rather than passing it by hand.
    """
    return SourceFact(
        id=fact_id,
        statement=statement,
        kind=kind,
        passages=(
            SourcePassage(
                id=passage_id,
                text=passage_text or f"{statement} It ships from Hamburg.",
                doc_sha256=document,
                language=language,
                section_path=section_path,
                topic_id=topic_id,
                document_title=document_title,
                ordinal=ordinal or passage_id,
                lemmas=lemmas,
            ),
            *rests_on,
        ),
        units=units,
    )


def resting(
    passage_id: int,
    document: str = "doc-b",
    text: str | None = None,
    **columns: Any,
) -> SourcePassage:
    """One further passage a bridge rests on."""
    return SourcePassage(
        id=passage_id,
        text=text or "Urgent requests are answered within 4 hours.",
        doc_sha256=document,
        language=columns.pop("language", "en"),
        section_path=columns.pop("section_path", None),
        topic_id=columns.pop("topic_id", None),
        document_title=columns.pop("document_title", None),
        ordinal=columns.pop("ordinal", passage_id),
        lemmas=columns.pop("lemmas", ()),
    )


def bridge(*passages: SourcePassage, **columns: Any) -> SourceFact:
    """One validated bridge, resting on its anchor and the passages given."""
    return source(
        kind=FactKind.BRIDGE,
        rests_on=passages or (resting(2),),
        **columns,
    )


def group(*facts: SourceFact) -> FactGroup:
    """The facts one question is written from, defaulting to one."""
    return FactGroup(facts or (source(),))


def candidate(
    question_text: str = "How heavy is the device built at the Hamburg plant?",
    target_answer: str | None = "4 kg",
    answerable: bool = True,
    facts: FactGroup | None = None,
    thread: tuple[tuple[str, str | None], ...] = (),
    question_type: str = QuestionType.FACTOID,
    planned_difficulty: str = Difficulty.EASY,
) -> Candidate:
    """A question as the model wrote it, before any gate has read it.

    The default clears every free gate, so a test about a later one is not
    stopped by an earlier one. In particular it names something its fact
    does not - the plant, from the passage - because a question that is
    about only what its fact is about is the defect `restates_fact` exists
    to catch, and `What does the device weigh?` was one.

    `question_type` decides which gates read the answer how: a factoid's is
    a value and may carry no verb, a reason's is an explanation and must.
    """
    return Candidate(
        question_text=question_text,
        target_answer=target_answer,
        answerable=answerable,
        group=facts if facts is not None else group(),
        thread=thread,
        spec=SPECS[question_type],
        planned_difficulty=planned_difficulty,
    )


def plan(
    question_type: str = QuestionType.FACTOID,
    band: str = Difficulty.EASY,
    shape: str = Shape.SINGLE,
    answerable: bool = True,
) -> Plan:
    """One slot of a topic's plan, as the writer is handed it."""
    return Plan(
        spec=SPECS[question_type], band=band, shape=shape, answerable=answerable
    )


class Chunk:
    """A stand-in carrying the attribute the chunk reader reads off a chunk."""

    def __init__(self, text: str) -> None:
        """Initialises the chunk with its text."""
        self.text = text


def _numbered(ordinal: int, chunk: Chunk) -> Any:
    """Turns a chunk into the little of a passage a test reads."""
    return type("P", (), {"ordinal": ordinal, "text": chunk.text.strip()})()


def chunked(texts: Iterable[str], max_tokens: int = 10) -> Chunking:
    """Runs the chunk reader over some chunk texts, counting a word as a token."""
    return chunking_of(
        [Chunk(text) for text in texts],
        max_tokens=max_tokens,
        count_tokens=lambda text: len(text.split()),
        to_passage=_numbered,
    )
