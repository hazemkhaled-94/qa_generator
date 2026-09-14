"""Builders the tests share."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from extraction.models import PassageToExtract
from preprocessing.chunking.models import Chunking
from preprocessing.chunking.passages import chunking_of
from question_generation.models import Candidate, FactGroup, SourceFact

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
) -> SourceFact:
    """One validated fact, as question generation reads it off a topic.

    The passage is what the verifier is shown, so a test about whether an
    answer is recoverable sets it and one about grouping does not.
    """
    return SourceFact(
        id=fact_id,
        statement=statement,
        passage_id=passage_id,
        passage_text=passage_text or f"{statement} It ships from Hamburg.",
        doc_sha256=document,
        language=language,
    )


def group(*facts: SourceFact) -> FactGroup:
    """The facts one question is written from, defaulting to one."""
    return FactGroup(facts or (source(),))


def candidate(
    question_text: str = "What does the device weigh?",
    target_answer: str | None = "4 kg",
    answerable: bool = True,
    facts: FactGroup | None = None,
) -> Candidate:
    """A question as the model wrote it, before any gate has read it."""
    return Candidate(
        question_text=question_text,
        target_answer=target_answer,
        answerable=answerable,
        group=facts if facts is not None else group(),
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
