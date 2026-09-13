"""Builders the tests share."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from extraction.models import PassageToExtract
from preprocessing.chunking.models import Chunking
from preprocessing.chunking.passages import chunking_of

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
