"""What the chunk reader stores and what it counts.

Nothing is discarded on size: a chunk over the budget is stored and counted,
because the chunker splits on the budget it counts against.
"""

from __future__ import annotations

import pytest
from factories import chunked

from preprocessing.chunking.models import Chunking
from preprocessing.chunking.passages import NoPassages


def test_every_chunk_is_stored_and_numbered() -> None:
    """Five chunks become five passages with contiguous ordinals."""
    result = chunked(("x", "y " * 12, "z", "w", "v"))
    assert isinstance(result, Chunking)
    assert len(result.passages) == 5, "a chunk over the budget is still stored"
    assert [p.ordinal for p in result.passages] == [1, 2, 3, 4, 5]


def test_a_chunk_over_the_budget_is_counted() -> None:
    """`oversized` reports the chunks above the token budget."""
    assert chunked(("x", "y " * 12, "z", "w", "v")).oversized == 1


def test_a_passage_is_stored_as_the_text_it_was_counted_on() -> None:
    """Surrounding whitespace is gone; every sentence offset indexes this."""
    assert chunked(("v",)).passages[0].text == "v"
    stored = chunked(("  hello\n",)).passages[0].text
    assert stored == "hello", repr(stored)


def test_a_chunk_holding_only_whitespace_is_dropped() -> None:
    """Blank chunks yield no passage."""
    assert chunked(("keep", "   ", "\n")).passages[0].text == "keep"
    assert len(chunked(("keep", "   ", "\n")).passages) == 1


def test_a_document_yielding_no_chunk_fails() -> None:
    """A document with nothing in it is refused rather than stored empty."""
    with pytest.raises(NoPassages, match="no passages"):
        chunked(("", "   "))
