"""Sentence numbering, which every citation resolves against."""

from __future__ import annotations

import pytest
from factories import passage

pytestmark = pytest.mark.nlp


def test_a_sentence_slices_its_own_text() -> None:
    """Each sentence's offsets index back to the sentence's own text."""
    built = passage()
    assert len(built.sentences) == 2, [s.text for s in built.sentences]
    for sentence in built.sentences:
        assert built.text[sentence.start : sentence.end] == sentence.text


def test_the_sentences_are_numbered_in_order() -> None:
    """Indices run from zero, and the passage counts what asserts."""
    built = passage()
    assert [s.index for s in built.sentences] == [0, 1]
    assert built.claims == 2, built.claims
