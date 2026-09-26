"""Keeping a fitted model, and reading it back.

The factorisation is the one thing a fit produces that nothing can
recompute - the database keeps each topic's top terms and not the matrix
they were read off - so what is pinned here is that a stored model comes
back as the model that went in, and that a file this cannot read is
refused rather than guessed at.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from topic_modelling.models import TopicSpace
from topic_modelling.space import FORMAT, dump, load


def space(**overrides) -> TopicSpace:
    """One small fitted model: two topics over three passages."""
    fields = {
        "topic_term": [[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]],
        "doc_topic": [[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]],
        "doc_lengths": [12, 30, 7],
        "vocabulary": ["anfrage", "frist", "prüfung"],
        "term_frequency": [40, 18, 6],
    }
    return TopicSpace(**{**fields, **overrides})


def test_a_stored_model_comes_back_as_it_went_in() -> None:
    """Every array, and the values in them."""
    read = load(dump(space()))

    assert read == space()


def test_a_long_term_is_not_truncated() -> None:
    """The array is as wide as the longest term, not a guessed width.

    Half this corpus is German, where a compound noun runs to forty
    characters; a fixed width would cut one and leave a vocabulary that
    looks fine until somebody reads it.
    """
    word = "Verkehrsinfrastrukturfinanzierungsgesellschaft"
    read = load(dump(space(vocabulary=[word, "frist", "prüfung"])))

    assert read.vocabulary[0] == word


def test_the_matrices_survive_as_floats() -> None:
    """A weight read back as an integer is a topic nobody can rank."""
    read = load(dump(space()))

    assert read.topic_term[0][1] == pytest.approx(0.2)
    assert read.doc_topic[2][1] == pytest.approx(0.8)


def test_a_file_from_another_format_is_refused() -> None:
    """Read as this one it would draw a figure off misaligned axes.

    Which looks like a model rather than like an error, so it is refused
    at the boundary instead.
    """
    buffer = io.BytesIO()
    np.savez_compressed(buffer, format=np.array(FORMAT + 1))

    with pytest.raises(ValueError, match="format"):
        load(buffer.getvalue())


def test_nothing_in_the_file_is_a_pickle() -> None:
    """`allow_pickle=False` is what makes a stored model safe to load.

    The reason this is .npz and not `Nmf.save()`: gensim's own format is a
    pickle, and a pickle is both a promise about the writing class and an
    arbitrary-code hazard on the way back in.
    """
    with np.load(io.BytesIO(dump(space())), allow_pickle=False) as held:
        assert set(held) == {
            "format",
            "topic_term",
            "doc_topic",
            "doc_lengths",
            "vocabulary",
            "term_frequency",
        }
