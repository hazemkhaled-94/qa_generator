"""The embedding behind the dedup gate, against the real model.

Skipped unless EMBEDDING_MODEL is already in the Hugging Face cache. Nothing
here downloads it: it is 2.2 GB, and a test suite that fetches that on a cold
machine is a test suite nobody runs twice. `make dev` warms the cache the
first time the chunk worker or the question worker starts.
"""

from __future__ import annotations

import pytest

from question_generation.embedding import WIDTH, Embedder, cosine

MODEL = "intfloat/multilingual-e5-large"


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    """The real embedder, or a skip when its weights are not here."""
    from transformers import AutoConfig

    try:
        AutoConfig.from_pretrained(MODEL, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - any failure is the same answer
        pytest.skip(f"{MODEL} is not in the cache: {exc}")
    return Embedder(MODEL, max_tokens=512)


def test_a_question_embeds_at_the_width_the_column_holds(embedder) -> None:
    """questions.embedding is vector(1024); a mismatch fails at insert."""
    assert len(embedder.embed("What does the device weigh?")) == WIDTH


def test_an_embedding_is_a_unit_vector(embedder) -> None:
    """Which is what lets the gates compare by dot product."""
    vector = embedder.embed("What does the device weigh?")

    assert cosine(vector, vector) == pytest.approx(1.0, abs=1e-5)


def test_a_paraphrase_scores_above_an_unrelated_question(embedder) -> None:
    """The whole basis of the dedup gate, in one comparison."""
    asked = embedder.embed("What does the device weigh?")
    same = embedder.embed("How heavy is the device?")
    other = embedder.embed("Who is responsible for the quarterly risk report?")

    assert cosine(asked, same) > cosine(asked, other)


def test_the_same_question_in_two_languages_is_close(embedder) -> None:
    """The model is multilingual because this corpus is."""
    english = embedder.embed("Who must produce the risk report?")
    german = embedder.embed("Wer muss den Risikobericht erstellen?")
    unrelated = embedder.embed("What does the device weigh?")

    assert cosine(english, german) > cosine(english, unrelated)


def test_the_model_is_as_wide_as_the_column_that_holds_it(embedder) -> None:
    """The guard in the constructor, seen from the other side.

    Not tested against a model of the wrong width: that would mean
    downloading a second one to watch a four-line `if` fire.
    """
    assert len(embedder.embed("anything at all")) == WIDTH


def test_cosine_needs_two_vectors_of_the_same_width() -> None:
    """A silent zip would compare the first half of one with all of another."""
    with pytest.raises(ValueError):
        cosine([1.0, 0.0], [1.0, 0.0, 0.0])
