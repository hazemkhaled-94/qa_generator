"""Reading an NLI model's answer, under either head it may have.

No weights are loaded here. What is worth testing without them is the label
handling, because that is the part that fails SILENTLY: a checkpoint whose
labels are not an NLI model's scores every pair fluently and means nothing
by it, and a threshold will happily compare the numbers.

Read by NAME throughout. The transformers pipeline hands back the label
beside each score, so the order a checkpoint declares its outputs in - which
mDeBERTa-xnli and bart-mnli disagree about - cannot invert a verdict any
more. These cases stay because the labels still have to be understood.

The encoder against real weights is `tests/eval/`, which needs a model.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from nlp.entailment import Entailment, Verdict, verdict

pytestmark = pytest.mark.nlp

THREE_WAY = {"entailment": 0.7, "neutral": 0.2, "contradiction": 0.1}
REVERSED = {"contradiction": 0.7, "neutral": 0.2, "entailment": 0.1}
TWO_WAY = {"entailment": 0.8, "not_entailment": 0.2}


def test_a_three_way_head_is_read_by_name_not_by_position() -> None:
    """mDeBERTa-xnli and bart-mnli order them opposite ways round."""
    forward = verdict(THREE_WAY, "a/checkpoint")
    backward = verdict(REVERSED, "another/checkpoint")

    assert forward.entailment == pytest.approx(0.7)
    assert backward.entailment == pytest.approx(0.1)
    assert backward.contradiction == pytest.approx(0.7)


def test_a_label_is_read_whatever_case_the_checkpoint_wrote_it_in() -> None:
    """Some checkpoints shout them: ENTAILMENT, NEUTRAL, CONTRADICTION."""
    answered = verdict(
        {"ENTAILMENT": 0.6, "NEUTRAL": 0.3, "CONTRADICTION": 0.1}, "a/shouting-model"
    )

    assert answered.entailment == pytest.approx(0.6)
    assert answered.label == "entailment"


def test_a_two_way_head_reports_what_it_did_not_split() -> None:
    """bge-m3-zeroshot-v2.0 names entailment and not_entailment only.

    It cannot tell "says something different" from "does not address it",
    and both are refusals here, so the mass goes to neutral and
    contradiction stays 0.0 rather than being invented.
    """
    answered = verdict(TWO_WAY, "a/zeroshot")

    assert answered.entailment == pytest.approx(0.8)
    assert answered.neutral == pytest.approx(0.2)
    assert answered.contradiction == 0.0


@pytest.mark.parametrize(
    "scores",
    [
        {"positive": 0.9, "negative": 0.1},
        {"neutral": 0.9, "contradiction": 0.1},
        {},
    ],
)
def test_a_head_that_is_not_an_nli_model_is_refused(scores: dict) -> None:
    """Scored anyway it would answer fluently and mean nothing."""
    with pytest.raises(ValueError, match="neither a three-way"):
        verdict(scores, "a/sentiment-model")


def test_the_refusal_names_what_the_checkpoint_did_say() -> None:
    """A deployment that named the wrong model has to be told which one."""
    with pytest.raises(ValueError, match="negative.*positive"):
        verdict({"positive": 0.9, "negative": 0.1}, "a/sentiment-model")


def test_a_threshold_decides_support_rather_than_the_likeliest_label() -> None:
    """The reason to prefer an encoder to a model answering true or false.

    A boolean carries no confidence, so a deployment cannot decide how sure
    it wants a gate to be. Here the likeliest label is `neutral` and the
    caller still gets to choose.
    """
    answered = Verdict(entailment=0.45, neutral=0.5, contradiction=0.05)

    assert answered.label == "neutral"
    assert answered.supports(0.4)
    assert not answered.supports(0.5)


def test_nothing_to_judge_costs_no_forward_pass() -> None:
    """A gate with no passages must not load weights to say so.

    The model name is never resolved, so this also pins that construction
    alone downloads nothing.
    """
    assert Entailment("a/checkpoint-that-does-not-exist").judge_all([]) == []


# ── The window one setting gives two encoders ─────────────────────────────


@dataclass
class Tokenizer:
    """The one field `window` reads."""

    model_max_length: int
    name_or_path: str = "a/checkpoint"


def test_a_model_is_held_to_its_own_window() -> None:
    """ENCODER_MAX_TOKENS is one setting over encoders that disagree.

    bge-m3-zeroshot-v2.0 reads 8,192 and xlm-roberta-large-squad2 reads
    512, so a deployment raising the setting for the first would hand the
    second a length its position embeddings do not have.
    """
    from nlp.windows import window

    assert window(Tokenizer(512), 8192) == 512


def test_a_window_the_model_allows_is_left_alone() -> None:
    """The setting is a ceiling, not an instruction."""
    from nlp.windows import window

    assert window(Tokenizer(8192), 512) == 512
    assert window(Tokenizer(8192), 8192) == 8192


def test_a_tokenizer_declaring_nothing_is_not_believed() -> None:
    """A checkpoint that declares no window is not taken at its word.

    `model_max_length` comes back as a near-1e30 sentinel when the
    checkpoint is silent, and truncating to that truncates nothing.
    """
    from nlp.windows import window

    assert window(Tokenizer(1_000_000_000_000_000_019_884_624_838_656), 512) == 512
