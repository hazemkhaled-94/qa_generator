"""Reading an NLI model's answer, under either head it may have.

No weights are loaded here. What is worth testing without them is the label
handling, because that is the part that fails SILENTLY: a checkpoint ordering
its outputs differently, read by position, inverts every verdict and still
returns three plausible probabilities that a threshold will happily compare.

The encoder against real weights is `tests/eval/`, which needs a model.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from nlp.entailment import Entailment, Verdict, positions, verdict

pytestmark = pytest.mark.nlp


@dataclass
class Config:
    """The one field of a checkpoint's config this reads."""

    id2label: dict[int, str]


THREE_WAY = Config({0: "entailment", 1: "neutral", 2: "contradiction"})
REVERSED = Config({0: "contradiction", 1: "neutral", 2: "entailment"})
TWO_WAY = Config({0: "entailment", 1: "not_entailment"})


def test_a_three_way_head_is_read_by_name_not_by_position() -> None:
    """mDeBERTa-xnli and bart-mnli order them opposite ways round."""
    row = torch.tensor([0.7, 0.2, 0.1])

    forward = verdict(row, positions(THREE_WAY, "a/checkpoint"))
    backward = verdict(row, positions(REVERSED, "another/checkpoint"))

    assert forward.entailment == pytest.approx(0.7)
    assert backward.entailment == pytest.approx(0.1)
    assert backward.contradiction == pytest.approx(0.7)


def test_a_two_way_head_reports_what_it_did_not_split() -> None:
    """bge-m3-zeroshot-v2.0 names entailment and not_entailment only.

    It cannot tell "says something different" from "does not address it",
    and both are refusals here, so the mass goes to neutral and
    contradiction stays 0.0 rather than being invented.
    """
    answered = verdict(torch.tensor([0.8, 0.2]), positions(TWO_WAY, "a/zeroshot"))

    assert answered.entailment == pytest.approx(0.8)
    assert answered.neutral == pytest.approx(0.2)
    assert answered.contradiction == 0.0


@pytest.mark.parametrize(
    "config",
    [
        Config({0: "positive", 1: "negative"}),
        Config({0: "neutral", 1: "contradiction"}),
        Config({}),
    ],
)
def test_a_head_that_is_not_an_nli_model_is_refused(config: Config) -> None:
    """Read by position it would score fluently and mean nothing."""
    with pytest.raises(ValueError, match="neither a three-way"):
        positions(config, "a/sentiment-model")


def test_the_refusal_names_what_the_checkpoint_did_say() -> None:
    """A deployment that named the wrong model has to be told which one."""
    with pytest.raises(ValueError, match="negative.*positive"):
        positions(Config({0: "positive", 1: "negative"}), "a/sentiment-model")


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
