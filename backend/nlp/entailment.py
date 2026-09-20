"""Entailment, in this process, from an encoder rather than a served model.

Importing this loads torch and a model's weights. Nothing outside a worker
should name it, which `tests/static/test_api_stays_light.py` pins, and it
sits beside `nlp.embedding` for the same reason: loading a model is this
package's, and a service may not import another service to reach one.

One call answers ONE question. An NLI model is given a premise and a
hypothesis and returns three probabilities over them, and that is the whole
of what it knows. A caller wanting two judgements makes two calls - the thing
being replaced is a prompt that asked for four at once and confused them,
which is the failure this shape exists to prevent.

The model is named by a setting rather than written here. Which encoder reads
German best is a deployment's measurement to make and to re-make, and this
module has no opinion beyond the label order it reads off the model's own
config.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

log = logging.getLogger(__name__)

#: The three things an NLI model can say about a premise and a hypothesis.
#: Read off the model's own `id2label` rather than assumed: the order differs
#: between checkpoints, and assuming it silently inverts every verdict.
ENTAILMENT = "entailment"
NEUTRAL = "neutral"
CONTRADICTION = "contradiction"

#: What a two-way head calls everything that is not entailment. A model
#: trained this way cannot tell "says something different" from "does not
#: address it", and both are refusals here, so the distinction costs this
#: caller nothing - see `Verdict.contradiction`.
NOT_ENTAILMENT = "not_entailment"


def positions(config, model_name: str) -> dict[str, int]:
    """Where each label sits in one checkpoint's output.

    Read off the checkpoint's own `id2label` rather than assumed: the order
    differs between checkpoints - mDeBERTa-xnli runs entailment/neutral/
    contradiction and bart-mnli the other way round - and a hard-coded order
    inverts every verdict on half the models anybody would configure, while
    still returning three plausible probabilities.

    Two heads are accepted and both are wanted. A three-way NLI model names
    all of entailment/neutral/contradiction; a zero-shot head - such as
    bge-m3-zeroshot-v2.0, whose 8,194-token window is the reason to want it,
    since a premise here is a passage already sized to 512 - names
    entailment and not_entailment only. Either is enough, because the only
    thing a gate asks is how sure the model is of entailment.

    Raises:
        ValueError: If the checkpoint names neither, which means it is not
            an NLI model and would otherwise be read by position.
    """
    names = {
        index: str(label).lower()
        for index, label in getattr(config, "id2label", {}).items()
    }
    at = {label: index for index, label in names.items()}
    if ENTAILMENT not in at or not ({NEUTRAL, NOT_ENTAILMENT} & set(at)):
        raise ValueError(
            f"{model_name!r} labels its outputs {sorted(names.values())}, which "
            f"names neither a three-way NLI head "
            f"({ENTAILMENT}/{NEUTRAL}/{CONTRADICTION}) nor a two-way one "
            f"({ENTAILMENT}/{NOT_ENTAILMENT}). This asks one about a premise "
            f"and a hypothesis."
        )
    return at


@dataclass(frozen=True)
class Verdict:
    """What one premise says about one hypothesis.

    Attributes:
        entailment: Probability the premise supports the hypothesis.
        neutral: Probability it neither supports nor contradicts it.
        contradiction: Probability it says the opposite. Always 0.0 from a
            two-way head, which cannot tell this from neutral. Nothing here
            reads it - `supports` is the whole of what a gate asks - and it
            is kept because a three-way model knows it and a reader
            debugging a refusal wants to see it.
    """

    entailment: float
    neutral: float
    contradiction: float

    @property
    def label(self) -> str:
        """Whichever of the three is likeliest."""
        return max(
            (
                (self.entailment, ENTAILMENT),
                (self.neutral, NEUTRAL),
                (self.contradiction, CONTRADICTION),
            )
        )[1]

    def supports(self, threshold: float) -> bool:
        """Whether the premise backs the hypothesis at this confidence.

        A threshold rather than the likeliest label, which is the reason to
        prefer this to a model answering true or false: a boolean from a
        served model carries no confidence, so a deployment cannot decide how
        sure it wants a gate to be.
        """
        return self.entailment >= threshold


def verdict(row, at: dict[str, int]) -> Verdict:
    """One row of probabilities, under whichever head produced it.

    Args:
        row: The softmaxed logits of one pair.
        at: Where each label sits, as `positions` read it.

    Returns:
        The verdict. A two-way head reports everything that is not
        entailment as neutral rather than splitting it, because the model
        did not split it.
    """
    entailment = float(row[at[ENTAILMENT]])
    if NEUTRAL in at:
        return Verdict(
            entailment=entailment,
            neutral=float(row[at[NEUTRAL]]),
            contradiction=float(row[at[CONTRADICTION]]),
        )
    return Verdict(
        entailment=entailment,
        neutral=float(row[at[NOT_ENTAILMENT]]),
        contradiction=0.0,
    )


class Entailment:
    """One NLI model, asked about one premise and one hypothesis at a time."""

    def __init__(self, model_name: str, max_tokens: int = 512) -> None:
        """Names the model to load. Nothing is loaded until it is asked.

        Deferred, so building a service that may never reach this gate costs
        nothing: a worker draining a queue of questions that are all refused
        structurally should not pay for weights it never uses.
        """
        self._model_name = model_name
        self._max_tokens = max_tokens

    @cached_property
    def _loaded(self):
        """The tokenizer, the model, and where each label sits in its output."""
        log.info("loading %s", self._model_name)
        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
        model.eval()

        # Read off the checkpoint. mDeBERTa-xnli orders them
        # entailment/neutral/contradiction and bart-mnli the other way round;
        # a hard-coded order inverts every verdict on half the models anybody
        # would configure, and inverts them silently.
        return tokenizer, model, positions(model.config, self._model_name)

    @property
    def model(self) -> str:
        """The model doing the judging, recorded on what it judges."""
        return self._model_name

    def judge(self, premise: str, hypothesis: str) -> Verdict:
        """What one premise says about one hypothesis.

        Args:
            premise: The text taken as given - a passage, or several joined.
            hypothesis: The single proposition being judged against it.

        Returns:
            The three probabilities, which sum to 1.
        """
        return self.judge_all([(premise, hypothesis)])[0]

    def judge_all(self, pairs: list[tuple[str, str]]) -> list[Verdict]:
        """The same, over several pairs in one forward pass.

        Args:
            pairs: One (premise, hypothesis) per judgement wanted.

        Returns:
            One verdict per pair, in the order given.
        """
        if not pairs:
            return []
        tokenizer, model, at = self._loaded
        tokens = tokenizer(
            [premise for premise, _ in pairs],
            [hypothesis for _, hypothesis in pairs],
            max_length=self._max_tokens,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            scores = torch.softmax(model(**tokens).logits, dim=-1)
        return [verdict(row, at) for row in scores]
