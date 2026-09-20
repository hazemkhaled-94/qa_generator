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


@dataclass(frozen=True)
class Verdict:
    """What one premise says about one hypothesis.

    Attributes:
        entailment: Probability the premise supports the hypothesis.
        neutral: Probability it neither supports nor contradicts it.
        contradiction: Probability it says the opposite.
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
        names = {
            index: str(label).lower()
            for index, label in getattr(model.config, "id2label", {}).items()
        }
        missing = {ENTAILMENT, NEUTRAL, CONTRADICTION} - set(names.values())
        if missing:
            raise ValueError(
                f"{self._model_name!r} labels its outputs {sorted(names.values())}, "
                f"which does not name {sorted(missing)}. It is not a three-way NLI "
                f"model, and this asks one for a premise and a hypothesis."
            )
        return tokenizer, model, {label: index for index, label in names.items()}

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
        return [
            Verdict(
                entailment=float(row[at[ENTAILMENT]]),
                neutral=float(row[at[NEUTRAL]]),
                contradiction=float(row[at[CONTRADICTION]]),
            )
            for row in scores
        ]
