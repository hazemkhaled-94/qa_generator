"""Finding an answer's span in a passage, with an encoder rather than a model.

Importing this loads torch and a model's weights. Nothing outside a worker
should name it, which `tests/static/test_api_stays_light.py` pins, and it sits
beside `nlp.embedding` and `nlp.entailment` for the same reason.

This is SQuAD 2.0 exactly: given a question and a context, return the answer
span or say there is none. A model trained on that task has a no-answer head,
so "the passage does not answer this" is an answer it was taught to give
rather than a refusal it has to be talked into - which is the thing the
verifier's prompt spends four lines on.

**It pre-filters; it does not replace.** The verifier is allowed to compose an
answer the passages spread over two sentences, or state in one place and
qualify in another, and an extractive model cannot: every answer it gives is
one contiguous span of one passage. So a confident span is taken and the
served model is not called, and everything else falls through to it. What that
buys is the easy majority answered locally in milliseconds, with the hard
minority still answered by something that can read.

The span decode is written out here rather than left to
`pipeline("question-answering")`, which **transformers 5 removed**: the task
is not in `SUPPORTED_TASKS` any more and asking for it raises. What the
pipeline did is below - the null score against the best start/end pair, the
offsets back to characters, a window that slides so a passage longer than the
encoder is not silently cut off at its first 512 tokens.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from nlp.windows import window

log = logging.getLogger(__name__)

#: How many tokens of one window the next one repeats. A span sitting on a
#: window boundary is whole in one of the two.
_STRIDE = 128

#: The longest answer worth returning, in tokens. A SQuAD-style head scores
#: every start against every end, and without a cap the best pair is often
#: most of the passage - which is not an answer to anything.
_MAX_ANSWER_TOKENS = 40

#: How many starts and ends are paired. The decode is O(k^2) per window and
#: the logits past the twentieth are not the answer.
_TOP_K = 20


@dataclass(frozen=True)
class Span:
    """One answer found in one passage.

    Attributes:
        text: The span, exactly as the passage writes it.
        score: How sure the model is against answering nothing, in [0, 1].
        passage: Which passage it was found in, by position.
    """

    text: str
    score: float
    passage: int


class Extractive:
    """One extractive QA model, asked for a span and allowed to find none."""

    def __init__(self, model_name: str, max_tokens: int = 384) -> None:
        """Names the model to load. Nothing is loaded until it is asked.

        Deferred, as `nlp.entailment` defers: a worker whose questions are
        all refused structurally should not pay for weights it never uses.
        """
        self._model_name = model_name
        self._max_tokens = max_tokens

    @cached_property
    def _loaded(self):
        """The tokenizer and the model, held to the window the model has."""
        log.info("loading %s", self._model_name)
        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(self._model_name)
        model.eval()
        self._max_tokens = window(tokenizer, self._max_tokens)
        return tokenizer, model

    @property
    def model(self) -> str:
        """The model doing the reading, recorded beside what it read."""
        return self._model_name

    def answer(
        self, question: str, passages: Sequence[str], confidence: float
    ) -> Span | None:
        """The best span any one passage gives, or None for no answer.

        One passage per call and the best of them wins, for the reason
        `nlp.entailment` judges one at a time: a context of several passages
        joined is longer than the window and is not the question being
        asked. An answer is in the material when SOME passage holds it.

        Args:
            question: The question to answer.
            passages: The passages it cites, in citation order.
            confidence: The lowest score worth taking. Below it the caller
                falls through to something that can read.

        Returns:
            The best span at or above `confidence`, or None.
        """
        best: Span | None = None
        for position, passage in enumerate(passages):
            if not passage.strip():
                continue
            found = self._span(question, passage)
            if found is None:
                continue
            text, score = found
            if score < confidence:
                continue
            if best is None or score > best.score:
                best = Span(text=text, score=score, passage=position)
        return best

    def _span(self, question: str, context: str) -> tuple[str, float] | None:
        """The best answer in one passage, and how sure the model is of it.

        Returns None when the model would rather answer nothing, which is
        what the null span at position 0 means and what a SQuAD 2.0 head was
        trained to say.

        The score is a softmax over exactly two numbers - the best span and
        the null - so it reads as "how sure against saying nothing" rather
        than as a share of the whole vocabulary of spans.
        """
        tokenizer, model = self._loaded
        encoded = tokenizer(
            question,
            context,
            max_length=self._max_tokens,
            # The question is kept whole and the passage is what slides. A
            # question truncated to fit is a different question.
            truncation="only_second",
            stride=_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        encoded.pop("overflow_to_sample_mapping", None)
        with torch.no_grad():
            answered = model(**encoded)

        null = float("inf")
        best: tuple[float, str] | None = None
        for index in range(answered.start_logits.shape[0]):
            # Where the passage sits in this window. `None` is a special
            # token and `0` is the question; a span may only come from `1`.
            inside = [
                at for at, side in enumerate(encoded.sequence_ids(index)) if side == 1
            ]
            if not inside:
                continue
            starts = answered.start_logits[index]
            ends = answered.end_logits[index]
            null = min(null, float(starts[0]) + float(ends[0]))
            found = self._pair(starts, ends, inside, offsets[index], context)
            if found is not None and (best is None or found[0] > best[0]):
                best = found

        if best is None or null == float("inf"):
            return None
        score, text = best
        if score <= null or not text.strip():
            return None
        against = torch.softmax(torch.tensor([null, score]), dim=0)
        return text.strip(), float(against[1])

    @staticmethod
    def _pair(starts, ends, inside: list[int], offsets, context: str):
        """The best start/end pair inside one window, and the text it covers."""
        top_starts = [at for at in _best_of(starts, inside)]
        top_ends = [at for at in _best_of(ends, inside)]
        best: tuple[float, str] | None = None
        for start in top_starts:
            for end in top_ends:
                if end < start or end - start + 1 > _MAX_ANSWER_TOKENS:
                    continue
                score = float(starts[start]) + float(ends[end])
                if best is not None and score <= best[0]:
                    continue
                first, last = int(offsets[start][0]), int(offsets[end][1])
                if last <= first:
                    continue
                best = (score, context[first:last])
        return best


def _best_of(logits, inside: list[int]) -> list[int]:
    """The `_TOP_K` highest-scoring positions that sit in the passage."""
    scored = sorted(inside, key=lambda at: float(logits[at]), reverse=True)
    return scored[:_TOP_K]
