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
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Span:
    """One answer found in one passage.

    Attributes:
        text: The span, exactly as the passage writes it.
        score: How sure the model is, in [0, 1].
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
    def _pipeline(self) -> Any:
        """The transformers pipeline, built once.

        Any, because `pipeline` is overloaded per task and a checker cannot
        narrow a task named by a string literal to the class that serves it.

        The pipeline rather than `AutoModelForQuestionAnswering` directly, as
        `nlp.embedding` and `nlp.entailment` both use their model classes.
        The difference is what the extra code would be: pooling and a dot
        product for those two, and for this one the whole span decode - the
        null score against the best start/end pair, the offsets back to
        characters, the maximum answer length. That is the pipeline's job and
        it is where a hand-rolled version would get SQuAD 2.0's no-answer
        case subtly wrong.
        """
        from transformers import pipeline

        log.info("loading %s", self._model_name)
        # The overloads this version ships do not list the QA task, so the
        # checker narrows `task` to the last literal it saw.
        task: Any = "question-answering"
        return pipeline(task, model=self._model_name)

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
            found = self._pipeline(
                question=question,
                context=passage,
                # What makes None reachable. Without it a SQuAD 2.0 model is
                # forced to name its least-bad span, and a gate reading that
                # would find an answer in every passage ever shown to it.
                handle_impossible_answer=True,
                max_seq_len=self._max_tokens,
            )
            text = (found.get("answer") or "").strip()
            score = float(found.get("score") or 0.0)
            if not text or score < confidence:
                continue
            if best is None or score > best.score:
                best = Span(text=text, score=score, passage=position)
        return best
