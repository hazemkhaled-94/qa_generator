"""Ordering candidates a cheap retrieval found, with a cross-encoder.

Importing this loads torch and a model's weights. Nothing outside a worker
should name it, which `tests/static/test_api_stays_light.py` pins, and it sits
beside `nlp.embedding` for the same reason.

A bi-encoder embeds the two sides apart and compares directions, which is what
makes an index possible and what makes it approximate: the two texts never
meet. A cross-encoder reads them together and scores the pair, which costs a
forward pass per candidate and cannot be indexed.

So this never retrieves. It **reorders what a retrieval already narrowed** -
the top handful out of `passages.embedding` - which is the one place the extra
forward passes are affordable and the ordering is what matters.

The default model is a causal one scored on a single token, which is how
Qwen3-Reranker is built: the pair goes into a chat template that asks a yes-or-
no question, and the answer is the probability mass on `yes` against `no` at
the first generated position. That template is the model's own and is written
out below rather than hidden, because a reranker given the wrong template
scores fluently and means nothing.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import cached_property

import torch

log = logging.getLogger(__name__)

#: What the pair is being judged for, unless a caller says otherwise. The
#: model was trained to read an instruction, and the default one it ships
#: with is about web search.
SEARCH = (
    "Given a search query, judge whether the document is relevant to it "
    "and would help answer it."
)

#: Qwen3-Reranker's own chat template, and the reason this module names a
#: default model at all. The scoring below reads one token, so the template
#: has to put the model exactly where that token is the answer.
_SYSTEM = (
    "<|im_start|>system\nJudge whether the Document meets the requirements "
    "based on the Query and the Instruct provided. Note that the answer can "
    'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
_ASSISTANT = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Reranker:
    """One cross-encoder, asked how well a document answers a query."""

    def __init__(self, model_name: str, max_tokens: int = 2048) -> None:
        """Names the model to load. Nothing is loaded until it is asked."""
        self._model_name = model_name
        self._max_tokens = max_tokens

    @cached_property
    def _loaded(self):
        """The tokenizer, the model, and the two token ids that are the answer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("loading %s", self._model_name)
        tokenizer = AutoTokenizer.from_pretrained(self._model_name, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(self._model_name)
        model.eval()
        yes = tokenizer.convert_tokens_to_ids("yes")
        no = tokenizer.convert_tokens_to_ids("no")
        if yes is None or no is None:
            raise ValueError(
                f"{self._model_name!r} has no single token for 'yes' or 'no', so "
                f"it cannot be scored this way. It is not a Qwen3-Reranker."
            )
        return tokenizer, model, yes, no

    @property
    def model(self) -> str:
        """The model doing the ordering."""
        return self._model_name

    def scores(
        self, query: str, documents: Sequence[str], instruction: str = SEARCH
    ) -> list[float]:
        """How well each document answers the query, each in [0, 1].

        Args:
            query: What is being looked for.
            documents: The candidates a retrieval already narrowed to.
            instruction: What relevance means for this caller.

        Returns:
            One probability per document, in the order given.
        """
        if not documents:
            return []
        tokenizer, model, yes, no = self._loaded
        pairs = [
            f"{_SYSTEM}<Instruct>: {instruction}\n<Query>: {query}\n"
            f"<Document>: {document}{_ASSISTANT}"
            for document in documents
        ]
        tokens = tokenizer(
            pairs,
            max_length=self._max_tokens,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            # The last position, which is where the answer would be written.
            logits = model(**tokens).logits[:, -1, :]
        # Softmax over the two tokens alone rather than the whole vocabulary:
        # what is wanted is how sure the model is BETWEEN yes and no, and the
        # mass it puts on the rest of the vocabulary is not part of that
        # question.
        answer = torch.softmax(torch.stack([logits[:, no], logits[:, yes]], dim=1), 1)
        return answer[:, 1].tolist()

    def ordered(
        self, query: str, documents: Sequence[str], instruction: str = SEARCH
    ) -> list[tuple[int, float]]:
        """The documents' positions and scores, best first.

        Positions rather than the documents themselves, so a caller holding
        more about each one than its text can reorder whatever it is holding.
        """
        scored = list(enumerate(self.scores(query, documents, instruction)))
        return sorted(scored, key=lambda one: (-one[1], one[0]))
