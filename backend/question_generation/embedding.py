"""Embedding a question, for the gate that asks whether it is already asked.

Importing this loads torch and the model's weights. Nothing outside a worker
should name it: the api serves JSON and has no use for two gigabytes of
parameters.

The model is EMBEDDING_MODEL, which is also the tokenizer chunking sizes a
passage by. One name for both, so a question cannot be measured in a space
the corpus was never put in.
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoModel, AutoTokenizer

log = logging.getLogger(__name__)

#: What the questions.embedding column holds. A model of another width would
#: be refused by the database one row at a time, with nothing saying why.
WIDTH = 1024

#: e5 was trained on prefixed inputs and scores badly without one. Everything
#: embedded here is a question, so there is one prefix and no branch.
_PREFIX = "query: "


class Embedder:
    """One embedding model, asked for the vector of one question."""

    def __init__(self, model_name: str, max_tokens: int) -> None:
        """Loads the model and refuses one of the wrong width.

        Raises:
            ValueError: If the model's output is not as wide as the column
                that has to hold it.
        """
        log.info("loading %s", model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()
        self._max_tokens = max_tokens

        width = int(self._model.config.hidden_size)
        if width != WIDTH:
            raise ValueError(
                f"EMBEDDING_MODEL={model_name!r} is {width} wide and "
                f"questions.embedding holds {WIDTH}. Changing the model means a "
                f"migration for that column and re-embedding every row."
            )

    def embed(self, text: str) -> list[float]:
        """Returns one question's unit vector.

        Mean-pooled over the tokens the attention mask keeps, then normalised
        to length 1, which is what makes pgvector's cosine distance a
        subtraction rather than a division.
        """
        tokens = self._tokenizer(
            _PREFIX + text,
            max_length=self._max_tokens,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            hidden = self._model(**tokens).last_hidden_state

        mask = tokens["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)[0].tolist()


def cosine(left: list[float], right: list[float]) -> float:
    """How alike two unit vectors are, in [-1, 1].

    A plain dot product, because :meth:`Embedder.embed` returns unit vectors
    and dividing by two lengths of 1 is the arithmetic this saves. Written
    here rather than reached for from torch: the caller comparing a candidate
    against the handful accepted earlier in the same run should not build a
    tensor to do it.
    """
    return sum(a * b for a, b in zip(left, right, strict=True))
