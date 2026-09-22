"""The embedding model, shared by the stages that compare two pieces of text.

Importing this loads torch and the model's weights. Nothing outside a worker
should name it: the api serves JSON and has no use for two gigabytes of
parameters, and `tests/static/test_api_stays_light.py` pins that.

Here rather than beside one stage's code because three columns now hold its
output - `passages.embedding`, `facts.embedding` and `questions.embedding` -
and a service may not import another service to reach it. Loading a model is
this package's, as building an engine is database's.

The model is EMBEDDING_MODEL, which is also the tokenizer chunking sizes a
passage by. One name for both, so nothing is measured in a space the corpus
was never put in.

The pooling used to be written out here - mean over the attention mask, then
L2. That is what `sentence-transformers` is, and it reads the checkpoint's
own `1_Pooling` config rather than assuming the mean: a model trained to be
read off its CLS token was being averaged.
"""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

from nlp.windows import window

log = logging.getLogger(__name__)

#: What every embedding column holds. A model of another width would be
#: refused by the database one row at a time, with nothing saying why.
WIDTH = 1024

#: e5 was trained on prefixed inputs and scores badly without one. Every
#: comparison here is between two texts of the same kind - a fact against
#: facts, a question against questions - and e5 asks for one prefix on both
#: sides of a symmetric comparison, so there is one prefix and no branch.
_PREFIX = "query: "


class Embedder:
    """One embedding model, asked for the vector of one text."""

    def __init__(self, model_name: str, max_tokens: int) -> None:
        """Loads the model and refuses one of the wrong width.

        Raises:
            ValueError: If the model's output is not as wide as the columns
                that have to hold it.
        """
        log.info("loading %s", model_name)
        self._model = SentenceTransformer(model_name)
        # The ceiling a deployment asked for, held to what the checkpoint
        # can actually read - the same reading `nlp.entailment` and `nlp.qa`
        # make of ENCODER_MAX_TOKENS.
        self._model.max_seq_length = window(self._model.tokenizer, max_tokens)

        width = self._model.get_embedding_dimension()
        if width != WIDTH:
            raise ValueError(
                f"EMBEDDING_MODEL={model_name!r} is {width} wide and the "
                f"embedding columns hold {WIDTH}. Changing the model means a "
                f"migration for each of them and re-embedding every row."
            )

    def embed(self, text: str) -> list[float]:
        """Returns one text's unit vector."""
        return self.embed_all([text])[0]

    def embed_all(self, texts: list[str]) -> list[list[float]]:
        """Returns one unit vector per text, in one pass over the model.

        A batch, because the per-call cost is the forward pass and a passage's
        facts are embedded together.

        Normalised to length 1, which is what makes pgvector's cosine
        distance a subtraction rather than a division, and what lets
        :func:`cosine` be a dot product.
        """
        if not texts:
            return []
        return self._model.encode(
            [_PREFIX + text for text in texts],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()


def cosine(left: list[float], right: list[float]) -> float:
    """How alike two unit vectors are, in [-1, 1].

    A plain dot product, because :meth:`Embedder.embed` returns unit vectors
    and dividing by two lengths of 1 is the arithmetic this saves. Written
    here rather than reached for from torch: the caller comparing a candidate
    against the handful accepted earlier in the same run should not build a
    tensor to do it.
    """
    return sum(a * b for a, b in zip(left, right, strict=True))
