"""The models bucket."""

from __future__ import annotations

from typing import ClassVar

from blob_store.s3.bucket import Bucket


class ModelsBucket(Bucket):
    """What the pipeline fitted, as against what it drew from it.

    One object per language per fit, replaced by the next fit, holding the
    factorisation itself: the term-topic matrix, the passage-topic matrix,
    the vocabulary and the counts under both. The database keeps what a
    stage reads or a person queries - each topic's top terms, its label,
    whether it counts towards coverage - and the matrix behind those keeps
    no column, for the reason `parsed` holds Docling's complete output
    rather than the passages table holding it.

    A bucket of its own rather than a prefix in `export`, because the two
    have opposite durability contracts and a bucket is what a storage
    policy is set on. `export` holds a figure that is
    a VIEW of this and can be redrawn from it; this holds the thing being
    viewed, and losing it costs a re-fit of the corpus. That the figure was
    the only durable trace of a fit is what this bucket exists to end.
    """

    name: ClassVar[str] = "models"

    #: What a stored factorisation is written as. `.npz` rather than the
    #: gensim model's own `save()`: that is pickle underneath and a gensim
    #: upgrade can make last year's fit unloadable, which is the one thing
    #: keeping it was for. Arrays and a version tag load in anything that
    #: has numpy.
    TOPIC_MODEL_TYPE: ClassVar[str] = "application/octet-stream"

    @classmethod
    def topic_model_key(cls, language: str) -> str:
        """Builds the key one language's fitted topic model is stored under.

        The same shape as the figure's key in `export`, so the two halves
        of one fit are found the same way.
        """
        return f"topics/{language}.npz"
