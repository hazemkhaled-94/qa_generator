"""Keeping a fitted model, and reading one back.

The factorisation is what the fit produced; the pyLDAvis page in `export`
is a picture of it. Until this existed only the picture was stored, so a
fit's one durable trace was a view of itself with d3 inlined into it, and
the matrix it was drawn from went out of scope when `_score` returned.

`.npz` rather than `Nmf.save()`. Gensim's own serialisation is pickle
underneath, and a pickle is a promise about the class that wrote it: a
gensim upgrade can make last year's fit unloadable, which is the one thing
keeping it was for. Five arrays and a version tag load in anything that has
numpy, this decade or the next.

Compressed, because the term-topic matrix is mostly zeros - a factorisation
puts a term in few of its topics - and deflate is very good at that.
"""

from __future__ import annotations

import io
import logging

import numpy as np

from topic_modelling.models import TopicSpace

log = logging.getLogger(__name__)

#: What the arrays in one file mean. Bumped when their shape or meaning
#: changes, and refused rather than guessed on the way back in: an older
#: file read as a newer one is a figure drawn from misaligned axes, which
#: looks like a model rather than like an error.
FORMAT = 1


def dump(space: TopicSpace) -> bytes:
    """Writes one language's fitted model as the bytes of an .npz file.

    Bytes rather than a path, like the workbook: the caller puts them in a
    bucket and nothing here touches a filesystem.
    """
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        format=np.array(FORMAT),
        topic_term=np.asarray(space.topic_term, dtype=np.float64),
        doc_topic=np.asarray(space.doc_topic, dtype=np.float64),
        doc_lengths=np.asarray(space.doc_lengths, dtype=np.int64),
        # Unicode, and the width is the longest term rather than a guess:
        # a fixed one silently truncates a compound noun, and this corpus
        # is half German.
        vocabulary=np.asarray(space.vocabulary, dtype=np.str_),
        term_frequency=np.asarray(space.term_frequency, dtype=np.int64),
    )
    return buffer.getvalue()


def load(data: bytes) -> TopicSpace:
    """Reads a stored model back.

    Raises:
        ValueError: If the file was written in a format this does not know.
    """
    with np.load(io.BytesIO(data), allow_pickle=False) as held:
        found = int(held["format"])
        if found != FORMAT:
            raise ValueError(
                f"this topic model is in format {found} and this reads {FORMAT}"
            )
        return TopicSpace(
            topic_term=held["topic_term"].tolist(),
            doc_topic=held["doc_topic"].tolist(),
            doc_lengths=held["doc_lengths"].tolist(),
            vocabulary=held["vocabulary"].tolist(),
            term_frequency=held["term_frequency"].tolist(),
        )
