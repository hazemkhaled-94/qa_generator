"""The export bucket."""

from __future__ import annotations

from typing import ClassVar

from blob_store.s3.bucket import Bucket


class ExportBucket(Bucket):
    """One pyLDAvis figure per language, replaced by the next fit.

    Only that. The name says datasets, and the question export is not one
    of them: `question_generation.export.workbook` returns bytes and its
    three callers stream, write or download them, so no .xlsx has ever been
    stored here.

    Neither quota-bounded nor TTL-expired, which this used to claim. The
    `s3.bucket.quota` line for this collection sits commented in
    `configs/seaweedfs/bucket-init.sh` under the heading saying those
    commands are not run automatically, so nothing bounds this bucket but
    the next fit overwriting the figure and `topics-delete` archiving it.

    **Nothing stored can redraw a figure.** It is prepared from the whole
    term-topic matrix and the database keeps each topic's top terms, not
    the matrix, so the only thing that draws one is another fit.
    `--visualise` reads this bucket rather than rendering, which is what
    that distinction looks like in the command line.

    That is a re-fit and not a disaster: the factorisation is seeded with
    TOPIC_RANDOM_STATE, so the same corpus under the same settings comes
    back the same, and `TopicRepository.labelled_topics` carries the labels
    a person wrote and the coverage flags they set across by `topic_index`.
    What it costs is the fit, and what it risks is a corpus or a setting
    having moved since - in which case the topics move too, and the old
    figure was describing something that is gone anyway.

    It said it "also holds the encrypted hold-out sets, which are not
    regenerable", and there was neither a hold-out draw nor anything doing
    the encrypting - the same promise `questions.holdout_set_id` made and
    the migration that dropped that column removed. A promise nothing
    implements is worse than a missing feature, because a reader checking
    the store finds it and plans a retention policy around it.
    """

    name: ClassVar[str] = "export"

    #: What a topic visualisation is stored and served as.
    TOPIC_VISUALISATION_TYPE: ClassVar[str] = "text/html; charset=utf-8"

    @classmethod
    def topic_visualisation_key(cls, language: str) -> str:
        """Builds the key one language's topic visualisation is stored under."""
        return f"topics/{language}.html"
