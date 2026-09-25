"""The export bucket."""

from __future__ import annotations

from typing import ClassVar

from blob_store.seaweedfs.bucket import Bucket


class ExportBucket(Bucket):
    """Generated datasets and coverage reports, quota-bounded and TTL-expired.

    Everything here is regenerable: a fit redraws a topic visualisation and
    the question export is built per request. This said it "also holds the
    encrypted hold-out sets, which are not regenerable", and there was
    neither a hold-out draw nor anything doing the encrypting - the same
    promise `questions.holdout_set_id` made and the migration that dropped
    that column removed. A promise nothing implements is worse than a
    missing feature, because a reader checking the store finds it and plans
    a retention policy around it.
    """

    name: ClassVar[str] = "export"

    #: What a topic visualisation is stored and served as.
    TOPIC_VISUALISATION_TYPE: ClassVar[str] = "text/html; charset=utf-8"

    @classmethod
    def topic_visualisation_key(cls, language: str) -> str:
        """Builds the key one language's topic visualisation is stored under."""
        return f"topics/{language}.html"
