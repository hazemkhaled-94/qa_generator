"""The export bucket."""

from __future__ import annotations

from typing import ClassVar

from blob_store.seaweedfs.bucket import Bucket


class ExportBucket(Bucket):
    """Generated datasets and coverage reports, quota-bounded and TTL-expired.

    Also holds the encrypted hold-out sets, which are not regenerable.
    """

    name: ClassVar[str] = "export"

    #: What a topic visualisation is stored and served as.
    TOPIC_VISUALISATION_TYPE: ClassVar[str] = "text/html; charset=utf-8"

    @classmethod
    def topic_visualisation_key(cls, language: str) -> str:
        """Builds the key one language's topic visualisation is stored under."""
        return f"topics/{language}.html"
