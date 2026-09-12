"""The parsed bucket."""

from __future__ import annotations

from typing import ClassVar

from blob_store.seaweedfs.bucket import Bucket


class ParsedBucket(Bucket):
    """Structured output from the parsing stage.

    Keyed by the source document's digest, so re-parsing overwrites rather
    than accumulating versions. Regenerable from the documents bucket, so it
    tolerates lower replication and a TTL.
    """

    name: ClassVar[str] = "parsed"

    @classmethod
    def key_for(cls, sha256: str) -> str:
        """Builds the key a parsed document is stored under."""
        return cls.fanout_key(sha256, "json")
