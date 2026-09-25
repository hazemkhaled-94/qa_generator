"""The archive bucket."""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from blob_store.seaweedfs.bucket import Bucket
from blob_store.seaweedfs.client import s3_client

#: How many keys one delete request carries. The S3 limit.
_BATCH = 1000


class ArchiveBucket(Bucket):
    """What a deletion took out of another bucket, until somebody purges it.

    One bucket for all three rather than an archive beside each, because
    the retention is the same wherever the object came from: it is held
    until a person says otherwise. A key carries the bucket it came out of,
    so the origin is not lost by being kept here.
    """

    name: ClassVar[str] = "archive"

    @classmethod
    def key_for(cls, origin: str, key: str) -> str:
        """Builds the key an object out of `origin` is archived under."""
        return f"{origin}/{key}"

    def take(self, origin: str, key: str) -> bool:
        """Moves an object out of `origin` and into the archive.

        Copied and then deleted, in that order: a failure between the two
        leaves the object in both places, which is recoverable, and the
        other order loses it.

        Args:
            origin: The bucket the object is in.
            key: Its key there.

        Returns:
            Whether there was an object to move.

        Raises:
            botocore.exceptions.ClientError: If the store answered with
                anything other than "no such key".
        """
        client = s3_client()
        try:
            client.copy_object(
                Bucket=self.name,
                Key=self.key_for(origin, key),
                CopySource={"Bucket": origin, "Key": key},
            )
        except client.exceptions.ClientError as exc:
            if self._absent(exc):
                return False
            raise
        client.delete_object(Bucket=origin, Key=key)
        return True

    def held(self, *, before: datetime | None = None) -> list[tuple[str, int]]:
        """Lists what the archive holds, as (key, bytes).

        Args:
            before: Only objects last written before this, or all of them.
        """
        return [
            (entry["Key"], entry["Size"])
            for page in s3_client()
            .get_paginator("list_objects_v2")
            .paginate(Bucket=self.name)
            for entry in page.get("Contents", [])
            if before is None or entry["LastModified"] < before
        ]

    def purge(self, *, before: datetime | None = None) -> int:
        """Deletes what the archive holds. Irreversible, as the second one is.

        Args:
            before: Only objects last written before this, or all of them.

        Returns:
            How many objects were deleted.
        """
        keys = [key for key, _ in self.held(before=before)]
        client = s3_client()
        for start in range(0, len(keys), _BATCH):
            client.delete_objects(
                Bucket=self.name,
                Delete={
                    "Objects": [{"Key": key} for key in keys[start : start + _BATCH]]
                },
            )
        return len(keys)
