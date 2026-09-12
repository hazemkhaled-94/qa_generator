"""Shared behaviour for every SeaweedFS bucket."""

from __future__ import annotations

import logging
import re
from time import monotonic
from typing import Any, ClassVar
from urllib.parse import quote

from blob_store.seaweedfs.client import s3_client

log = logging.getLogger(__name__)

#: S3 user-metadata keys travel as HTTP header suffixes, so they are limited to
#: token characters. Anything else is rejected rather than silently mangled.
_METADATA_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

#: The error codes that mean "no such object", across the spellings S3
#: implementations use for a head request.
_ABSENT = frozenset({"404", "NoSuchKey", "NotFound"})

#: How long an object count may be reused, in seconds.
_COUNT_TTL = 30.0

#: The last count taken per bucket, as (when, how many).
_COUNTS: dict[str, tuple[float, int]] = {}


class Bucket:
    """One bucket in the SeaweedFS S3 gateway.

    Holds put, get, remove and count. Key layout is not part of this
    interface: a bucket that is written to defines its own.

    One class per bucket rather than a name passed as a string, because a
    bucket is a SeaweedFS collection and a collection is the unit of storage
    policy - replication, TTL, disk type and erasure coding.

    A concrete base, not an ABC: every method here works, and there is
    nothing for a subclass to implement. What a subclass must do is name its
    bucket, which `__init_subclass__` enforces - an `ABC` with no abstract
    method would not have.
    """

    #: The bucket's name in the object store. A ClassVar rather than an
    #: abstract property, because a subclass would answer a property with a
    #: ClassVar anyway - which shadows the descriptor rather than
    #: implementing it.
    name: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Refuses a bucket that has not said what it is called.

        Raises:
            TypeError: If the subclass declares no name. Every method here
                addresses `self.name`, so without one the first call fails
                with an AttributeError from inside boto3.
        """
        super().__init_subclass__(**kwargs)
        if not isinstance(getattr(cls, "name", None), str):
            raise TypeError(
                f"{cls.__name__} is a Bucket but does not declare name: ClassVar[str]."
            )

    @staticmethod
    def fanout_key(sha256: str, extension: str) -> str:
        """Builds a content-addressed key with a 2x2-hex fanout.

        The fanout is not cosmetic: the filer keys filemeta on (dirhash,
        name), so one flat directory becomes a write hotspot.
        """
        return f"{sha256[:2]}/{sha256[2:4]}/{sha256}.{extension}"

    def put(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Stores an object.

        Raises:
            ValueError: If a metadata key is not a valid header token.
        """
        s3_client().put_object(
            Bucket=self.name,
            Key=key,
            Body=data,
            ContentType=content_type,
            Metadata={
                self._checked_key(k): quote(v) for k, v in (metadata or {}).items()
            },
        )
        log.info("put %s/%s (%d bytes)", self.name, key, len(data))

    @staticmethod
    def _checked_key(key: str) -> str:
        """Validates one metadata key.

        Refused rather than escaped: a percent-encoded header name is a
        different header.

        Raises:
            ValueError: If the key is not a valid header token.
        """
        if not _METADATA_KEY.match(key):
            raise ValueError(
                f"invalid S3 metadata key {key!r}: "
                "use letters, digits, dot, dash or underscore"
            )
        return key

    def get(self, key: str) -> bytes:
        """Reads an object.

        Raises:
            botocore.exceptions.ClientError: If the object does not exist.
        """
        return s3_client().get_object(Bucket=self.name, Key=key)["Body"].read()

    def remove(self, key: str) -> bool:
        """Deletes an object, if it is there.

        The head request is what makes the answer possible: S3's own delete
        is silent about a key that was never there.

        Raises:
            botocore.exceptions.ClientError: If the store answered with
                anything other than "no such key". Refusing a delete and
                being unreachable are not the same as being empty, and
                reporting them as empty left the object in place with
                nothing recording that it was still there.
        """
        client = s3_client()
        try:
            client.head_object(Bucket=self.name, Key=key)
        except client.exceptions.ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in _ABSENT:
                return False
            raise
        client.delete_object(Bucket=self.name, Key=key)
        return True

    def count(self) -> int:
        """Counts the objects in the bucket.

        Walks every page of the listing, so it costs one request per thousand
        objects. Answered from a short-lived cache because the status panel
        polls it: without one, every render of every open page paid the full
        walk again.

        ponytail: a count that can be up to _COUNT_TTL seconds stale. The
        filer's own per-collection statistics are the upgrade.
        """
        now = monotonic()
        cached = _COUNTS.get(self.name)
        if cached is not None and now - cached[0] < _COUNT_TTL:
            return cached[1]
        total = sum(
            len(page.get("Contents", []))
            for page in s3_client()
            .get_paginator("list_objects_v2")
            .paginate(Bucket=self.name)
        )
        _COUNTS[self.name] = (now, total)
        return total
