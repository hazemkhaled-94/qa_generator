"""The export bucket."""

from __future__ import annotations

from typing import ClassVar

from blob_store.seaweedfs.bucket import Bucket


class ExportBucket(Bucket):
    """Generated datasets and coverage reports, quota-bounded and TTL-expired.

    Also holds the encrypted hold-out sets, which are not regenerable: a
    hold-out draw is a random sample, and redrawing it defeats its purpose.

    Nothing writes to it yet, so it declares no key layout. Created and
    health-checked with the others, so it exists before the release that
    needs it.
    """

    name: ClassVar[str] = "export"
