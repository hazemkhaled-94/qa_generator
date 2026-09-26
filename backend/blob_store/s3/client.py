"""Connection to the S3 object store.

The only module that knows boto3 exists or where the store lives.

Any S3 implementation, named by its endpoint: SeaweedFS and MinIO on a
path-style endpoint of their own, AWS S3 on none, R2 and B2 and Spaces on
theirs. What differs between them is an address, a region and an addressing
style, so those are the three settings; every operation the buckets use is
core S3 and spelled the same everywhere.

An empty value means **let boto3 decide**, and that is what makes AWS
reachable: it resolves its own endpoint from the region and finds its own
credentials from the instance role. The names must still be SET, empty or
not - a deployment that forgot the file gets a KeyError here rather than a
quiet fallback to somebody's real AWS account.
"""

from __future__ import annotations

import os
from functools import lru_cache

import boto3
from botocore.config import Config as BotoConfig


@lru_cache(maxsize=1)
def s3_client():
    """Builds the S3 client, once per process.

    Raises:
        KeyError: If S3_ENDPOINT, S3_ACCESS_KEY or S3_SECRET_KEY is unset.
            Empty is a value and means the SDK's own default; absent is a
            missing configuration.
    """
    return boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT"] or None,
        aws_access_key_id=os.environ["S3_ACCESS_KEY"] or None,
        aws_secret_access_key=os.environ["S3_SECRET_KEY"] or None,
        region_name=os.getenv("S3_REGION") or "local",
        config=BotoConfig(
            signature_version="s3v4",
            # Path style is what a self-hosted gateway expects, because the
            # virtual-host style boto3 prefers resolves buckets as
            # subdomains. AWS wants `auto`, and says so in .env.example.
            s3={"addressing_style": os.getenv("S3_ADDRESSING_STYLE") or "path"},
            retries={"max_attempts": 3, "mode": "standard"},
            connect_timeout=5,
            read_timeout=60,
        ),
    )


def bucket_names() -> list[str]:
    """Lists every bucket the store exposes."""
    return [b["Name"] for b in s3_client().list_buckets()["Buckets"]]
