"""Connection to the SeaweedFS S3 gateway.

The only module that knows boto3 exists or where the gateway lives.
"""

from __future__ import annotations

import os
from functools import lru_cache

import boto3
from botocore.config import Config as BotoConfig


@lru_cache(maxsize=1)
def s3_client():
    """Builds the S3 client, once per process.

    Path-style addressing and s3v4 signing are what SeaweedFS expects; the
    virtual-host style boto3 prefers resolves buckets as subdomains.

    Raises:
        KeyError: If S3_ENDPOINT, S3_ACCESS_KEY or S3_SECRET_KEY is unset.
    """
    return boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT"],
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        region_name="local",
        config=BotoConfig(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            retries={"max_attempts": 3, "mode": "standard"},
            connect_timeout=5,
            read_timeout=60,
        ),
    )


def bucket_names() -> list[str]:
    """Lists every bucket the gateway exposes."""
    return [b["Name"] for b in s3_client().list_buckets()["Buckets"]]
