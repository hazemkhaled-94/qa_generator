"""Buckets of the SeaweedFS object store.

    documents   source of truth, never expires
    parsed      parser output, regenerable, TTL-expired
    export      datasets and reports, regenerable, quota-bounded

Nothing outside this package builds an S3 client or knows the endpoint.
"""

from blob_store.seaweedfs.bucket import Bucket
from blob_store.seaweedfs.client import bucket_names, s3_client
from blob_store.seaweedfs.documents import DocumentsBucket
from blob_store.seaweedfs.export import ExportBucket
from blob_store.seaweedfs.parsed import ParsedBucket

__all__ = [
    "Bucket",
    "DocumentsBucket",
    "ExportBucket",
    "ParsedBucket",
    "bucket_names",
    "s3_client",
]
