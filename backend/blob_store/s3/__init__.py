"""Buckets of the S3 object store.

    documents   source of truth, never expires
    parsed      parser output, overwritten by the next parse
    models      what the pipeline fitted, replaced by the next fit
    export      views drawn from a model, replaced by the next fit
    archive     what a deletion took, until it is purged

`models` and `export` are one fit's two halves and the order matters:
`models` holds the factorisation and `export` holds a picture of it, so the
picture is derivable and the factorisation is not. The figure used to be
the only durable trace of a fit, which made a disposable view the thing
being kept.

**Nothing here expires on its own.** This said `parsed` was "TTL-expired"
and `export` "quota-bounded", and neither is: the only TTL and quota in the
project are the commented `s3.bucket.quota` lines in
`configs/seaweedfs/bucket-init.sh`, which that file says in as many words
are not run automatically. A bucket is a SeaweedFS collection and a
collection is where such a policy would go, so it can be applied - but
until an operator does, every bucket here grows until something replaces or
purges an object. What bounds each one is in the table above and in the
README beside this file, and it is a rewrite or a deletion in every case.

Nothing outside this package builds an S3 client or knows the endpoint.
"""

from blob_store.s3.archive import ArchiveBucket
from blob_store.s3.bucket import Bucket
from blob_store.s3.client import bucket_names, s3_client
from blob_store.s3.documents import DocumentsBucket
from blob_store.s3.export import ExportBucket
from blob_store.s3.models import ModelsBucket
from blob_store.s3.parsed import ParsedBucket

__all__ = [
    "ArchiveBucket",
    "Bucket",
    "DocumentsBucket",
    "ExportBucket",
    "ModelsBucket",
    "ParsedBucket",
    "bucket_names",
    "s3_client",
]
