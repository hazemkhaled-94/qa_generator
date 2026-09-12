#!/bin/sh
# Creates the buckets, then hands control back to the S3 gateway's entrypoint.
# Idempotent — s3.bucket.create is a no-op when the bucket already exists.
set -eu

MASTER="${SEAWEEDFS_MASTER:-seaweedfs-master:9333}"

# Sends one or more weed-shell commands to the master.
shell() { printf '%s\n' "$@" | weed shell -master="$MASTER"; }

# ── Bucket creation ────────────────────────────────────────────────────────
#
# In SeaweedFS, a bucket is a collection. Collection is the unit of storage
# policy: replication factor, TTL, disk type, and erasure coding are all
# configured per collection. Separating documents/parsed/export into distinct
# buckets is therefore not cosmetic — each has a different durability contract.
#
#   documents   source of truth.   Never expires. Erasure-coding candidate.
#   parsed      Docling output.    Regenerable from documents.
#   export      Generated reports. Regenerable from the database.
#
# Adding a bucket: append its name to S3_BUCKETS in .env and restart.

for b in $(echo "${S3_BUCKETS}" | tr ',' ' '); do
    shell "s3.bucket.create -name ${b}"
done

echo "buckets ready: ${S3_BUCKETS}"

# ── Production storage policy notes ───────────────────────────────────────
#
# The commands below are NOT run automatically. They are executed once by the
# operator at the appropriate stage of the deployment lifecycle. Each is
# documented here so the intent survives alongside the code that depends on it.
#
#
# DOCUMENTS — immutable source of truth
#
#   Replication is chosen at volume creation and cannot be changed after the
#   fact. The collection-level override below must be applied before any
#   documents are ingested into the cluster:
#
#     shell "volume.configure.replication -collection documents -replication 010"
#
#   Erasure coding reduces storage overhead to ~1.4x while maintaining
#   durability comparable to 2x full replication. It is suited to the
#   documents collection because objects are written once and read rarely
#   after parsing. Apply after ingestion is substantially complete:
#
#     shell "ec.encode -collection documents -fullPercent 95 -quietFor 24h"
#
#
# PARSED — regenerable Docling output
#
#   Objects are keyed by the source document's digest alone, so re-parsing
#   overwrites in place rather than accumulating versions. Nothing here grows
#   without a document behind it, and a quota is what stops a runaway
#   re-parse from filling the cluster:
#
#     shell "s3.bucket.quota -name parsed -op=set -sizeMB=51200"
#
#   Lower replication is acceptable — every object can be rebuilt from the
#   documents collection in the time it takes to re-run the parser.
#
#
# EXPORT — generated datasets and reports
#
#   Rebuilt from the database, with the same durability policy as parsed:
#   quota-bounded and lower replication. The encrypted hold-out sets are the
#   exception; they are a random draw and cannot be regenerated.
#
#     shell "s3.bucket.quota -name export -op=set -sizeMB=20480"

# ── S3 object metadata contract ────────────────────────────────────────────
#
# S3 user metadata (x-amz-meta-*) is immutable without rewriting the object,
# so it carries only facts that are known at PUT time. Mutable state —
# extraction_status, title, language — lives in Postgres and is not
# duplicated here. Duplicated mutable state drifts.
#
# The fields below are chosen to satisfy one requirement: if the qa_generator
# database is lost entirely, the documents bucket alone is sufficient to
# rebuild it. Everything else is re-derivable by re-running the pipeline.
#
# Written by the ingestion service, on every object in `documents`:
#
#   x-amz-meta-sha256              content hash (doubles as the object key)
#   x-amz-meta-original-filename   filename as submitted by the user
#   x-amz-meta-ingested-at         ISO-8601 UTC timestamp
#   x-amz-meta-pipeline-version    version of the code that wrote the object
#   Content-Type                   MIME type detected at ingest
#
# Written by the parsing service, on every object in `parsed`:
#
#   x-amz-meta-sha256              digest of the source document
#   Content-Type                   application/json
