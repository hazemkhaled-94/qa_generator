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
#   models      Fitted topic model. Regenerable only by another fit.
#   export      Topic figures.     Redrawn from models, or by another fit.
#   archive     Deleted objects.   Never expires: a TTL here would finish the
#               deletion the archive exists to postpone.
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
# MODELS — the fitted topic model
#
#   One .npz per language, holding the factorisation the topics were read
#   off: the term-topic and passage-topic matrices, the vocabulary and the
#   counts under both. The database keeps each topic's top terms, its label
#   and its coverage flag, which is what a stage reads and a person
#   queries; the matrix behind those is here, for the reason Docling's
#   complete output is in `parsed` rather than in the passages table.
#
#   The highest-value collection after documents, and the only one whose
#   loss costs a re-fit of the corpus rather than a re-read. Treat it like
#   documents rather than like parsed:
#
#     shell "s3.bucket.quota -name models -op=set -sizeMB=10240"
#
#
# EXPORT — the topic visualisations
#
#   The same durability policy as parsed: quota-bounded and lower
#   replication.
#
#     shell "s3.bucket.quota -name export -op=set -sizeMB=20480"
#
#   Lower replication is right HERE and was not before `models` existed.
#   pyLDAvis is prepared from the whole term-topic matrix and the database
#   keeps only each topic's top terms, so when the figure was the only
#   stored trace of a fit, losing it cost a re-fit of the corpus. It is now
#   a view of an object in `models` and cheap to lose.
#
#   This said the bucket is "rebuilt from the database". It is not, and was
#   not: nothing in the database can draw a figure. It also said the bucket
#   "holds the encrypted hold-out sets, which cannot be regenerated". There
#   was never a hold-out draw and nothing ever encrypted anything -
#   `questions.holdout_set_id` made the same promise and the 2026-09-24
#   migration dropped it as a column no code had written. Both are out now,
#   because a reader sizing these collections would have planned around
#   bytes that never existed and a rebuild path that never worked.
#
#
# ARCHIVE — what a deletion took
#
#   The same durability as documents, because it holds the same bytes: an
#   archived upload is the only remaining copy of it. No TTL and no quota —
#   a TTL would complete the deletion this bucket exists to postpone, and a
#   full quota would make the next deletion fail rather than the next
#   `make archive-purge` happen. It is bounded by purging it instead.

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
