# Ingestion

Takes a file somebody uploaded, decides whether the corpus will hold it, and
if so puts the bytes in an object store and writes one row.

This is the only stage a person waits on, so it does only what can be done in
about a second. Everything expensive happens later, in a worker, off a queue.

**A document's identity is the SHA-256 of its bytes.** It is the primary key
of `documents`, the object key in both buckets, and what a duplicate is
detected by. No filename, declared content type or extension is evidence of
anything.

**Every attempt is recorded, refused ones included.** `ingest_events` is the
record of what was offered; `documents` is the record of what was kept.

The other two stages that turn a PDF into passages are
[parsing](../preprocessing/parsing/README.md) and
[chunking](../preprocessing/chunking/README.md).

## What it does

Synchronous, inside the API process. There is no queue and no worker: the
route calls `IngestService.ingest` and returns its answer. The document
arrives with every stage at `new`, and a worker only claims `pending`.

### Step 1 — Refuse before reading the body

If the declared `Content-Length` already exceeds `MAX_FILE_SIZE_MB`, the body
is never read. [`refuse_oversized`](service.py) writes the event and stops:
with no bytes there is no digest and no document to point at.

`Content-Length` covers the whole multipart body, so it is an upper bound. A
request declaring no length is read in full and measured by the check below.

### Step 2 — Run the checks, cheapest first

Nothing is written until every check passes.

| Order | Check | Refusal |
|---|---|---|
| 1 | `size_bytes > MAX_FILE_SIZE_MB` | `too_large`, carrying both figures |
| 2 | leading bytes are in `ALLOWED_MIME_TYPES` | `unsupported_type`, naming what was detected |
| 3 | the digest is not already in `documents` | `duplicate_bytes` |
| 4 | the PDF opens and is not password protected | `unsupported_type`, carrying the reader's reason |

Hashing bytes already in memory is cheaper than walking every page, so a
re-upload of something the corpus holds never reads the document.

The type check is on the leading bytes. PyMuPDF is not the type check: handed
bytes no magic number claims, it returns a one-page empty document rather
than an error.

### Step 3 — Write the object, then the row

A fixed order, because the key is content-addressed: an object with no row
holds exactly the bytes that key means and is harmless, where a row pointing
at a missing object fails every later stage. If the row cannot be written,
the object is taken back out.

The stored object carries four pieces of S3 user metadata — the digest, the
original filename, the time and the pipeline version — which is enough for
the bucket alone to rebuild the row. Nothing mutable is included, because S3
metadata cannot change without rewriting the object.

### Step 4 — Delete

[`removal.py`](removal.py). Two operations, because they are two decisions.

`delete` removes the object in `documents`, the object in `parsed`, and the
row — which cascades to passages, and from those to facts, memberships and
questions. Only the `ingest_events` rows survive, on `ON DELETE SET NULL`.
Nothing is atomic across two stores, so the order is fixed: objects first,
row last.

`delete_derived` drops only the passages and what cascades from them, and
returns `chunk_status` to `new`. Both objects stay.

Neither destroys anything: the rows are copied into `archived_rows` by an
AFTER DELETE trigger and the objects are **moved** to the `archive` bucket.
See [`archive/`](../archive/README.md).

## Inputs and outputs

**In:** one PDF, up to `MAX_FILE_SIZE_MB`, born-digital. Posted to
`POST /documents` or picked on the Upload page.

**Written, in PostgreSQL:**

| Table | Holds |
|---|---|
| `ingest_events` | every upload attempt, refused ones included: the submitted filename, the size, the outcome, the reason, and the document it became |
| `documents` | the digest, the media type, the original filename, the page and character counts, and every stage's status at `new` |

**Written, in SeaweedFS:** `documents`, at
`{sha[0:2]}/{sha[2:4]}/{sha}.pdf` — the uploaded bytes exactly as they
arrived, never expiring. The key is derived in code and never stored. The
two-by-two hex fanout is not cosmetic: the filer keys its metadata on
(directory hash, name), so one flat directory becomes a write hotspot.

| Route | Answers |
|---|---|
| `POST /documents` | Upload one file, and what became of it |
| `GET /documents` | The corpus, with each document's stage statuses and passage counts |
| `GET /documents/names` | Digests and filenames, for the pickers |
| `GET /documents/{sha256}/file` | The stored bytes back |
| `DELETE /documents/{sha256}` | Remove the document and everything derived from it |
| `DELETE /documents/{sha256}/derived` | Remove the passages and what cascades, keep both objects |

**Command line:** `make documents`, `make delete SHA=…`,
`make delete-derived SHA=…`, `make wipe`.

**Read by:** [parsing](../preprocessing/parsing/README.md).

## Configuration

Read from the environment at start-up, with no defaults in code. Tuning is in
[`configs/env/backend.env`](../../configs/env/backend.env).

| Setting | What it does |
|---|---|
| `MAX_FILE_SIZE_MB` | Largest upload accepted, by both the `Content-Length` check and the measured size. Keep in step with `server.maxUploadSize` in `frontend/.streamlit/config.toml` |
| `ALLOWED_MIME_TYPES` | Comma-separated. May only name types this build can detect from leading bytes. Anything else stops the service at start-up |

The `pipeline-version` written into every stored object's metadata is not a
setting: it is read from `version` in `pyproject.toml`, because it describes
the build rather than the deployment. Documents already held keep the version
they arrived under.

Connection settings are not here: `database` and `blob_store` own theirs.

## Tests

```sh
poetry run pytest tests/unit/intake/test_intake_ingestion.py
poetry run pytest tests/unit/ingestion tests/integration/test_ingest.py
```

`IngestDriver` and `RemovalDriver` live in
[`tests/unit/intake/intake_drivers.py`](../../tests/unit/intake/intake_drivers.py),
over `MemoryDocuments`, `MemoryParsed` and `DocumentRows`. Only the database
and the two buckets are stood in for.

| File | Covers |
|---|---|
| [`test_uploaded_file.py`](../../tests/unit/ingestion/test_uploaded_file.py) | What an upload's bytes say about it: magic-byte detection, the digest, the size |
| [`test_intake_ingestion.py`](../../tests/unit/intake/test_intake_ingestion.py) | The check order, both size boundaries, the object metadata, the two store-failure paths, the insert race, and the deletion ordering across three stores |
| [`test_intake_cli.py`](../../tests/unit/intake/test_intake_cli.py) | Which flag combinations the command line refuses |
| [`test_ingest.py`](../../tests/integration/test_ingest.py), [`test_removal.py`](../../tests/integration/test_removal.py) | Both flows against a real PostgreSQL and SeaweedFS |
| [`test_documents.py`](../../tests/integration/api/test_documents.py) | The HTTP surface: status codes, the digest guard, the refusal shapes |
| [`test_orphans.py`](../../tests/integration/test_orphans.py) | That deleting a document leaves nothing derived from it behind |
| [`test_views.py`](../../tests/frontend/test_views.py) | The Upload and Documents pages under Streamlit's own runner |
| [`test_intake_pinned.py`](../../tests/static/test_intake_pinned.py) | The settings, routes, answer shapes and modules, pinned |

## Limits

- **A re-upload's filename is recorded and never shown.** The same bytes
  under a new name are refused as `duplicate_bytes`, and the only reader of
  that column takes the earliest row.
- **A duplicate by *content* gets through here.** Different bytes carrying
  the same text pass every check, because nothing knows the text yet. Parsing
  refuses them on `content_sha256`.
- **`Content-Length` is an upper bound.** It covers the whole multipart body.
- **An upload that fails on the object store writes no `ingest_events`
  row.** Only refusals are recorded; an unreachable bucket is a 500.
