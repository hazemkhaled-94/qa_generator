# Ingestion

Takes a file somebody uploaded, decides whether the corpus will hold it, and
if so puts the bytes in an object store and writes one row.

This is the only stage a person waits on. Somebody presses Submit and wants
an answer while they are still looking at the screen, so it does only what
can be done in about a second: read what the file actually is from its
leading bytes, check it against a size limit and an allowlist, check whether
the corpus already holds those exact bytes, count the pages and the
extractable characters, store the object, write the row. Everything
expensive happens later, in a worker, off a queue.

Two ideas run through it and most of the design follows from them.

**A document's identity is the SHA-256 of its bytes.** It is the primary key
of `documents`, it is the object key in both buckets, and it is what a
duplicate is detected by. No filename, no declared content type and no
extension is evidence of anything: all three are things the uploader chose.

**Every attempt is recorded, including the refused ones.** A refusal writes
no document at all and would otherwise leave no trace that anybody tried.
`ingest_events` is the record of what was offered; `documents` is the record
of what was kept.

Ingestion is the first of three stages that turn a PDF into passages. The
other two are [parsing](../preprocessing/parsing/README.md) and
[chunking](../preprocessing/chunking/README.md).

## What it does

### The shape of a run

Synchronous, inside the API process. There is no queue and no worker: the
route calls `IngestService.ingest` and returns its answer. Nothing it writes
starts anything — the document arrives with every stage at `new`, and a
worker only claims `pending`.

### Step 1 — Refuse before reading the body

**In:** the request's `Content-Length`.
**Out:** an `ingest_events` row and a refusal, or nothing at all.

If the declared length already exceeds `MAX_FILE_SIZE_MB`, the body is never
read. [`refuse_oversized`](service.py) writes the event and stops there:
with no bytes there is no digest and no document to point at.

`Content-Length` covers the whole multipart body rather than the file alone,
so it is an upper bound. A request declaring no length at all is read in
full and measured afterwards, by the check below.

### Step 2 — Run the checks, cheapest first

**In:** a filename and the raw bytes.
**Out:** an `IngestResult` — `stored`, `duplicate_bytes`, `too_large` or
`unsupported_type` — and one `ingest_events` row either way.

Nothing is written until every check passes:

| Order | Check | Refusal |
|---|---|---|
| 1 | `size_bytes > MAX_FILE_SIZE_MB` | `too_large`, carrying both figures |
| 2 | leading bytes are in `ALLOWED_MIME_TYPES` | `unsupported_type`, naming what was detected |
| 3 | the digest is not already in `documents` | `duplicate_bytes` |
| 4 | the PDF opens and is not password protected | `unsupported_type`, carrying the reader's reason |

The order of the last two matters. Hashing bytes already in memory is
cheaper than walking every page, so a re-upload of something the corpus
already holds never reads the document.

The type check is on the leading bytes and not on the filename or the
declared type. PyMuPDF is not the type check either: handed bytes no magic
number claims, it returns a one-page empty document rather than an error,
which is why the magic-byte check runs first.

### Step 3 — Write the object, then the row

**In:** the bytes and everything the checks read.
**Out:** one object in the `documents` bucket and one `documents` row.

A fixed order, because the key is content-addressed. An object with no row
holds exactly the bytes that key means and is harmless. A row pointing at a
missing object fails every later stage. If the row cannot be written, the
object is taken back out.

The stored object carries four pieces of S3 user metadata — the digest, the
original filename, the time, and `PIPELINE_VERSION` — which is enough for
the bucket alone to rebuild the row. Nothing mutable is included, because S3
metadata cannot change without rewriting the object.

The row is written with `page_count`, `char_count` and the media type, and
with every stage status at `new`.

### Step 4 — Delete

**Where:** [`removal.py`](removal.py).

Two operations, because they are two different decisions.

`delete` removes everything: the object in `documents`, the object in
`parsed`, and the row — which cascades to passages, and from those to facts,
topic memberships and questions. Only the `ingest_events` rows survive, on
`ON DELETE SET NULL`, so the record that the upload happened outlives the
document. Nothing is atomic across two stores, so the order is fixed:
objects first, row last.

`delete_derived` drops only the passages and what cascades from them, and
returns `chunk_status` to `new`. Both objects stay, so the next run rebuilds
the passages without paying to parse again.

## Tools, and where each is used

| Tool | Where | Why this one |
|---|---|---|
| **PyMuPDF** | [`pdf.py`](pdf.py) | Counts pages and extractable characters in about a second, which is what lets the upload answer while somebody is watching. It also detects encryption, so a password-protected file is refused by that name. Summed page by page, so a long document's text is never all held at once. |
| **boto3** | `blob_store/seaweedfs/` | The S3 API of the SeaweedFS gateway. |
| **SQLAlchemy** | [`repository.py`](repository.py) | The insert, the duplicate lookup and the listings. Column comments live on the models and are carried into the database by the migrations. |
| **OpenTelemetry** | [`service.py`](service.py) | One span per upload with the outcome on it, so one trace query finds every refusal and why. |

## Inputs and outputs

**In:** one PDF, up to `MAX_FILE_SIZE_MB`, born-digital. Posted to
`POST /documents` or picked on the Upload page.

**Written, in PostgreSQL:**

| Table | Holds |
|---|---|
| `ingest_events` | every upload attempt, refused ones included: the submitted filename, the size, the outcome, the reason, and the document it became when it became one |
| `documents` | the digest, the media type, the original filename, the page and character counts, and every stage's status at `new` |

**Written, in SeaweedFS:**

| Bucket | Key | Holds |
|---|---|---|
| `documents` | `{sha[0:2]}/{sha[2:4]}/{sha}.pdf` | the uploaded bytes, exactly as they arrived, never expiring |

The key is derived in code and never stored. The two-by-two hex fanout is
not cosmetic: the filer keys its metadata on (directory hash, name), so one
flat directory becomes a write hotspot.

**Serves**

| Route | What it answers |
|---|---|
| `POST /documents` | Upload one file, and what became of it |
| `GET /documents` | The corpus, with each document's stage statuses and passage counts |
| `GET /documents/names` | Just the digests and filenames, for the pickers |
| `GET /documents/{sha256}/file` | The stored bytes back |
| `DELETE /documents/{sha256}` | Remove the document and everything derived from it |
| `DELETE /documents/{sha256}/derived` | Remove the passages and what cascades, keep both objects |

**Command line:** `make documents`, `make delete SHA=…`,
`make delete-derived SHA=…`, `make wipe`.

**Read by:** [parsing](../preprocessing/parsing/README.md), which claims the
row and fetches the object it names.

## Configuration

Every setting is read from the environment at start-up and there are no
defaults in code: a missing variable stops the service rather than running
with a value nobody chose. Tuning lives in
[`configs/env/backend.env`](../../configs/env/backend.env), which is in git;
credentials, ports and addresses live in `.env`, which is not.

| Setting | What it does |
|---|---|
| `PIPELINE_VERSION` | Written into every stored object's metadata. Bump it when what the pipeline produces changes meaning. |
| `MAX_FILE_SIZE_MB` | Largest upload accepted, by both the `Content-Length` check and the measured size. Keep in step with `server.maxUploadSize` in `frontend/.streamlit/config.toml`, or Streamlit refuses the file before the API sees it. |
| `ALLOWED_MIME_TYPES` | Comma-separated. May only name types this build can detect from leading bytes — currently `application/pdf`. Anything else stops the service at start-up rather than reading as support that does not exist. |

Connection settings are not here: `database` and `blob_store` each own their
own, which is why `Settings` in [`config.py`](config.py) holds none of them.

## Tests

```sh
poetry run pytest tests/unit/intake/test_intake_ingestion.py
poetry run pytest tests/unit/ingestion tests/integration/test_ingest.py
```

Non-UI tests follow the same page-object structure the UI tests do: one
**driver** per thing under test, exposing the operations a reader cares
about and holding the wiring out of the test body. `IngestDriver` and
`RemovalDriver` live in
[`tests/unit/intake/intake_drivers.py`](../../tests/unit/intake/intake_drivers.py),
over `MemoryDocuments`, `MemoryParsed` and `DocumentRows`. Only the database
and the two buckets are stood in for; the checks, the PDF reader and the
removal ordering are the real code.

| File | Covers |
|---|---|
| [`tests/unit/ingestion/test_uploaded_file.py`](../../tests/unit/ingestion/test_uploaded_file.py) | what an upload's bytes say about it: magic-byte detection, the digest, the size |
| [`tests/unit/intake/test_intake_ingestion.py`](../../tests/unit/intake/test_intake_ingestion.py) | the check order, both size boundaries, the object metadata, the two store-failure paths, the insert race, and the deletion ordering across three stores |
| [`tests/unit/intake/test_intake_cli.py`](../../tests/unit/intake/test_intake_cli.py) | which flag combinations the command line refuses, and that it builds no service until a flag needs one |
| [`tests/integration/test_ingest.py`](../../tests/integration/test_ingest.py), [`test_removal.py`](../../tests/integration/test_removal.py) | both flows against a real PostgreSQL and a real SeaweedFS |
| [`tests/integration/api/test_documents.py`](../../tests/integration/api/test_documents.py) | the HTTP surface: status codes, the digest guard, the refusal shapes |
| [`tests/integration/test_orphans.py`](../../tests/integration/test_orphans.py) | that deleting a document leaves nothing derived from it behind |
| [`tests/frontend/test_views.py`](../../tests/frontend/test_views.py) | the Upload and Documents pages under Streamlit's own runner, including the answers that are refusals |
| [`tests/static/test_intake_pinned.py`](../../tests/static/test_intake_pinned.py) | the settings, the routes, the answer shapes and the modules, pinned |

Statement coverage of `backend/ingestion` is **99%**. The uncovered lines
are the `if __name__ == "__main__"` guard and the body of `build_service`,
which constructs real buckets.

## Known edges

Things that are true, are not bugs, and have surprised somebody.

- **A re-upload's filename is recorded and never shown.** The same bytes
  under a new name are refused as `duplicate_bytes`, so there is no second
  document — but the name they were refused under is written to
  `ingest_events`, and the only reader of that column takes the earliest
  row. Somebody who was told "already held" and later searches for the name
  they used will find nothing.
- **A duplicate by *content* gets through here.** Different bytes carrying
  the same text pass every check on this side, because nothing knows the
  text yet. Parsing refuses them, on `content_sha256`.
- **`Content-Length` is an upper bound.** It covers the whole multipart
  body, so the early size refusal can in principle fire on a file just under
  the limit.
- **An upload that fails on the object store writes no `ingest_events`
  row.** Only refusals are recorded; an unreachable bucket is a 500.
