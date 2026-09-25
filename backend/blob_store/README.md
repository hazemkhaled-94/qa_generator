# Blob store

Object storage, one module per bucket, over the SeaweedFS S3 gateway.

One of the three packages that build a connection — the others are
[`database/`](../database/README.md) and [`nlp/`](../nlp/README.md). A
service asks for a bucket; it never builds a client or learns a credential.

## The four buckets

| Bucket | Key | Holds | Expires |
|---|---|---|---|
| `documents` | `{sha[0:2]}/{sha[2:4]}/{sha}.pdf` | The uploaded bytes, exactly as they arrived | never |
| `parsed` | `{sha[0:2]}/{sha[2:4]}/{sha}.json` | The converter's complete output | never |
| `export` | `topics/{language}.html` | Each language's pyLDAvis figure | replaced by the next fit |
| `archive` | `{origin}/{the key it had}` | What a deletion took out of the other three | `make archive-purge` |

`archive` is the only one written by a **deletion**. `ArchiveBucket.take`
copies an object across and then deletes the original — in that order, so a
failure between the two leaves it in both places rather than neither. The key
keeps the bucket it came out of as a prefix. See
[`archive/`](../archive/README.md).

A key is **derived in code and never stored**: it is a pure function of the
digest and the media type, so a row holding a stale path is not a state this
can reach.

The two-by-two hex prefix is not cosmetic. The SeaweedFS filer keys its
metadata on (directory hash, name), so one flat directory becomes a write
hotspot; two levels of 256 spread a corpus over 65,536 directories.

`parsed` keeps the converter's **complete** output, so nothing it produced is
ever lost. The database holds only what a later stage reads or a person
queries; figures, formulas as LaTeX, code blocks, per-line geometry and the
heading tree stay here, reachable through `passages.doc_item_refs`. That is
what makes re-chunking cheap.

## Ordering

Nothing is atomic across an object store and a database, so every flow fixes
an order and accepts the one failure that order leaves.

**Writing:** object first, row last. An object with no row holds exactly the
bytes its key means and is harmless; a row pointing at a missing object fails
every later stage. If the row cannot be written, the object is taken back
out.

**Deleting:** objects first, row last. The reverse leaves an object nothing
names.

The bucket classes are shared behaviour in
[`seaweedfs/bucket.py`](seaweedfs/bucket.py) and one subclass per bucket, so
a caller names the bucket it means rather than a string.

## Configuration

Addresses and credentials only, all from `.env`, none configurable at
runtime:

| Setting | What it is |
|---|---|
| `S3_ENDPOINT` | Where the gateway answers |
| `S3_ACCESS_KEY`, `S3_SECRET_KEY` | The credential the gateway was created with |
| `S3_BUCKETS` | The buckets [`configs/seaweedfs/bucket-init.sh`](../../configs/seaweedfs/bucket-init.sh) creates at first boot |

`GET /status` reports whether each required bucket exists, as classes rather
than names.

## Tests

```sh
poetry run pytest tests/unit/blob_store tests/integration/blob
```

| File | Covers |
|---|---|
| [`test_keys.py`](../../tests/unit/blob_store/test_keys.py) | The object key a stored document is found under |
| [`test_bucket.py`](../../tests/integration/blob/test_bucket.py) | Storing and reading objects against a real S3 gateway |
| [`test_removal.py`](../../tests/integration/test_removal.py) | Deleting a document across the three stores that hold one |

## Limits

- **A media type with no known file extension raises rather than storing.**
  A key nothing could find again is worse than a refusal.
- **Rewriting an object in `documents` or `parsed` is harmless.** The key is
  content-addressed.
- **`export` is the one bucket a fit overwrites.** A language modelled before
  the figure was drawn has no object until the next fit, and the route 404s.
- **An upload that fails on the object store writes no `ingest_events`
  row.** Only refusals are recorded; an unreachable bucket is a 500.
