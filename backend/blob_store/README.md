# Blob store

Object storage, one module per bucket, over any S3 implementation.

One of the three packages that build a connection — the others are
[`database/`](../database/README.md) and [`nlp/`](../nlp/README.md). A
service asks for a bucket; it never builds a client or learns a credential.

## The five buckets

| Bucket | Key | Holds | Expires |
|---|---|---|---|
| `documents` | `{sha[0:2]}/{sha[2:4]}/{sha}.pdf` | The uploaded bytes, exactly as they arrived | never |
| `parsed` | `{sha[0:2]}/{sha[2:4]}/{sha}.json` | The converter's complete output | never |
| `models` | `topics/{language}.npz` | The fitted factorisation each language's topics were read off | replaced by the next fit |
| `export` | `topics/{language}.html` | Each language's pyLDAvis figure, and nothing else | replaced by the next fit |
| `archive` | `{origin}/{the key it had}` | What a deletion took out of the other four | `make archive-purge` |

`models` and `export` are one fit's two halves, and which is which matters.
`models` holds the factorisation; `export` holds a picture drawn from it. The
picture is derivable from the model and the model is derivable from nothing
short of another fit, so they have opposite durability contracts despite
being written in the same pass. Until `models` existed only the picture was
kept, which made a disposable view the one durable trace of an hour's work.

The same split as `parsed`, one level up: the database holds what a stage
reads or a person queries — each topic's top terms, its label, its coverage
flag — and the matrix those were read off stays in the object store.

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

The bucket classes are shared behaviour in [`s3/bucket.py`](s3/bucket.py)
and one subclass per bucket, so a caller names the bucket it means rather
than a string.

## Configuration

Addresses and credentials only, all from `.env`, none configurable at
runtime:

| Setting | What it is |
|---|---|
| `S3_ENDPOINT` | Where the store answers. Empty for AWS, which resolves its own |
| `S3_ACCESS_KEY`, `S3_SECRET_KEY` | The credential. Empty to let boto3 find one — an instance role, a profile |
| `S3_REGION` | `local` for a self-hosted gateway, the real region for AWS, `auto` for R2 |
| `S3_ADDRESSING_STYLE` | `path` for a self-hosted gateway, `auto` for AWS |
| `S3_BUCKETS` | The buckets [`configs/seaweedfs/bucket-init.sh`](../../configs/seaweedfs/bucket-init.sh) creates at first boot |

**Empty is a value and absent is a mistake.** Empty means the SDK's own
default, which is what makes AWS reachable; a name that is not set at all
raises rather than falling back, so a deployment missing its env file fails
here instead of addressing somebody's real account.

Buckets are created by [`bucket-init.sh`](../../configs/seaweedfs/bucket-init.sh)
on SeaweedFS and by the provider's own tooling anywhere else. Nothing here
creates one; `GET /status` reports which are missing.

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
- **No bucket expires on its own.** The `s3.bucket.quota` lines in
  [`bucket-init.sh`](../../configs/seaweedfs/bucket-init.sh) are operator
  notes that file says are not run automatically, and no TTL is set anywhere.
  Each bucket is bounded by what overwrites or purges it, which is the
  `Expires` column above and nothing else.
- **A topic figure cannot be redrawn from the database.** pyLDAvis is
  prepared from the whole term-topic matrix and the database keeps only each
  topic's top terms. It can be redrawn from the `models` object for its
  language, which is what that bucket is for; losing both costs a re-fit.
  `topics-visualise` reads the `export` bucket — it does not render.
- **`make topics-render` is the only thing that reads `models` back.** It
  redraws a language's figure from its stored model, which is what makes
  `export` disposable. No route serves a model; a reader wanting the matrix
  itself fetches the key by hand.
- **There is one model per language, and no way to ask for an older one.**
  The next fit replaces it, because the database holds one fit's topics —
  `replace` drops every row and writes the new fit's. A figure drawn from a
  previous model would number its topics off rows that are gone, so
  `--language` picks which language to redraw, not which fit.
- **An upload that fails on the object store writes no `ingest_events`
  row.** Only refusals are recorded; an unreachable bucket is a 500.
