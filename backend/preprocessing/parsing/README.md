# Parsing

Turns the uploaded PDF into a structured document: headings in a hierarchy,
tables as cell grids, text in reading order.

This is the expensive stage — layout and table-structure models over every
page — so it runs in a worker, one document at a time, off a queue.

The converted document is written to its bucket **before** anything reads
values out of it, so a bug in the reading does not mean paying for the
conversion twice. Two refusals therefore fire after the object exists.

Nothing this stage does starts the next one. `complete` writes neither
`page_count`, which is ingestion's, nor `chunk_status`, which is chunking's,
so a re-parse leaves the old passages standing.

The other two stages that turn a PDF into passages are
[ingestion](../../ingestion/README.md) and [chunking](../chunking/README.md).

## What it does

One worker process, `parse-worker`, running
`python -m preprocessing.parsing.run --watch`. It claims the next document
whose `parse_status` is `pending` with `FOR UPDATE SKIP LOCKED`. A worker
killed mid-document leaves its row `in_progress`; the next run's sweep fails
any claim older than the lease — four times `PARSING_TIMEOUT_SECONDS`.

### Step 1 — Dispatch on the media type

[`PipelineRegistry`](pipelines/__init__.py) picks the pipeline that owns the
format ingestion detected from the leading bytes. A type no pipeline handles
fails the document with the list of types that are handled.

### Step 2 — Decide whether the document is scanned

`char_count / page_count < PARSING_OCR_CHAR_THRESHOLD`, read from counts that
already exist. A document with no counts counts as scanned.

OCR is not implemented, so a scanned document is refused here, where the
reason is still known.

### Step 3 — Convert

Docling, with heading hierarchy and table structure on and OCR off. Any
failure becomes a `ConversionFailed` carrying the exception's type and
message, and the call is bounded by `PARSING_TIMEOUT_SECONDS` — Docling has
no timeout of its own.

### Step 4 — Repair the text

[`_mend`](pipelines/pdf.py) rewrites every string before anything reads it:

- compose to NFC
- rejoin words the page broke across two lines — both the soft-hyphen
  spelling and a real hyphen before a lower-case continuation, unless a
  coordinating conjunction follows, because `Zoll- und Steuerrecht` is not a
  broken word
- drop control and zero-width characters
- fold every other kind of space to U+0020

It runs on the text items *and* on the table grid. A character a model cannot
reproduce is one no quote of it can match.

### Step 5 — Store the converted document

The converter's own exported dictionary, one JSON object in the `parsed`
bucket keyed by the source digest. Re-parsing overwrites. This is what
chunking reads, so re-chunking never re-parses.

### Step 6 — Read the values the row holds

[`DocumentAnalyser`](analysis.py) renders the body as text — table values
included, running headers and footers excluded — finds the title, detects the
language, and hashes the body with whitespace collapsed and case folded. A
conversion with no body text fails as `EmptyDocument`.

### Step 7 — Two refusals after the fact

A conversion whose **lower-bound** confidence is under
`PARSING_MIN_CONFIDENCE` fails; the lower bound rather than the mean, because
it says how bad the worst of the document is.

A document whose `content_sha256` another document already holds fails too.
Ingestion catches a file uploaded twice by its bytes; the same report
released as a second PDF has different bytes and the same text.
`content_sha256` is not written on the copy that fails, so every later copy
fails against the one that parsed.

Both run after Step 5, so a document failing either leaves a converted form
in the `parsed` bucket with no row pointing at it. It is collected when the
document is deleted.

## Inputs and outputs

**In:** one claimed row of `documents` where `parse_status = 'pending'`, and
the object it names in the `documents` bucket.

**Written, in PostgreSQL:** on `documents` — `title`, `language`,
`content_sha256`, `parse_confidence`, `parse_confidence_low`, and the
`parse_status` / `parse_error` / `parse_claimed_at` triple. Nothing else.

**Written, in SeaweedFS:** `parsed`, at
`{sha[0:2]}/{sha[2:4]}/{sha}.json` — the converted document, regenerable from
the uploaded bytes.

| Route | Answers |
|---|---|
| `GET /parsing/status` | The queue depth, and whether a worker is on it |
| `POST /parsing/{action}` | `start`, `stop`, `retry`, `rerun` |
| `GET\|POST /parsing/{scope}/{value}/…` | The same, narrowed to one document |

The routes only move rows between statuses; the worker picks up whatever
became claimable on its next poll.

**Command line:** `make parse-status`, `parse-start`, `parse-stop`,
`parse-retry`, `parse-rerun`, `parse`; `--only document=<sha>` narrows any of
them.

**Read by:** [chunking](../chunking/README.md).

## Configuration

Read from the environment at start-up, with no defaults in code. Tuning is in
[`configs/env/backend.env`](../../../configs/env/backend.env).

| Setting | What it does |
|---|---|
| `PARSING_OCR_CHAR_THRESHOLD` | Characters per page below which a document is treated as scanned and refused |
| `PARSING_MIN_CONFIDENCE` | Lowest lower-bound confidence a conversion may carry and be kept, in [0, 1] |
| `PARSING_TABLE_MODE` | TableFormer mode: `fast` or `accurate`. Anything else is refused when the pipeline is constructed |
| `PARSING_HEADING_HIERARCHY` | Whether the converter rebuilds the heading tree. Turning it on also turns on `generate_parsed_pages` |
| `PARSING_TIMEOUT_SECONDS` | Longest one conversion may run. The lease is four times this |
| `DOCLING_ARTIFACTS_PATH` | Where Docling's weights are. Must be a real path or absent — an empty value resolves to the working directory |

The language settings are shared with chunking and described there.
Connection settings belong to `database` and `blob_store`.

Docling is configured with its own `docling-parse` backend rather than
pypdfium2, because that is the backend reporting the font style heading
levels are ranked by. `enforce_same_font` is turned **off**; left on, a text
cell is cut wherever the font changes, and an f-ligature is set in a font of
its own.

## Tests

```sh
poetry run pytest tests/unit/intake/test_intake_parsing.py
poetry run pytest tests/unit/parsing
```

Non-UI tests act through a driver. `ParseDriver` lives in
[`tests/unit/intake/intake_drivers.py`](../../../tests/unit/intake/intake_drivers.py),
over `MemoryDocuments`, `MemoryParsed` and `ParseRows`. Only the database,
the two buckets and Docling itself are stood in for.

| File | Covers |
|---|---|
| [`test_intake_parsing.py`](../../../tests/unit/intake/test_intake_parsing.py) | Dispatch, the scanned decision at its boundaries, the store-before-read ordering, all five failure paths, the confidence floor, the content-digest refusal, and the analyser |
| [`test_text_repair.py`](../../../tests/unit/parsing/test_text_repair.py) | The text repair, case by case, on real corpus strings |
| [`test_intake_regression.py`](../../../tests/unit/intake/test_intake_regression.py) | Fixes nothing at run time would report if they were undone |
| [`test_intake_cli.py`](../../../tests/unit/intake/test_intake_cli.py) | Which flag combinations the command line refuses |
| [`tests/property/test_invariants.py`](../../../tests/property/test_invariants.py) | The repair is idempotent and leaves no control character |
| [`tests/integration/test_intake_queues.py`](../../../tests/integration/test_intake_queues.py) | Which columns this stage writes and which it must leave alone |
| [`tests/integration/database/test_queue.py`](../../../tests/integration/database/test_queue.py) | The claim mechanics, including `SKIP LOCKED` and the abandoned-claim sweep |
| [`tests/e2e/test_pipeline.py`](../../../tests/e2e/test_pipeline.py) | One document through every stage, with the converter stood in for |
| [`tests/static/test_intake_pinned.py`](../../../tests/static/test_intake_pinned.py) | The settings, routes, answer shapes and modules, pinned |

The regression suite holds one test each for: every space codepoint folding
to U+0020 and every zero-width mark removed, named one at a time; every
control character except newline and tab dropped; the repair reaching `orig`
and the table grid; `enforce_same_font` off and the backend Docling's own;
`generate_parsed_pages` following `PARSING_HEADING_HIERARCHY`; `do_ocr` off;
table structure on with mode, timeout and weights path passed through; a bad
table mode refused at construction; and the converter built once.

Not covered: the five lines of `PdfPipeline.convert` that call Docling, whose
models are a multi-gigabyte download. OCR and any format other than PDF are
not covered because neither exists.

## Limits

- **A re-parse does not re-chunk.** After re-parsing, redo chunking.
- **A failed parse can leave an object in the `parsed` bucket.** It is
  collected when the document is deleted.
- **A duplicate by content costs a stored file and a failed row.**
