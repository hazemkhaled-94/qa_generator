# Parsing

Turns the uploaded PDF into a structured document: headings in a hierarchy,
tables as cell grids, text in reading order.

This is the expensive stage. Converting a PDF means running layout and
table-structure models over every page; on two CPU cores a 56-page document
has taken over 990 seconds. So it runs in a worker, one document at a time,
claimed off a queue — never in the process serving JSON, where a conversion
once held a request for sixteen minutes.

The order of operations inside it is the thing worth knowing. The converted
document is written to its bucket **before** anything reads values out of
it, because a bug in the reading should not mean paying for the conversion
twice. Two of the refusals therefore fire after the object exists, and leave
a converted form with no row pointing at it.

Nothing this stage does starts the next one. A row arrives `new` and no
worker looks at a `new` row; somebody moves it to `pending`. `complete`
deliberately writes neither `page_count`, which is ingestion's column, nor
`chunk_status`, which is chunking's — so a re-parse leaves the old passages
standing until somebody asks for them to be rebuilt.

Parsing is the second of three stages that turn a PDF into passages. The
others are [ingestion](../../ingestion/README.md) and
[chunking](../chunking/README.md).

## What it does

### The shape of a run

One worker process, `parse-worker` in `compose.yaml`, running
`python -m preprocessing.parsing.run --watch`. It claims the next document
whose `parse_status` is `pending` with `FOR UPDATE SKIP LOCKED`, so scaling
it with `--scale` means a second worker takes the following row rather than
blocking on the one the first holds.

A worker killed mid-document leaves its row `in_progress` with a timestamp.
The next run's sweep fails any claim older than the lease — four times
`PARSING_TIMEOUT_SECONDS` — with "the worker did not finish".

### Step 1 — Dispatch on the media type

**In:** the media type ingestion detected from the leading bytes.
**Out:** the pipeline that owns that format.

[`PipelineRegistry`](pipelines/__init__.py) picks it. A type no pipeline
handles fails the document with the list of types that are handled, which is
what says `ALLOWED_MIME_TYPES` has drifted from the registry.

### Step 2 — Decide whether the document is scanned

**In:** the page and character counts ingestion already took.
**Out:** a refusal, or nothing.

`char_count / page_count < PARSING_OCR_CHAR_THRESHOLD`. Read from counts
that already exist, so it costs nothing. A document with no counts at all
counts as scanned.

OCR is not implemented, so a scanned document is refused *here*, where the
reason is still known, rather than as "no extractable body text" three steps
later.

### Step 3 — Convert

**In:** the object in the `documents` bucket.
**Out:** a `DoclingDocument` and two confidence figures.

Docling, with heading hierarchy and table structure both on and OCR off. The
call is wrapped so that any failure becomes a `ConversionFailed` carrying
the exception's type and message, and it is bounded by
`PARSING_TIMEOUT_SECONDS`: Docling has no timeout of its own, so one
pathological file would occupy a worker indefinitely.

### Step 4 — Repair the text

**In:** every string in the converted document.
**Out:** the same document, mended in place.

Before anything reads it, [`_mend`](pipelines/pdf.py) rewrites every string:

- compose to NFC
- rejoin words the page broke across two lines — both the soft-hyphen
  spelling and a real hyphen before a lower-case continuation, unless a
  coordinating conjunction follows, because `Zoll- und Steuerrecht` is not a
  broken word
- drop control and zero-width characters
- fold every other kind of space to U+0020

It runs on the text items *and* on the table grid, because a passage's text
comes from the first and its `table_cells` from the second.

The reason this matters is downstream. A character a model cannot reproduce
is one no quote of it can match, and the evidence gate reports that as the
model inventing a quote.

### Step 5 — Store the converted document

**In:** the mended document.
**Out:** one JSON object in the `parsed` bucket, keyed by the source digest.

The converter's own exported dictionary. Re-parsing overwrites rather than
accumulating versions. This is what chunking reads: re-chunking never
re-parses.

### Step 6 — Read the handful of values the row holds

**In:** the stored document.
**Out:** `title`, `language`, `content_sha256`, `parse_confidence`,
`parse_confidence_low`.

[`DocumentAnalyser`](analysis.py) renders the body as text — which includes
table values and excludes running headers and footers — finds the title,
detects the language, and hashes the body with whitespace collapsed and case
folded. A conversion with no body text fails as `EmptyDocument`.

### Step 7 — Two refusals after the fact

**In:** the confidence figures and the content digest.
**Out:** a failed document, or `parse_status = 'parsed'`.

A conversion whose **lower-bound** confidence is under
`PARSING_MIN_CONFIDENCE` fails. The lower bound rather than the mean,
because it is the figure that says how bad the worst of the document is.

A document whose `content_sha256` another document already holds fails too.
Ingestion catches a file uploaded twice by its bytes; the same report
released as a second PDF has different bytes and the same text, and is only
knowable here. `content_sha256` is deliberately not written on the copy that
fails, so the copy that parsed keeps the text and every later copy fails
against that one.

Both run after Step 5, so a document that fails either leaves a converted
form in the `parsed` bucket with no row pointing at it. It is collected when
the document is deleted.

## Tools, and where each is used

| Tool | Where | Why this one |
|---|---|---|
| **Docling** | [`pipelines/pdf.py`](pipelines/pdf.py) | The conversion: layout, reading order, heading hierarchy and TableFormer cell grids, with a confidence score per document and per page. Configured with its own `docling-parse` backend, not pypdfium2, because that is the backend that reports the font style heading levels are ranked by and the word cells table structure is matched against. `enforce_same_font` is turned **off**: left on, a text cell is cut wherever the font changes, and an f-ligature is set in a font of its own — 397 splits on one 73-page document, none with the flag off. |
| **lingua** | `nlp/language.py`, via [`analysis.py`](analysis.py) | The document's language, over the languages `NLP_MODELS` names and no others: detecting a language this deployment has no pipeline for tells nobody anything. Web addresses are stripped before detection. |
| **SQLAlchemy** | [`repository.py`](repository.py) | The queue (`FOR UPDATE SKIP LOCKED`), the claim sweep, and the columns this stage owns. |
| **boto3** | `blob_store/seaweedfs/` | Both buckets: read from `documents`, write to `parsed`. |
| **OpenTelemetry** | [`service.py`](service.py) | One span per document with `stage.name` and `stage.outcome`, so one trace query finds everything that failed whichever stage failed it. |

## Inputs and outputs

**In:** one claimed row of `documents` where `parse_status = 'pending'`, and
the object it names in the `documents` bucket.

**Written, in PostgreSQL:** on `documents` — `title`, `language`,
`content_sha256`, `parse_confidence`, `parse_confidence_low`, and the
`parse_status` / `parse_error` / `parse_claimed_at` triple. Nothing else.

**Written, in SeaweedFS:**

| Bucket | Key | Holds |
|---|---|---|
| `parsed` | `{sha[0:2]}/{sha[2:4]}/{sha}.json` | the converted document, regenerable from the uploaded bytes |

**Serves**

| Route | What it answers |
|---|---|
| `GET /parsing/status` | The queue depth, and whether a worker is on it |
| `POST /parsing/{action}` | `start`, `stop`, `retry`, `rerun` |
| `GET\|POST /parsing/{scope}/{value}/…` | The same, narrowed to one document |

The API never does the work: the routes only move rows between statuses, and
the worker picks up whatever has become claimable on its next poll
(`WORKER_POLL_SECONDS`).

**Command line:** `make parse-status`, `make parse-start`, `make parse-stop`,
`make parse-retry`, `make parse-rerun`, `make parse`; `--only document=<sha>`
narrows any of them.

**Read by:** [chunking](../chunking/README.md), which reads the object in
`parsed` and never the original PDF.

## Configuration

Every setting is read from the environment at start-up and there are no
defaults in code. Tuning lives in
[`configs/env/backend.env`](../../../configs/env/backend.env), which is in
git; credentials, ports and addresses live in `.env`, which is not.

| Setting | What it does |
|---|---|
| `PARSING_OCR_CHAR_THRESHOLD` | Characters per page below which a document is treated as scanned and refused. Also the switch OCR will read when it is added. |
| `PARSING_MIN_CONFIDENCE` | Lowest lower-bound confidence a conversion may carry and still be kept, in [0, 1]. Measured over this corpus the lower bounds run 0.72 to 0.94, so 0.5 refuses a genuinely broken conversion and nothing merely untidy. A document below it fails with the measurement. |
| `PARSING_TABLE_MODE` | TableFormer mode: `fast` or `accurate`. Anything else is refused when the pipeline is constructed. |
| `PARSING_HEADING_HIERARCHY` | Whether the converter rebuilds the heading tree. Off, every section header comes back at level one. Turning it on also turns on `generate_parsed_pages`, which the hierarchy reads font style from. |
| `PARSING_TIMEOUT_SECONDS` | Longest one conversion may run. It must sit well above an ordinary document on the slowest hardware it will run on. The parsing lease is four times this. |
| `DOCLING_ARTIFACTS_PATH` | Where Docling's model weights are. Read by Docling itself, so it must be a real path or absent — an empty value resolves to the working directory and every conversion fails. Commented out lets Docling download the weights on first use, which fails in a container with no network. |

The language settings — `NLP_MODELS`, `NLP_DEFAULT_LANGUAGE` — are shared
with chunking and are described there. Connection settings belong to
`database` and `blob_store`.

## Tests

```sh
poetry run pytest tests/unit/intake/test_intake_parsing.py
poetry run pytest tests/unit/intake/test_intake_regression.py
poetry run pytest tests/unit/parsing
```

Non-UI tests follow the same page-object structure the UI tests do: one
**driver** per thing under test. `ParseDriver` lives in
[`tests/unit/intake/intake_drivers.py`](../../../tests/unit/intake/intake_drivers.py),
over `MemoryDocuments`, `MemoryParsed` and `ParseRows`. Only the database,
the two buckets and Docling itself are stood in for — the dispatch, the
scanned guard, the repair and the analyser are the real code, over real
`DoclingDocument`s.

| File | Covers |
|---|---|
| [`tests/unit/intake/test_intake_parsing.py`](../../../tests/unit/intake/test_intake_parsing.py) | dispatch, the scanned decision at its boundaries, the store-before-read ordering, all five failure paths, the confidence floor, the content-digest refusal, and the analyser (title, body, language, content hash, page count) |
| [`tests/unit/parsing/test_text_repair.py`](../../../tests/unit/parsing/test_text_repair.py) | the text repair, case by case, on real corpus strings |
| [`tests/unit/intake/test_intake_regression.py`](../../../tests/unit/intake/test_intake_regression.py) | the regression suite — see below |
| [`tests/unit/intake/test_intake_cli.py`](../../../tests/unit/intake/test_intake_cli.py) | which flag combinations the command line refuses, the exit codes, and that it builds no service until a flag needs one |
| [`tests/unit/nlp/`](../../../tests/unit/nlp) | language detection on its own |
| [`tests/property/test_invariants.py`](../../../tests/property/test_invariants.py) | Hypothesis over arbitrary text: the repair is idempotent and leaves no control character and no exotic space |
| [`tests/integration/test_intake_queues.py`](../../../tests/integration/test_intake_queues.py) | the repository against a real PostgreSQL: which columns this stage writes and which it must leave alone |
| [`tests/integration/database/test_queue.py`](../../../tests/integration/database/test_queue.py) | the claim mechanics, including `SKIP LOCKED` under concurrent workers and the abandoned-claim sweep |
| [`tests/integration/api/test_stages.py`](../../../tests/integration/api/test_stages.py) | the queue verbs and their narrowed forms over HTTP |
| [`tests/e2e/test_pipeline.py`](../../../tests/e2e/test_pipeline.py) | one document through every stage, with the converter stood in for |
| [`tests/static/test_intake_pinned.py`](../../../tests/static/test_intake_pinned.py) | the settings, the routes, the answer shapes and the modules, pinned; and that one pipeline module exists per format |

### The regression suite

[`test_intake_regression.py`](../../../tests/unit/intake/test_intake_regression.py)
holds only fixes that nothing at run time would report if they were undone.
Each is one test:

- every one of the nineteen space codepoints folds to U+0020, and every one
  of the five zero-width marks is removed, named one at a time — the
  character class once named only some of them and three survived into
  passage text
- every control character except newline and tab is dropped, over the whole
  range
- the repair reaches `orig` as well as `text`, and the table grid as well as
  the text items
- `enforce_same_font` is off, and the backend is Docling's own
- `generate_parsed_pages` follows `PARSING_HEADING_HIERARCHY`
- `do_ocr` is off in the converter as well as in the guard
- table structure is on, and the mode, the timeout and the weights path are
  passed through
- a table mode that is not one is refused when the pipeline is constructed
- the converter is built once and reused

### What is not covered, and why

The five lines of `PdfPipeline.convert` that call Docling itself, whose
layout and table models are a multi-gigabyte download. What those lines hand
to and take from Docling is covered — `_repaired` and `_score` on the way
out, the scanned guard on the way in — and the e2e layer exercises the
surrounding flow with the converter stood in for.

Not covered at all, by choice: OCR, which is not implemented, and any format
other than PDF, for which no pipeline exists.

## Known edges

- **A re-parse does not re-chunk.** `parse_status` returning to `parsed`
  leaves the old passages in place. After re-parsing, redo chunking.
- **A failed parse can leave an object in the `parsed` bucket.** The
  converted document is stored before the confidence and duplicate-text
  checks run. It is collected when the document is deleted.
- **A duplicate by content costs a stored file and a failed row.** The
  document stays at `parse_status = failed`, with its object in both
  buckets, until somebody deletes it.
