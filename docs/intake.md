# Intake: ingestion, parsing and chunking

**Status: frozen.** The three stages documented here are complete and are
not being developed further. Changes should be confined to defect fixes,
with a test added beside the fix. Everything downstream — extraction, topic
modelling, question generation — reads what this produces and is documented
in the root `README.md`.

- Code: [`backend/ingestion/`](../backend/ingestion),
  [`backend/preprocessing/parsing/`](../backend/preprocessing/parsing),
  [`backend/preprocessing/chunking/`](../backend/preprocessing/chunking)
- Tests: [`tests/unit/intake/`](../tests/unit/intake),
  [`tests/integration/test_intake_queues.py`](../tests/integration/test_intake_queues.py)

---

## What it does

Intake turns a PDF somebody uploaded into a numbered set of passages that
the rest of the pipeline can cite. That is the whole of its job, and the
reason it is three stages rather than one is that the three have completely
different costs and completely different failure modes.

**Ingestion** is synchronous and cheap. Somebody presses Submit and wants an
answer while they are still looking at the screen, so this stage only does
what can be done in about a second: decide what the file actually is from
its leading bytes, check it against a size limit and an allowlist, check
whether the corpus already holds those exact bytes, read the page and
character counts with a PDF library, put the bytes in an object store, and
write one row. Every attempt is recorded — including the refused ones, which
write no document at all and would otherwise leave no trace that anybody
tried.

**Parsing** is asynchronous and expensive. Converting a PDF into a
structured document means running layout and table-structure models over
every page; on two CPU cores a 56-page document has taken over 990 seconds.
So it runs in a worker, one document at a time, claimed off a queue. It
dispatches on the media type recorded at ingest, converts with the pipeline
that owns that format, writes the converted document to a second bucket, and
only then reads out of it the handful of values the database holds: the
title, the language, a digest of the normalised body text, and the
converter's own confidence in its work. Storing before reading means a bug
in the reading does not mean paying for the conversion twice.

**Chunking** is asynchronous and moderate. It reads the converted document
back out of the bucket — never the original PDF, so re-chunking never
re-parses — and cuts it into passages. A passage is the unit everything
downstream works in: it is what gets embedded, what a fact is drawn from,
what a topic model is fitted over, and what a question is ultimately traced
back to. Cutting is done by the converter's own hybrid chunker, which splits
on the document's structure first and only then merges or divides by token
count, so a passage keeps the heading trail it sat under. Each passage is
then read for three more things: which language it is actually in, the
numbered units a fact may cite, and the content lemmas the topic model needs.

Three properties hold across all three stages and are worth stating up
front, because most of the design follows from them.

1. **A document's identity is the SHA-256 of its bytes.** It is the primary
   key, it is the object key in both buckets, and it is what a duplicate is
   detected by. No filename, no declared content type and no extension is
   evidence of anything.

2. **Nothing starts by itself.** A row arrives `new`, and no worker looks at
   a `new` row. Somebody — a person on the Documents page, a `make` target,
   an HTTP call — moves it to `pending`, which is the only status a worker
   claims. A stage therefore cannot set the next stage going, which is why
   parsing does not touch `chunk_status` and chunking never reads
   `parse_status`.

3. **A citation is an index, not a quote.** Chunking numbers every sentence
   of every passage and records its character offsets. A fact says "sentence
   4"; the evidence span is then resolved from the offsets. A citation cannot
   be approximately right, and no similarity threshold exists anywhere in
   this service. For a table, which has no sentences, the rendered Markdown
   rows are numbered the same way, so a table cell is citable by exactly the
   same mechanism.

---

## The steps

### 1. Accept an upload

**Where** [`ingestion/service.py`](../backend/ingestion/service.py) ·
`IngestService.ingest`
**Runs in** the API process, synchronously
**In** a filename and the raw bytes
**Out** an `IngestResult` (`stored`, `duplicate_bytes`, `too_large` or
`unsupported_type`), one `ingest_events` row always, and on success one
`documents` row plus one object in the `documents` bucket

The checks run cheapest first and nothing is written until all of them pass:

| Order | Check | Refusal |
| --- | --- | --- |
| 1 | `size_bytes > MAX_FILE_SIZE_MB` | `too_large`, carrying both figures |
| 2 | leading bytes are in `ALLOWED_MIME_TYPES` | `unsupported_type`, naming what was detected |
| 3 | the digest is not already in `documents` | `duplicate_bytes` |
| 4 | the PDF opens and is not password protected | `unsupported_type`, carrying the reader's reason |

The order of the last two matters: hashing bytes already in memory is
cheaper than walking every page, so a re-upload never reads the document.

Then the two writes, in a fixed order. The object goes first, because the
key is content-addressed — an object with no row holds exactly the bytes
that key means and is harmless, while a row pointing at a missing object
fails every later stage. If the row cannot be written, the object is taken
back out.

The API adds one check in front of all of this: if `Content-Length` already
exceeds the limit, the body is never read.
[`refuse_oversized`](../backend/ingestion/service.py) writes the event and
nothing else, since with no bytes there is no digest and no document.
`Content-Length` covers the whole multipart body, so it is an upper bound
rather than the file's own size.

The stored object carries four pieces of S3 user metadata — the digest, the
original filename, the time, and `PIPELINE_VERSION` — which is enough for
the bucket alone to rebuild the row. Nothing mutable is included, because S3
metadata cannot change without rewriting the object.

### 2. Convert to a structured document

**Where** [`parsing/service.py`](../backend/preprocessing/parsing/service.py) ·
`ParsingService.process_next`
**Runs in** the `parse-worker`
**In** one claimed row of `documents` where `parse_status = 'pending'`, and
the object it names in the `documents` bucket
**Out** one JSON object in the `parsed` bucket, and `title`, `language`,
`content_sha256`, `parse_confidence`, `parse_confidence_low`,
`parse_status = 'parsed'` on the row

In order:

1. **Dispatch.** [`PipelineRegistry`](../backend/preprocessing/parsing/pipelines/__init__.py)
   picks the pipeline for the media type recorded at ingest. A type no
   pipeline handles fails the document with the list of types that are
   handled, which is what says `ALLOWED_MIME_TYPES` has drifted from the
   registry.

2. **Decide whether the document is scanned.** `char_count / page_count <
   PARSING_OCR_CHAR_THRESHOLD`, read from the counts ingestion already took,
   so it costs nothing. A document with no counts at all counts as scanned.
   OCR is not implemented, so a scanned document is refused *here*, where the
   reason is still known, rather than as "no extractable body text" three
   steps later.

3. **Convert.** Docling, with heading hierarchy and table structure both on
   and OCR off. The conversion is wrapped so that any failure becomes a
   `ConversionFailed` carrying the exception's type and message.

4. **Repair the text.** Before anything reads the document,
   [`_mend`](../backend/preprocessing/parsing/pipelines/pdf.py) rewrites every
   string in it: compose to NFC, rejoin words the page broke across two lines
   (both the soft-hyphen spelling and a real hyphen before a lower-case
   continuation, unless a coordinating conjunction follows — `Zoll- und
   Steuerrecht` is not a broken word), drop control and zero-width
   characters, and fold every other kind of space to U+0020. This runs on the
   text items *and* on the table grid, because a passage's text comes from
   the first and its `table_cells` from the second. The reason it matters is
   downstream: a character a model cannot reproduce is one no quote of it can
   match, and the evidence gate reports that as the model inventing a quote.

5. **Store the converted document** in the `parsed` bucket, keyed by the
   source digest, as the converter's own exported dictionary. Re-parsing
   overwrites rather than accumulating versions.

6. **Read out of it.** [`DocumentAnalyser`](../backend/preprocessing/parsing/analysis.py)
   renders the body as text (which includes table values and excludes running
   headers and footers), finds the title, detects the language, and hashes the
   body with whitespace collapsed and case folded. A conversion with no body
   text fails as `EmptyDocument`.

7. **Two refusals after the fact.** A conversion whose *lower-bound*
   confidence is under `PARSING_MIN_CONFIDENCE` fails — the lower bound
   rather than the mean, because it is the figure that says how bad the worst
   of the document is. And a document whose `content_sha256` another document
   already holds fails as well: ingestion catches a file uploaded twice by
   its bytes, but the same report released as a second PDF has different
   bytes and the same text, and is only knowable here.

Both of those run *after* the object is written, so a document that fails
either one leaves a converted form in the `parsed` bucket with no row
pointing at it. It is collected when the document is deleted.

`complete` deliberately writes neither `page_count` (ingestion's column) nor
`chunk_status` (chunking's). A re-parse therefore leaves the old passages
standing until somebody asks for them to be rebuilt.

### 3. Cut into passages

**Where** [`chunking/passages.py`](../backend/preprocessing/chunking/passages.py) ·
`PassageBuilder.build`
**Runs in** the `chunk-worker`
**In** one claimed row of `documents` where `chunk_status = 'pending'`, and
the JSON object it names in the `parsed` bucket
**Out** every row of `passages` for that document, replaced, plus
`oversized` and `chunk_status = 'chunked'` on the document

1. **Cut.** Docling's `HybridChunker`, sized by the tokenizer of
   `EMBEDDING_MODEL` rather than by its own English default, with
   `CHUNKING_MERGE_PEERS` deciding whether undersized neighbours sharing a
   heading are combined. Every kind of block is serialised into the passage
   text as Markdown — tables included — because a block either renders into
   the text or it does not exist, and evidence has to be a span of the
   passage text.

2. **Number and count.** [`chunking_of`](../backend/preprocessing/chunking/passages.py)
   drops chunks holding only whitespace, numbers the rest from 1
   contiguously, and counts how many exceed the token budget. Nothing is
   discarded on size: `oversized` is a tripwire, because the chunker splits
   on the same budget it is counted against, so anything above it means the
   split did not happen. A document that yields no passages at all fails as
   `NoPassages` rather than being stored empty.

3. **Read each chunk into a row.** `_passage` records the stripped text (the
   string every offset indexes), the heading trail, the block type, the
   converter's item references, one bounding box per page spanned normalised
   to a top-left origin, and the cell grid of every table.

   The table handling is the fiddliest part of the service and is worth
   spelling out. A table is rendered into the passage text as Markdown rows,
   and the grid is kept separately because the rendering loses the header
   flags, the positions and the spans. Each data row of the grid is then
   paired with the rendered line that shows it, matched *by position* —
   every one of a row's values must sit at its own column index, or a table
   of repeated counts matches nearly every row. Lines are consumed in order,
   so two identical rows take two different lines. A data row this passage
   does not render is dropped, because it belongs to another piece of a table
   the chunker split and keeping it would let the extractor describe a value
   with a row the passage never shows. Header rows are kept whether matched
   or not and carry no line. A `row_header` cell does *not* make its row a
   header row: it marks the stub in the first column of an ordinary data row,
   and reading it as a header left 82% of this corpus's cells with nothing
   quotable.

4. **Read the linguistic surface.** `_read` detects each passage's language
   on its own text, falling back to the document's when the passage is too
   short to judge — these documents carry an English summary of a German
   report and a language switcher in every link, so a document-wide label
   sends half the passages to the wrong pipeline. Then, one spaCy batch per
   language: the numbered units a fact may cite (sentences for prose with
   their finite-verb count, rendered rows for a table) and the content lemmas.
   Every passage is read for vocabulary, a table included, since its headings
   and cell values are what it is about.

5. **Store.** One transaction deletes the document's old passages, inserts
   the new ones, writes `oversized` and sets the status, so a document can
   never read as chunked while holding another run's passages. The delete
   cascades to the facts drawn from the old passages and the topic
   memberships they held — which is why re-chunking is a destructive
   operation and is offered separately from re-parsing.

### 4. Re-read stored passages (a side door)

**Where** [`chunking/service.py`](../backend/preprocessing/chunking/service.py) ·
`revocabulary`
**In** every stored passage's text and its document's language
**Out** `passages.language` and `passages.lemmas`, replaced in place

Applies a change to how language or vocabulary is read without re-chunking,
which would delete every passage and take its facts with it. Sentence
offsets are deliberately left alone: a fact cites one by index, so moving
them would point every citation in the corpus at different text.

### 5. Delete

**Where** [`ingestion/removal.py`](../backend/ingestion/removal.py)

Two operations, because they are different decisions.

`delete` removes everything: the object in `documents`, the object in
`parsed`, and the row — which cascades to passages, facts, topic memberships
and questions. Only the `ingest_events` rows survive, on `ON DELETE SET
NULL`, so the record that the upload happened outlives the document. Nothing
is atomic across two stores, so the order is fixed: objects first, row last.

`delete_derived` drops only the passages (and what cascades from them) and
returns `chunk_status` to `new`. Both objects stay, so the next run rebuilds
the passages without re-parsing.

---

## Inputs and outputs

**In:** one PDF, up to `MAX_FILE_SIZE_MB`, born-digital (carrying a text
layer). Posted to `POST /documents` or picked on the Upload page.

**Out, in PostgreSQL:**

| Table | Written by | Holds |
| --- | --- | --- |
| `ingest_events` | ingestion | every upload attempt, refused ones included: the submitted filename, the size, the outcome, the reason |
| `documents` | ingestion, then parsing | the digest, media type, page and character counts; then title, language, content digest, both confidences; plus one status/error/claim triple per stage and `oversized` |
| `passages` | chunking | the text, the language, the numbered units, the lemmas, the pages, the heading trail, the block type, the converter's item references, the bounding boxes and the table grids |

**Out, in SeaweedFS:**

| Bucket | Key | Holds |
| --- | --- | --- |
| `documents` | `{sha[0:2]}/{sha[2:4]}/{sha}.pdf` | the uploaded bytes, exactly as they arrived, never expiring |
| `parsed` | `{sha[0:2]}/{sha[2:4]}/{sha}.json` | the converted document, regenerable from the above |

Both keys are derived in code and never stored. The two-by-two hex fanout is
not cosmetic: the filer keys its metadata on (directory hash, name), so one
flat directory becomes a write hotspot.

**Read by:** extraction (the passage text, its numbered units, its table
grids and its language), topic modelling (`passages.lemmas`, partitioned by
`passages.language`), question generation (through facts, back to the
passage). Nothing downstream reads either bucket.

---

## How it is driven

Every operation exists in three places, and they are the same operation.

| | HTTP | Command line | UI |
| --- | --- | --- | --- |
| upload | `POST /documents` | — | Upload page |
| list | `GET /documents`, `GET /documents/names` | `make documents` | Documents page |
| fetch the file | `GET /documents/{sha}/file` | — | Documents page, "View the file" |
| delete | `DELETE /documents/{sha}` | `make delete SHA=…` | Documents page, danger zone |
| delete derived | `DELETE /documents/{sha}/derived` | `make delete-derived SHA=…` | Documents page, danger zone |
| queue depth | `GET /parsing/status`, `GET /chunking/status` | `make parse-status`, `make chunk-status` | the bars on Documents and Passages |
| start / stop / retry / rerun | `POST /{stage}/{action}` | `make parse-start` etc. | the per-document control rows |
| the same, for one document | `POST /{stage}/document/{sha}/{action}` | `--only document=<sha>` | the same controls |
| re-read vocabulary | — | `make chunk-revocabulary` | — |
| read the passages | `GET /passages`, `GET /passages/{id}`, `GET /passages/types` | — | Passages page |

The API never does the work. A conversion running in the process serving
JSON held it for sixteen minutes; the routes only move rows between
statuses, and the workers pick up whatever has become claimable on their
next poll (`WORKER_POLL_SECONDS`).

The workers are `parse-worker` and `chunk-worker` in `compose.yaml`, each
running `python -m preprocessing.{stage}.run --watch`. Scaling either is
`--scale`: claims use `FOR UPDATE SKIP LOCKED`, so a second worker takes the
following row rather than blocking on the one the first holds. A worker
killed mid-document leaves its row `in_progress` with a timestamp; the next
run's sweep fails any claim older than the stage's lease (two hours for
parsing, thirty minutes for chunking) with "the worker did not finish".

---

## Tools, and why each one is here

| Tool | Where | Why this one |
| --- | --- | --- |
| **PyMuPDF** | `ingestion/pdf.py` | Counts pages and extractable characters in about a second, which is what lets the upload answer while somebody is watching. It also detects encryption, so a password-protected file is refused by that name. Summed page by page, so a long document's text is never all held at once. It is *not* a type check: handed bytes no magic number claims it returns a one-page empty document, so the leading-byte check is what refuses those, and it runs first. |
| **Docling** | `parsing/pipelines/pdf.py` | The conversion: layout, reading order, heading hierarchy and TableFormer cell grids, with a confidence score per document and per page. Configured with its own `docling-parse` backend, not pypdfium2, because that is the backend that reports the font style heading levels are ranked by and the word cells table structure is matched against. `enforce_same_font` is turned **off**: left on, a text cell is cut wherever the font changes, and an f-ligature is set in a font of its own — 397 splits on one 73-page document, none with the flag off. |
| **docling-core** | `chunking/passages.py` | `HybridChunker`, which cuts on document structure before merging or dividing by token count, so a passage keeps its heading trail; and the Markdown serialisers, which put table and picture content into the passage text. |
| **transformers** (tokenizer only) | `chunking/passages.py` | Sizes a passage with the tokenizer of the model that will embed it. One name for both, so a passage cannot be sized by one tokenizer and embedded by another — that is silent truncation at embed time, and a benchmark built on truncated passages misreports its own coverage. No model weights are loaded here, only the tokenizer. |
| **lingua** | `nlp/language.py` | Per-passage language detection, over the languages `NLP_MODELS` names and no others: detecting a language this deployment has no pipeline for tells nobody anything. Web addresses are stripped before detection and the length floor is measured on what is left. |
| **spaCy** | `nlp/analysis.py` | Sentence boundaries, the finite-verb count per sentence, and the content lemmas. See the note below. |
| **SQLAlchemy** | every `repository.py` | The queue (`FOR UPDATE SKIP LOCKED`), the transactional passage replacement, and the listings. Column comments live on the models and are carried into the database by the migrations. |
| **boto3** | `blob_store/seaweedfs/` | The S3 API of the SeaweedFS gateway. |
| **OpenTelemetry** | all three services | One span per document with `stage.name` and `stage.outcome`, so one trace query finds everything that failed whichever stage failed it. |

### A note on spaCy

spaCy is used in exactly three places in this service, and it is worth
naming what it is and is not doing.

- **Sentence segmentation** ([`nlp.analysis.read`](../backend/nlp/analysis.py)),
  called once per passage batch by chunking. This is the load-bearing use:
  the sentence offsets it produces are what every fact in the corpus cites.
  A trained pipeline rather than the rule-based splitter, because the
  rule-based one breaks German legal text on `Abs.`, `Nr.` and `z. B.`.
- **The finite-verb count per sentence**, stored as `predicates` on each
  numbered unit. It is what tells a sentence that asserts something from a
  heading or a caption, which is how extraction decides a passage is not
  worth a model call at all.
- **Content lemmas** — nouns, proper nouns and adjectives, lemmatised — which
  are the vocabulary the topic model is fitted over. Written here so one
  segmentation serves both stages and a fit reads a column instead of
  re-tokenising the corpus.

Three deliberate choices sit inside that third one. Web addresses are
excluded, because Markdown renders a link as `[text](url)` and the
tokenizer offers every path segment as a word — one repeated link put
`publikationen`, `fokus`, `im` and `en` among the most widespread terms in
the corpus. A lemma must be alphabetic and longer than one character,
because a lemmatiser handed a word from another language returns the
placeholder `--`. And in a language named in `NLP_CAPITALISED_NOUNS`, a
lower-case word tagged as a noun is treated as foreign and dropped: the
German pipeline tags the English `the`, `and` and `of` as proper nouns, and
they reached the top terms of two of the twelve German topics.

Everything else in `nlp/analysis.py` — `claim`, `vocabulary`, `content`,
`_units`, `_references` — belongs to extraction and question generation, not
to this service. The medium models are required rather than the small ones:
`de_core_news_sm` does not tag a German modal as a finite verb, so
`Ein Risikobericht muss erstellt werden` reads as no claim at all. Both
models must be present in the image; nothing downloads one at run time.

`spacy.__version__` is recorded on every fact, because the parser decides
the verdicts and two parsers are two datasets.

---

## Configuration

Every setting is read from the environment at start-up and there are no
defaults in code: a missing variable stops the service rather than running
with a value nobody chose. `tests/static/test_settings_documented.py` gates
that every name the code reads is declared in a file.

- **Tuning** lives in [`configs/env/backend.env`](../configs/env/backend.env),
  which is in git. `compose` hands it to each service with `env_file` and the
  `Makefile` sources it for the host commands, so one value reaches both.
- **Credentials, ports and addresses** live in `.env`, which is not, with
  [`.env.example`](../.env.example) as the template.

### Ingestion

| Setting | Read by | What it does |
| --- | --- | --- |
| `PIPELINE_VERSION` | `IngestService` | Written into every stored object's metadata. Bump it when what the pipeline produces changes meaning. |
| `MAX_FILE_SIZE_MB` | `IngestService`, and the route's `Content-Length` check | Largest upload accepted. Keep in step with `server.maxUploadSize` in `frontend/.streamlit/config.toml`, or Streamlit refuses the file before the API sees it. |
| `ALLOWED_MIME_TYPES` | `IngestService` | Comma-separated. May only name types this build can detect from leading bytes (currently `application/pdf`); anything else stops the service at start-up rather than reading as support that does not exist. |

### Parsing

| Setting | What it does |
| --- | --- |
| `PARSING_OCR_CHAR_THRESHOLD` | Characters per page below which a document is treated as scanned and refused. Also the switch OCR will read when it is added. |
| `PARSING_MIN_CONFIDENCE` | Lowest lower-bound confidence a conversion may carry and still be kept, in [0, 1]. Measured over this corpus the lower bounds run 0.72 to 0.94, so 0.5 refuses a genuinely broken conversion and nothing merely untidy. A document below it fails with the measurement. |
| `PARSING_TABLE_MODE` | TableFormer mode: `fast` or `accurate`. Anything else is refused when the pipeline is constructed. |
| `PARSING_HEADING_HIERARCHY` | Whether the converter rebuilds the heading tree. Off, every section header comes back at level one. Turning it on also turns on `generate_parsed_pages`, which the hierarchy reads font style from. |
| `PARSING_TIMEOUT_SECONDS` | Longest one conversion may run; Docling has no timeout of its own, so one pathological file would occupy a worker indefinitely. It must sit well above an ordinary document on the slowest hardware it will run on. The parsing lease is four times this. |
| `DOCLING_ARTIFACTS_PATH` | Where Docling's model weights are. Read by Docling itself, so it must be a real path or absent — an empty value resolves to the working directory and every conversion fails. Commented out lets Docling download the weights on first use, which fails in a container with no network. |

### Chunking

| Setting | What it does |
| --- | --- |
| `EMBEDDING_MODEL` | The model whose tokenizer sizes a passage, and the model that will embed it. One name for both. |
| `EMBEDDING_MAX_TOKENS` | The token budget, and so the longest passage the chunker emits. Keep it equal to the model's context window: larger silently truncates at embed time, smaller wastes the window and fragments tables for nothing. |
| `CHUNKING_MERGE_PEERS` | Whether undersized neighbours sharing a heading are combined. Off leaves one-line passages, which retrieve poorly and carry too little to draw a fact from. |

There is deliberately no character floor and no character ceiling. A
character count is a different unit from the token budget, and being a
different unit is how a ceiling came to fire before the budget it was meant
to back up and discard whole tables that were within it.

### Language

| Setting | What it does |
| --- | --- |
| `NLP_MODELS` | `language:model` pairs. The spaCy pipeline per language, and also the list the detector may answer with. Both models must be in the image. |
| `NLP_DEFAULT_LANGUAGE` | Used for a passage too short to detect and for any language `NLP_MODELS` does not name. Must be one of the languages it names. |
| `NLP_CAPITALISED_NOUNS` | Languages that write every noun with a capital. In one of these, a lower-case word tagged as a noun is treated as a foreign word rather than a subject. Empty is valid. |

### Shared

`DATABASE_URL`, `DATABASE_POOL_SIZE`, `DATABASE_POOL_OVERFLOW`,
`S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKETS`,
`WORKER_POLL_SECONDS`, `LOG_LEVEL`. None of these belongs to this service:
`database` and `blob_store` each own their own connection settings, which is
why no `Settings` class here has a single one of them.

### What is not configurable, on purpose

There are no similarity thresholds anywhere in this service. A fact cites a
numbered sentence rather than quoting one, so a citation is either exact or
names a sentence that does not exist. Nothing here is tuned by a cut-off
somebody has to pick.

---

## Tests

335 tests are about these three packages alone; 507 run when the layers they
share with the rest of the pipeline are included. Together they put statement
coverage of `backend/ingestion` and `backend/preprocessing` at **99%** (932
statements, 10 uncovered).

```sh
make test                                  # everything that gates
poetry run pytest tests/unit/intake        # the driver-based suites
poetry run pytest tests/integration/test_intake_queues.py
```

### Layout

Non-UI tests follow the same page-object structure the UI tests do: one
**driver** per thing under test, exposing the operations a reader cares about
and holding the wiring out of the test body. The drivers and the in-memory
doubles are in
[`tests/unit/intake/intake_drivers.py`](../tests/unit/intake/intake_drivers.py):
`IngestDriver`, `RemovalDriver`, `ParseDriver`, `ChunkDriver` and
`RevocabularyDriver`, over `MemoryDocuments`, `MemoryParsed`,
`DocumentRows`, `ParseRows`, `ChunkRows` and `PassageRows`. Only the database
and the two buckets are stood in for. Everything else — the checks, the
analyser, the chunk reader, the spaCy pipelines, real `DoclingDocument`s — is
the real code.

| File | Covers |
| --- | --- |
| [`tests/unit/ingestion/test_uploaded_file.py`](../tests/unit/ingestion/test_uploaded_file.py) | what an upload's bytes say about it: magic-byte detection, the digest, the size |
| [`tests/unit/intake/test_intake_ingestion.py`](../tests/unit/intake/test_intake_ingestion.py) | the ingest flow and the removal flow, plus the PDF reader itself: the check order, both size boundaries, the metadata, the two store-failure paths, the insert race, and the deletion ordering across three stores |
| [`tests/unit/intake/test_intake_parsing.py`](../tests/unit/intake/test_intake_parsing.py) | dispatch, the scanned decision at its boundaries, the store-before-read ordering, all five failure paths, the confidence floor, the content-digest refusal, and the analyser (title, body, language, content hash, page count) |
| [`tests/unit/intake/test_intake_chunking.py`](../tests/unit/intake/test_intake_chunking.py) | numbering, the token budget, block-type labelling, the row-to-line pairing, the cell grids, the bounding boxes, one chunk into one passage, the flow's failure paths, and the re-read |
| [`tests/unit/intake/test_intake_surface.py`](../tests/unit/intake/test_intake_surface.py) | the spaCy-backed reading, with the real pipelines: per-passage language, sentence numbering and offsets, predicate counts, table rows, and what does and does not become vocabulary |
| [`tests/unit/intake/test_intake_cli.py`](../tests/unit/intake/test_intake_cli.py) | the three command lines: which flag combinations are refused, the exit codes, and that neither stage builds its service until a flag needs it |
| [`tests/unit/intake/test_intake_regression.py`](../tests/unit/intake/test_intake_regression.py) | the regression suite — see below |
| [`tests/unit/chunking/test_chunking.py`](../tests/unit/chunking/test_chunking.py), [`test_lines.py`](../tests/unit/chunking/test_lines.py) | the chunk reader's decisions and the numbering of a rendered table |
| [`tests/unit/parsing/test_text_repair.py`](../tests/unit/parsing/test_text_repair.py) | the text repair, case by case, on real corpus strings |
| [`tests/unit/nlp/`](../tests/unit/nlp) | sentence numbering, claim counting and language detection on their own |
| [`tests/property/test_invariants.py`](../tests/property/test_invariants.py) | Hypothesis over arbitrary text: every non-blank chunk is stored, ordinals run from 1 without a gap, `oversized` never exceeds the passages stored, the repair is idempotent and leaves no control character and no exotic space, and a search never reaches the database as a wildcard |
| [`tests/integration/test_intake_queues.py`](../tests/integration/test_intake_queues.py) | the three repositories against a real PostgreSQL: which columns each stage writes and which it must leave alone, the transactional passage replacement and its cascade to facts, `NULL` versus JSON `null`, the language CHECK constraint, and every listing filter |
| [`tests/integration/test_ingest.py`](../tests/integration/test_ingest.py), [`test_removal.py`](../tests/integration/test_removal.py) | the same two flows against a real PostgreSQL and a real SeaweedFS |
| [`tests/integration/database/test_queue.py`](../tests/integration/database/test_queue.py) | the claim mechanics, including `SKIP LOCKED` under concurrent workers and the abandoned-claim sweep |
| [`tests/integration/api/test_documents.py`](../tests/integration/api/test_documents.py), [`test_catalogue.py`](../tests/integration/api/test_catalogue.py), [`test_stages.py`](../tests/integration/api/test_stages.py) | the HTTP surface: status codes, the digest guard, the queue verbs and their narrowed forms |
| [`tests/e2e/test_pipeline.py`](../tests/e2e/test_pipeline.py) | one document through every stage in this process, against the real stores: the handovers, that nothing starts until asked, and that dropping the derived data lets chunking repeat without re-parsing |
| [`tests/frontend/test_views.py`](../tests/frontend/test_views.py) | the Documents, Passages and Upload pages under Streamlit's own runner, against a scripted backend, including the answers that are refusals |
| [`tests/contract/test_openapi.py`](../tests/contract/test_openapi.py) | the published schema, so a route cannot change shape unnoticed |

### The regression suite

[`test_intake_regression.py`](../tests/unit/intake/test_intake_regression.py)
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

The 10 uncovered statements are: four `if __name__ == "__main__"` guards;
the bodies of the three `build_service` factories, which construct real
buckets and download a tokenizer; and the five lines of `PdfPipeline.convert`
that call Docling itself, whose layout and table models are a multi-gigabyte
download. What those five lines hand to and take from Docling is covered —
`_repaired` and `_score` on the way out, and the scanned guard on the way in
— and the e2e layer exercises the surrounding flow with the converter stood
in for.

Not covered at all, by choice: OCR (not implemented), any format other than
PDF (no pipeline exists), and the behaviour of the real embedding tokenizer
on a corpus (a property of the model, not of this code).

---

## Known edges

Things that are true, are not bugs, and have surprised somebody:

- **A re-parse does not re-chunk.** `parse_status` returning to `parsed`
  leaves the old passages in place. After re-parsing, redo chunking.
- **A re-chunk is destructive.** It deletes every passage of the document,
  which cascades to the facts drawn from them, the topic memberships they
  held and the questions resting on those facts.
- **A failed parse can leave an object in the `parsed` bucket.** The
  converted document is stored before the confidence and duplicate-text
  checks run. It is collected when the document is deleted.
- **A document is only searchable by the name it first arrived under.** The
  listing carries one filename per document, the earliest; a re-upload under
  another name is recorded in `ingest_events` and does not become a second
  way to find the document.
- **`Content-Length` is an upper bound.** It covers the whole multipart body,
  so the early size refusal can in principle fire on a file just under the
  limit. A request declaring no length at all is read in full and measured
  afterwards.
- **An upload that fails on the object store writes no `ingest_events` row.**
  Only refusals are recorded; an unreachable bucket is a 500.
- **`oversized` should always read 0.** Anything above 0 means the chunker's
  split did not happen and those passages will be truncated when embedded.
- **A passage too short to detect a language for gets `NULL`**, which puts it
  outside every topic model. That is intended; the topic model is fitted per
  language.
