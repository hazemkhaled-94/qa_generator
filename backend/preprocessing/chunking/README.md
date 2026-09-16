# Chunking

Cuts the converted document into passages, and reads each one for the
language, the numbered units a fact may cite, and the vocabulary a topic
model is fitted over.

A passage is the unit everything downstream works in. It is what gets
embedded, what a fact is drawn from, what a topic model is fitted over, and
what a question is ultimately traced back to. This stage is where that unit
comes into existence, which makes it the stage whose decisions are hardest
to undo: re-chunking deletes every passage of a document, and the facts, the
topic memberships and the questions resting on them go with it.

It reads the converted document out of the `parsed` bucket and never the
original PDF, so re-chunking never re-parses.

One idea it is built around:

**A citation is an index, not a quote.** Every sentence of every passage is
numbered and its character offsets recorded. A fact says "sentence 4"; the
evidence span is then resolved from the offsets. A citation cannot be
approximately right, and there is no similarity threshold anywhere in this
stage. For a table, which has no sentences, the rendered Markdown rows are
numbered the same way, so a table cell is citable by exactly the same
mechanism.

Chunking is the last of three stages that turn a PDF into passages. The
others are [ingestion](../../ingestion/README.md) and
[parsing](../parsing/README.md).

## What it does

### The shape of a run

One worker process, `chunk-worker` in `compose.yaml`, running
`python -m preprocessing.chunking.run --watch`. It claims the next document
whose `chunk_status` is `pending` with `FOR UPDATE SKIP LOCKED`. A worker
killed mid-document leaves its row `in_progress`; the next run's sweep fails
any claim older than the lease — thirty minutes — with "the worker did not
finish".

Nothing this stage does starts extraction. The passages arrive with
`extract_status = 'new'`.

### Step 1 — Cut

**In:** the JSON object in the `parsed` bucket.
**Out:** the converter's chunks.

Docling's `HybridChunker`, sized by the tokenizer of `EMBEDDING_MODEL`
rather than by its own English default, with `CHUNKING_MERGE_PEERS` deciding
whether undersized neighbours sharing a heading are combined. It splits on
the document's structure first and only then merges or divides by token
count, so a passage keeps the heading trail it sat under.

Every kind of block is serialised into the passage text as Markdown, tables
included, because a block either renders into the text or it does not exist
— and evidence has to be a span of the passage text.

### Step 2 — Number and count

**In:** the chunks.
**Out:** a `Chunking`: the passages, numbered, and `oversized`.

[`chunking_of`](passages.py) drops chunks holding only whitespace, numbers
the rest from 1 contiguously, and counts how many exceed the token budget.

Nothing is discarded on size. `oversized` is a tripwire: the chunker splits
on the same budget it is counted against, so anything above it means the
split did not happen, and those passages will be truncated when embedded. A
document that yields no passages at all fails as `NoPassages` rather than
being stored empty.

### Step 3 — Read each chunk into a row

**In:** one chunk.
**Out:** the stripped text (the string every offset indexes), the heading
trail, the block type, the converter's item references, one bounding box per
page spanned normalised to a top-left origin, and the cell grid of every
table.

The table handling is the fiddliest part of the stage and is worth spelling
out. A table is rendered into the passage text as Markdown rows, and the
grid is kept separately because the rendering loses the header flags, the
positions and the spans. Each data row of the grid is then paired with the
rendered line that shows it, matched **by position** — every one of a row's
values must sit at its own column index, or a table of repeated counts
matches nearly every row. Lines are consumed in order, so two identical rows
take two different lines.

A data row this passage does not render is dropped: it belongs to another
piece of a table the chunker split, and keeping it would let the extractor
describe a value with a row the passage never shows. Header rows are kept
whether matched or not and carry no line. A `row_header` cell does *not*
make its row a header row — it marks the stub in the first column of an
ordinary data row, and reading it as a header left 82% of this corpus's
cells with nothing quotable.

### Step 4 — Read the linguistic surface

**In:** every passage's text, and the document's language as a fallback.
**Out:** `language`, `sentences` and `lemmas` on each.

`_read` detects each passage's language on its own text, falling back to the
document's when the passage is too short to judge. Per passage rather than
per document because these documents carry an English summary of a German
report and a language switcher in every link, so a document-wide label sends
half the passages to the wrong pipeline.

Then one spaCy batch per language, producing:

- the **numbered units** a fact may cite — sentences for prose, each with
  its finite-verb count; the rendered rows for a table
- the **content lemmas**, which are the vocabulary the topic model is fitted
  over

Every passage is read for vocabulary, a table included, since its headings
and cell values are what it is about.

### Step 5 — Store

**In:** everything above.
**Out:** every row of `passages` for that document, replaced, plus
`oversized` and `chunk_status = 'chunked'` on the document.

One transaction deletes the document's old passages, inserts the new ones,
writes `oversized` and sets the status, so a document can never read as
chunked while holding another run's passages.

The delete cascades to the facts drawn from the old passages and the topic
memberships they held — which is why re-chunking is a destructive operation
and is offered separately from re-parsing.

### Step 6 — Re-read stored passages (a side door)

**Where:** [`service.py`](service.py) · `revocabulary`.
**In:** every stored passage's text and its document's language.
**Out:** `passages.language` and `passages.lemmas`, replaced in place.

Applies a change to how language or vocabulary is read without re-chunking,
which would delete every passage and take its facts with it.

Sentence offsets are deliberately left alone. A fact cites one by index, so
moving them would point every citation in the corpus at different text.

## Tools, and where each is used

| Tool | Where | Why this one |
|---|---|---|
| **docling-core** | [`passages.py`](passages.py) | `HybridChunker`, which cuts on document structure before merging or dividing by token count, so a passage keeps its heading trail; and the Markdown serialisers, which put table and picture content into the passage text. |
| **transformers** (tokenizer only) | [`passages.py`](passages.py) | Sizes a passage with the tokenizer of the model that will embed it. One name for both, so a passage cannot be sized by one tokenizer and embedded by another — that is silent truncation at embed time, and a benchmark built on truncated passages misreports its own coverage. No model weights are loaded here, only the tokenizer. |
| **lingua** | `nlp/language.py` | Per-passage language detection, over the languages `NLP_MODELS` names and no others. Web addresses are stripped before detection and the length floor is measured on what is left. |
| **spaCy** | `nlp/analysis.py` | Sentence boundaries, the finite-verb count per sentence, and the content lemmas. See the note below. |
| **SQLAlchemy** | [`repository.py`](repository.py) | The queue (`FOR UPDATE SKIP LOCKED`), the transactional passage replacement, and the listings. |
| **boto3** | `blob_store/seaweedfs/` | Reads the `parsed` bucket. Writes nothing. |
| **OpenTelemetry** | [`service.py`](service.py) | One span per document with `stage.name` and `stage.outcome`. |

### A note on spaCy

spaCy is used in exactly three places here, and it is worth naming what it
is and is not doing.

- **Sentence segmentation** (`nlp.analysis.read`), called once per passage
  batch. This is the load-bearing use: the sentence offsets it produces are
  what every fact in the corpus cites. A trained pipeline rather than the
  rule-based splitter, because the rule-based one breaks German legal text
  on `Abs.`, `Nr.` and `z. B.`.
- **The finite-verb count per sentence**, stored as `predicates` on each
  numbered unit. It is what tells a sentence that asserts something from a
  heading or a caption, which is how extraction decides a passage is not
  worth a model call at all.
- **Content lemmas** — nouns, proper nouns and adjectives, lemmatised —
  written here so one segmentation serves both stages and a fit reads a
  column instead of re-tokenising the corpus.

Three deliberate choices sit inside that third one. Web addresses are
excluded, because Markdown renders a link as `[text](url)` and the tokenizer
offers every path segment as a word — one repeated link put `publikationen`,
`fokus`, `im` and `en` among the most widespread terms in the corpus. A
lemma must be alphabetic and longer than one character, because a lemmatiser
handed a word from another language returns the placeholder `--`. And in a
language named in `NLP_CAPITALISED_NOUNS`, a lower-case word tagged as a
noun is treated as foreign and dropped: the German pipeline tags the English
`the`, `and` and `of` as proper nouns, and they reached the top terms of two
of the twelve German topics.

Everything else in `nlp/analysis.py` — `claim`, `vocabulary`, `content`,
`interrogatives` — belongs to extraction and question generation, not to
this stage. The medium models are required rather than the small ones:
`de_core_news_sm` does not tag a German modal as a finite verb, so
`Ein Risikobericht muss erstellt werden` reads as no claim at all. Both
models must be present in the image; nothing downloads one at run time.

`spacy.__version__` is recorded on every fact, because the parser decides
the verdicts and two parsers are two datasets.

## Inputs and outputs

**In:** one claimed row of `documents` where `chunk_status = 'pending'`, and
the JSON object it names in the `parsed` bucket.

**Written, in PostgreSQL:**

| Table | Holds |
|---|---|
| `passages` | the text, the language, the numbered units, the lemmas, the pages, the heading trail, the block type, the converter's item references, the bounding boxes and the table grids |
| `documents` | `oversized`, and the `chunk_status` / `chunk_error` / `chunk_claimed_at` triple. Nothing else. |

**Written, in SeaweedFS:** nothing.

**Serves**

| Route | What it answers |
|---|---|
| `GET /chunking/status` | The queue depth, and whether a worker is on it |
| `POST /chunking/{action}` | `start`, `stop`, `retry`, `rerun` |
| `GET\|POST /chunking/{scope}/{value}/…` | The same, narrowed to one document |
| `GET /passages` | One page of passages, narrowed by document, block type and a text search |
| `GET /passages/types` | The block types the corpus holds, for the picker |
| `GET /passages/{passage_id}` | One passage in full, with its table cells and its numbered sentences |

**Command line:** `make chunk-status`, `make chunk-start`, `make chunk-stop`,
`make chunk-retry`, `make chunk-rerun`, `make chunk`,
`make chunk-revocabulary`; `--only document=<sha>` narrows any of them.

**Read by:** extraction (the passage text, its numbered units, its table
grids and its language), topic modelling (`passages.lemmas`, partitioned by
`passages.language`), question generation (through `fact_passages`, back to
the passage).

## Configuration

Every setting is read from the environment at start-up and there are no
defaults in code. Tuning lives in
[`configs/env/backend.env`](../../../configs/env/backend.env), which is in
git; credentials, ports and addresses live in `.env`, which is not.

| Setting | What it does |
|---|---|
| `EMBEDDING_MODEL` | The model whose tokenizer sizes a passage, and the model that will embed it. One name for both. |
| `EMBEDDING_MAX_TOKENS` | The token budget, and so the longest passage the chunker emits. Keep it equal to the model's context window: larger silently truncates at embed time, smaller wastes the window and fragments tables for nothing. |
| `CHUNKING_MERGE_PEERS` | Whether undersized neighbours sharing a heading are combined. Off leaves one-line passages, which retrieve poorly and carry too little to draw a fact from. |

There is deliberately no character floor and no character ceiling. A
character count is a different unit from the token budget, and being a
different unit is how a ceiling came to fire before the budget it was meant
to back up and discard whole tables that were within it.

### Language

Shared with parsing, and read by the surface this stage produces.

| Setting | What it does |
|---|---|
| `NLP_MODELS` | `language:model` pairs. The spaCy pipeline per language, and also the list the detector may answer with. Both models must be in the image. |
| `NLP_DEFAULT_LANGUAGE` | Used for a passage too short to detect and for any language `NLP_MODELS` does not name. Must be one of the languages it names. |
| `NLP_CAPITALISED_NOUNS` | Languages that write every noun with a capital. In one of these, a lower-case word tagged as a noun is treated as a foreign word rather than a subject. Empty is valid. |

### What is not configurable, on purpose

There are no similarity thresholds. A fact cites a numbered sentence rather
than quoting one, so a citation is either exact or names a sentence that
does not exist. Nothing here is tuned by a cut-off somebody has to pick.

## Tests

```sh
poetry run pytest tests/unit/intake/test_intake_chunking.py
poetry run pytest tests/unit/intake/test_intake_surface.py
poetry run pytest tests/unit/chunking
```

Non-UI tests follow the same page-object structure the UI tests do: one
**driver** per thing under test. `ChunkDriver` and `RevocabularyDriver` live
in
[`tests/unit/intake/intake_drivers.py`](../../../tests/unit/intake/intake_drivers.py),
over `MemoryParsed`, `ChunkRows` and `PassageRows`. Only the database and
the bucket are stood in for; the chunk reader, the table pairing and the
real spaCy pipelines are the real code.

| File | Covers |
|---|---|
| [`tests/unit/intake/test_intake_chunking.py`](../../../tests/unit/intake/test_intake_chunking.py) | numbering, the token budget, block-type labelling, the row-to-line pairing, the cell grids, the bounding boxes, one chunk into one passage, the flow's failure paths, and the re-read |
| [`tests/unit/intake/test_intake_surface.py`](../../../tests/unit/intake/test_intake_surface.py) | the spaCy-backed reading with the real pipelines: per-passage language, sentence numbering and offsets, predicate counts, table rows, and what does and does not become vocabulary |
| [`tests/unit/chunking/test_chunking.py`](../../../tests/unit/chunking/test_chunking.py), [`test_lines.py`](../../../tests/unit/chunking/test_lines.py) | the chunk reader's decisions and the numbering of a rendered table |
| [`tests/unit/intake/test_intake_cli.py`](../../../tests/unit/intake/test_intake_cli.py) | which flag combinations the command line refuses, the exit codes, and that it builds no service until a flag needs one |
| [`tests/unit/nlp/`](../../../tests/unit/nlp) | sentence numbering and claim counting on their own |
| [`tests/property/test_invariants.py`](../../../tests/property/test_invariants.py) | Hypothesis over arbitrary text: every non-blank chunk is stored, ordinals run from 1 without a gap, `oversized` never exceeds the passages stored, and a search never reaches the database as a wildcard |
| [`tests/integration/test_intake_queues.py`](../../../tests/integration/test_intake_queues.py) | the repository against a real PostgreSQL: which columns this stage writes and which it must leave alone, the transactional passage replacement and its cascade to facts, `NULL` versus JSON `null`, the language CHECK constraint, and every listing filter |
| [`tests/integration/database/test_queue.py`](../../../tests/integration/database/test_queue.py) | the claim mechanics, including `SKIP LOCKED` under concurrent workers and the abandoned-claim sweep |
| [`tests/integration/api/test_catalogue.py`](../../../tests/integration/api/test_catalogue.py), [`test_stages.py`](../../../tests/integration/api/test_stages.py) | the passage listing and the queue verbs over HTTP |
| [`tests/e2e/test_pipeline.py`](../../../tests/e2e/test_pipeline.py) | one document through every stage against the real stores, and that dropping the derived data lets chunking repeat without re-parsing |
| [`tests/frontend/test_views.py`](../../../tests/frontend/test_views.py) | the Passages page under Streamlit's own runner, against a scripted backend |
| [`tests/static/test_intake_pinned.py`](../../../tests/static/test_intake_pinned.py) | the settings, the routes, the answer shapes and the modules, pinned |

Statement coverage of `backend/preprocessing` is **99%**. What is uncovered
is the `if __name__ == "__main__"` guard and the body of `build_service`,
which downloads a tokenizer.

Not covered at all, by choice: the behaviour of the real embedding tokenizer
on a corpus, which is a property of the model rather than of this code.

## Known edges

- **A re-chunk is destructive.** It deletes every passage of the document,
  which cascades to the facts drawn from them, the topic memberships they
  held and the questions resting on those facts.
- **`oversized` should always read 0.** Anything above 0 means the chunker's
  split did not happen and those passages will be truncated when embedded.
- **A passage too short to detect a language for gets `NULL`**, which puts
  it outside every topic model and outside question generation. That is
  intended; the topic model is fitted per language.
- **`revocabulary` does not renumber sentences.** A change to how sentences
  are split reaches the corpus only through a re-chunk, and a re-chunk takes
  the facts with it.
