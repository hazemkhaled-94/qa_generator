# Chunking

Cuts the converted document into passages, and reads each one for the
language, the numbered units a fact may cite, and the vocabulary a topic
model is fitted over.

A passage is the unit everything downstream works in, which makes this the
stage whose decisions are hardest to undo: re-chunking deletes every passage
of a document, and the facts, topic memberships and questions resting on them
go with it.

It reads the converted document out of the `parsed` bucket and never the
original PDF, so re-chunking never re-parses.

**A citation is an index, not a quote.** Every sentence is numbered and its
character offsets recorded; a fact says "sentence 4" and the evidence span is
resolved from the offsets. There is no similarity threshold anywhere in this
stage. For a table, the rendered Markdown rows are numbered the same way.

The other two stages that turn a PDF into passages are
[ingestion](../../ingestion/README.md) and [parsing](../parsing/README.md).

## What it does

One worker process, `chunk-worker`, running
`python -m preprocessing.chunking.run --watch`. It claims the next document
whose `chunk_status` is `pending` with `FOR UPDATE SKIP LOCKED`. A worker
killed mid-document leaves its row `in_progress`; the next run's sweep fails
any claim older than the lease.

Passages arrive with `extract_status = 'new'`, so nothing this stage does
starts extraction.

### Step 1 — Cut

Docling's `HybridChunker`, sized by the tokenizer of `EMBEDDING_MODEL` rather
than its own English default, with `CHUNKING_MERGE_PEERS` deciding whether
undersized neighbours sharing a heading are combined. It splits on the
document's structure first and only then merges or divides by token count, so
a passage keeps its heading trail.

Every kind of block is serialised into the passage text as Markdown, tables
included, because evidence has to be a span of the passage text.

### Step 2 — Number and count

[`chunking_of`](passages.py) drops chunks holding only whitespace, numbers the
rest from 1 contiguously, and counts how many exceed the token budget.

Nothing is discarded on size. `oversized` is a tripwire: the chunker splits on
the same budget it is counted against, so anything above it means the split
did not happen. A document yielding no passages fails as `NoPassages`.

### Step 3 — Read each chunk into a row

Out: the stripped text (the string every offset indexes), the heading trail,
the block type, the converter's item references, one bounding box per page
spanned normalised to a top-left origin, and the cell grid of every table.

A table is rendered into the passage text as Markdown rows, and the grid is
kept separately because the rendering loses the header flags, positions and
spans. Each data row is paired with the rendered line that shows it, matched
**by position** — every value must sit at its own column index — and lines
are consumed in order, so two identical rows take two different lines.

A data row this passage does not render is dropped: it belongs to another
piece of a table the chunker split. Header rows are kept whether matched or
not and carry no line. A `row_header` cell does **not** make its row a header
row — it marks the stub in the first column of an ordinary data row.

### Step 4 — Read the linguistic surface

`_read` detects each passage's language on its own text, falling back to the
document's when the passage is too short to judge. Per passage rather than per
document, because a document carrying an English summary of a German report
would send half its passages to the wrong pipeline.

Then one spaCy batch per language, producing the **numbered units** a fact may
cite — sentences for prose, each with its finite-verb count; the rendered rows
for a table — and the **content lemmas** the topic model is fitted over.

Every passage is read for vocabulary, a table included, since its headings and
cell values are what it is about.

Three choices sit inside the lemma reading: web addresses are excluded,
because Markdown renders a link as `[text](url)` and the tokenizer offers
every path segment as a word; a lemma must be alphabetic and longer than one
character, because a lemmatiser handed a word from another language returns
the placeholder `--`; and in a language named in `NLP_CAPITALISED_NOUNS`, a
lower-case word tagged as a noun is treated as foreign and dropped.

### Step 5 — Store

One transaction deletes the document's old passages, inserts the new ones,
writes `oversized` and sets the status, so a document can never read as
chunked while holding another run's passages.

The delete cascades to the facts drawn from the old passages and the topic
memberships they held.

### Step 6 — Re-read stored passages

[`service.py`](service.py) · `revocabulary`. Applies a change to how language
or vocabulary is read without re-chunking, replacing `passages.language` and
`passages.lemmas` in place.

Sentence offsets are deliberately left alone: a fact cites one by index, so
moving them would point every citation in the corpus at different text.

## spaCy here

Used in exactly three places, and the medium models are required:
`de_core_news_sm` does not tag a German modal as a finite verb.

- **Sentence segmentation**, once per passage batch. The offsets it produces
  are what every fact in the corpus cites. A trained pipeline rather than the
  rule-based splitter, which breaks German legal text on `Abs.`, `Nr.` and
  `z. B.`
- **The finite-verb count per sentence**, stored as `predicates` on each
  numbered unit. It is what tells a sentence that asserts something from a
  heading or caption.
- **Content lemmas**, written here so one segmentation serves both stages and
  a fit reads a column instead of re-tokenising the corpus.

Everything else in `nlp/analysis.py` belongs to extraction and question
generation. Both models must be present in the image; nothing downloads one
at run time. `spacy.__version__` is recorded on every fact.

## Inputs and outputs

**In:** one claimed row of `documents` where `chunk_status = 'pending'`, and
the JSON object it names in the `parsed` bucket.

**Written, in PostgreSQL:**

| Table | Holds |
|---|---|
| `passages` | the text, the language, the numbered units, the lemmas, the pages, the heading trail, the block type, the converter's item references, the bounding boxes and the table grids |
| `documents` | `oversized`, and the `chunk_status` / `chunk_error` / `chunk_claimed_at` triple. Nothing else |

**Written, in SeaweedFS:** nothing.

| Route | Answers |
|---|---|
| `GET /chunking/status` | The queue depth, and whether a worker is on it |
| `POST /chunking/{action}` | `start`, `stop`, `retry`, `rerun`, `reclaim` |
| `GET\|POST /chunking/{scope}/{value}/…` | The same, narrowed to one document |
| `GET /passages` | One page of passages, narrowed by document, block type and a text search |
| `GET /passages/types` | The block types the corpus holds |
| `GET /passages/{passage_id}` | One passage in full, with its table cells and numbered sentences |

**Command line:** `make chunk-status`, `chunk-start`, `chunk-stop`,
`chunk-retry`, `chunk-rerun`, `chunk`, `chunk-revocabulary`;
`--only document=<sha>` narrows any of them.

**Read by:** extraction, topic modelling and question generation.

## Configuration

Read from the environment at start-up, with no defaults in code.

| Setting | What it does |
|---|---|
| `EMBEDDING_MODEL` | The model whose tokenizer sizes a passage, and the model that will embed it. One name for both |
| `EMBEDDING_MAX_TOKENS` | The token budget, and so the longest passage the chunker emits. Keep it equal to the model's context window |
| `CHUNKING_MERGE_PEERS` | Whether undersized neighbours sharing a heading are combined |

There is deliberately no character floor and no character ceiling: a
character count is a different unit from the token budget.

### Language

Shared with parsing.

| Setting | What it does |
|---|---|
| `NLP_MODELS` | `language:model` pairs. The spaCy pipeline per language, and the list the detector may answer with. Both models must be in the image |
| `NLP_DEFAULT_LANGUAGE` | For a passage too short to detect. Must be one of the languages `NLP_MODELS` names |
| `NLP_CAPITALISED_NOUNS` | Languages that write every noun with a capital. Empty is valid |

## Tests

```sh
poetry run pytest tests/unit/intake/test_intake_chunking.py
poetry run pytest tests/unit/intake/test_intake_surface.py
poetry run pytest tests/unit/chunking
```

`ChunkDriver` and `RevocabularyDriver` live in
[`tests/unit/intake/intake_drivers.py`](../../../tests/unit/intake/intake_drivers.py),
over `MemoryParsed`, `ChunkRows` and `PassageRows`. Only the database and the
bucket are stood in for; the chunk reader, the table pairing and the real
spaCy pipelines are the real code.

| File | Covers |
|---|---|
| [`test_intake_chunking.py`](../../../tests/unit/intake/test_intake_chunking.py) | Numbering, the token budget, block-type labelling, the row-to-line pairing, the cell grids, the bounding boxes, the failure paths and the re-read |
| [`test_intake_surface.py`](../../../tests/unit/intake/test_intake_surface.py) | The spaCy-backed reading with real pipelines: per-passage language, sentence offsets, predicate counts, table rows, and what becomes vocabulary |
| [`tests/unit/chunking/`](../../../tests/unit/chunking) | The chunk reader's decisions and the numbering of a rendered table |
| [`test_intake_cli.py`](../../../tests/unit/intake/test_intake_cli.py) | Which flag combinations the command line refuses |
| [`tests/property/test_invariants.py`](../../../tests/property/test_invariants.py) | Every non-blank chunk is stored, ordinals run from 1 without a gap, `oversized` never exceeds the passages stored |
| [`tests/integration/test_intake_queues.py`](../../../tests/integration/test_intake_queues.py) | Which columns this stage writes, the transactional replacement and its cascade, and every listing filter |
| [`tests/integration/database/test_queue.py`](../../../tests/integration/database/test_queue.py) | The claim mechanics under concurrent workers |
| [`tests/e2e/test_pipeline.py`](../../../tests/e2e/test_pipeline.py) | One document through every stage, and that dropping the derived data lets chunking repeat without re-parsing |
| [`tests/static/test_intake_pinned.py`](../../../tests/static/test_intake_pinned.py) | The settings, routes, answer shapes and modules, pinned |

Not covered, by choice: the behaviour of the real embedding tokenizer on a
corpus, which is a property of the model rather than of this code.

## Limits

- **A re-chunk is destructive.** It deletes every passage of the document,
  which cascades to the facts, memberships and questions resting on them.
- **`oversized` should always read 0.** Above 0 means the chunker's split did
  not happen and those passages will be truncated when embedded.
- **A passage too short to detect a language for gets `NULL`**, which puts it
  outside every topic model and outside question generation.
- **`revocabulary` does not renumber sentences.** A change to how sentences
  are split reaches the corpus only through a re-chunk.
- **`start` and `rerun` skip a document parsing has not finished.** This is
  the only stage that queues over rows it did not create — a document exists
  from the moment it is uploaded — so it is the only one that has to check.
  A start over a half-parsed corpus used to queue the rest and fail every
  row of it on the missing parsed object. The guard is `ChunkQueue.ready`;
  see [`backend/stages/`](../../stages/README.md).
