# Facts

Turns passages into facts: short written statements, each carrying a pointer
back to the exact text that supports it.

A statement that fails a check is stored with the code that refused it rather
than dropped, because the share that failed is how the reading is measured.

The corpus is read four ways, all stored in the same table:

| Kind | What it is |
|---|---|
| `atomic` | One claim, from one sentence, citing it by number |
| `summary` | Two or three sentences standing in for a whole passage |
| `outline` | The points that passage makes, one bullet each |
| `bridge` | One claim no single passage states |

## What it does

Two passes, and they are independent.

The **passage pass** is a queue. One passage is claimed at a time, routed to
a reader by its block type, read for its claims and then for its digest.
Everything both readers propose is checked, and the lot is written in one
transaction that also marks the passage done.

The **bridge pass** is not a queue: its unit of work is a group of passages
the topic model put together. `make extract-bridge` walks every topic, forms
groups, and writes what they yield. Topics must be fitted first.

`make extract-revalidate` re-judges every stored fact without calling a
model, which is how a change to the checks reaches a corpus extracted before
it.

### Step 1 — Claim a passage

`SELECT … FOR UPDATE SKIP LOCKED` on `passages.extract_status`. The claim is
timestamped rather than held as a row lock, and a later run sweeps a claim
that outlived its lease. The lease is derived from `LLM_TIMEOUT_SECONDS` and
`LLM_MAX_ATTEMPTS`.

### Step 2 — Skip what asserts nothing

A passage is skipped, and marked read with nothing found, when:

- it has **no numbered sentences** — nothing can cite it;
- it **is its own heading**;
- **none of its sentences carries a finite verb** — a caption, a navigation
  line, a bare list fragment.

A table is exempt from the last two, because it is read from its cell grid.
It is not exempt from one more: a grid whose filled cells are more than half
identifiers — `TA-BO1`, `K2`, `1.7`, a tick — states nothing a reader would
look up. The threshold is 0.5; see [measurements](../../docs/measurements.md).

### Step 3 — Read the claims

Routed by block type:

- **`table` → the deterministic reader.** Walks the cell grid and labels each
  value cell with every header above it and to the left. No model. Both a row
  and a column label are required. Each fact cites the numbered Markdown line
  its value sits in.
- **everything else → the model.** Shown one passage's numbered sentences and
  its heading trail, it answers with a statement plus the *numbers* of the
  sentences it came from. The evidence is resolved from the numbers
  afterwards; nothing is copied.

### Step 4 — Read the digest

One call produces both the `summary` and the `outline`, so naming both in
`EXTRACTION_KINDS` costs no more than naming one. A passage carrying fewer
than two claims is not digested.

`EXTRACTION_DIGEST_MODEL` names a model of its own; unset it uses whichever
model reads the passage.

### Step 5 — Check everything

Every check is structural rather than lexical. A citation is an index, so it
cannot be half-right, and there are no similarity thresholds in this service.

The citation is resolved first — a citation naming nothing that exists is
`evidence_absent` and no further check runs. Then the checks the kind calls
for, first failure winning:

| Check | Refuses | `atomic` | `summary` | `outline` | `bridge` |
|---|---|:-:|:-:|:-:|:-:|
| citation resolves | `evidence_absent` | ● | ● | ● | ● |
| not copied | `copied` | ● | | | |
| asserts something | `asserts_nothing` | ● | ● | | ● |
| exactly one claim | `not_atomic` | ● | | | ● |
| nothing invented | `unsupported_addition` | ● | ● | ● | ● |
| no dangling pronoun | `unresolved_reference` | ● | | | ● |
| shorter than its passage | `not_condensed` | | ● | ● | |
| two points or more | `not_listed` | | | ● | |
| rests on two passages | `not_bridging` | | | | ● |

Two more refusals are the service refusing what the checks passed:

- `duplicate` — the corpus already holds the statement.
  `EXTRACTION_DUPLICATE_COSINE` is how alike two may be, and a candidate is
  probed against every validated fact in the corpus through the HNSW index
  **and** against the facts this passage already kept.
- `over_cap` — `EXTRACTION_MIN_OTHER_SHARE` is a floor on the share of a
  passage's facts that are not atomic, which works out to a cap on the atomic
  ones. The facts asserting a number, date or name are kept first.
  `make extract-recap` re-applies it with no model call.

The gaps are the design: a `summary` and an `outline` carry several claims by
definition, an `outline` is a fragment the parser cannot read as a sentence,
a `summary` resolves its own pronoun, and a statement a deterministic reader
composed faces only `copied`.

What `nothing invented` compares is **numbers and proper nouns only** — both
are things a writer reports rather than chooses. A number keeps its surface
form; a proper noun is lemmatised.

### Step 6 — Store

One transaction. The passage's existing facts are deleted and the new ones
written, scoped to that passage. Bridge facts resting on it are not deleted.

Every fact gets one `fact_passages` row per passage it rests on. That link is
the only route from a fact to a passage, and so to a document and a topic.

### Step 7 — The bridge pass

Passages are grouped by the topic they carry most strongly, then paired on
their vectors — each head takes whichever unused passage it most nearly
meets. Within a topic the documents are taken in turn, and the groups are
strided over the whole topic.

The model is shown the group as `[P0]`, `[P1]`, … with sentences numbered
inside each, and answers with a claim plus, per passage, the numbers of the
sentences it read it in. It is told not to compute: a total nobody wrote down
would be refused as `unsupported_addition`.

A trigger deletes a fact once any passage it rests on is gone. Re-running the
pass replaces what the previous one wrote.

## Inputs and outputs

**Reads:** `passages` (text, numbered sentences, section path, block type,
table grids), `documents.language`, `passage_topics` for the bridge pass, and
one served model.

**Writes:**

- `facts` — one row per statement, refused ones included, with the verdict,
  the rejection code, and the model, prompt version, temperature and spaCy
  version it was produced under.
- `fact_passages` — one row per passage the fact rests on, with the sentences
  and the span the claim rests on there.
- `passages.extract_status`, `extract_error`, `extract_claimed_at`.

**Invariant:** for every `fact_passages` row whose citation was recorded,
`passages.text[evidence_start:evidence_end]` is the text the claim was read
in. `facts.evidence_text` is those spans joined in position order.

**Serves**

| Route | Answers |
|---|---|
| `GET /facts` | One page of facts, narrowed by document, kind, method and a text search |
| `GET /facts/quality` | The figures under that same filter |
| `GET /facts/{id}/passages` | The passages one fact rests on |
| `GET /extraction/status` | The queue depth, and whether a worker is on it |
| `POST /extraction/{action}` | `start`, `stop`, `retry`, `rerun` |
| `GET\|POST /extraction/{scope}/{value}/…` | The same, narrowed to one document or passage |

## Prompts

Each of the three model-backed readers declares its own `PROMPT_VERSION` and
writes its composed prompt to the `prompts` table before the first claim.
[`prompts.py`](prompts.py) is the catalogue. The atomic prompt is recorded
with the cap paragraph `EXTRACTION_MIN_OTHER_SHARE` works out to.

Read one back with `GET /prompts?service=extraction`, or in Phoenix, where
this stage publishes whatever it records when it starts.

A worker asks the configured model one question before it claims anything and
refuses to start if it will not answer. See
[`backend/llm/`](../llm/README.md).

## Configuration

Nine settings of its own, in `configs/env/backend.env`. Everything else is
shared: `LLM_*`, `NLP_MODELS`, `EMBEDDING_MODEL`, `DATABASE_*`.

| Setting | What it does |
| --- | --- |
| `EXTRACTION_KINDS` | Which kinds a run writes. `atomic` is mandatory |
| `EXTRACTION_DIGEST_MAX_SHARE` | The longest a digest may be, as a share of its passage |
| `EXTRACTION_DIGEST_MIN_CHARS` | Shortest passage worth digesting |
| `EXTRACTION_BRIDGES_PER_TOPIC` | Groups one topic is worth |
| `EXTRACTION_BRIDGE_PASSAGES` | Passages one group holds. At least 2 |
| `EXTRACTION_MIN_OTHER_SHARE` | The floor a passage's non-atomic kinds keep |
| `EXTRACTION_DUPLICATE_COSINE` | How alike two facts may be |
| `EXTRACTION_MODEL` | A different reader. Unset means `LLM_MODEL` |
| `EXTRACTION_DIGEST_MODEL` | The model the digest call goes to |

Naming `summary` or `outline` adds one model call per passage. Naming
`bridge` costs nothing on the passage queue — it enables a separate pass.

Changing `EXTRACTION_KINDS` or either bridge setting affects the next run;
`EXTRACTION_DIGEST_MAX_SHARE` can be applied to stored facts with
`make extract-revalidate`.

### Commands

```bash
make extract-start                  # queue every passage never asked for
make extract                        # drain the queue here, in the foreground
make extract-status                 # passages by state, and facts held
make extract-stop                   # take back whatever has not begun
make extract-retry                  # return every failed passage to the queue
make extract-rerun                  # read every passage again
make extract-revalidate             # re-judge stored facts; no model call
make extract-bridge                 # read every topic's groups for bridges
make extract-recap                  # re-apply the atomic cap; no model call
make extract-embed                  # write vectors onto rows already stored

make extract-retry PASSAGE=<id>     # any of them, narrowed to one passage
make extract-rerun SHA=<sha256>     # or to one document
```

`extract-embed` does **not** apply the dedup gate: a fact accepted before
that gate existed was accepted.

## Tests

```sh
poetry run pytest tests/unit/extraction
```

Twenty-two files over seven layers. Non-UI tests act through a driver —
`Checker`, `Model`, `Queue`, `Catalogue`, `Extraction` in `tests/drivers.py`,
and `FactStore` in `tests/integration/facts.py`.

| Layer | Holds |
|---|---|
| Unit | Every check on every kind, both readers, the passage gate, the per-passage flow, the bridge pass, grouping, revalidation and the config refusals |
| Integration | The SQL against a real Postgres, what the database refuses, and the HTTP surface |
| Property | A verdict and its code always agree; a span always resolves; a bridge never records a passage it was not offered |
| Regression | A golden table of every candidate anybody has seen, in both languages, with the verdict it should get |
| Static | `tests/static/test_facts_pinned.py` — settings, routes, dataclass fields, modules, kinds, rejection codes, the check matrix and the three prompt versions |
| Eval | A real served model against golden passages. Prints; never gates |

```bash
make test-fast          # everything but spaCy, pyright and containers
make test-unit
make test-integration   # needs a container runtime
make test-eval          # needs LLM_MODEL and LLM_BASE_URL; never gates
```

## Limits

- **A spelled-out number counts as a unit.** spaCy's `like_num` is true of
  *one*, *two*, *both* and *dozen*, so a statement saying `two parties` drawn
  from a source writing `2 parties` is refused as an unsupported addition.
- **A bridge cannot aggregate.** A total, difference or average appears in no
  passage, so it is refused. Bridges relate what is written.
- **An outline is checked on its shape, not its content.** The parser cannot
  read a fragment.
- **The bridge pass depends on topic modelling.** With no fit it finds no
  groups and says so in the log rather than failing.
