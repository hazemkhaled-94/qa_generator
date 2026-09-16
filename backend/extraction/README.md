# Facts

Turns passages into facts: short written statements, each one carrying a
pointer back to the exact text that supports it.

This is the stage that decides what the benchmark is made of. Everything
downstream — the questions, their answers, the score a chatbot gets — rests
on a fact being true of the corpus and traceable to a place in it. So the
service is built around one idea: **a fact is not what a model said, it is
what a model said that survived a check against its source.** Nothing is
thrown away. A statement that fails a check is stored with the code that
refused it, because the share that failed is the only honest measure of how
well the reading went.

The corpus is read four ways, and all four are stored in the same table:

| Kind | What it is | What it is for |
|---|---|---|
| `atomic` | One claim, from one sentence, citing it by number | The unit a question is generated from and scored against |
| `summary` | Two or three sentences standing in for a whole passage | Read instead of the passage to know what it covers |
| `outline` | The points that passage makes, one bullet each | The same reading as a list rather than as prose |
| `bridge` | One claim no single passage states | Questions a retriever cannot answer from one chunk |

Atomic facts are the important ones and the reason the service exists. The
other three were added because a corpus of atomic facts alone is a pile of
sentences with no shape: nothing in it says what a section is about, and
nothing in it needs two places at once.

## What it does

### The shape of a run

Two passes, and they are independent.

The **passage pass** is a queue. Extraction claims one passage at a time —
not one document — so the work divides evenly between workers and a passage
nobody can read fails only itself. Each claimed passage is routed to a reader
by its block type, read for its claims, and then read again for its digest.
Everything both readers propose is checked, and the whole lot is written in
one transaction that also marks the passage done.

The **bridge pass** is not a queue. Its unit of work is a *group* of
passages the topic model put together, so it cannot be driven by a per-row
status column. It is an operation of its own — `make extract-bridge` — that
walks every topic, forms groups, and writes what they yield. Topics must be
fitted before it will find anything.

A third operation, `make extract-revalidate`, re-judges every stored fact
without calling a model. It re-reads what the checks read and rewrites only
that, leaving the statement, the kind, the method and the model's own
provenance alone. This is how a change to the checks reaches a corpus that
was extracted before it.

### Step 1 — Claim a passage

**In:** the `passages` table, `extract_status = 'pending'`.
**Out:** one `PassageToExtract`, marked `in_progress` with the time of the
claim.

`SELECT … FOR UPDATE SKIP LOCKED` on this stage's own status column, so a
second worker takes the following row rather than blocking on this one. The
claim is timestamped rather than held as a row lock, because the work that
follows runs for minutes; a later run sweeps a claim that outlived its lease
and fails it. The lease is derived from `LLM_TIMEOUT_SECONDS` and
`LLM_MAX_ATTEMPTS`, so a healthy worker taking the timeout on every attempt
is never mistaken for a dead one.

The passage's language comes from its document, and it selects the spaCy
pipeline everything below is read with.

### Step 2 — Skip what asserts nothing

**In:** the claimed passage.
**Out:** a reason to skip, or nothing.

One model call costs minutes. A passage is skipped when:

- it has **no numbered sentences** — nothing can cite it, whatever it is.
  This is checked before the table exemption: a table read without line
  numbers once produced a fact per cell, every one citing a row that did not
  exist;
- it **is its own heading** — the text equals a segment of its own section
  path;
- **none of its sentences carries a finite verb** — a caption, a navigation
  line, a bare list fragment. It names things; it asserts nothing.

A table is exempt from the last two: it is read from its cell grid, not from
its prose.

A skipped passage is marked *read with nothing found*, not failed. There was
nothing there, which is an answer.

### Step 3 — Read the claims

**In:** the passage.
**Out:** zero or more `CandidateFact`s of kind `atomic`.

Routed by block type:

- **`table` → the deterministic reader.** Walks the stored cell grid and
  labels each value cell with every header above it and to the left of it.
  No model is involved. Both a row label and a column label are required; a
  cell repeating one of its own labels states nothing and is dropped. Every
  header above a cell is used rather than only the nearest, because a grouped
  table would otherwise give several columns the same label — and so several
  facts identical in wording and different in value. Each fact cites the
  numbered Markdown line its own value sits in, so a table citation is an
  index like any other and the checker needs no rule of its own.

- **everything else → the model.** It is shown one passage's numbered
  sentences and its heading trail, never the whole document, and answers with
  a statement plus the *numbers* of the sentences it came from. It copies
  nothing: the evidence is resolved from the numbers afterwards. The heading
  is fenced off and labelled as context, because run together with the
  excerpt the model cited it as a sentence that does not exist.

The prompt asks for one claim per statement, the subject named rather than
pronouned, and nothing added. Returning no facts is stated to be a correct
answer, because a weak fact is worse than a missing one.

### Step 4 — Read the digest

**In:** the passage, if it carries at least two claims.
**Out:** up to two `CandidateFact`s, of kinds `summary` and `outline`.

One call produces both. They are the same reading of the same passage — *what
is this about* — so asking twice would cost twice for nothing. That is also
why naming both kinds in `EXTRACTION_KINDS` costs no more than naming one.

A passage carrying fewer than two claims is not digested at all: it is
already as short as its own summary.

### Step 5 — Check everything

**In:** a candidate and the passage or passages it came from.
**Out:** a `CheckedFact`, validated or refused with a code and a
measurement.

Every check is structural rather than lexical. A citation is an index, so it
cannot be half-right. What a statement asserts is read off its spaCy parse
rather than guessed from how many words it shares with its source. There are
no similarity thresholds anywhere in this service.

First the citation is resolved. An atomic fact cites sentences; a summary, an
outline and a bridge stand in for whole passages and cite all of them. If the
citation names nothing that exists, that is `evidence_absent` and no further
check runs.

Then the checks the candidate's kind calls for, in order, first failure
winning:

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

The gaps are the design, not oversights:

- **A summary and an outline are exempt from `not_atomic`.** Carrying several
  claims is what they are for.
- **An outline is exempt from `asserts_nothing`.** A bullet point is written
  as a fragment — *Weighs 4 kg* — and spaCy reads a fragment as a noun phrase
  with no finite verb and, in fact, no verb at all. Requiring a verb would
  refuse every well-formed outline. So an outline is checked on its *shape*
  instead: at least two lines opening with `- `.
- **A summary is exempt from `unresolved_reference`.** It names its subject
  in its first sentence and may refer back to it in its second; the pronoun
  resolves inside the fact.
- **A summary and an outline are exempt from `copied`.** `not_condensed`
  already catches a copy, which is a digest at 100% of its source.
- **A statement a deterministic reader composed faces only `copied`.** It is
  neither written nor a sentence — `Table 1: Models - Compact - Mass: 4` —
  so nothing can be read off its grammar.

What `nothing invented` compares is narrow on purpose: **numbers and proper
nouns only.** Both are things a writer reports rather than chooses, so one
appearing in a statement and nowhere in its source was invented. Common
nouns are excluded because a statement is supposed to name its subject in its
own words; verbs because the statement is written rather than quoted, so its
predicate is the model's to choose; named entities because the tagger's
entity decision moves with the surrounding words, so the two sides never
agree. A number keeps its surface form, because `12,5` and `12.5` are
different values; a proper noun is lemmatised, so an inflection is not an
addition.

### Step 6 — Store

**In:** everything the passage yielded.
**Out:** rows in `facts`, and the passage marked `extracted`.

One transaction. The passage's existing facts are deleted and the new ones
written, scoped to that passage so two workers on two passages of one
document do not delete each other's work. Bridge facts anchored to the
passage are *not* deleted: they belong to the bridge pass.

### Step 7 — The bridge pass

**In:** every topic's passages, and the model.
**Out:** rows in `facts` of kind `bridge`, plus one `fact_passages` row per
passage each rests on.

Passages are grouped by **the topic they carry most strongly**. That is the
corpus's own statement that two passages are about the same thing, and it is
a far better reason to put them together than that they happen to be
adjacent. Within a topic the documents are taken in turn, so a group spans as
many files as the topic does; the groups are then strided over the whole
topic rather than taken from its start, so a large topic is sampled across
rather than at its head.

The model is shown the group as `[P0]`, `[P1]`, … and answers with a claim
plus the *numbers* of the passages it rests on. It is told explicitly not to
compute: a total nobody wrote down is not in the material, and would be
refused as `unsupported_addition` anyway, since the arithmetic result appears
in no passage.

A bridge is stored anchored to the first passage it rests on. That anchor is
its `passage_id`, so it lists, filters, cascades and joins exactly like every
other fact; the full group is in `fact_passages`. A trigger deletes a bridge
that falls below two passages, because re-chunking one side of it cascades to
the link table and not to the fact, which would otherwise leave a claim
bridging nothing.

Re-running the pass replaces what the previous one wrote rather than adding
to it.

## Tools, and where each is used

| Tool | Where | Why |
|---|---|---|
| **spaCy** | every check | The verdicts are read off a dependency parse: finite verbs to count claims, POS tags to find numbers and proper nouns, morphology to tell a referring pronoun from a reflexive or an expletive. This is the whole reason the checks are structural rather than a similarity score. The model and its version are recorded on every fact, because the parser decides the verdict. |
| **litellm + instructor** | the three model-backed readers | One client for any provider, and a declared answer shape the runtime is constrained to produce. A malformed answer is a retry, not a parse. |
| **Pydantic** | the answer shapes | What the model is asked to return, field descriptions included. The descriptions are part of the prompt. |
| **SQLAlchemy** | the repository | The queue's `FOR UPDATE SKIP LOCKED` claim, the filters shared between a listing and its count, and the bulk write a re-judgement makes. |
| **PostgreSQL** | storage | `CHECK` constraints on the kind, the method, the rejection code and the verdict, so a hand-run `UPDATE` cannot write a row the service would never write. Trigram indexes for the search box. One trigger, for bridges that lose a passage. |
| **OpenTelemetry** | the passage pass | One span per passage, carrying the passage id, its block type, how many facts it yielded, how many passed, and — when it was skipped — why. |

**spaCy is not used** to decide *what* to extract; only to judge what was
extracted. The reading is the model's job and the judging is the parser's,
and keeping those apart is what makes a re-judgement possible without a model
call.

## Inputs and outputs

**Reads**

- `passages` — text, numbered sentences, section path, block type, table cell
  grids, document digest.
- `documents` — language.
- `passage_topics` — which topic each passage carries most strongly. The
  bridge pass only.
- One served model, over HTTP.

**Writes**

- `facts` — one row per statement, refused ones included. Statement,
  evidence text and span, cited sentence ids, kind, method, verdict,
  rejection code and message, everything the checks read, and full
  provenance: model, prompt version, temperature, spaCy model and version.
- `fact_passages` — one row per passage a bridge rests on, anchor first.
- `passages.extract_status`, `extract_error`, `extract_claimed_at`.

**Serves**

| Route | What it answers |
|---|---|
| `GET /facts` | One page of facts, narrowed by document, kind, method and a text search |
| `GET /facts/quality` | The figures under that same filter |
| `GET /facts/{id}/passages` | The passages a bridge rests on |
| `GET /extraction/status` | The queue depth, and whether a worker is on it |
| `POST /extraction/{action}` | `start`, `stop`, `retry`, `rerun` |
| `GET\|POST /extraction/{scope}/{value}/…` | The same, narrowed to one document or one passage |

**Invariant every reader can rely on:** for every fact,
`passages.text[evidence_start:evidence_end] == evidence_text`, where the
passage is `facts.passage_id`. It holds for all four kinds; a bridge's
evidence is the whole of its anchor, and the rest of its group is reached
through `fact_passages`.

## Configuration

Four settings of its own, in `configs/env/backend.env`. Everything else this
stage needs is shared: `LLM_*` for the model, `NLP_MODELS` and
`NLP_DEFAULT_LANGUAGE` for the pipelines, `DATABASE_*` for the connection.

### `EXTRACTION_KINDS`

Which kinds a run writes, comma-separated. `atomic` is mandatory; naming
anything else is refused at start-up with the list of what is valid.

```
EXTRACTION_KINDS=atomic,summary,outline,bridge
```

- Naming `summary` or `outline` adds **one model call per passage**, so it
  roughly doubles what a corpus costs to read. Naming *both* costs no more
  than naming one: they come from the same call.
- Naming `bridge` costs nothing on the passage queue. It enables a separate
  pass, run by hand.
- `atomic` alone is the cheapest configuration and the one that produces
  exactly what earlier versions of this service produced.

### `EXTRACTION_DIGEST_MAX_SHARE`

The longest a summary or an outline may be, as a share of the passage it
stands in for. Measured in characters after collapsing whitespace, so it
reads the same in both languages. Above it the fact is refused as
`not_condensed`.

```
EXTRACTION_DIGEST_MAX_SHARE=0.6
```

Lower is stricter. At 1.0 nothing is refused for length, and a summary that
is its passage copied out becomes valid — which defeats the point. The floor
is practical rather than principled: a passage of two short sentences cannot
be condensed much, which is why a passage carrying fewer than two claims is
not digested at all.

### `EXTRACTION_BRIDGES_PER_TOPIC` and `EXTRACTION_BRIDGE_PASSAGES`

How many groups one topic is worth, and how many passages one group holds.

```
EXTRACTION_BRIDGES_PER_TOPIC=5
EXTRACTION_BRIDGE_PASSAGES=2
```

The first multiplied by the topic count is the number of model calls a bridge
run makes. The second must be at least 2; at 3 and above the model tends to
write a claim resting on two of the passages and name the third anyway, which
the checks cannot catch because the claim is supported.

### Changing any of them

None of these is read at check time except the digest share, so:

- changing `EXTRACTION_KINDS` affects the **next** run. Existing facts are
  left as they are; `make extract-rerun` re-reads the corpus;
- changing `EXTRACTION_DIGEST_MAX_SHARE` can be applied to facts already
  stored with `make extract-revalidate`, which calls no model;
- changing either bridge setting affects the next `make extract-bridge`,
  which replaces every bridge it wrote last time.

### Commands

```bash
make extract-start                  # queue every passage never asked for
make extract                        # drain the queue here, in the foreground
make extract-status                 # passages by state, and facts held
make extract-stop                   # take back whatever has not begun
make extract-retry                  # return every failed passage to the queue
make extract-rerun                  # read every passage again
make extract-revalidate             # re-judge stored facts; no model is called
make extract-bridge                 # read every topic's groups for bridges

make extract-retry PASSAGE=<id>     # any of them, narrowed to one passage
make extract-rerun SHA=<sha256>     # or to one document
```

## Tests

Twenty-two files, 349 tests, over seven layers. Coverage of
`backend/extraction` is **99%**: everything but the `--watch` loop in
`run.py`, which is a process that does not return.

| Layer | File | Marker | What it holds |
|---|---|---|---|
| Unit | `tests/unit/extraction/test_fact_checker.py` | `nlp` | Every check, on every kind. One test per refusal, plus the accepted cases each check must *not* refuse — a German reflexive, an expletive, a year with a full stop attached, a common noun echoed from the source. Ends by asserting every rejection code is reachable and distinct. |
| Unit | `tests/unit/extraction/test_llm_extractor.py` | `nlp` | The message a passage becomes, and what a candidate is built from what comes back. |
| Unit | `tests/unit/extraction/test_digest_extractor.py` | — | One call, two readings; bullet normalisation; only the kinds asked for. |
| Unit | `tests/unit/extraction/test_bridge_extractor.py` | — | The numbered group, positions deduped in order, a group of one never sent. |
| Unit | `tests/unit/extraction/test_table_extractor.py` | `nlp` | Header stacks, spans, the row-header fallback, a row the passage does not render. |
| Unit | `tests/unit/extraction/test_passage_gate.py` | `nlp` | Which passages never reach a model. |
| Unit | `tests/unit/extraction/test_extraction_service.py` | `nlp` | The per-passage flow: both readings stored, the digest skipped when there is nothing to condense, a model that will not answer failing the row, an unexpected exception named by its type. |
| Unit | `tests/unit/extraction/test_bridge_pass.py` | `nlp` | The pass: clear then write, one failing group not stopping the rest. |
| Unit | `tests/unit/extraction/test_grouping.py` | `nlp` | Documents paired across rather than within, groups strided over the topic. |
| Unit | `tests/unit/extraction/test_revalidate.py` | `nlp` | Re-judging in batches, a bridge judged against its whole group. |
| Unit | `tests/unit/extraction/test_replay.py` | `nlp` | A second judgement reaches the first verdict, for every kind. |
| Unit | `tests/unit/extraction/test_registry.py` | — | Routing, the default, two readers claiming one block type. |
| Unit | `tests/unit/extraction/test_config.py` | — | Every refusal a bad setting gets, naming itself. |
| Unit | `tests/unit/extraction/test_extraction_wiring.py` | — | The settings reach the objects that read them; every operation is on the command line. |
| Integration | `tests/integration/database/test_fact_store.py` | `integration` | The SQL, against a real Postgres: which rows a store replaces and which it leaves, the listing and its count agreeing about the filter, the quality figures, the groups a topic offers, a bridge read back whole. |
| Integration | `tests/integration/database/test_fact_constraints.py` | `integration` | What the database refuses: an unknown kind, an unknown code, a duplicate link, a link to nothing. Plus the two cascades and the trigger. |
| Integration | `tests/integration/api/test_facts.py` | `integration` | The HTTP surface against the real wiring. |
| Property | `tests/property/test_fact_invariants.py` | `nlp` | Over inputs nobody wrote down: a verdict and its code always agree, a span always resolves in its passage, a group never repeats a passage, a bridge never records one it was not offered. |
| Regression | `tests/regression/test_fact_verdicts.py` | `nlp` | A golden table of every candidate anybody has actually seen, in both languages, with the verdict it is meant to get. A change to one check that quietly moves another shows up here as a diff. Asserts that every kind and every rejection code appears in the table. |
| Static | `tests/static/test_facts_pinned.py` | — | The pinned surface: settings, routes, dataclass fields, modules, kinds, rejection codes, the check matrix, the three prompt versions, and that this README documents all of it. |
| Frontend | `tests/frontend/test_facts_page.py` | `frontend` | The Facts page against a scripted backend: every reading offered and explained, the kind filter reaching the request, every check listed, decomposition withheld on a mixed set, a bridge naming its passages. |
| Eval | `tests/eval/test_extraction_quality.py` | `eval` | A real served model against golden passages. Prints; never gates. |

### How they are written

Non-UI tests act through a **driver** — the same idea as a page object. A
test says `checker.summary(...)` or `run.next()`; it never assembles a
`CandidateFact` or wires a service. The drivers are in `tests/drivers.py`
(`Checker`, `Model`, `Queue`, `Catalogue`, `Extraction`) and
`tests/integration/facts.py` (`FactStore`). The Facts page has a real page
object in `tests/frontend/facts_page.py`.

### Running them

```bash
make test-fast          # everything but spaCy, pyright and containers
make test-unit          # every layer that needs no container
make test-integration   # needs a container runtime
make test-coverage      # the gating layers, with a report
make test-eval          # needs LLM_MODEL and LLM_BASE_URL; never gates
```

## Known limits

These are properties of the approach, recorded rather than fixed.

- **A spelled-out number counts as a unit.** spaCy's `like_num` is true of
  *one*, *two*, *both* and *dozen*, so a statement saying `two parties` drawn
  from a source that writes `2 parties` is refused as an unsupported
  addition. Narrowing the check to digit-bearing tokens would fix it, at the
  cost of missing an invented spelled-out number — and the field it reads is
  shared with question generation, so the change is not this service's alone
  to make.
- **A bridge cannot aggregate.** A total, a difference or an average is a
  value that appears in no passage, so it is refused. Bridges relate what is
  written; they do not compute over it. This is deliberate — nothing here can
  verify arithmetic — but it does mean the `aggregation` question type is
  served by the writer reasoning over several atomic facts, not by a bridge.
- **An outline is checked on its shape, not its content.** Two lines of
  labels with nothing invented will pass. The parser cannot read a fragment,
  so there is nothing stricter available that does not refuse good outlines.
- **A digest of a very short passage is refused for length.** The floor is
  the two-claim gate before the call; between one and two claims' worth of
  text there is a band where a summary cannot be 40% shorter and still be a
  sentence.
- **The bridge pass depends on topic modelling.** With no fit, it finds no
  groups and writes nothing. It says so in the log rather than failing.
