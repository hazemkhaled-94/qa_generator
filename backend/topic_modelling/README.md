# Topic modelling

Finds what a corpus is about: the groups of words that tend to appear
together, one model per language, over the corpus's own vocabulary.

Nothing here is pretrained and nothing is embedded, so no word list, ontology
or domain model has to be supplied.

Question generation works one topic at a time, so a topic is the unit of
work, the unit of coverage reporting, and what makes a question's subject
measurable. A corpus with no topics generates no questions.

`tests/static/test_topics_pinned.py` pins this service's surface — the
settings it reads, the routes it serves, the fields it answers with and the
modules it is made of.

## What it does

Chunking has already stored each passage's language and its content lemmas.
For each language separately this service builds a vocabulary, drops the
terms too rare or too common to distinguish anything, weights the rest by
tf-idf, and factorises the passage-by-term matrix into a small number of
non-negative components. Each component is a topic. Every passage is then
scored against the fitted model, a served model names each topic from its
terms, and the whole model is drawn as an interactive figure.

**One model per language**, because a mixed corpus fitted as one model spends
its topics telling the languages apart — "delivery" and "Lieferung" are the
same subject and share no characters. The cost is that topic numbers are only
meaningful within a language.

**A fit is always the whole corpus.** Non-negative matrix factorisation
estimates every component jointly, so there is no refitting one topic: adding
a document makes all of them stale. That is why this stage has no per-item
controls and no `start` — a fit is asked for, and one ask replaces the whole
model. Gensim's `Nmf.update()` is deliberately not used: the vocabulary is
fixed when the model is constructed, so a new document's novel terms would be
silently dropped.

Factorisation rather than LDA, and a passage rather than a sentence, are both
measured choices — see [measurements](../../docs/measurements.md).

### The steps

`topics` is both the queue and the result: a row whose `topic_index` is NULL
is an outstanding request, and a row that has one is a topic.

1. **Claim the request.** The oldest row that is both outstanding and
   `pending`, with `FOR UPDATE SKIP LOCKED`.
2. **List the languages** present in `passages.language`. A passage with no
   language is in no model; `GET /topics/fit` reports how many there are. If
   no passage carries one, the fit fails saying to chunk the corpus first.
3. **Build the vocabulary** from `passages.lemmas`, streamed by id in batches
   of 500. The lemmas are read, not computed. What the fitter takes is a
   callable returning a fresh walk, because a fit reads the corpus once to
   build the vocabulary, once per pass to fit, and once more to score.
4. **Filter it.** `filter_extremes` drops terms below `TOPIC_NO_BELOW`
   passages and above `TOPIC_NO_ABOVE` of them. Two failures are raised
   rather than fitted through: an empty vocabulary, and a vocabulary whose
   every surviving term has an idf of zero, which would make every topic
   `NaN`.
5. **Fit.** `Nmf` over `TOPIC_NUM_TOPICS` components and `TOPIC_PASSES`
   passes, seeded with `TOPIC_RANDOM_STATE`.
6. **Read the topics out.** `TOPIC_TOP_TERMS` highest-weighted terms each.
   This is also the only signature a label is matched on across a refit.
7. **Score every passage.** One walk produces the memberships (every weight
   above `TOPIC_MIN_WEIGHT`, clamped to 1.0), the count of unplaced passages,
   and the whole fitted space the figure needs.
8. **Carry the labels over.** `carry_labels` matches on shared top terms —
   Jaccard overlap of at least 0.4 — and each previous label is used at most
   once. A topic whose terms moved too far loses its name. `ponytail:` a
   greedy pass; the upgrade is to solve it as an assignment problem.
9. **Name the unnamed topics.** Only topics holding no label are offered to
   the model, which is shown the terms and up to four excerpts and told which
   language to answer in. `Mixed` is stored as no name. `labelled_by` records
   `person` or the model identifier.
10. **Replace everything, in one transaction** per language. Topic ids are
    read back after the insert rather than from `RETURNING`, whose row order
    for an `executemany` is not guaranteed, and are keyed by
    `(language, topic_index)`.
11. **Draw the figure**, after the topics are written, so a failure here is
    logged and left. `pyLDAvis` is imported inside the function. Two shims
    are marked `ponytail:` — the complex residue from `js_PCoA`'s
    eigendecomposition, and a log of zero in the relevance ranking.

On failure the previous topics stay exactly as they were, readable beside the
reason. One language the fitter refuses is logged and left out; the run fails
only if no language produced a model.

Where `TOPIC_MODEL` or `LLM_MODEL` resolves to something, the worker asks it
one question before claiming the request. See
[`backend/llm/`](../llm/README.md).

### Where a topic sits

`topics.embedding` is the mean of the vectors of the passages a topic holds,
normalised to length 1, written in SQL out of `passages.embedding`.
`TopicCatalog.crowded(threshold)` returns the pairs of one language's topics
whose centroids sit closer than that. Nothing merges them — a coverage report
read by its names needs to be told, and what to do is a person's.

No HNSW index on that column: a fit is tens of rows.

A topic is offered to the labeller only when its passages carry facts.
`TOPIC_LABEL_MIN_FACT_SHARE` is the floor. It reads facts rather than terms,
which is what keeps it language-neutral. The labeller is also shown the names
already given and refuses one it repeats.

### Limits of the output

Properties of the input, not defects here. Figures in
[measurements](../../docs/measurements.md).

- **Title-cased foreign text reaches the German vocabulary.** Every word in a
  title-cased English heading is capitalised, so the rule that drops
  lower-case nouns as foreign does not fire. Fixing it needs a signal other
  than case in `backend/nlp/analysis.py`.
- **German compounds fragment the vocabulary.** `Gewerbeimmobilienmärkte` is
  one term where English writes three the factorisation can share across
  topics. A decompounder in chunking would be the fix.
- **Some German topics are document apparatus** — table headers, figure
  captions, financial-statement scaffolding. That is what
  `include_in_coverage` is for, and a refit needs a few minutes of review.

## Inputs and outputs

**Reads:** `topics` (the queue), `passages.language`, `passages.lemmas`,
`passages.text` for excerpts, `passages.block_type` and `doc_sha256` and
`facts.validated` for reporting.

**Writes:** `topics`, one row per topic per language; `passage_topics`, one
row per passage-topic pair above the weight floor; `topics/<language>.npz`
in the `models` bucket; and `topics/<language>.html` in the `export` bucket.

### The model and the picture of it are two artefacts

`models` holds the **factorisation**; `export` holds a pyLDAvis page drawn
from it. Until `models` existed, a fit's one durable trace was a view of
itself with d3 inlined, and the matrix it was drawn from went out of scope
when scoring returned — so the disposable half was the kept half.

They have opposite durability contracts: the picture is derivable from the
model, and the model is derivable from nothing short of another fit. See
[`blob_store/`](../blob_store/README.md#the-five-buckets).

`.npz` rather than gensim's own `Nmf.save()`, which is pickle underneath. A
pickle is a promise about the class that wrote it, so a gensim upgrade can
make last year's fit unloadable — which is the one thing keeping it was
for. Five arrays and a version tag load in anything that has numpy. It is
compressed because a term-topic matrix is mostly zeros: a factorisation
puts a term in few of its topics, and deflate is very good at that.

| Route | Answers |
|---|---|
| `GET /topics/status` | Queue depth by status |
| `GET /topics` | Every topic with how much of the corpus it holds |
| `GET /topics/fit` | The model's state, one entry per language |
| `GET /topics/visualisation/{language}` | One language's figure. 400 `invalid_language`, 404 `no_visualisation` |
| `PATCH /topics/{topic_id}` | Records a label and a coverage choice |
| `POST /topics/discover` | Queues a fit. 202 |
| `POST /topics/stop` | Withdraws a queued fit |
| `POST /topics/retry` | Returns a failed fit to the queue |
| `DELETE /topics` | Deletes every topic, membership, stored model and figure |

There is no `POST /topics/start` and no `/topics/rerun`: asking is what
creates the work, and a fit has no scope to narrow to.

**Command line:** `python -m topic_modelling.run`, one flag per route.
`--discover` queues and drains in one go; `--watch` is what the worker
container runs. Every flag has a `make` target.

### The database

`topics` carries two queues at once: `status` is this stage's and
`question_status` is question generation's. A CHECK named
`topics_modelled_is_a_topic` enforces that a `modelled` row has an index, a
language and at least one term. `(language, topic_index)` is unique; NULLs do
not collide.

`topics` has no foreign key to anything. `passage_topics` cascades from both
sides, so re-chunking a document takes its memberships with it — which is
what leaves a model stale, and what `GET /topics/fit` makes visible.

The dominant topic is derived, not stored. It is declared once as `DOMINANT`
beside the table because two services read it, with `DISTINCT ON` so two
topics tied at the same weight yield one row.

## Configuration

Read once at start-up from the environment, with no default in code. Changing
any of the first eight changes what the topics are, and there is no partial
refit.

| Setting | Default | What it does |
|---|---|---|
| `TOPIC_PASSAGES_PER_TOPIC` | `40` | How many passages one topic is worth. Above 0 this turns `TOPIC_NUM_TOPICS` into a ceiling and fits `passages / this`, floored at 2. `0` turns it off |
| `TOPIC_NUM_TOPICS` | `40` | Topics per language, and the ceiling above. At least 2 |
| `TOPIC_PASSES` | `10` | Times the factorisation walks the corpus |
| `TOPIC_RANDOM_STATE` | `42` | Seed. Fixed, so the same corpus gives the same topics |
| `TOPIC_TOP_TERMS` | `12` | Terms stored as a topic's signature, and what a label is matched on |
| `TOPIC_MIN_WEIGHT` | `0.05` | Smallest membership weight stored, exclusive. In `(0, 1]` |
| `TOPIC_NO_BELOW` | `3` | Drop a term appearing in fewer than this many passages |
| `TOPIC_NO_ABOVE` | `0.5` | Drop a term appearing in more than this share. In `(0, 1]` |
| `TOPIC_LANGUAGE_NAMES` | `de:German,en:English` | ISO code to language name, for the naming prompt |
| `TOPIC_LABEL_MIN_FACT_SHARE` | `0.15` | Below this share of validated facts, a topic is not sent to the model to be named |
| `TOPIC_MODEL` | `ollama_chat/gemma4:12b` | The model that names a topic |
| `LLM_MODEL` | — | What `TOPIC_MODEL` falls back to. With neither, topics are fitted but not named |

Not configurable, on purpose: the Jaccard threshold for carrying a label
(0.4), how many excerpts the labeller sees (4), how much of each (400
characters), the figure's bar-chart length (30) and the streaming batch size
(500). `include_in_coverage` is a per-topic human decision with no rule
behind it.

## The user interface

The Topics page is the whole service in one screen: the stored model's
figures, a model-health row per thing that has to hold, every topic with its
terms and size, the pyLDAvis map, one topic's own controls, the three
corpus-wide fit controls, and delete behind a confirmation.

## Tests

`make test` runs all of them.

| Layer | Covers |
|---|---|
| Unit | The fitter, the labeller, the visualisation, the service flow, the wiring, the command line, reporting and what reaches the vocabulary |
| Regression | One test per defect this service has had, named for the symptom |
| Property | Invariants over generated corpora: every weight a probability, every passage placed or counted, a seeded fit reproducible |
| Integration | Both repositories against a migrated Postgres, and all nine routes |
| Frontend | The page against a scripted backend, including every refusal |
| Static | The pinned surface |

Every test acts through an object rather than the code under test directly —
`FitterDriver`, `FittingDriver`, `LabellerDriver`, `ServiceDriver` and
`CommandDriver` in `tests/unit/topics/topic_drivers.py`, `TopicStore` and
`TopicsApi` in `tests/integration/topics.py`, and `TopicsPage` in
`tests/frontend/pages.py`.

The quality of the topics themselves is not asserted. Whether twelve is the
right number is a measurement, and a threshold on coherence would fail on
somebody else's corpus rather than on a regression. What is asserted is that
the prompt carries what it should, that `Mixed` is stored as no name, and
that an unreachable model does not lose the fit.
