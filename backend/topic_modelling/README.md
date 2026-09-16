# Topic modelling

> This README is the service's description, and
> `tests/static/test_topics_pinned.py` pins its
> surface — the settings it reads, the routes it serves, the fields it answers
> with, and the modules it is made of. Changing any of those fails that test
> on purpose. Unfreezing means editing that file in the same commit and saying
> why.

## What it does

This service answers one question about a corpus: **what is it about?** Not
by asking a language model, and not by comparing the corpus to anything
outside it. It reads the words the corpus itself uses, finds the groups of
words that tend to appear together, and calls each group a topic. A passage is
then described by how much of each topic it holds.

Nothing here is pretrained and nothing is embedded. The vocabulary is the
corpus's own, which is what makes the service industry-agnostic: point it at
a bank's procedures and the topics are about banking; point it at a
warehouse's manuals and they are about warehousing. No word list, no ontology
and no domain model has to be supplied or maintained.

The result matters to the rest of the pipeline for one reason. Question
generation works one topic at a time: the facts a question may be written from
are the ones whose passage has that topic as its strongest. So a topic is the
unit of work, the unit of coverage reporting, and the thing that makes a
question's subject measurable. A corpus with no topics generates no questions.

### How, in one paragraph

Chunking has already split every document into passages, detected each
passage's language, and stored the *content lemmas* of each — the nouns,
proper nouns and adjectives, reduced to dictionary form. This service reads
those lemmas. For each language separately it builds a vocabulary, drops the
terms that are too rare to mean anything and the terms that are too common to
distinguish anything, weights what is left by tf-idf, and factorises the
resulting passage-by-term matrix into a small number of non-negative
components. Each component is a topic: a weighting over the vocabulary, read
out as its highest-weighted terms. Each passage is then scored against the
fitted model, producing a weight per topic, and the weights above a floor are
stored. A served language model is asked to name each topic from its terms and
a few of its strongest passages. Finally the whole model is drawn as an
interactive figure and stored as a page.

### Why one model per language

A corpus of German and English passages fitted as one model spends its topics
on telling the two languages apart rather than on telling subjects apart:
"delivery" and "Lieferung" are the same subject and share no characters, so
the factorisation sees two unrelated terms and the strongest signal in the
matrix becomes the language itself. Fitting each language over its own
vocabulary removes that signal. The cost is that topic numbers are only
meaningful within a language — `de #3` and `en #3` are unrelated — and that a
topic present in both languages is two topics.

### Why a fit is always the whole corpus

Non-negative matrix factorisation estimates every component jointly. Each
topic is defined relative to all the others over one vocabulary, so there is
no such thing as refitting one topic: adding a document does not make one
topic stale, it makes all of them stale. That is why this stage, alone among
the pipeline's six, has no per-item controls and no `start` — a fit is
**asked for**, and one ask replaces the whole model.

Gensim's `Nmf.update()` would allow incremental training and is deliberately
not used. The vocabulary is fixed when the model is constructed, so a new
document's novel terms would be silently dropped, and the result would come to
depend on the order documents were ingested in — which is exactly what
`TOPIC_RANDOM_STATE` exists to prevent.

### Why factorisation rather than LDA

LDA is defined over counts: words drawn from a multinomial. A tf-idf weighted
input contradicts its own likelihood, and measured on this corpus, feeding it
one was worse than feeding it counts (c_v coherence 0.514 against 0.522).
Factorisation carries no such assumption, and weighting the input is what
stops one dominant vocabulary spreading across every topic. Measured on 501
German passages at twelve topics, weighting the input dropped term overlap
between topics from 32% to 11% and raised coherence from 0.52 to 0.66.

### Why a passage rather than a sentence

The unit handed to the factorisation is a passage, which was measured too. A
sentence averages 5.7 content tokens — too few to express a mixture of topics
at all — and 6% of sentences hold none of the vocabulary. German, twelve
topics: c_v 0.522 per passage against 0.396 per sentence.

---

## The steps, in order

A fit is one row on a queue. `topics` is both the queue and the result: a row
whose `topic_index` is `NULL` is an outstanding request, and a row that has
one is a topic. Everything below happens inside `TopicModellingService`.

### 1. Claim the request

**In:** the `topics` table. **Out:** a fit id, or nothing.

`TopicQueue.claim` takes the oldest row that is both outstanding and
`pending`, with `FOR UPDATE SKIP LOCKED`, and marks it `in_progress` with a
timestamp. Both conditions are load-bearing: selecting on status alone would
let a fitted topic that somehow reached `pending` be claimed as a request, and
running it would replace the whole table.

Nothing else on the queue is contended, so `topic-worker` can be scaled but
there is no point: a fit is one row and only one worker can hold it.

### 2. List the languages

**In:** `passages.language`. **Out:** the ISO 639-1 codes present, commonest
first.

A passage with no language is in no model. Chunking leaves `language` NULL
when a passage is too short for the detector to judge — under 40 characters of
prose once addresses are stripped — so those passages, and every fact drawn
from them, sit outside every topic-weighted report. `GET /topics/fit` reports
how many there are, and the Topics page flags it.

If no passage carries a language at all, the fit fails with a message saying
to chunk the corpus first, because chunking is what detects it.

### 3. Build the vocabulary

**In:** `passages.lemmas` for one language, streamed by id in batches of 500.
**Out:** a `gensim` dictionary, and a passage count.

The lemmas are read, not computed: chunking stored them. That matters for
reproducibility — the same corpus gives the same vocabulary without reloading
spaCy — and it is what keeps grammar out. There is no stopword list and no
minimum token length in this service's configuration, because the vocabulary
is already only nouns, proper nouns and adjectives.

The corpus is *streamed*, not held. `TopicQueue.passages` is a generator over
an ordered query, and what the fitter takes is a **callable returning a fresh
walk** — not an iterable. A fit reads the corpus once to build the vocabulary,
once per pass to fit, and once more to score, so a spent generator would fit a
model over nothing and report success. The fitter checks for exactly that and
refuses a corpus it cannot walk twice.

Ordering by id is what makes a seeded fit reproducible.

### 4. Filter the vocabulary

**In:** the dictionary and the passage count. **Out:** a smaller dictionary.

`filter_extremes` drops any term appearing in fewer than `TOPIC_NO_BELOW`
passages and any term appearing in more than `TOPIC_NO_ABOVE` of them.

The low end removes the typos, the one-off product names and the OCR debris
that would otherwise each get a topic to themselves. The high end catches the
boilerplate no word list could know about: a heading repeated in every
document, a source note under every table.

Two failures are raised here rather than fitted through:

- **Nothing left.** If the filter empties the vocabulary, the message says how
  many terms were found, over how many passages, and what the two bounds were,
  so a reader knows which to move.
- **Nothing to separate.** A term appearing in *every* passage has an idf of
  zero. If every surviving term is one — reachable with `TOPIC_NO_ABOVE=1`, or
  on a single-passage corpus — the whole weighted matrix is zero, the
  factorisation divides by its norm, and every topic comes out `NaN`. The fit
  refuses instead, naming `TOPIC_NO_ABOVE`.

### 5. Fit the model

**In:** the filtered vocabulary and a re-walkable corpus. **Out:** an `Nmf`
model.

Each passage becomes a bag of words (`dictionary.doc2bow`), weighted by tf-idf
read off the dictionary's own document frequencies and applied lazily as the
corpus is walked. `Nmf` factorises that into `TOPIC_NUM_TOPICS` components
over `TOPIC_PASSES` passes, seeded with `TOPIC_RANDOM_STATE`.

Memory follows the batch and the vocabulary, not the corpus.

### 6. Read the topics out

**In:** the fitted model. **Out:** one `FittedTopic` per component.

`show_topic` gives each component's highest-weighted terms;
`TOPIC_TOP_TERMS` of them are kept. These are the topic's readable identity
and — importantly — the **only** signature a label is matched on across a
refit, so a very short list makes two unrelated topics look alike.

### 7. Score every passage

**In:** the model and one more walk over the corpus. **Out:** the memberships,
the count of unplaced passages, and the whole fitted space.

Each passage's full topic distribution is read, and everything the rest of the
service needs comes out of this single walk:

- **Memberships.** Every topic whose weight is strictly above
  `TOPIC_MIN_WEIGHT`, clamped to 1.0. The clamp is not cosmetic: gensim
  returns float32, and a distribution's last member can round to 1.0000001,
  which breaches the CHECK on `passage_topics.weight` and rolls back the whole
  batch.
- **Unplaced passages.** A passage with no membership at all is counted. There
  are two ways to be one: holding none of the surviving vocabulary, or having
  every weight fall below the floor. Both are counted here and both are
  reported as `passages_without_topics`, which is why the page names both
  settings.
- **The space.** The term-by-topic matrix, the topic-by-passage matrix, each
  passage's term-occurrence count, the vocabulary and each term's corpus
  frequency — everything the figure needs, which the database does not keep.

### 8. Carry the labels over

**In:** the new topics, and the previous fit's topics that carried a label or
sat out of coverage. **Out:** the new topics, some now labelled.

A fit deletes every topic, so a name somebody typed survives only by being
re-attached. `carry_labels` matches on shared top terms — Jaccard overlap of
at least 0.4 — and each previous label is used at most once, on its best
match. A topic whose terms moved too far loses its name rather than keeping it
wrongly, because a coverage report is read by its names. The
`include_in_coverage` flag travels with the label.

This is a greedy pass, marked `ponytail:` in the code: a label can land on the
second-best topic when two compete for it. The upgrade is to solve it as an
assignment problem.

### 9. Name the unnamed topics

**In:** a topic's terms and the text of up to four of its strongest passages.
**Out:** a short noun phrase, or nothing.

Only topics that hold no label are offered to the model. A name a person typed
is never overwritten, and neither is one just carried over.

The terms alone are a thin prompt — a dozen words can read as several subjects
at once — so a few excerpts go in with them, truncated to 400 characters each.
The model is told which language to answer in, because the coverage report the
name appears in is read in that language. It is also told to answer exactly
`Mixed` when the terms share no subject, and `Mixed` is stored as no name: a
wrong name is worse than none.

Whatever named a topic is recorded in `labelled_by` — `person` or the model
identifier. A model that cannot be reached is logged and the topic is left
unnamed; losing the names is better than losing the fit. This step is skipped
entirely when no model is configured.

### 10. Replace everything, in one transaction

**In:** every language's fitting. **Out:** the new `topics` and
`passage_topics` rows, and a membership count.

One transaction for every language, because a corpus holding one language's
new topics beside another's old memberships is a corpus whose weights point at
the wrong subjects. The claimed request row goes with the old topics, which is
why the new rows are written `modelled` outright rather than the stage
finishing the row afterwards. A request queued *behind* this fit is left on
the queue.

Topic ids are read back after the insert rather than taken from `RETURNING`,
whose row order for an `executemany` is not guaranteed to match the
parameters, and they are keyed by `(language, topic_index)` because an index
identifies a topic only within a language.

A fit replaces every row, so every topic returns to `new` on question
generation's queue. The questions themselves are not a topic's and survive it.

### 11. Draw the figure

**In:** the fitted space. **Out:** one HTML page per language in the `export`
bucket, under `topics/<language>.html`.

This runs *after* the topics are written, so every failure here is logged and
left: a model with no picture is still a model. `pyLDAvis` is imported inside
the function for the same reason, so an image built without the `viz` group
still fits models.

The old figure is taken away before the new one is drawn, so a language whose
figure fails to render has none rather than a stale one; the route answers 404
for that. Topics keep the numbers they were fitted with (`sort_topics=False`),
so topic 3 in the figure is topic 3 in the table beside it. d3, the LDAvis
script and its stylesheet are all inlined, so the page reaches no network when
it is opened.

Two shims are marked `ponytail:`. `pyLDAvis.js_PCoA` eigendecomposes a
symmetric matrix with `np.linalg.eig`, which returns complex values that
`json.dumps` then refuses, so the imaginary part is dropped; the upgrade is
`eigh`, the routine for a symmetric matrix. And the relevance ranking takes a
log of a term weight that the factorisation leaves at exactly zero for a term
absent from a topic, so that divide is silenced.

### On failure

Anything raised is recorded on the request row, with the exception's type name
unless it is `NoVocabulary`. The previous topics stay exactly as they were,
readable beside the reason. `POST /topics/retry` returns the request to the
queue. A worker that dies holding a fit has its claim swept by the next run,
which fails the row with a reason rather than leaving it claimed and
invisible.

One language the fitter refuses is logged and left out; the run fails only if
*no* language produced a model.

---

## Tools, and where each is used

| Tool | Where | What for |
|---|---|---|
| **gensim** `corpora.Dictionary` | `topics.py` | Builds the vocabulary incrementally from batches, holds the document frequencies, and turns a passage into a bag of words. |
| **gensim** `TfidfModel` | `topics.py` | Weights the bags of words, lazily, off the dictionary's own frequencies. Weighting is what stops one dominant vocabulary spreading across every topic. |
| **gensim** `Nmf` | `topics.py` | The factorisation itself. Chosen over `LdaModel` because LDA's likelihood is defined over counts and contradicts a tf-idf input. |
| **spaCy** | *not here* | Never imported by this service. The lemmas it reads were produced by `backend/nlp/analysis.py` during chunking and stored on the passage. See below. |
| **pyLDAvis** | `visualisation.py` | Draws the fitted model as an interactive page. Imported inside `_draw` so the service runs without it. |
| **numpy** | `visualisation.py` | Discards the imaginary residue pyLDAvis's PCoA returns, and silences the log-of-zero in its relevance ranking. |
| **SQLAlchemy** | `repository.py` | The queue, the corpus stream and the reporting reads. |
| **litellm** (via `llm.client`) | `labels.py` | Asks the served model for a topic's name, in a declared shape. Optional. |
| **pydantic** | `labels.py` | Declares the shape of that one answer. |
| **boto3** (via `blob_store`) | `service.py` | Stores each language's figure in the `export` bucket. |
| **OpenTelemetry** | `service.py` | One span per fit, annotated with the run's shape. |

### Why spaCy is not in this service

This is worth stating plainly, because it looks like an omission. Topic
modelling never loads a spaCy pipeline. The division is:

- **spaCy's job** is deciding what counts as a term. Its part-of-speech tagger
  is what separates subjects from grammar — nouns, proper nouns and
  adjectives are kept, everything else is not — and its lemmatiser is what
  folds the inflections German spreads one term across, so `Institut`,
  `Instituts` and `Institute` are one term the frequency filter can count.
  Document frequency alone cannot do either of those things.
- **gensim's job** is deciding which terms belong together, over the terms
  spaCy already chose.

Doing that work here would mean loading two language models in the topic
worker and re-tagging the whole corpus on every fit, to arrive at the lemmas
already sitting in `passages.lemmas`. Instead chunking pays for it once, and
the lemmas are stored precisely so this service can re-read them cheaply on
every refit. It also means the vocabulary is stable: two fits of the same
corpus cannot disagree because a spaCy version changed between them.

The corollary is that this service's output quality is bounded by that
filtering, and the filtering has its own tests in
`tests/unit/topics/test_vocabulary.py` — which live here, rather than under
`tests/unit/nlp/`, because what reaches the vocabulary is a topic-modelling
concern even though the code that decides it is not.

---

## Known limits of the output

These are properties of the input, not defects in this service, and they bound
what the topics can be. They are recorded here because the figures below were
read off a real fit and somebody reading the topics will meet them.

**Title-cased foreign text reaches the German vocabulary.** `NLP_MODELS` names
German as a language that capitalises every noun, so `nlp.analysis` treats a
lower-case noun as a word from another language and drops it. A *title-cased*
English heading defeats that: every word is capitalised, so nothing is
dropped, and English function words enter the German vocabulary as subjects.
The stored German figure shows the result — topic #2 is
`market, due, concentration, financial, with, serious, incidents,
consequences, of`, 9.3% of the German corpus spent on an English topic.
`tests/unit/topics/test_vocabulary.py` carries a strict `xfail` reproducing
it. Fixing it needs a signal other than case in `backend/nlp/analysis.py`,
because every German noun is capitalised; it is not fixable here.

**German compounds fragment the vocabulary.** `Gewerbeimmobilienmärkte`,
`Terrorismusfinanzierung` and `Unternehmenskredite` are each one term, where
the English corpus writes `commercial real estate`, `terrorist financing` and
`corporate credit` as words the factorisation can share across topics. This is
why the English topics separate more cleanly than the German ones on the same
corpus. A decompounder in the chunking pipeline would be the fix, and it would
be a substantial change in another service.

**Some German topics are document apparatus rather than subjects.** Three of
the twelve — the `euro, million, milliarde, prozent, abbildung` one, the
`quelle, gesamt, zahl, eingang` one, and the `vorjahr, veränderung, vermerk,
stellenwegfall` one — are table headers, figure captions and
financial-statement scaffolding, and together they hold about 19% of the
German corpus. This is anticipated rather than surprising: it is what
`include_in_coverage` is for, the Topics page puts the table share and
facts-per-passage in the picked topic's own table beside the checkbox to spot
them by, and the exclusion is carried across a refit with the label. It does mean a refit needs a few
minutes of review.

**The English model is in good shape** on the same corpus, for comparison:
twelve nameable subjects (AI regulation, capital requirements, licensing,
real-estate risk, ICT third parties, money laundering, consumer credit,
sustainability in insurance, and so on), evenly sized between 5.5% and 11.4%,
none of them apparatus.

---

## Inputs and outputs

### Read

| From | What |
|---|---|
| `topics` | The queue: a row with a NULL `topic_index`. |
| `passages.language` | Which model a passage belongs to. |
| `passages.lemmas` | The vocabulary, per passage. |
| `passages.text` | Excerpts, for naming a topic. |
| `passages.block_type`, `passages.doc_sha256` | Reporting only. |
| `facts.validated` | Reporting only: how many questions a topic could yield. |

### Written

| To | What |
|---|---|
| `topics` | One row per topic per language: its index, its top terms, its label and what named it, its coverage flag, and what the fit was over. |
| `passage_topics` | One row per passage-topic pair above the weight floor. |
| `export` bucket | `topics/<language>.html`, one pyLDAvis page per language. |

### Served

| Route | Answers |
|---|---|
| `GET /topics/status` | Queue depth by status. `modelled` counts topics; the rest describe a fit. |
| `GET /topics` | Every topic with how much of the corpus it holds. |
| `GET /topics/fit` | The model's state: one entry per language, with the live counts beside the fitted ones. |
| `GET /topics/visualisation/{language}` | One language's figure. 400 `invalid_language`, 404 `no_visualisation`. |
| `PATCH /topics/{topic_id}` | Records a label and a coverage choice. 404 `unknown_topic`. |
| `POST /topics/discover` | Queues a fit. 202. |
| `POST /topics/stop` | Withdraws a queued fit. |
| `POST /topics/retry` | Returns a failed fit to the queue. |
| `DELETE /topics` | Deletes every topic, membership and figure. |

There is no `POST /topics/start` and no `/topics/rerun`: asking is what
creates the work, so a refit and a first fit are the same request. There are
no narrowed routes either — a fit has no scope to narrow to.

### Command line

`python -m topic_modelling.run`, one flag per route. `--discover` on its own
queues and drains in one go; `--watch` is what the worker container runs.
`--visualise` writes each stored figure to `topics/<language>.html` on disk.
Every flag has a `make` target: `topics`, `topics-status`, `topics-discover`,
`topics-visualise`, `topics-stop`, `topics-retry`, `topics-delete`.

### The database, in detail

`topics` carries two queues at once. `status` is this stage's — `pending`,
`in_progress`, `modelled`, `failed` — and `question_status` is question
generation's, over topics rather than over fits. A `CHECK` named
`topics_modelled_is_a_topic` enforces that a `modelled` row has an index, a
language and at least one term, so the queue and the result cannot be
confused. `(language, topic_index)` is unique; NULLs do not collide, so any
number of requests can coexist.

`topics` has no foreign key to anything: a topic spans passages in many
documents. `passage_topics` cascades from both sides, so re-chunking a
document deletes its passages and takes the memberships with them — which is
what leaves a model *stale*, and what `GET /topics/fit` exists to make
visible. `passage_topics.weight` is constrained to `(0, 1]`.

The dominant topic — a passage's highest weight — is derived, not stored. It
is declared once as `DOMINANT` beside the table rather than in either
service, because two services read it, and with `DISTINCT ON` rather than a
max-weight join so two topics tied at the same weight yield one row.

---

## Configuration

Every setting is read once at start-up from the environment, with no default
in code: a missing one stops the container naming itself, rather than failing
the first fit an hour in. Tuning lives in `configs/env/backend.env`, which is
in git because these are decisions rather than credentials.

Changing any of the first seven changes what the topics *are*, so changing one
without refitting leaves stored topics that describe settings no longer in
force. There is no partial refit to reach for.

| Setting | Default | What it does |
|---|---|---|
| `TOPIC_NUM_TOPICS` | `12` | Topics per language. Must be at least 2. |
| `TOPIC_PASSES` | `10` | Times the factorisation walks the corpus. More is a better fit and a longer wait. |
| `TOPIC_RANDOM_STATE` | `42` | Seed. Fixed on purpose. |
| `TOPIC_TOP_TERMS` | `12` | Terms stored as a topic's signature. Must be at least 1. |
| `TOPIC_MIN_WEIGHT` | `0.05` | Smallest membership weight stored, exclusive. Must be in `(0, 1]`. |
| `TOPIC_NO_BELOW` | `3` | Drop a term appearing in fewer than this many passages. |
| `TOPIC_NO_ABOVE` | `0.5` | Drop a term appearing in more than this share of passages. Must be in `(0, 1]`. |
| `TOPIC_LANGUAGE_NAMES` | `de:German,en:English` | ISO code to language name, for the naming prompt. |
| `LLM_MODEL` | — | Shared with every other stage. Unset means topics are fitted but not named. |

### How each default was chosen

**`TOPIC_NUM_TOPICS=12`.** Measured over 501 German and 240 English passages,
three seeds each:

| k | de coherence | biggest topic | en coherence | biggest topic |
|---|---|---|---|---|
| 8 | 0.704 | 71% | 0.561 | 44% |
| 12 | 0.656 | 46% | 0.553 | 37% |
| 16 | 0.599 | 40% | 0.591 | 29% |
| 20 | 0.613 | 30% | 0.582 | 26% |

Coherence alone always prefers fewer topics, because broad topics share terms
trivially — at four, one German topic held 71% of the corpus, which is no
partition at all. Read beside that share and beside inter-topic overlap,
twelve is the best German point: the lowest overlap (11%) and the most stable
across seeds. English is within its own noise of its best there. Re-measure if
the corpus grows a lot.

**`TOPIC_RANDOM_STATE=42`.** The same corpus and settings must give the same
topics, or a reference dataset's topic weighting moves under its own feet
between releases. Change it only to see how stable a fit is.

**`TOPIC_MIN_WEIGHT=0.05`.** The model gives a passage a weight on many
topics, so with no floor `passage_topics` fills with rows that are almost all
noise. The weights are normalised, so this reads as a share. It must be above
zero because the column's CHECK requires it.

**`TOPIC_NO_ABOVE=0.5`.** Lowering it does not clean up a noisy fit. Swept
from 0.02 to 1.0 on this corpus: below 0.5 the widespread *real* terms go
first, which promotes the noise into the top terms rather than removing it,
and at 0.02 coherence collapses and 120 passages fall out of every topic.
Above 0.5 nothing changes, because no term reaches half the corpus. Note that
`1.0` is legal but degenerate on a small corpus — see step 4.

**`TOPIC_TOP_TERMS=12`.** Also what a label is matched on across a refit, so a
very short list makes two unrelated topics look alike and a label lands on the
wrong one.

### Not configurable, on purpose

The Jaccard threshold for carrying a label (0.4), how many excerpts the
labeller is shown (4), how much of each it sees (400 characters), how many
terms the figure's bar chart lists (30), and the streaming batch size (500)
are all constants in the code. Each is named and commented where it is
defined. None of them is a decision a deployment has been asked to make.

`include_in_coverage` is deliberately a per-topic human decision with no rule
behind it: what counts as a subject belongs to the corpus, not to the
pipeline. The Topics page puts the two figures worth reading it off — the
share of a topic's passages that are tables, and validated facts per passage
— beside the checkbox.

---

## The user interface

The Topics page (`frontend/views/topics.py`) is the whole service in one
screen, top to bottom:

1. **The stored topic model.** Five figures: topics, language models,
   memberships, passages placed, and model age. Every one carries a tooltip
   saying what it counts.
2. **Model health.** One row per thing that has to hold for the topics to
   describe the corpus, each reading `OK` or `Attention` with an explanation
   of what to do. This is where a stale model, unplaced passages, passages
   with no language, and a failed fit all surface.
3. **Every topic in the model.** One row per topic per language, with its
   terms, its size, its table share and its facts-per-passage.
4. **Topic map.** The pyLDAvis figure for a chosen language, with a caption
   explaining how to read it — including what the λ slider does, which is the
   one control in the figure nobody guesses.
5. **One topic.** Its own figures, its top terms, and the two controls that
   change it: a label and the coverage checkbox.
6. **Fit the model.** Three corpus-wide controls — fit, stop, retry — with a
   note saying why none of them can be per-topic, and a line saying what the
   stored model was fitted over.
7. **Delete.** Behind a confirmation, with the warning that labels go with the
   topics because nothing else stores one.

---

## Tests

327 tests across six layers. `make test` runs all of them.

| Layer | File | Tests | Covers |
|---|---|---|---|
| Unit | `tests/unit/topics/test_fitting.py` | 35 | The fitter: weights, vocabulary filtering, reproducibility, subject separation, and every setting it refuses. |
| Unit | `tests/unit/topics/test_labelling.py` | 36 | The prompt, what the model's answer is stripped of, what counts as a refusal, and every path through `carry_labels`. |
| Unit | `tests/unit/topics/test_visualisation.py` | 14 | The shape of the space handed to pyLDAvis, and that the page is self-contained, numbered correctly and parseable. |
| Unit | `tests/unit/topics/test_topic_service.py` | 27 | The flow: claim, multi-language fit, partial failure, drawing, naming, and every failure that must not lose the topics. |
| Unit | `tests/unit/topics/test_topic_wiring.py` | 22 | The settings read from the environment, and what the factory builds with and without a served model. |
| Unit | `tests/unit/topics/test_run.py` | 13 | Every command-line flag, and the exit status each gives. |
| Unit | `tests/unit/topics/test_reporting.py` | 12 | `stale`, the one derived answer on this service's boundary. |
| Unit | `tests/unit/topics/test_vocabulary.py` | 9 | What reaches the vocabulary: grammar dropped, inflections folded, URLs and foreign function words excluded. Needs spaCy (`nlp` marker). |
| Regression | `tests/unit/topics/test_regressions.py` | 15 | One test per defect this service has had. Named for the symptom. |
| Property | `tests/property/test_topic_invariants.py` | 15 | Invariants over generated corpora: every weight a probability, every passage placed or counted, every matrix rectangular and every row a distribution, and a seeded fit reproducible. |
| Integration | `tests/integration/database/test_topic_store.py` | 50 | Both repositories against a migrated Postgres: which rows a fit replaces, which it leaves, and what every reporting query answers. |
| Integration | `tests/integration/api/test_topics.py` | 34 | All nine routes, with the object store behind them. |
| Frontend | `tests/frontend/test_topics_page.py` | 36 | The page against a scripted backend, including every refusal and every nullable column. |
| Static | `tests/static/test_topics_pinned.py` | 9 | The pinned surface: settings, routes, answer shapes, modules, and that this README exists. |

Coverage of `backend/topic_modelling` is **99% of statements** — 538 of 539.
The one uncovered line is `run.py`'s `if __name__ == "__main__"`.

### How the tests are built

Every test acts through an object rather than through the code under test
directly — the page object pattern, applied to a service:

- `tests/unit/topics/topic_drivers.py` holds `FitterDriver`, `FittingDriver`,
  `LabellerDriver`, `ServiceDriver` and `CommandDriver`, plus in-memory
  doubles for the queue and the bucket. A test says
  `fitter.with_passage(99, "zzz").fit("de")` and asks
  `fitting.unplaced`; it never builds a `TopicFitter` or a corpus callable.
- `tests/integration/topics.py` holds `TopicStore`, which owns the seeding and
  the SQL, and `TopicsApi`, which owns the paths.
- `tests/frontend/pages.py` holds `TopicsPage`, which owns the selectors, the
  scripted answers, and what a reader does — `press("Save")`,
  `type_label(1, "Shipping")`.

A layout or schema change is one edit in a driver rather than one per test.

### What the layers are for

The split is by what can go wrong where. The unit layer runs the real gensim
and the real pyLDAvis over a four-passage corpus, so it covers the whole fit
in under a second and there is no reason for a fitter test to reach a
database. The integration layer covers what only SQL can get wrong: which rows
a `DELETE` takes, whether a membership points at the right primary key,
whether a `CHECK` fires. The property layer covers the corpora nobody thought
to write down — it is what found the `NaN` model in step 4. The frontend layer
covers the half no integration test reaches: what a page does with an answer,
including the answers that are refusals.

### What is deliberately not tested

The quality of the topics themselves. Whether twelve is the right number, and
whether the topics a corpus produces are the subjects a reader would name, are
measurements rather than assertions — the numbers are in this README and were
taken by sweeping the settings over a real corpus. A threshold on coherence
would fail on somebody else's corpus rather than on a regression.

The served model's taste in names is likewise not asserted. What *is* asserted
is that the prompt carries what it should, that `Mixed` is stored as no name,
and that an unreachable model does not lose the fit.
