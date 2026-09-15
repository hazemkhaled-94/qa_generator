# Q&A Reference Dataset Generator

Generates a verified question-and-answer dataset from a corpus of documents, so
that a retrieval-augmented chatbot can be measured against a fixed benchmark
rather than assessed by impression.

Documents are uploaded, parsed into passages, broken down into atomic facts
that each cite one sentence of their source, and turned into questions with
known answers. Questions that deliberately have no answer in the corpus are
included, to test whether a chatbot recognises the limits of its knowledge.

The pipeline runs entirely on local infrastructure. No document content leaves
the deployment.

**Status:** ingestion, parsing, chunking, fact extraction, topic modelling and
question generation are implemented. Quality assurance and the evaluation
harness are not yet built.

Nothing here is bound to a subject or an industry. The parser, the chunker and
the topic model work over whatever the documents say; the extraction prompt's
worked example is deliberately about nothing in particular, because an example
drawn from the corpus at hand teaches the model to expect it. German and English are the two languages
configured, in `NLP_MODELS` — one list, naming both the pipeline each
language is read with and the languages the detector may answer with.

Scanned documents are refused rather than parsed: OCR is not enabled. See the
placeholder in `backend/preprocessing/parsing/pipelines/pdf.py`.

## Prerequisites

| | |
|---|---|
| Python | 3.12 or 3.13 |
| [Poetry](https://python-poetry.org/docs/#installation) | 2.0 or later |
| [Podman](https://podman.io/docs/installation) or Docker | with Compose |

The Makefile invokes `podman compose`. For Docker, set `COMPOSE=docker compose`
in the environment or edit the variable at the top of the Makefile.

## Install

```bash
git clone <repository-url> qa_generator && cd qa_generator
cp .env.example .env
```

That copies the credentials, ports and addresses. How the pipeline behaves is
in `configs/env/`, which comes with the clone and needs no copying.

Edit `.env` and replace every `change_me_*` value. Two have constraints:
`PHOENIX_ADMIN_SECRET` needs at least 32 characters including a digit and a
lower-case letter, and `PHOENIX_DEFAULT_ADMIN_INITIAL_PASSWORD` is applied only
when Phoenix first creates its admin user — changing it afterwards means
dropping the `phoenix` database.

```bash
make install
```

This installs the dependencies and downloads the spaCy pipelines named in
`NLP_MODELS`. They are also baked into the backend image, because the runtime
has no network.

`EMBEDDING_MODEL` is not baked in. One image serves the api and all five
workers, and its weights are 2.2 GB that four of those processes never load —
and that every CI build of the image would carry. It is fetched on first use
instead, into the `models` volume: chunking wants its tokenizer and question
generation wants its weights, so whichever starts first pays for it once.

## Run

```bash
make dev
```

This generates TLS certificates, starts every service, waits for PostgreSQL,
and creates the database schema. First run pulls several images and takes a few
minutes.

Once it reports ready:

| Service | URL | Purpose |
|---|---|---|
| Frontend | http://localhost:8501 | Upload documents, browse passages and facts, view system status |
| API | http://localhost:8000/docs | OpenAPI documentation |
| Phoenix | http://localhost:6006 | Traces — sign in as `admin@localhost` with `PHOENIX_DEFAULT_ADMIN_INITIAL_PASSWORD` |
| Grafana | http://localhost:3001 | Logs — sign in with `GRAFANA_ADMIN_USER` and `GRAFANA_ADMIN_PASSWORD` |
| Adminer | http://localhost:9001 | Database browser |

Upload a PDF from the frontend, then pick it in the table on the Documents
page and press **Start** on the Parsing row: nothing runs until it is asked
to. Every control there acts on the document you picked and on nothing else,
so one document can be re-run or deleted while the rest of a corpus is
mid-flight. The System health page reports the state of every component.

## Common commands

```bash
make up              # start services
make down            # stop services, keep data
make down-volumes    # stop services and delete all data
make logs            # follow all logs
make logs-api        # follow the API log
make logs-frontend   # follow the frontend log
make logs-shipper    # follow the log shipper, when Grafana shows nothing
make test            # everything that gates; starts containers of its own
make test-fast       # only the fast layers: no spaCy, no pyright, no containers
make test-smoke      # build both images and look inside them
make test-eval       # score the served model against the golden passages
make test-coverage   # the gating layers, with a coverage report
make lint            # ruff check and format check
make typecheck       # pyright over backend, frontend and telemetry
make audit           # known advisories against the two locks
make format          # apply every fix ruff can make
make lock            # rewrite backend/api/requirements.lock
make certs           # generate the TLS certificates Elasticsearch needs
```

Every stage answers the same five verbs, and each is the same operation as the
route beside it. Run any of them with `--help` for the flags.

```bash
make parse-start     # queue every document parsing has not been asked to do
make parse           # drain the parsing queue here, in the foreground
make parse-status    # show documents by parse state
make parse-stop      # take back whatever has not begun
make parse-retry     # return every failed document to the queue
make parse-rerun     # queue every document again, finished ones included
make chunk-start     # the same five, over documents chunking has not read
make chunk           #   ... and `chunk-status`, `chunk-stop`, `chunk-retry`,
                     #       `chunk-rerun`
make extract-start   # the same five, over passages
make extract         #   ... and `extract-status`, `extract-stop`,
                     #       `extract-retry`, `extract-rerun`
make extract-revalidate  # judge stored facts again; the model is not called
make chunk-revocabulary  # read stored passages' language and vocabulary again
make topics-discover # ask for a fit over the whole corpus, and run it
make topics          # run any queued fit here, in the foreground
make topics-status   # show the topics held, and any queued fit
make topics-visualise # write each language's pyLDAvis page to ./topics/
make topics-stop     # withdraw a queued fit
make topics-retry    # return a failed fit to the queue
make topics-delete   # delete every topic and membership
make questions-start # the same five, over topics
make questions       #   ... and `questions-status`, `questions-stop`,
                     #       `questions-retry`, `questions-rerun`
make questions-reverify  # check stored questions again; no model is called
```

Any of those five verbs narrows to a single item with `SHA`, `PASSAGE` or
`TOPIC`, which is the same operation against fewer rows:

```bash
make parse-start SHA=<sha256>     # queue one document for parsing
make chunk-rerun SHA=<sha256>     # rebuild one document's passages
make extract-start SHA=<sha256>   # queue every passage of one document
make extract-retry PASSAGE=<id>   # return one failed passage to the queue
make extract-status SHA=<sha256>  # that document's passages by extract state
make questions-start TOPIC=<id>   # write the questions for one topic
```

Parsing and chunking narrow to a document, extraction to a document or a
passage, question generation to a topic, topic modelling to nothing — a fit is
all-or-nothing over one vocabulary, so there is no single topic to start, stop
or refit.

### Replaying a stage without redoing it

Three operations re-derive what a stage computed, over rows already stored,
without the expensive part:

| | |
|---|---|
| `make extract-revalidate` | Judges every stored fact again. The model is not called and no statement changes — only what the checks read off one. |
| `make chunk-revocabulary` | Reads every stored passage's language and vocabulary again. Only `passages.language` and `passages.lemmas` change. |
| `make questions-reverify` | Puts every stored question through the gates that need no model: its facts still pass their own checks, its evidence is still spread the way its difficulty says, it is still well formed, and no earlier question already asks it. |

The first two take `SHA`, `extract-revalidate` takes `PASSAGE`, and
`questions-reverify` takes `TOPIC`, like the five queue verbs.

`questions-reverify` only ever rejects. Accepting is a person's decision, and
a re-check that un-rejected would overturn one on its next run. It is what
carries a change made further up the pipeline through to the questions resting
on it: `extract-revalidate` can turn a fact that passed into one that does not,
and re-extracting a single document can take a cross-document question's second
citation away without taking the question. Neither leaves any sign on the
question itself, which is what this finds.

These exist because the obvious way to apply a change is the destructive one.
`extract-rerun` calls the model again over the whole corpus, which costs hours
and returns the same statements when only a check changed. `chunk-rerun`
deletes every passage of a document, and the facts drawn from them go with it.
When what changed is a fact check or how vocabulary is read, neither is needed.

What each leaves alone is the point. A re-judgement keeps the statement, the
method and the model's provenance — the record of one extraction — and
replaces only the verdict, the resolved span and the counts the checks read. A
re-read keeps the passages and their sentence offsets, so every citation still
resolves to the text it was checked against. Fit the topics afterwards for a
re-read to show.

Documents are ingestion's, not a stage's:

```bash
make documents                    # list every document with its parse state
make delete SHA=<sha256>          # delete a document and everything from it
make delete-derived SHA=<sha256>  # delete only its passages and facts
```

Both deletions are irreversible, and both have a route: `DELETE /documents/{sha}`
and `DELETE /documents/{sha}/derived`.

## Pipeline stages

Every stage is a queue in the database, a worker that drains it, and a way to
put rows on or off it:

| Stage | Queue depth | Start | Stop | Retry failures | Do it all again |
|---|---|---|---|---|---|
| Parsing | `GET /parsing/status` | `POST /parsing/start` | `POST /parsing/stop` | `POST /parsing/retry` | `POST /parsing/rerun` |
| Chunking | `GET /chunking/status` | `POST /chunking/start` | `POST /chunking/stop` | `POST /chunking/retry` | `POST /chunking/rerun` |
| Extraction | `GET /extraction/status` | `POST /extraction/start` | `POST /extraction/stop` | `POST /extraction/retry` | `POST /extraction/rerun` |
| Topic modelling | `GET /topics/status` | `POST /topics/discover` | `POST /topics/stop` | `POST /topics/retry` | — |
| Question generation | `GET /questions/status` | `POST /questions/start` | `POST /questions/stop` | `POST /questions/retry` | `POST /questions/rerun` |

Each verb comes twice. The routes above act on the whole queue; the same
verb under `/{scope}/{value}` acts on one item, which is what the frontend's
per-item controls call:

| Stage | Narrows to | Example |
|---|---|---|
| Parsing | `document` | `POST /parsing/document/{sha256}/start` |
| Chunking | `document` | `POST /chunking/document/{sha256}/rerun` |
| Extraction | `document`, `passage` | `POST /extraction/passage/{id}/retry` |
| Question generation | `topic` | `POST /questions/topic/{id}/rerun` |
| Topic modelling | — | a fit is all-or-nothing |

`GET /{stage}/{scope}/{value}/status` reports that item's queue the same way
`/{stage}/status` reports the whole one. A scope a stage does not accept is a
404 `unknown_scope`; a value its column cannot hold is a 400 `invalid_value`.
Narrowing is a `WHERE` on the stage's own table and nothing more, so a
narrowed verb and a whole-queue one cannot disagree about what they do.

**Nothing starts by itself.** A row arrives `new`, which no worker looks at.
`start` moves it to `pending`, which is the only status a worker claims, and
`stop` moves it back. A stage never sets another stage going: the previous
stage finishing leaves a row `new`, and somebody — the Start button, the route,
the `make` target, later the orchestrator — decides it should run.

The API only ever reads and writes the queue; the work happens in the stage's
worker container. There is no `run` route, and that is not an omission — a
conversion running inside the process that serves JSON held it for sixteen
minutes at a stretch.

The `make` targets run one drain on the host instead, in the foreground,
against the same database. That is what you want while developing a stage.

Topic modelling differs in one way: it has no `new` rows and so no `start`.
Every topic is estimated jointly over one vocabulary, so no topic can be
rediscovered on its own and a new document does not make one topic stale — it
makes all of them stale. One model is fitted per language, over that language's passages
only; a run fits every language and replaces the lot in one transaction. Asking is what creates the work, by the Start button on the Topics
page, `POST /topics/discover` or `make topics-discover`. That request is a row
in `topics` with a NULL `topic_index`: the table is both the queue and the
result. A fit that succeeds replaces every row; a fit that fails leaves the
working topics in place with the reason beside them. `DELETE /topics` removes
them entirely.

Every failure is recoverable. A row a worker died holding is failed by the next
run of that stage, with the reason recorded, rather than being left claimed and
invisible; `retry` then returns it to the queue. There is no state a row can
reach that nothing can move it out of.

## Scaling

`parse-worker`, `chunk-worker` and `extract-worker` all scale horizontally:
each claims one row at a time with `FOR UPDATE SKIP LOCKED`, so two workers
never take the same row. `topic-worker` can be scaled too, though there is no
point: a fit is one request row, and only one worker can claim it.

`question-worker` scales the same way and has far less to divide. Its queue is
one row per topic, so twelve topics per language is two dozen rows for the
whole corpus, and a worker past that has nothing to claim.

Its first start is slow and looks like nothing happening: it downloads
`EMBEDDING_MODEL` before it claims anything, which is 2.2 GB into the `models`
volume. The topics sit `pending` until that finishes, and `make logs` is where
it says so. Later starts read the volume and claim immediately.

A stage added since a stack came up has no container until `make up` creates
one. The queue fills, `/questions/status` reports it, and nothing drains it —
which reads exactly like a broken worker rather than an absent one.

```bash
podman compose up -d --scale extract-worker=4
```

Every claim is timestamped, and a stage's lease says how long one may go
unfinished before another run sweeps it. Extraction derives its lease from
`LLM_TIMEOUT_SECONDS` and `LLM_MAX_ATTEMPTS`, so raising either
one never makes a healthy worker look abandoned.

The model is the bottleneck, not the pipeline: a median passage measured at
473 s on a 31B model. A passage whose sentences carry no finite verb — a
heading, a caption, a navigation line — is skipped before the call rather than
sent and rejected afterwards.

A topic's lease is derived from what a topic actually costs, not from what one
model call costs: `QUESTIONS_PER_TOPIC` candidates, each of them a writer call
and a verifier call. Sized any smaller and a run would fail workers that are
only halfway through their first topic.

Each process opens its own connection pool, sized by `DATABASE_POOL_SIZE` and
`DATABASE_POOL_OVERFLOW`; the total across every worker has to stay under
PostgreSQL's `max_connections`.

## Reading what the pipeline produced

| Route | Shows |
|---|---|
| `POST /documents` | Upload one document |
| `GET /documents` | Documents with the state of each stage, one page at a time |
| `GET /documents/names` | Every document by name, for a picker |
| `GET /documents/{sha256}/file` | The file itself, as uploaded |
| `DELETE /documents/{sha256}` | The document, its file, its converted form and everything derived |
| `DELETE /documents/{sha256}/derived` | Only its passages and facts; chunking returns to `new` |
| `GET /passages` | Passages, with their heading trail, pages and tables |
| `GET /passages/types` | The block types the corpus actually holds |
| `GET /passages/{id}` | One passage in full, its numbered sentences and cell grids |
| `GET /facts` | Facts, accepted and rejected alike, with what they cite |
| `GET /facts/quality` | How many facts hold up, how far the passages were decomposed, and why the rest were rejected |
| `GET /topics` | Topics, their terms, and how much of the corpus each holds |
| `PATCH /topics/{id}` | Name a topic, or take it out of coverage reporting |
| `GET /topics/fit` | When the topics were fitted, over what, and whether they still describe the corpus |
| `GET /topics/visualisation/{language}` | One language's model as a self-contained pyLDAvis page |
| `DELETE /topics` | Every topic and membership |
| `GET /questions` | Questions, accepted and rejected alike, with what each cites |
| `GET /questions/{id}` | One question with the facts it was written from |
| `GET /questions/quality` | How many questions hold up, which gate stopped the rest, and how much of the corpus's subject matter is covered |
| `PATCH /questions/{id}` | Accept or reject one question |
| `GET /health` | That the process is up, for the container healthcheck |
| `GET /status` | Every component behind the API, and what each service holds |

`/documents`, `/passages`, `/facts` and `/questions` take `q` for a
case-insensitive substring and `limit`/`offset` to page; the last three also
take `document` to narrow to one digest, and `/questions` takes `topic` as
well. The frontend renders each as a page of its own and filters nothing
itself.

`/questions` is the one path that is both a stage and its output. Every other
stage is a verb with its product under a noun — `/extraction` and `/facts` —
but a question is what generation produces and what it is called, so the queue
routes and the read routes share the router. The queue routes are declared
first, which is what keeps `/questions/status` from being read as a question
with the id `status`.

A document moves through the stages by its status columns, one request at a
time: ingestion leaves `parse_status = 'new'`, starting parsing moves it to
`'pending'`, parsing sets it to `'parsed'`, and chunking selects on its own
`chunk_status` — never on `parse_status`, because a stage knows of no other
stage. Extraction does the same over the passage's `extract_status`. Nothing
schedules the stages; a person or the orchestrator does. Topic modelling stands
outside that chain: it reads every passage whatever stage its document has
reached, and runs when a request row in `topics` asks it to. Question
generation is back inside it, over a row of its own in the same table: a
fitted topic arrives `question_status = 'new'` and waits to be asked, like
everything else.

`topics` therefore carries two queues. A row with a NULL `topic_index` is a
request to refit and is topic modelling's; a row that is a topic is question
generation's. Every operation on the second carries `topic_index IS NOT NULL`,
because without it `start` would queue the asking as though it were a subject
and the two workers would fight over one row.

A fact or a question reaches its topics by joining through its passage —
`facts.passage_id` to `passage_topics` — rather than holding a topic of its
own, so no two rows can disagree about which topic a passage is in.

## What a fact is

A fact is one claim drawn out of a passage, not the passage said differently.

Chunking splits every passage into numbered sentences and stores them. A fact
**cites one of those numbers** rather than quoting text, so its source span is
exact by construction: there is no quote to search for, nothing to match
character for character, and no near miss. A citation is either a sentence the
passage has or one it does not.

What the model writes is the statement, and spaCy is what judges it. Extraction
needs a served model; the checks need only the pipelines in the image. A fact
that fails a check is stored with the reason rather than dropped: the rate at
which that happens is how the model is judged, so it belongs in the data and
not in a log.

| Check | Rejects a statement that |
|---|---|
| `evidence_absent` | cites a sentence the passage does not have |
| `copied` | is its cited sentence repeated rather than a claim drawn out of it |
| `not_atomic` | has more or fewer than one finite verb, so it carries several claims or none |
| `unsupported_addition` | asserts a number, name, date or predicate the cited sentence does not contain |
| `unresolved_reference` | leaves a pronoun a reader who cannot see the passage is unable to resolve |

The last two are the ones no lexical measure could make. `unsupported_addition`
is a hallucination check: every accepted fact has an empty `units_added`, which
is what says nothing was invented. `unresolved_reference` is what decides
whether a question written from the fact can be answered on its own.

`not_atomic` replaced a similarity threshold. Counting finite verbs is what
"one claim" actually means; measuring how many words a statement shares with
its source rejected genuine narrowings — a statement that drops a qualifier and
keeps the subject's wording scored as a reword — and let a paraphrase through
for swapping a noun.

`GET /facts/quality` reports the rejections by code, and the two numbers that
say whether the model is decomposing or summarising: claims per statement,
which should be 1, and claims per statement against claims per cited sentence,
which should sit well above 1. A sentence usually carries several claims; if a
statement keeps all of them, the passage was restated rather than broken up.
The Facts page shows all of it and says plainly when a number is wrong.

A table is read by a deterministic cell reader, not by the model. It cites the
numbered rendered row its own value sits in, so a table citation is an index
like any other and the checks need no rule of their own. Only the checks that
apply to a written sentence are applied to a composed one.

## What a question is

A question is written from one or more **facts**, and the writer is shown the
**passages** they came from as well. The facts are what the answer must rest
on; the passage is what the question can be phrased from.

Both halves are load-bearing, and the first version of this stage had only
one. Shown a single atomic statement and nothing else, a model has one triple
to work with, so the only question available is that statement with one part
replaced by a question word — `Geopolitische Konflikte schüren Unsicherheit`
asked as `Was schüren geopolitische Konflikte?`. Nobody searching a corpus of
thousands of pages types that. The passage and its heading trail are where the
institution, the document and the period come from, which is what a question
has to name to be one somebody could have asked.

The unit of work is a **topic**, because a question's subject is one. The facts
a topic may be asked about are the facts of the passages that topic is
strongest in; a passage belongs a little to many topics, and writing a question
for every one of them asks the same thing under a dozen subjects. A topic with
`include_in_coverage = false` is skipped, which is the neutral switch for "this
is not a subject" — nothing in the code decides that.

### The plan comes first

Before anything is written, a topic gets a **plan**: one slot per question,
each carrying the kind of question to write, the difficulty band to aim for,
and whether it is meant to have an answer at all. Two settings decide it, and
both are proportions rather than counts:

```
QUESTIONS_TYPE_MIX=factoid:3,reason:2,procedure:2,...
QUESTIONS_DIFFICULTY_MIX=easy:2,medium:2,hard:1
```

The weights are spread over the slots by **highest averages**, so the counts
are exact over a whole run and interleaved along it rather than run in blocks.
That second property matters: the unanswerable share and the follow-up share
are both taken by position, so a mix run in blocks would always perturb the
same kind of question.

A weight of `0`, or a name left out, is never written. That is the switch for
choosing what a run produces.

### The eleven kinds

Every one of them is a question **form**, never a subject, so the same list
applies to a manual, a contract, a policy or a report. Nothing in any prompt
names a domain.

| Kind | Asks for | Answer | Passages |
|---|---|---|---|
| `factoid` | one checkable value — how many, how much, by when | value | 1 |
| `definition` | what a named thing or status is, as the material defines it | explanation | 1 |
| `entity` | who does, decides, owns or must be told something | value | 1 |
| `enumeration` | which things belong to a named set | list | 1 |
| `condition` | when, or under what circumstances, something applies | list | 1 |
| `reason` | why something is required, done, or the way it is | explanation | 1 |
| `procedure` | how something is done, or in what order | explanation | 1 |
| `consequence` | what happens when something is or is not done | explanation | 1 |
| `comparison` | how two named things differ | list | 2 |
| `aggregation` | a total no single fact states on its own | value | 2 |
| `temporal` | what changed between two periods | list | 2 |

The last three need facts from two passages and are never planned `easy`: a
comparison drawn from one passage is a question about one thing.

One shared rule block holds what is true of every question — do not name the
source, name the subject, one question, the language of the facts — and each
kind adds what it asks for, what its answer looks like, and one worked example.
Two prompts for one rule is how the two come to disagree.

### The answer form, and why it is a column

`answer_form` is `value`, `list` or `explanation`, declared by the kind. Every
gate that reads an answer reads it against that form.

This is the column the old set did not have, and not having it is why every
question in it was a lookup. One prompt demanded *"a short noun phrase, a few
words at most … never anything with a verb in it"*, and one structural gate
enforced it on every answer. Measured against six realistic answers, five were
refused as malformed:

```
'weil die Risiken im Bankensektor gestiegen sind'                   -> malformed
'because risks in the banking sector increased'                     -> malformed
'by notifying the authority within four hours through the portal'   -> malformed
'submit the application, provide the business plan, and pay the fee' -> malformed
'EUR 15,000'                                                        -> accepted
```

Why, how, what-happens-if and which-things were not badly written. They were
**unwritable**. The verb rule now applies to a `value` alone; an `explanation`
is refused for carrying *no* verb, which is the opposite failure; and each form
has its own length bounds in `QUESTIONS_ANSWER_CHARS`.

### The three criteria, and the band they feed

Each is read off the facts the question reported citing. Each says something
different about what a chatbot has to do, so they are three columns rather
than one:

| | |
|---|---|
| `passage_scope` | `single_passage` or `multi_passage` — how many passages hold the answer |
| `document_scope` | `single_document` or `cross_document` — the one a retriever cannot fake: no single chunk holds the answer |
| `topic_scope` | `single_topic` or `multi_topic` — whether the question bridges two subjects |

`difficulty` is then **derived** rather than judged. Five things each count
one point — the three scopes above, an answer past
`QUESTIONS_LONG_ANSWER_CHARS`, and following another question — and the band is
the total: 0–1 `easy`, 2 `medium`, 3+ `hard`.

Three and not four, because `cross_document` implies `multi_passage`: two
documents are two passages, so the three scopes total at most three. A
threshold of four would have made a question spanning two documents and two
subjects — the hardest thing a retriever faces — only medium. Nothing is
weighted, because a weighting is an opinion and the point of deriving
difficulty rather than judging it is that nobody has to hold one.

### How a band is asked for without being judged

`QUESTIONS_DIFFICULTY_MIX` asks for a band. Nothing judges one. What the plan
actually chooses is the **shape of the sample** the writer is offered, and the
shape is what makes a band reachable at all — a question drawn from one passage
cannot be cross-document however it is phrased:

| Band | Shape | What the deal offers | Worth |
|---|---|---|---|
| `easy` | `single` | one passage | 0 |
| `medium` | `cross` | a passage in another document, ranked by shared vocabulary | 2 |
| `hard` | `bridge` | a bridging passage in another document — another file *and* another subject | 3 |

The band the plan asked for is stored as `planned_difficulty` beside the
`difficulty` the question turned out to be. The two disagree when the writer
cited fewer facts than it was offered, and the share that agree is on the
Questions page: it is a measurement of the plan, not a fault in the row.

A topic sitting in one document has no cross-document question in it. The deal
falls back to the widest sample it can give — the nearest passage of the same
document, by ordinal — rather than writing nothing about that subject.

### What the deal offers, and what it used to

Within a topic the writer is offered a **sample** of facts. `QUESTIONS_FACT_SAMPLE`
caps how many, **divided between the passages the sample holds**, so a wide
sample offers both sides of what it is asking about.

That cap is the second defect the old set had, and it is worth recording
because nothing about it looked broken. The cap used to *flush* a group and
then add the next passage **whole**. On a corpus whose median passage carries
ten validated facts and whose cap was six, that meant every sample was exactly
one passage. Measured over the 24 topics of this corpus, 214 samples:

| | before | after |
|---|---|---|
| multi-passage | 5 (2%) | 183 (50%) |
| cross-document | 3 (1%) | 122 (33%) |
| multi-topic | 2 (1%) | 52 (14%) |
| facts per sample | 1–54 | 1–6 |

`QUESTIONS_BRIDGE_SHARE` asked for a bridge on 35% of samples and landed two,
for the same reason: a bridge was refused when it would take the sample over
the cap, and the sample was already over it. The multi-topic scope existed and
was unreachable. There is no bridge share now — the `hard` shape reaches for
one directly.

Passages are dealt **strided over the whole topic** rather than from its start,
and each is offered once. Ten questions used to mean the first ten passages in
document order, so two thirds of a large topic was never asked about at all.

Which facts a question actually cites is the **writer's** answer, not the
sample's. That distinction was missing at first and produced a measurable lie:
facts paired only because they came from different documents had nothing to do
with each other, the writer answered one and ignored the rest as its prompt
told it to, and the row was stored with a `cross_document` label earned by a
fact the question never used — 91 of the first 140 rows. A sample wider than
one passage now also carries an instruction to use both halves.

### Follow-up threads

A share of accepted questions get a **thread**: the question somebody would
ask next, up to `QUESTIONS_MAX_FOLLOWUPS` deep. `follows_id` and
`thread_position` carry it.

Each turn takes the next kind in `QUESTIONS_FOLLOWUP_TYPES`, cycled, so a
conversation moves from a value to the circumstances it applies in to the
reason behind it rather than asking the same kind of thing three times.

A follow-up **may lean on the conversation** — *"And for an urgent one?"* — and
that is the point: a chatbot answering one has to carry the thread, which is a
real capability and one no single-turn question tests. Two consequences follow
from it. The phrasing gate is **not** applied to a follow-up, because not
standing alone is what it is for; and the verifier is shown the conversation
when it judges recoverability, because read alone *"And for an urgent one?"*
has no answer in any passage.

Only an accepted, **answerable** root is followed. A thread whose first turn
has no answer has nothing to follow on from — the chatbot was supposed to say
it did not know — and one whose root a gate refused would be a conversation
starting with a question nobody would ask. A thread stops at the first
follow-up a gate refuses: the refused one is stored as drop-rate evidence like
any other, but a third turn after a discarded second is a conversation with a
hole in it.

Each follow-up is another writer call and another verifier call, so a thread
multiplies what a topic costs. At the defaults a topic of twenty questions
costs about thirty-two.

Some questions are written to have **no answer in the corpus**, by perturbing a
verified fact just out of reach. These test whether a chatbot says it does not
know instead of inventing something, which is half of what this dataset is for.
`QUESTIONS_UNANSWERABLE_SHARE` sets how many are attempted, spread by position
rather than drawn at random, so a share of 0.25 is exactly one in four and is
the same one in four on a re-run. An unanswerable question is always planned
`easy` and from one passage: it is written by moving one fact out of reach, so
a second passage has nothing to do with it.

### The gates

Applied cheapest first, because each one that fires saves the cost of those
behind it. A question that fails one is stored with the gate's name rather than
dropped: the rate at which that happens is how the writer is judged, so it
belongs in the data and not in a log.

| Gate | Rejects a question that | Costs |
|---|---|---|
| `malformed` | is not a question, asks two things, carries no target answer when it claims one, is in the wrong language, or is one of its own facts handed back | nothing |
| `answer_too_short` | is scored against an answer below its form's floor | nothing |
| `answer_too_long` | answers past its form's ceiling — a value answered with a paragraph | nothing |
| `wrong_form` | answers in the wrong shape: a value describing an action, an explanation explaining nothing | nothing |
| `leaks_source` | quotes the title of the document its answer is in | nothing, or the round trip |
| `duplicate` | is a near twin of one already accepted | one index probe |
| `answerable_after_all` | was written to have no answer and turns out to have one | the same probe, or the round trip |
| `unanchored` | nobody could have asked without the passage in front of them | the round trip |
| `wrong_type` | is not the kind of question it was asked to be | the round trip |
| `not_recoverable` | cites evidence its own answer is not in | the round trip |

The length bounds are per form and measured rather than guessed: a floor of 15
on values refused 41% of the answers this corpus had accepted — `70%`, `2025`
and `Bafin` among them, which are the most unambiguously scoreable answers
there are. One line in `backend.env` changes any of them, and the share each
refuses is on the Questions page either way.

There is **no gate for "the question is its own fact rearranged"**, and that is
a finding rather than an omission. One was written, in two formulations, and
measured against 61 real rows: both refused questions like *"Wie hoch war die
Arbeitslosenquote im August 2025?"* → `6,4 Prozent`, which is as good as a
benchmark question gets. For a single atomic fact, a good question *is* the
fact minus its answer — that is what asking about a fact means. What separates
a good one from a bad one is whether the answer is determinate, and that is not
lexical either. `not_recoverable` already carries the judgement where it can be
made.

`not_recoverable` is the one no similarity measure makes. A second model is
shown **only the cited passages** and asked to answer; the question survives if
what comes back carries every number, name and date the target answer asserts,
and enough of what it is about. How much is enough depends on the form: a
`value` is compared whole, because every word of one is the answer, while a
`list` or an `explanation` is compared by overlap — `QUESTIONS_ANSWER_OVERLAP`
— because demanding that every lemma of prose survive a paraphrase refuses
answers the verifier plainly found. Numbers are always exact whatever the form,
so `4 hours` never passes for `48 hours`.

`unanchored`, `leaks_source` and `wrong_type` ride on that same call, for
nothing extra. Each is a judgement rather than a measurement, and no structural
check makes any of them — but a model already looking at the question and the
material can.

They are also the only gates here that are **opinions**, and an opinion needs
an independent holder. With `QUESTIONS_VERIFIER_MODEL` unset the writer marks
its own work, and one measured run rejected *"According to the ECB and NCAs,
who can conduct the due diligence check for an outsourcing arrangement?"* for
naming nothing — three of four rejections in that topic were false positives.
So the factory turns those three off when no second model is named, and logs
the verdict instead. Recoverability stays on regardless, because that one is
checkable against the passage rather than a matter of taste.

Three things about the round trip are load-bearing. The verifier is a
**different** model, because a model marking its own work recovers what it just
wrote and the gate then passes everything. The escape hatch is explicit — the
verifier answers whether the passage states it at all, not just what it says —
or the model confabulates rather than declining. And it sees only the cited
passages, never the corpus: what is being measured is the dataset, not a
retriever.

It catches what nothing else did. *"Ein Liquiditätsmanagementtool ist eine
einjährige Rückgabefrist"* survived an NLI model, an LLM judge and a structural
check, because it reads exactly like its passage. It does not survive being
asked, because recoverability is not similarity — and a paraphrase, a
decomposition and a resolved pronoun all survive it, which a similarity
threshold does not let them do.

### What a refit and a re-extraction do to them

Questions belong to their facts, not to a topic. A fit replaces every row in
`topics`, so refitting returns every topic to `new` and the questions are
written again — but the questions themselves survive, and selection skips the
facts an accepted question already rests on, so the second run writes only about
what the first did not reach. Without that skip the writer would be paid for
once per duplicate before the dedup gate could throw the duplicate away.

Re-extracting is the destructive one. `question_facts` cascades from `facts`
and a trigger deletes a question once its last citation is gone, so
`extract-rerun` over the corpus takes the questions with it. Re-extracting a
*single* document is quieter and worse: a cross-document question that loses one
of its two citations is not deleted, and its stored difficulty stops being true.
That is what `questions-reverify` exists to find — it re-reads the stored form
and bounds, so a question is never re-judged under a kind it was not written
as.

## Topic modelling

One model per language, fitted over the lemmas chunking stored for that
language's passages. A gensim *document* is one passage; the *corpus* is an
object that re-walks the database, because the fit reads it once per pass and
a generator would be spent after the first. Vectorisation is
`dictionary.doc2bow(lemmas)` weighted by tf-idf — a sparse vector over the
vocabulary the corpus itself defines. Nothing is pretrained and nothing is
embedded, which is what keeps the result industry-agnostic.

The model is non-negative matrix factorisation, not LDA. LDA is defined over
counts — words drawn from a multinomial — so a tf-idf weighted input
contradicts its own likelihood, and measured on this corpus feeding it one was
worse than counts (c_v 0.514 against 0.522). Factorisation carries no such
assumption, and weighting the input is what stops one dominant vocabulary
spreading across every topic. Measured at twelve topics, German: term overlap
between topics fell from 32% to 11%, coherence rose from 0.52 to 0.66. The
weights it returns are normalised, so a membership still reads as a share.

The corpus is streamed throughout: the vocabulary is built in batches of 500,
the tf-idf weighting is read off the vocabulary's own document frequencies and
applied lazily, each pass re-reads the rows, and scoring reads them once more.
Memory follows the batch and the vocabulary, not the corpus.

A passage is the document rather than a sentence, which was measured too: a
sentence averages 5.7 content tokens, too few to express the mixture of topics
the model is about, and 6% of them hold none of the vocabulary at all. German,
twelve topics: c_v 0.522 per passage against 0.396 per sentence.

### Seeing the model

Each fit also draws every language it fitted as a
[pyLDAvis](https://github.com/bmabey/pyLDAvis) figure and stores it in the
`export` bucket under `topics/<language>.html`. Read it on the Topics page,
at `GET /topics/visualisation/{language}`, or as a file with
`make topics-visualise`.

The figure is drawn during the fit rather than on demand, because it needs the
weight of every term in every topic and the database keeps only a topic's top
terms. A language modelled before a fit that draws has no figure until the next
one; the route answers 404 and the page says so.

The pages are self-contained: d3 and the LDAvis script are inlined, so nothing
is fetched when one is opened. Topics are numbered as `topics.topic_index`
numbers them, so topic 3 in the figure is topic 3 in the table beside it.

A fit is always every language and always the whole corpus. Gensim's `update()`
would allow incremental training, and incremental training is deliberately not
used: the vocabulary is fixed at construction, so a new document's novel terms
would be silently dropped, and the result would come to depend on ingestion
order — which is exactly what `TOPIC_RANDOM_STATE` exists to prevent.

## What is kept where

Parsing writes the converter's complete output to the `parsed` bucket, so
nothing it produced is ever lost. The database holds what a later stage reads
or a person queries:

| In Postgres | Why it is not left in the bucket |
|---|---|
| `passages.text` | The unit facts are drawn from and scored against |
| `passages.sentences` | What a fact cites, and what resolves the citation to a span |
| `passages.lemmas` | The vocabulary the topic model is fitted over |
| `passages.language` | Which spaCy pipeline reads it, and which topic model covers it |
| `passages.table_cells` | A serialised table loses its header flags, row and column positions and spans, and the cell reader has nothing to walk |
| `passages.bbox` | Highlighting a citation must not mean fetching and parsing a multi-megabyte document |
| `passages.doc_item_refs` | The only non-fuzzy way back to the converted document |
| `passages.section_path`, `block_type`, `page_from`, `page_to` | Read on every query that places or routes a passage |
| `documents.oversized` | Passages stored above the token budget, which the embedder would truncate. Expected to be 0 |

`language`, `sentences` and `lemmas` are written by chunking rather than by the
stages that read them, so one segmentation serves both and a topic fit reads a
column instead of re-tokenising the corpus.

The language is detected on the **passage**, not inherited from its document:
45 of this corpus's passages are German inside an English-labelled file, and a
document-wide label read every one of them with the wrong pipeline.

Everything else — figures, formulas as LaTeX, code blocks, per-line geometry,
the heading tree — stays in the bucket and is reachable through
`doc_item_refs`. Formulas, code and list items do reach the database as
`block_type`, because that is what routes them to an extractor.

Two deliberate gaps: figures become no passage of their own, which needs a
vision model to be worth anything, and key-value form regions are not modelled,
because no document in the corpus has any.

## Tests

Seven layers, each a directory and a marker. `make test` runs everything that
gates a merge; the two that do not are excluded from it.

| Directory | What it covers | Needs |
|---|---|---|
| `tests/static/` | The repository against itself: settings declared where they are read, the migration chain, the extensions the schema needs, the locks, the workflows, and pyright at zero | nothing |
| `tests/unit/` | One module at a time, no I/O | spaCy, for some |
| `tests/property/` | Invariants over generated input, with hypothesis | spaCy, for some |
| `tests/contract/` | The OpenAPI surface, the paths the frontend builds, and the refusals each route declares | a container |
| `tests/integration/` | The database, the object store and the HTTP surface, against the images compose runs | a container |
| `tests/e2e/` | One document through every stage in this process, with the converter and the model stood in for | a container |
| `tests/frontend/` | Each Streamlit page against a scripted backend | nothing |
| `tests/smoke/` | Both images built and looked inside, and the compose file resolved | a container engine |
| `tests/eval/` | How a real served model reads the golden passages, and whether the round-trip gate splits the golden questions | a served model |

The integration layers start a PostgreSQL and a SeaweedFS of their own through
testcontainers and skip, with a reason, where no container engine answers.
Nothing they do touches a running stack.

`tests/eval/` never gates: a model's answers move between versions and between
runs at the same temperature, so a threshold there would fail on somebody
else's Tuesday rather than on a regression. It prints its numbers. The one
thing it does assert is that the round-trip gate splits its golden questions
the right way round, which is not a measurement of the model's taste — it is
whether the gate is wired up at all, and a gate that accepts everything cannot
be told from no gate.

Two tests would need a 2.2 GB download to run and skip instead of taking one:
`tests/unit/questions/test_embedding.py` skips unless `EMBEDDING_MODEL` is
already in the Hugging Face cache, and the end-to-end pipeline stands the
embedder in for with a digest.

### Continuous integration

`.github/workflows/ci.yml` runs on every pull request and on every push to
`main`, as four jobs that together are the whole suite: `static`, `unit`,
`integration`, and `smoke`. A fifth, `gate`, waits for the rest and is the one
thing a branch protection rule needs to require.

`smoke` builds both images, so it is the slow one: it is also the only layer
that can see what an image contains, and a lock that does not install is not
worth finding out about after the merge.

`.github/workflows/nightly.yml` runs what is worth knowing but not worth
blocking on: advisories against both locks, every layer including the images,
and the model evaluation, which skips itself unless `LLM_MODEL` and
`LLM_BASE_URL` are set as repository variables.

The first run of any job installs the dependencies and caches the virtualenv
against `poetry.lock` and the Makefile. Later runs restore it.

torch arrives through docling and nothing here uses a GPU, so `pyproject.toml`
declares PyTorch's CPU index as an explicit source and names torch and
torchvision against it for Linux. Without that, the wheel PyPI serves on Linux
brings the whole CUDA runtime with it — eighteen packages and some three
gigabytes, downloaded on every cache miss and shipped in nothing. macOS keeps
PyPI's build, which has no CUDA variant to avoid.

## Layout

| Path | Contents |
|---|---|
| `backend/database/` | SQLAlchemy models, one module per table, plus the engine |
| `backend/database/migrations/` | Alembic: one revision per schema change |
| `backend/blob_store/` | Object storage clients, one module per bucket |
| `backend/nlp/` | The spaCy pipelines, and reading sentences, claims and vocabulary out of text |
| `backend/api/` | The HTTP surface the frontend and the orchestrator call |
| `backend/ingestion/` | Upload validation, hashing and storage |
| `backend/preprocessing/parsing/` | A stored file becomes a structured document |
| `backend/preprocessing/chunking/` | That document becomes passages, with their sentences and lemmas |
| `backend/extraction/` | Those passages become facts citing a sentence |
| `backend/topic_modelling/` | Each language becomes topics over its own vocabulary |
| `backend/question_generation/` | Each topic's facts become questions with known answers |
| `backend/stages/` | The queue, drain loop, command line and watch loop every stage shares |
| `backend/settings/` | Reading configuration out of the environment, and nowhere else |
| `tests/` | The test suite, one directory per layer; see Tests below |
| `.github/workflows/` | What CI runs, and when |
| `telemetry/` | Logging and OpenTelemetry configuration |
| `configs/filebeat/` | What the log shipper reads and where it puts it |
| `configs/grafana/` | The log datasource and dashboard, provisioned |
| `frontend/` | Streamlit application |
| `configs/env/` | The settings that are decisions rather than credentials, and so live in git |
| `configs/` | Service configuration and init scripts |

The frontend holds one address, `BACKEND_URL`, and no knowledge of the
database, the object store or the services behind the API.

No backend service imports another. Each owns a repository over the shared
models and passes plain dataclasses across its own boundaries; `database/`,
`blob_store/` and `nlp/` are the only packages that build a connection or load
a model, so a service can neither configure the infrastructure nor reach around
another service to it. Stages hand work to each other through their own status
column, and share only what is in `backend/stages/`.

A service package re-exports nothing. A caller names the submodule it wants —
`from extraction.repository import PassageQueue` — so it pays for that
submodule and no more. That is what keeps litellm, Docling, gensim and spaCy
out of the API process, which loads none of them.

Each stage splits its database access in two: a queue, which claims rows and
records outcomes, and a catalogue, which reads back what the stage produced.
The API imports only the catalogue for a read route, and only the queue for a
queue route.

They are independent modules, not independent services. They share one
PostgreSQL schema, one `Status` enum and one container image, and two stages
own different columns of the same `documents` row. That is the right shape at
this size; splitting them further would mean a schema and a migration history
each, and a contract between them that is not a table.

## Logs

One configuration, in `telemetry/`, and every process calls it before it does
anything else. Each record is rendered twice: as a line of text on stdout,
which is what `make logs` shows, and as one JSON object per line in a file,
which is what reaches Elasticsearch. Same record, same fields; the JSON
carries the ones a text line has no room for.

    api, 5 workers, streamlit  ──▶  logs volume  ──▶  filebeat  ──▶  elasticsearch  ──▶  grafana

Nothing is aggregated and nothing is dropped on the way. Every level from
`LOG_LEVEL` upwards is shipped, `DEBUG` included when it is set that low. An
exception is written whole: `error.type`, `error.message` and the entire
traceback in `error.stack_trace`, so a failure is readable in Grafana without
going back to the container. Every stage that records a failure against a row
also logs it with its traceback — the row's error column is one line for a
person reading the Documents page, not the whole story.

The field names are [ECS](https://www.elastic.co/guide/en/ecs/current/index.html).
That is the only reason none of this needs an index template of its own:
`log.level`, `service.name`, `log.logger` and `error.type` are names
Elasticsearch's own template already maps as keywords, so Grafana can group on
them out of the box. `trace.id` is on every line too, which is what ties a log
line to its span in Phoenix.

Only this project's processes are shipped. Postgres, SeaweedFS, Redis and
Elasticsearch itself keep the `json-file` driver and are read with `make logs`.
Collecting those too would mean reading the engine's own log store, which is in
a different place under Docker and Podman and, on macOS, inside a virtual
machine a bind mount cannot see; one more input in
`configs/filebeat/filebeat.yml` is all it takes once that path is known for a
given machine.

A worker writes to `{stage}-{container}.log` on a shared volume rather than to
one file per stage, because a scaled stage runs several containers over one
volume and two processes rotating one file take each other's lines with them.
Files rotate at 50 MB, three kept; the shipper has read a line long before it
is deleted. `LOG_DIR` is what turns the file on — it is set for the containers
and unset for every `make` target, so a host command logs to the terminal and
nowhere else.

Elasticsearch is the one Argilla already uses. That is a deliberate reuse
rather than a second node, and it is a shared heap: `ES_JAVA_OPTS` in
`configs/env/elasticsearch.env` is where to raise it if a long run makes either
slow.

```bash
make logs-shipper    # why nothing is arriving, when nothing is arriving
```

## Configuration

All configuration is environment variables, split by what the value is rather
than by which service reads it:

| File | In git | Holds |
|---|---|---|
| `configs/env/backend.env` | yes | How the pipeline behaves: the parsing, chunking, extraction, topic and question settings the api and the five workers read |
| `configs/env/elasticsearch.env` | yes | The Elasticsearch node's certificate paths, security flags and heap. One node serves both Argilla and the logs |
| `configs/env/seaweedfs-filer.env` | yes | Which metadata store the filer uses |
| `.env` | no | Credentials, ports, and the addresses a host reaches a service at. `.env.example` lists it |

compose hands each file to the services that need it with `env_file`, and the
Makefile sources `backend.env` and `.env` for the host commands, so one value
reaches both. A variable given on the command line beats both:

```bash
make topics-discover TOPIC_PASSES=20
```

The values most likely to need changing:

| Variable | Where | Default | Purpose |
|---|---|---|---|
| `MAX_FILE_SIZE_MB` | `backend.env` | 100 | Largest upload accepted |
| `ALLOWED_MIME_TYPES` | `backend.env` | `application/pdf` | Types with a parser behind them |
| `EMBEDDING_MODEL` | `backend.env` | `intfloat/multilingual-e5-large` | The one embedding model, and the only tokenizer in the project |
| `EMBEDDING_MAX_TOKENS` | `backend.env` | 512 | That model's context window, and so the longest passage |
| `NLP_MODELS` | `backend.env` | `de:de_core_news_md,en:en_core_web_md` | The spaCy pipeline per language, and the languages the detector may answer with. Must be in the image. Medium, not small: the small German model does not tag a modal as a finite verb |
| `LLM_MODEL` | `.env` | `ollama_chat/gemma4:31b` | LiteLLM model id; the prefix picks the provider |
| `LLM_BASE_URL` | `.env` | — | Where that model is served |
| `QUESTIONS_VERIFIER_MODEL` | `.env` | unset | The second model, which checks that a question's answer is in the passages it cites. Unset means the writer marks its own work, which it will always pass; the worker warns on every start |
| `QUESTIONS_PER_TOPIC` | `backend.env` | 20 | How many questions to aim for per topic, and so how many of its passages are asked about. This times the topic count is what a full run costs. Not below the number of kinds with a weight, or a topic never sees some of them |
| `QUESTIONS_FACT_SAMPLE` | `backend.env` | 6 | How many of a topic's facts are offered per call, divided between the passages the sample holds. The writer picks which of them one question needs |
| `QUESTIONS_TYPE_MIX` | `backend.env` | eleven kinds | Which kinds of question are written and in what proportion, as `kind:weight`. A weight of 0, or a name left out, is never written |
| `QUESTIONS_DIFFICULTY_MIX` | `backend.env` | `easy:2,medium:2,hard:1` | Which bands the plan aims for, as `band:weight`. A request for a shape of sample; the band itself stays derived |
| `QUESTIONS_ANSWER_CHARS` | `backend.env` | `value:1:80,list:3:300,explanation:20:600` | The shortest and longest target answer per form, as `form:min:max` |
| `QUESTIONS_ANSWER_OVERLAP` | `backend.env` | 0.6 | How much of a list or an explanation has to come back for the verifier to have recovered it. Numbers are always exact |
| `QUESTIONS_UNANSWERABLE_SHARE` | `backend.env` | 0.25 | What share of questions are written to have no answer in the corpus |
| `QUESTIONS_FOLLOWUP_SHARE` | `backend.env` | 0.3 | What share of accepted questions get a follow-up thread |
| `QUESTIONS_MAX_FOLLOWUPS` | `backend.env` | 2 | How far a thread may run past its root |
| `QUESTIONS_FOLLOWUP_TYPES` | `backend.env` | `condition,reason,comparison` | The kinds the turns of a thread take, cycled |
| `QUESTIONS_LONG_ANSWER_CHARS` | `backend.env` | 60 | Where an answer starts counting towards the difficulty band. Not a gate |
| `QUESTIONS_DUPLICATE_COSINE` | `backend.env` | 0.93 | How alike two questions must be before the later one is thrown away |
| `LOG_LEVEL` | `.env` | `INFO` | Log level for every service, the frontend included. Everything at or above it reaches Grafana |
| `OTEL_CONTAINER_ENDPOINT` | `.env` | `http://phoenix:4317` | Trace collector |

`MAX_FILE_SIZE_MB` must be kept in step with `server.maxUploadSize` in
`frontend/.streamlit/config.toml`, or Streamlit rejects the file before the API
sees it.

The frontend comes in light and dark. That file gives Streamlit a palette
under `[theme.light]` and another under `[theme.dark]`; it starts from the
browser's `prefers-color-scheme` and the toolbar menu, top right, switches
between them per page. Put a colour in `[theme]` itself, or set `theme.base`,
and it applies to both themes — which is what pinned the app to light before.

`frontend/styles.css` reads its own palette off `light-dark()`, which resolves
against the `color-scheme` Streamlit sets on the app container from the theme
it actually settled on. So the custom styling follows the chrome whichever way
the chrome was decided — a `prefers-color-scheme` media query would have got
the menu wrong, staying light while everything around it went dark. Only the
five accent hues are written twice; the neutrals are mixed from `currentColor`
and each border and wash from its own hue, so they need no second value.

The topic map stays on white in either theme. It is a pyLDAvis document inside
an iframe, so nothing outside it can restyle it; it is framed and given a
background of its own so it reads as a figure printed on white.

Changing `NLP_MODELS` changes what a fact is, in the same way changing the
prompt does. Both are recorded on every fact — `spacy_model`, `spacy_version`,
`extraction_model`, `prompt_version` — so two generations of the dataset can be
told apart.

## Schema changes

Alembic owns the schema. The models say what the tables should be; a revision
under `backend/database/migrations/versions/` says how to get an existing
database there.

```bash
make migration m="add the dropped counts"  # write a revision from the models
make schema                                # apply everything outstanding
make schema-status                         # where the database is
make schema-down                           # take the newest revision back off
make schema-reset                          # drop every table and rebuild from
                                           #   the revisions. Irreversible
make schema-stamp                          # adopt a database that already
                                           #   holds the tables
```

Read the generated revision before applying it. Autogenerate compares tables,
columns, indexes and constraints; it does not see a trigger, a data backfill or
anything that has to happen in a particular order. Both such cases in this
history are written out by hand: the one trigger, in the initial revision, and
the deletion of every fact in the revision that changed what a citation is.
