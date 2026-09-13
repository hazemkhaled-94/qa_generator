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

**Status:** ingestion, parsing, chunking, fact extraction and topic modelling
are implemented. Question generation, quality assurance and the evaluation
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
| Python | 3.12–3.14 |
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
make check           # run the self-checks; no database, no served model
make lint            # ruff check and format check
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
```

Any of those five verbs narrows to a single item with `SHA` or `PASSAGE`,
which is the same operation against fewer rows:

```bash
make parse-start SHA=<sha256>     # queue one document for parsing
make chunk-rerun SHA=<sha256>     # rebuild one document's passages
make extract-start SHA=<sha256>   # queue every passage of one document
make extract-retry PASSAGE=<id>   # return one failed passage to the queue
make extract-status SHA=<sha256>  # that document's passages by extract state
```

Parsing and chunking narrow to a document, extraction to a document or a
passage, topic modelling to neither — a fit is all-or-nothing over one
vocabulary, so there is no single topic to start, stop or refit.

### Replaying a stage without redoing it

Two operations re-derive what a stage computed, over rows already stored,
without the expensive part:

| | |
|---|---|
| `make extract-revalidate` | Judges every stored fact again. The model is not called and no statement changes — only what the checks read off one. |
| `make chunk-revocabulary` | Reads every stored passage's language and vocabulary again. Only `passages.language` and `passages.lemmas` change. |

Both take `SHA` and `extract-revalidate` takes `PASSAGE` too, like the five
queue verbs.

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

Each verb comes twice. The routes above act on the whole queue; the same
verb under `/{scope}/{value}` acts on one item, which is what the frontend's
per-item controls call:

| Stage | Narrows to | Example |
|---|---|---|
| Parsing | `document` | `POST /parsing/document/{sha256}/start` |
| Chunking | `document` | `POST /chunking/document/{sha256}/rerun` |
| Extraction | `document`, `passage` | `POST /extraction/passage/{id}/retry` |
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
| `GET /health` | That the process is up, for the container healthcheck |
| `GET /status` | Every component behind the API, and what each service holds |

`/documents`, `/passages` and `/facts` take `q` for a case-insensitive
substring and `limit`/`offset` to page; the last two also take `document` to
narrow to one digest. The frontend renders each as a page of its own and
filters nothing itself.

A document moves through the stages by its status columns, one request at a
time: ingestion leaves `parse_status = 'new'`, starting parsing moves it to
`'pending'`, parsing sets it to `'parsed'`, and chunking selects on its own
`chunk_status` — never on `parse_status`, because a stage knows of no other
stage. Extraction does the same over the passage's `extract_status`. Nothing
schedules the stages; a person or the orchestrator does. Topic modelling stands
outside that chain: it reads every passage whatever stage its document has
reached, and runs when a request row in `topics` asks it to.

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
| `backend/stages/` | The queue, drain loop, command line and watch loop every stage shares |
| `backend/settings/` | Reading configuration out of the environment, and nowhere else |
| `backend/checks.py` | Self-checks for the logic that would fail silently |
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

    api, 4 workers, streamlit  ──▶  logs volume  ──▶  filebeat  ──▶  elasticsearch  ──▶  grafana

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
| `configs/env/backend.env` | yes | How the pipeline behaves: the parsing, chunking, extraction and topic settings the api and the four workers read |
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
| `LOG_LEVEL` | `.env` | `INFO` | Log level for every service, the frontend included. Everything at or above it reaches Grafana |
| `OTEL_CONTAINER_ENDPOINT` | `.env` | `http://phoenix:4317` | Trace collector |

`MAX_FILE_SIZE_MB` must be kept in step with `server.maxUploadSize` in
`frontend/.streamlit/config.toml`, or Streamlit rejects the file before the API
sees it.

The frontend follows the dark mode of whatever is showing it. That file gives
Streamlit a palette under `[theme.light]` and another under `[theme.dark]`, and
`frontend/styles.css` branches on the same `prefers-color-scheme` the browser
reports, so the chrome and the custom styling cannot disagree about which theme
is up. There is deliberately no in-app switch: a second way to choose would be
a second source of truth. Put a colour in `[theme]` itself, or set `theme.base`,
and it applies to both themes — which is what pinned the app to light before.
The topic map stays on white in either theme; it is a pyLDAvis document inside
an iframe, so nothing outside it can restyle it.

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
