# API

The HTTP surface the frontend and the orchestrator call, and the only address
either of them holds.

It does two things and deliberately not a third. It **reads back** what the
stages produced — documents, passages, facts, topics, questions — and it
**moves rows on and off a stage's queue**. It never runs a stage. There is no
`run` route, and that is not an omission: a conversion running inside the
process that serves JSON held it for sixteen minutes at a stretch.

The work happens in the stage's worker container, off the queue this API
writes to. See [`backend/stages/`](../stages/README.md) for the queue and the
worker that drains it.

## What it does

### The shape of the process

One FastAPI application, assembled in [`main.py`](main.py): tracing is
attached to the instance, the refusal handler is installed, and every router
in [`routes/`](routes/) is included. Wiring — the repositories and buckets a
route needs — is built once in [`dependencies.py`](dependencies.py) and
injected; a route module builds nothing itself.

The process **loads no model, no converter and no inference library**. It
imports a stage's catalogue for a read route and a stage's queue for a queue
route, and nothing else of that stage. Because a service package re-exports
nothing, importing `extraction.repository` does not drag litellm, Docling,
gensim or spaCy into this process. `tests/static/test_api_stays_light.py`
pins that.

### The two kinds of route

**Read routes** are a noun: `/documents`, `/passages`, `/facts`, `/topics`,
`/questions`. Each answers a page of rows with the filters that page offers.

**Queue routes** are a verb under a stage's name: `/parsing`, `/chunking`,
`/extraction`, `/questions`, `/topics`. Four of the five are built by one
factory — [`stage_router`](routes/stage.py) — so a verb cannot mean two things
depending on which stage answered it.

`/questions` is the one path that is both. Every other stage is a verb with
its product under a different noun — `/extraction` produces `/facts` — but a
question is both what generation produces and what it is called, so the queue
routes and the read routes share a router. The queue routes are declared
first, which is what keeps `/questions/status` from being read as a question
with the id `status`.

### The queue surface every stage shares

```
GET  /{stage}/status                      how much work is waiting
POST /{stage}/{action}                    start | stop | retry | rerun
GET  /{stage}/{scope}/{value}/status      the same, for one item
POST /{stage}/{scope}/{value}/{action}    the same verb, against fewer rows
```

Every verb comes twice. The narrowed form is a `WHERE` on the stage's own
table and nothing more, so a narrowed verb and a whole-queue one cannot
disagree about what they do. It is what the frontend's per-item controls
call.

| Verb | Moves |
|---|---|
| `start` | `new` → `pending`, which is the only status a worker claims |
| `stop` | `pending` → `new`, taking back whatever has not begun |
| `retry` | a failed row back to `pending`, clearing the reason |
| `rerun` | every row to `pending`, finished ones included, skipping what a worker holds right now |

All four answer **202** and return at once. None of them does the work.

| Stage | Narrows to | Example |
|---|---|---|
| `parsing` | `document` | `POST /parsing/document/{sha256}/start` |
| `chunking` | `document` | `POST /chunking/document/{sha256}/rerun` |
| `extraction` | `document`, `passage` | `POST /extraction/passage/{id}/retry` |
| `questions` | `topic` | `POST /questions/topic/{id}/rerun` |
| `topics` | — | a fit is all-or-nothing over one vocabulary |

Topic modelling has no `start` and no `rerun`, and its routes are written out
in [`routes/topics.py`](routes/topics.py) rather than built by the factory.
Asking is what creates the work: `POST /topics/discover` writes a request row,
and `/topics/stop` and `/topics/retry` are the only other verbs that mean
anything over one.

## The full surface

**Documents** — served by [ingestion](../ingestion/README.md).

| Route | Answers |
|---|---|
| `POST /documents` | Upload one file, and what became of it |
| `GET /documents` | The corpus, with each document's stage statuses |
| `GET /documents/names` | Digests and filenames, for a picker |
| `GET /documents/{sha256}/file` | The stored bytes back |
| `DELETE /documents/{sha256}` | The document and everything derived from it |
| `DELETE /documents/{sha256}/derived` | Only its passages and facts; chunking returns to `new` |

**Passages** — served by [chunking](../preprocessing/chunking/README.md).

| Route | Answers |
|---|---|
| `GET /passages` | Passages, with their heading trail, pages and tables |
| `GET /passages/types` | The block types the corpus actually holds |
| `GET /passages/{id}` | One passage in full, its numbered sentences and cell grids |

**Facts** — served by [extraction](../extraction/README.md).

| Route | Answers |
|---|---|
| `GET /facts` | Facts, accepted and rejected alike, with what they cite |
| `GET /facts/quality` | How many hold up, how far the passages were decomposed, and why the rest were rejected |
| `GET /facts/{fact_id}/passages` | The passages one fact rests on, and the span it cited in each |

**Topics** — served by [topic modelling](../topic_modelling/README.md).

| Route | Answers |
|---|---|
| `GET /topics` | Every topic, its terms, and how much of the corpus it holds |
| `PATCH /topics/{id}` | Name a topic, or take it out of coverage reporting |
| `GET /topics/fit` | When the topics were fitted, over what, and whether they still describe the corpus |
| `GET /topics/visualisation/{language}` | One language's model as a self-contained pyLDAvis page |
| `DELETE /topics` | Every topic and membership |

**Questions** — served by
[question generation](../question_generation/README.md).

| Route | Answers |
|---|---|
| `GET /questions` | Questions, accepted and rejected alike, with what each cites |
| `GET /questions/{id}` | One question with the facts it was written from |
| `GET /questions/plan` | What a topic is planned to be asked, before anything is written |
| `GET /questions/quality` | How many hold up, which gate stopped the rest, and how much of the subject matter is covered |
| `PATCH /questions/{id}` | Accept or reject one question |

**Settings** — served by [`backend/settings/`](../settings/README.md).

| Route | Answers |
|---|---|
| `GET /settings/{service}` | Each setting's value, what the files say it would be, its type, its bounds, and whether somebody changed it |
| `PATCH /settings/{service}` | Change it. A null value returns one setting to what the files say |

**The platform itself.**

| Route | Answers |
|---|---|
| `GET /health` | That the process is up, for the container healthcheck |
| `GET /status` | Every component behind the API, and what each service holds |

`GET /docs` is FastAPI's own OpenAPI page. Nothing else is served.

### Filtering and paging

`/documents`, `/passages`, `/facts` and `/questions` all take `q` for a
case-insensitive substring and `limit`/`offset` to page. The last three also
take `document` to narrow to one digest, and `/questions` takes `topic` as
well. Each takes the filters its page offers — `parse_status` on
`/documents`, `block_type` on `/passages`, `kind` and `method` on `/facts`,
and six on `/questions`.

`/topics` takes none and returns every topic at once, because one fit produces
a list a person can read.

The frontend renders each of these as a page and filters nothing itself.
`tests/static/test_api_vocabularies.py` checks the filters the API offers
against the values the database can actually hold.

## Refusals

Every deliberate refusal answers with a stable `code` beside its message, so
a caller branches on the code rather than on English:

```json
{ "code": "unknown_document", "detail": "No document with that digest." }
```

| Code | Status | Raised when |
|---|---|---|
| `invalid_digest` | 400 | The path segment is not a SHA-256 |
| `invalid_language` | 400 | No such language was fitted |
| `invalid_value` | 400 | A narrowed verb was given a value its column cannot hold |
| `unknown_document` | 404 | No document with that digest |
| `unknown_passage` | 404 | No passage with that id |
| `unknown_question` | 404 | No question with that id |
| `unknown_topic` | 404 | No topic with that id |
| `unknown_scope` | 404 | A stage was narrowed to a scope it does not accept |
| `unknown_setting` | 404 | A service has no setting by that name |

Two things keep their own shape. FastAPI's 422 for a malformed query is left
alone, and **a stage's own failures are not answered here at all** — they are
recorded against the row and read back through `/{stage}/status`. A document
that failed to parse is a 200 with a reason in it, not a 500.

Each refusal is logged once, in the handler rather than at each `raise`, at
**warning** — a 404 for an unknown id is the API working. That is what makes
the refusal a caller branched on also a refusal a dashboard can count.

## Tools, and where each is used

| Tool | Where | Why this one |
|---|---|---|
| **FastAPI** | [`main.py`](main.py), [`routes/`](routes/) | Generates the OpenAPI document the frontend and the contract test both read, from the annotations the routes already carry |
| **Pydantic** | route signatures | Validates a query before a repository sees it, and is what answers 422 |
| **SQLAlchemy** | through each stage's catalogue | The API writes no SQL of its own; it calls the stage that owns the table |
| **OpenTelemetry** | [`main.py`](main.py) | Attached to the instance after it exists, so a caller's trace continues here rather than a new one beginning |

## Configuration

The API reads the same [`configs/env/backend.env`](../../configs/env/backend.env)
the workers do, because it serves the same settings they read. It holds no
configuration of its own.

Its own address and pool sizes are the deployment's:
`DATABASE_POOL_SIZE` and `DATABASE_POOL_OVERFLOW` are served read-only and
marked `fixed`, because they are read before a service could ask a database
for anything. See [docs/configuration.md](../../docs/configuration.md).

## Tests

```sh
poetry run pytest tests/contract tests/integration/api
poetry run pytest tests/unit/api tests/static/test_api_stays_light.py
```

| File | Covers |
|---|---|
| [`tests/contract/test_openapi.py`](../../tests/contract/test_openapi.py) | The published surface against [`openapi.json`](../../tests/contract/openapi.json): every path, its parameters and the refusals it declares |
| [`tests/unit/api/test_errors.py`](../../tests/unit/api/test_errors.py) | The shape every deliberate refusal takes |
| [`tests/integration/api/test_stages.py`](../../tests/integration/api/test_stages.py) | The queue surface every stage shares, over HTTP |
| [`tests/integration/api/test_catalogue.py`](../../tests/integration/api/test_catalogue.py) | Reading back what the stages produced, and the platform's own routes |
| [`tests/integration/api/test_documents.py`](../../tests/integration/api/test_documents.py), [`test_facts.py`](../../tests/integration/api/test_facts.py), [`test_questions.py`](../../tests/integration/api/test_questions.py), [`test_topics.py`](../../tests/integration/api/test_topics.py) | Each noun's routes against the database that answers them |
| [`tests/integration/api/test_settings.py`](../../tests/integration/api/test_settings.py) | Reading and changing what a service is configured to do, over HTTP |
| [`tests/static/test_api_stays_light.py`](../../tests/static/test_api_stays_light.py) | That this process loads no model, no converter and no inference library |
| [`tests/static/test_api_vocabularies.py`](../../tests/static/test_api_vocabularies.py) | The filters offered against the values the database holds |

`tests/contract/openapi.json` is a committed fixture. A route added without
updating it fails the contract test, which is the point: the frontend and the
orchestrator both build paths against this surface.

## Known edges

Things that are true, are not bugs, and have surprised somebody.

- **A queue verb answering 202 has done nothing yet.** It moved rows between
  statuses. The worker picks them up on its next poll, and `/status` is the
  only thing that says whether it has.
- **`POST /{stage}/{action}` accepts any of the four verbs as a path
  segment.** An unknown one is FastAPI's own 422 against the `Action`
  literal, not an `ApiError` with a code.
- **`rerun` skips what a worker holds right now.** It is not a way to
  interrupt a running conversion; it queues everything a worker is not
  already in the middle of.
- **`GET /topics/visualisation/{language}` 404s for a language fitted before
  the figure was drawn.** The figure is written during a fit, not on demand,
  because it needs the weight of every term in every topic and the database
  keeps only a topic's top terms.
- **The API can be reached while no worker exists.** A stage added since the
  stack came up has no container until `make up` creates one: the queue
  fills, `/status` reports it, and nothing drains it — which reads exactly
  like a broken worker rather than an absent one.
