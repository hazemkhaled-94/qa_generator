# API

The HTTP surface the frontend and the orchestrator call, and the only address
either of them holds.

It **reads back** what the stages produced and **moves rows on and off a
stage's queue**. It never runs a stage: there is no `run` route, because a
conversion running inside the process that serves JSON held one request for
sixteen minutes. The work happens in the stage's worker container, off the
queue this API writes to — see [`backend/stages/`](../stages/README.md).

## What it does

One FastAPI application, assembled in [`main.py`](main.py): tracing is
attached to the instance, the refusal handler is installed, and every router
in [`routes/`](routes/) is included. Wiring is built once in
[`dependencies.py`](dependencies.py) and injected; a route module builds
nothing itself.

The process **loads no model, no converter and no inference library**. It
imports a stage's catalogue for a read route and a stage's queue for a queue
route, and nothing else of that stage. Because a service package re-exports
nothing, importing `extraction.repository` does not drag litellm, Docling,
gensim or spaCy into this process.
`tests/static/test_api_stays_light.py` pins that.

### The two kinds of route

**Read routes** are a noun: `/documents`, `/passages`, `/facts`, `/topics`,
`/questions`. **Queue routes** are a verb under a stage's name: `/parsing`,
`/chunking`, `/extraction`, `/questions`, `/topics`, `/assessment`. Five of
the six are built by one factory — [`stage_router`](routes/stage.py) — so a
verb cannot mean two things depending on which stage answered it.

`/questions` and `/assessment` are the paths that are both, because in each
case the stage and what it produces share the word. The queue routes are
declared first, which keeps `/questions/status` from being read as a question
with the id `status`.

### The queue surface every stage shares

```
GET  /{stage}/status                      how much work is waiting
POST /{stage}/{action}                    start | stop | retry | rerun
GET  /{stage}/{scope}/{value}/status      the same, for one item
POST /{stage}/{scope}/{value}/{action}    the same verb, against fewer rows
```

The narrowed form is a `WHERE` on the stage's own table and nothing more, so
a narrowed verb and a whole-queue one cannot disagree.

| Verb | Moves |
|---|---|
| `start` | `new` → `pending`, the only status a worker claims |
| `stop` | `pending` → `new` |
| `retry` | a failed row back to `pending`, clearing the reason |
| `rerun` | every row to `pending`, skipping what a worker holds right now |

All four answer **202** and return at once. None does the work.

There is a fifth verb on the command line and deliberately not here:
`reclaim` moves an `in_progress` row back to `pending` without waiting out
its lease, which is safe only when a person knows the worker holding it is
gone. See [`backend/stages/`](../stages/README.md).

| Stage | Narrows to | Example |
|---|---|---|
| `parsing` | `document` | `POST /parsing/document/{sha256}/start` |
| `chunking` | `document` | `POST /chunking/document/{sha256}/rerun` |
| `extraction` | `document`, `passage` | `POST /extraction/passage/{id}/retry` |
| `questions` | `topic` | `POST /questions/topic/{id}/rerun` |
| `topics` | — | a fit is all-or-nothing over one vocabulary |
| `assessment` | `kind` | `POST /assessment/kind/question/start` |

Topic modelling's routes are written out in
[`routes/topics.py`](routes/topics.py) rather than built by the factory:
`POST /topics/discover` writes a request row, and `/topics/stop` and
`/topics/retry` are the only other verbs that mean anything.

## The full surface

**Documents** — served by [ingestion](../ingestion/README.md).

| Route | Answers |
|---|---|
| `POST /documents` | Upload one file, and what became of it |
| `GET /documents` | The corpus, with each document's stage statuses |
| `GET /documents/names` | Digests and filenames, for a picker |
| `GET /documents/{sha256}/file` | The stored bytes back |
| `DELETE /documents/{sha256}` | The document and everything derived from it |
| `DELETE /documents/{sha256}/derived` | Only its passages and facts |

**Passages** — served by [chunking](../preprocessing/chunking/README.md).

| Route | Answers |
|---|---|
| `GET /passages` | Passages, with their heading trail, pages and tables |
| `GET /passages/types` | The block types the corpus holds |
| `GET /passages/{id}` | One passage in full, its numbered sentences and cell grids |

**Facts** — served by [extraction](../extraction/README.md).

| Route | Answers |
|---|---|
| `GET /facts` | Facts, accepted and rejected alike, with what they cite |
| `GET /facts/quality` | How many hold up, and why the rest were rejected |
| `GET /facts/{fact_id}/passages` | The passages one fact rests on, and the span it cited |

**Topics** — served by [topic modelling](../topic_modelling/README.md).

| Route | Answers |
|---|---|
| `GET /topics` | Every topic, its terms, and how much of the corpus it holds |
| `PATCH /topics/{id}` | Name a topic, or take it out of coverage reporting |
| `GET /topics/fit` | When the topics were fitted, over what, and whether they still describe the corpus |
| `GET /topics/visualisation/{language}` | One language's model as a pyLDAvis page |
| `DELETE /topics` | Every topic and membership |

**Questions** — served by
[question generation](../question_generation/README.md).

| Route | Answers |
|---|---|
| `GET /questions` | Questions, accepted and rejected alike |
| `GET /questions/{id}` | One question with the facts it was written from |
| `GET /questions/plan` | What a topic is planned to be asked |
| `GET /questions/quality` | How many hold up, which gate stopped the rest, and coverage |
| `GET /questions/export` | The questions a filter selects, as an `.xlsx`. No default scope |
| `PATCH /questions/{id}` | Accept or reject one question |

**Assessment** — served by [the evaluation phase](../assessment/README.md).

| Route | Answers |
|---|---|
| `GET /assessment` | What an LLM judge made of each fact, topic and question |
| `GET /assessment/plan` | Whether the phase is on, who judges, and what each kind is asked |
| `GET /assessment/quality` | How much was judged, how each metric did, and where the judge and the pipeline disagree |

There is no PATCH, deliberately. A judgement is a record of what a model
said; a person who disagrees has `reviewed_verdict`, through
[`review/`](../../review/README.md) or the Facts and Questions pages.

**Prompts** — served by [`backend/stages/`](../stages/README.md).

| Route | Answers |
|---|---|
| `GET /prompts` | Every recorded prompt, narrowed by `service`, `version` and `name` |

`facts.prompt_version` and `questions.prompt_version` name a version, and
until this table nothing resolved one: the span carrying the prompt belongs
to a Phoenix project with a retention of its own, and the row outlives it.

Read-only, and there is no route that writes one. A prompt is changed in the
source and recorded by the stage that sends it. It serves the table rather
than asking the code, which is what keeps litellm out of this process. The
text comes with the listing rather than behind a second call.

**Settings** — served by [`backend/settings/`](../settings/README.md).

| Route | Answers |
|---|---|
| `GET /settings/{service}` | Each setting's value, what the files say, its type, its bounds, and whether somebody changed it |
| `PATCH /settings/{service}` | Change it. A null value returns one setting to the file |

**The platform.**

| Route | Answers |
|---|---|
| `GET /health` | That the process is up, for the container healthcheck |
| `GET /status` | Every component behind the API, and what each service holds |
| `GET /services` | Every container, whether it is listening, and where to open it |

`/status` reports what the pipeline **holds**, counted out of the database;
`/services` reports what is **up**. A page wanting both asks twice.

What `/services` proves is that something accepted a TCP connection on the
port — one code path for a web UI, a database and a broker, with no auth and
no TLS. The alternative was nine health protocols, nine sets of credentials,
and a page reporting Argilla as failed because it answered 401. A worker
serves no port and is reported as neither up nor down. `SERVICE_URLS` is
where the links come from; a service nobody published gets no link.

`GET /docs` is FastAPI's own OpenAPI page. Nothing else is served.

### How one artefact was produced

| Route | Answers |
|---|---|
| `GET /lineage/{kind}/{id}` | Everything one document, passage, fact, topic or question was produced from, in pipeline order |

The one route that crosses every service, and the only one whose query
lives here rather than in a stage's repository — [`lineage.py`](lineage.py)
reads the same models the service repositories read, and the api is
already the only layer above all seven.

It walks **upwards**: a question to its facts, a fact to its passages, a
passage to its document. What came of an artefact is the listing each page
already answers with a filter.

Each step is numbered by the stage that produced it, carries that stage's
verdict and what the judge made of it, and links each artefact to its own
span. A step is capped at twenty artefacts and reports its own total,
because one topic in a real corpus holds hundreds of passages.

For a question it also reports `gates_ran` — every gate that read it, at
its fixed position. `gates_recorded` is false for a question written
before that column existed.

### Filtering and paging

`/documents`, `/passages`, `/facts` and `/questions` all take `q` for a
case-insensitive substring and `limit`/`offset` to page. The last three also
take `document`, and `/questions` takes `topic` as well. Each takes the
filters its page offers — `parse_status`, `block_type`, `kind` and `method`,
and on `/questions` every column a question is classified by.

Each is spelled as a `Literal` so the OpenAPI document lists its values and
the frontend's pickers cannot drift.
`tests/static/test_api_vocabularies.py` keeps those two copies equal.

`/topics` takes none and returns every topic at once.

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

FastAPI's 422 for a malformed query is left alone, and **a stage's own
failures are not answered here at all** — they are recorded against the row
and read back through `/{stage}/status`.

Each refusal is logged once, in the handler rather than at each `raise`, at
**warning**.

## Configuration

The API reads the same
[`configs/env/backend.env`](../../configs/env/backend.env) the workers do and
holds no configuration of its own. Its address and pool sizes are the
deployment's: `DATABASE_POOL_SIZE` and `DATABASE_POOL_OVERFLOW` are served
read-only and marked `fixed`. See
[docs/configuration.md](../../docs/configuration.md).

## Tests

```sh
poetry run pytest tests/contract tests/integration/api
poetry run pytest tests/unit/api tests/static/test_api_stays_light.py
```

| File | Covers |
|---|---|
| [`tests/contract/test_openapi.py`](../../tests/contract/test_openapi.py) | The published surface against [`openapi.json`](../../tests/contract/openapi.json) |
| [`tests/unit/api/test_errors.py`](../../tests/unit/api/test_errors.py) | The shape every deliberate refusal takes |
| [`tests/integration/api/test_stages.py`](../../tests/integration/api/test_stages.py) | The queue surface every stage shares, over HTTP |
| [`tests/integration/api/test_catalogue.py`](../../tests/integration/api/test_catalogue.py) | Reading back what the stages produced, and the platform's routes |
| [`tests/integration/api/test_documents.py`](../../tests/integration/api/test_documents.py), [`test_facts.py`](../../tests/integration/api/test_facts.py), [`test_questions.py`](../../tests/integration/api/test_questions.py), [`test_topics.py`](../../tests/integration/api/test_topics.py) | Each noun's routes against the database that answers them |
| [`tests/integration/api/test_settings.py`](../../tests/integration/api/test_settings.py) | Reading and changing what a service is configured to do |
| [`tests/static/test_api_stays_light.py`](../../tests/static/test_api_stays_light.py) | That this process loads no model, converter or inference library |
| [`tests/static/test_api_vocabularies.py`](../../tests/static/test_api_vocabularies.py) | The filters offered against the values the database holds |

`tests/contract/openapi.json` is a committed fixture: a route added without
updating it fails the contract test.

## Limits

- **A queue verb answering 202 has done nothing yet.** It moved rows between
  statuses; `/status` is the only thing that says whether a worker picked
  them up.
- **`POST /{stage}/{action}` accepts any of the four verbs as a path
  segment.** An unknown one is FastAPI's own 422, not an `ApiError`.
- **`rerun` skips what a worker holds right now.** It is not a way to
  interrupt a running conversion.
- **`GET /topics/visualisation/{language}` 404s for a language fitted before
  the figure was drawn.** The figure is written during a fit, not on demand.
- **The API can be reached while no worker exists.** A stage added since the
  stack came up has no container until `make up` creates one.
