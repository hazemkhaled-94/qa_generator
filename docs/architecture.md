# Architecture

Four graphs: what a document becomes, what a worker does with one row, what a
module may import, and what runs in a container.

## 1. What a document becomes

Six packages, in the order a document moves. Each writes its own tables and
reads the previous one's. A seventh, `assessment`, runs after all of them and
writes nothing any of them reads.

```mermaid
flowchart TD
    upload([Upload]) --> ing[ingestion]
    ing -->|documents row + object| par[parsing]
    par -->|parsed document| chu[chunking]
    chu -->|passages, sentences, lemmas| ext[extraction]
    ext -->|facts, each citing a sentence| top[topic_modelling]
    top -->|topics, passage memberships| qgn[question_generation]
    qgn --> out([questions with known answers])
    qgn -->|every fact, topic and question| ass[assessment]
    ass --> opinion([a judgement beside each verdict])

    ext -.calls.-> llm{{served model}}
    top -.calls.-> llm
    qgn -.calls.-> llm
    qgn -.verifies with.-> vfy{{verifier model}}
    ass -.judges with.-> jdg{{judge model}}

    classDef model fill:#fff3cd,stroke:#b8860b
    class llm,vfy,jdg model
```

| Stage | Claims | Reads | Writes | Done value |
|---|---|---|---|---|
| `parsing` | `documents.parse_status` | the stored object | parsed document, `parsed` bucket | `parsed` |
| `chunking` | `documents.chunk_status` | the parsed document | `passages` | `chunked` |
| `extraction` | `passages.extract_status` | one passage | `facts`, `fact_passages` | `extracted` |
| `topic_modelling` | `topics.status` | every passage's vocabulary | `topics`, `passage_topics` | `modelled` |
| `question_generation` | `topics.question_status` | one topic's facts | `questions`, `question_facts` | `generated` |
| `assessment` | `assessments.assess_status` | one fact, topic or question | `assessments`, `assessment_metrics` | `assessed` |

Ingestion owns no queue: an upload writes a row and an object, and parsing is
what is then asked to run.

Assessment owns a queue with nothing upstream to fill it. No stage hands it an
artefact, so `start` **enrols** whatever has no assessment and then queues it —
topic modelling's arrangement, where asking is what creates the work, reached
from the other direction. It is also the one stage that can be switched off
entirely, with `ASSESSMENT_ENABLED`, because it is the one stage the dataset
does not depend on.

A stage selects on its own status column and on nothing else. Chunking never
reads `parse_status`, which is how parsing and chunking own different columns
of the same `documents` row.

The five buckets are `documents`, `parsed`, `models`, `export` and `archive` —
`S3_BUCKETS` in `.env`. `models` holds what the topic fit produced and
`export` a figure drawn from it, which is why they are two. See
[`backend/blob_store/`](../backend/blob_store/README.md).

## 2. What a worker does with one row

The same queue for all five stages, written once in
[`backend/stages/`](../backend/stages/README.md).

```mermaid
stateDiagram-v2
    [*] --> new: the previous stage finished
    new --> pending: start
    pending --> new: stop
    pending --> in_progress: claimed
    in_progress --> done: finished
    in_progress --> failed: raised
    in_progress --> failed: lease expired
    failed --> pending: retry
    done --> [*]
```

A row arrives `new`, which no worker looks at. Only `pending` is claimed, so
nothing starts by itself — the Start button, a route, a `make …-start` or the
[orchestrator](../orchestration/README.md) moves it.

Claiming is `SELECT … FOR UPDATE SKIP LOCKED`, one row at a time:

```bash
podman compose up -d --scale extract-worker=4
```

Every claim is timestamped and every lease is derived from what the stage
costs — extraction's from `LLM_TIMEOUT_SECONDS` × `LLM_MAX_ATTEMPTS`. A row a
worker died holding is failed by the next run of that stage, and `retry`
returns it.

## 3. What a module may import

Enforced by [`.importlinter`](../.importlinter). Run `make lint-imports`.
Higher may import lower; nothing imports upward.

```mermaid
flowchart TD
    api --> stages
    subgraph services [the seven services, none importing another]
        direction LR
        assessment
        extraction
        ingestion
        preprocessing
        question_generation
        topic_modelling
        archive
    end
    api --> services
    services --> stages
    stages --> shared
    subgraph shared [llm / nlp / blob_store]
        direction LR
        llm
        nlp
        blob_store
    end
    shared --> settings
    settings --> database
    database --> telemetry
```

| Contract | Says |
|---|---|
| `layers` | A stage sits above what it shares and below nothing but the api |
| `stages-are-independent` | No backend service imports another |

There are two deliberate edges the contracts name. `settings.changes`
imports each service's `config` submodule, because refusing a setting that
would stop a service means calling that service's own `Settings.load`. And
`question_generation.export` imports `assessment.repository` — one import,
in a function body, in the module that is a command line rather than a
library: `workbook()` takes the judgements as an argument and imports
nothing, which is what lets the api call it holding its own catalogue.

**One module sits outside the graph.** `backend/confidence.py` is a leaf
imported by both extraction and question generation — how close a row came
to the verdict that would have refused it, which is a reading both stages
take and neither owns. It imports nothing but the standard library, so it
cannot create an edge between them; it is not in `.importlinter`'s
`root_packages`, so nothing enforces that.

A service package re-exports nothing, so a caller naming
`extraction.repository` pays for that submodule and no more. The contracts
cannot check that on their own — grimp counts imports inside function bodies,
and a deferred import is how the topic service keeps pyLDAvis out of
everything else.
[`tests/static/test_api_stays_light.py`](../tests/static/test_api_stays_light.py)
is what checks it.

## 4. What runs in a container

```mermaid
flowchart LR
    browser([browser]) --> streamlit
    streamlit --> api
    orchestration --> api
    api --> pg[(postgres)]
    api --> s3[(seaweedfs)]

    subgraph workers [six workers, one image with the api]
        parse-worker
        chunk-worker
        extract-worker
        topic-worker
        question-worker
        assess-worker
    end
    workers --> pg
    workers --> s3
    workers -.-> model{{served model}}

    workers --> logs[/logs volume/]
    api --> logs
    host([a make target]) --> hostlogs[/./logs/]
    hostlogs --> filebeat
    logs --> filebeat --> elasticsearch --> grafana
    host -.spans.-> phoenix
    workers -.spans.-> phoenix
    api -.spans.-> phoenix
    elasticsearch --> argilla
```

The frontend talks only to the api. The served model is the one outbound
call, and a local one makes even that internal.

`make services` asks the api which of these are listening; `make open` opens
the application.

### The same stage, run on the host

`make extract`, `make topics` and `make questions` run the identical code in
a host process against the same database, and the Makefile sources the same
files compose hands the containers.

Some credentials exist only where a person is — an interactive cloud login, a
key in a login keychain — and a container holds none of them.
`LLM_CONTAINER_MODEL` is the split: the containers call a model they can
authenticate to, the host calls the one it can, and both drain the same
queues because claiming is `FOR UPDATE SKIP LOCKED`. That is two halves of
one corpus, not an A/B — comparing two models means running them over the
same rows under two `RUN_ID`s.

`LOG_DIR` names `./logs` on the host and the `logs` volume in a container,
and filebeat reads both into the same data stream. `host.name` tells a
machine from a container id.

## 5. Where the signals go

| Signal | Path | Read with |
|---|---|---|
| Logs | process → `logs` volume or `./logs` (JSON, ECS fields) → Filebeat → Elasticsearch | Grafana, `make logs` |
| Traces | process → OTLP → Phoenix, one project per `<position>-<stage>-<run id>` | Phoenix |
| Cost and tokens | on the span, and on the log line beside it | Phoenix, Grafana, `make spend LOG=` |
| Gate verdicts | the row in Postgres, an attribute on the span, one annotation per gate | the Questions page, Phoenix's Evaluations view |
| Which gates ran | `questions.gates_ran`, in the order they read it | the Questions page, `GET /lineage/question/<id>` |
| Lineage | the link tables — `fact_passages`, `question_facts`, `passage_topics` | `GET /lineage/<kind>/<id>`, the **How this was produced** fold |
| The join between them | `trace_id` and `span_id`, on every artefact table | every artefact page links to both |
| Prompts | composed in the source, recorded to the `prompts` table, published to Phoenix by the stage that recorded them, and on each span | the Questions page, `GET /prompts`, Phoenix |
| Scores | golden cases run against the served model | `make eval-score`, Phoenix |
| Human review | Postgres → a disposable copy in Argilla → the answers back | Argilla, `make review-*` |
| Runs | which stage ran when | Dagster |
| State | the tables themselves | Grafana's state dashboard, the pipeline pages |

`trace.id` is on every log line and is the join between a span and its lines.
See [`telemetry/`](../telemetry/README.md).

### One number, in all five

[`telemetry/pipeline.py`](../telemetry/pipeline.py) holds the seven stages
once, and the number in front of a name is the stage's place among them.
All four tools sort their own names alphabetically, which is not the order
a corpus moves, so `4-extraction` in Phoenix, `4-facts` in Argilla and
`stage_4_extraction` in Dagster are the same step seen from three sides.

Each tool then has one job no other has: **Phoenix** the calls, **Grafana**
every line, **Argilla** what a person judged, **Dagster** which run, and the
**application** the chain between the artefacts and the links out to the
other four. Two static tests keep the copies of those names equal —
`tests/static/test_tool_names.py`, because neither `orchestration` nor
`review` can be imported from where the names are needed.

Nothing is deleted until `make logs-retention` and `make logs-prune` have
run. The first ages the Elasticsearch index, the second the files.

## 6. Where to change things

| To change | Go to |
|---|---|
| Any behaviour setting | [`backend/settings/catalog.py`](../backend/settings/catalog.py) |
| Which model, or which provider | `LLM_MODEL` in `.env`; credentials in `configs/env/provider.env` |
| Credentials, ports, addresses | `.env` |
| How the pipeline behaves, by default | `configs/env/backend.env` |
| What a stage does | that stage's package, and nothing else |

A setting is declared once. Two static tests refuse a setting the code reads
and the catalogue omits, and a catalogue entry nothing reads. See
[configuration.md](configuration.md).

## What keeps this page true

| Claim | Checked by |
|---|---|
| The layer order and the independence rule | `make lint-imports` |
| The layers named here match the contracts | `tests/static/test_architecture.py` |
| The api loads no model or converter | `tests/static/test_api_stays_light.py` |
| Every setting is declared once and read | `tests/static/test_settings_catalogued.py` |
| The dashboards match the fields that serve them | `tests/static/test_dashboards.py` |
| One stage goes by the same name in all four tools | `tests/static/test_tool_names.py` |
| Every artefact records the run and trace that made it | `tests/integration/database/test_provenance.py` |
| Every relative link on these pages resolves | `tests/static/test_doc_links.py` |
