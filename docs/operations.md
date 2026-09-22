# Operations

Running the stack, watching it, and the things that go wrong.

## The services

Once `make dev` reports ready:

| Service | URL | Purpose |
|---|---|---|
| Frontend | <http://localhost:8501> | Run each stage, browse what it produced, view system status |
| API | <http://localhost:8000/docs> | OpenAPI documentation |
| Phoenix | <http://localhost:6006> | Traces, and the golden-set experiments — sign in as `admin@localhost` with `PHOENIX_DEFAULT_ADMIN_INITIAL_PASSWORD` |
| Grafana | <http://localhost:3001> | Logs and pipeline dashboards — `GRAFANA_ADMIN_USER` / `GRAFANA_ADMIN_PASSWORD` |
| Argilla | <http://localhost:6900> | Review what the models decided — `ARGILLA_USERNAME` / `ARGILLA_PASSWORD` |
| Adminer | <http://localhost:9001> | Database browser |
| Dagster | <http://localhost:3000> | The asset graph and run history |

Behind them: PostgreSQL, SeaweedFS (master, two volumes, filer, S3 gateway,
UI), Elasticsearch, Redis, Filebeat, and the five stage workers.

## Scaling

`parse-worker`, `chunk-worker` and `extract-worker` all scale horizontally:
each claims one row at a time with `FOR UPDATE SKIP LOCKED`, so two workers
never take the same row.

```bash
podman compose up -d --scale extract-worker=4
```

`topic-worker` can be scaled too, though there is no point: a fit is one
request row, and only one worker can claim it.

`question-worker` scales the same way and has far less to divide. Its queue is
one row per topic, so twelve topics per language is two dozen rows for the
whole corpus, and a worker past that has nothing to claim.

**Its first start is slow and looks like nothing happening.** It downloads
`EMBEDDING_MODEL` before it claims anything — 2.2 GB into the `models` volume.
The topics sit `pending` until that finishes, and `make logs` is where it says
so. Later starts read the volume and claim immediately. `question-worker` is
given 6 GB because of it, and the weights are **per container, not per lane**.

### The one that reads like a broken worker

A stage added since a stack came up has no container until `make up` creates
one. The queue fills, `/{stage}/status` reports it, and nothing drains it —
which reads exactly like a broken worker rather than an absent one.

## What the model costs

The model is the bottleneck, not the pipeline: a median passage measured at
**473 s** on a 31B model. A passage whose sentences carry no finite verb — a
heading, a caption, a navigation line — is skipped before the call rather than
sent and rejected afterwards.

Live, that is the **Pipeline throughput** dashboard below and Phoenix, each
per run. Over a run that is already finished, off a log that was captured
while it ran:

```bash
make spend LOG=run.log                      # calls, tokens and cost
make spend LOG=run.log SINCE=2026-09-19
```

`LOG` is required. There is no default: the shipped files are JSON inside a
container, and the pattern this reads is the text one a terminal saw.

## Dashboards

Three, provisioned into Grafana from `configs/grafana/`, over two datasources.

| Dashboard | Reads | Shows |
|---|---|---|
| Pipeline state | PostgreSQL | Queue depth per stage, failures with reasons, fact acceptance by rejection code, question acceptance by gate, topics and their coverage |
| Pipeline throughput | Elasticsearch | Units finished per interval, model latency at p50/p95/p99, facts and questions accepted against refused, and which queue verb was asked for over HTTP |
| Pipeline logs | Elasticsearch | Lines per level, what failed and where, every line |

**The split is the point.** Logs say what happened once; the tables say what
is true now. A row a worker died holding logged nothing and is still counted
in Pipeline state, which is the difference that matters when a stage has gone
quiet.

The panel worth watching during a long run is the **model latency p99**. The
extraction lease is derived from `LLM_TIMEOUT_SECONDS` and `LLM_MAX_ATTEMPTS`,
so a p99 climbing towards the timeout is healthy workers about to start
looking abandoned.

Grafana reads the application database as `grafana_reader`, which holds
`SELECT` and nothing else. The role is created by `configs/postgres/init.sh`,
which only runs on the first boot of an empty volume; on a stack that already
has one, re-run it by hand:

```bash
podman compose exec postgres bash /docker-entrypoint-initdb.d/init.sh
```

## Logs

```bash
make logs                # everything, from the container engine
make logs-api
make logs-shipper        # why nothing is arriving, when nothing is arriving
make logs-retention      # how long Elasticsearch keeps them. Run once
make logs-prune          # delete the files on the volume nothing writes to
```

Only this project's processes are shipped to Elasticsearch. Postgres,
SeaweedFS, Redis and Elasticsearch itself keep the `json-file` driver and are
read with `make logs`.

**Until `make logs-retention` has run, nothing is ever deleted.** That is the
state the stack ships in.

Retention has two halves, and that target is one of them. It ages what
**Elasticsearch** holds; `make logs-prune` ages the **files** the shipper
read them out of. Each process writes `{service}-{container}.log`, so a
recreated container starts a new file and leaves the old one for ever —
nothing else on the volume deletes anything.

See [`telemetry/`](../telemetry/README.md) for the field names, the data
stream, and the two Filebeat settings that are load-bearing.

## Failure, and why every one is recoverable

A row a worker died holding is failed by the next run of that stage, with the
reason recorded, rather than being left claimed and invisible; `retry` then
returns it to the queue.

**There is no state a row can reach that nothing can move it out of.**

A stage's failure is never an HTTP 500. It is recorded against the row and
read back through `/{stage}/status` — a document that failed to parse is a 200
with a reason in it.

## Things that go wrong

| Symptom | Usually |
|---|---|
| Topics sit `pending`, nothing in the logs | `question-worker` is downloading 2.2 GB of embedding weights. `make logs` says so |
| A queue fills and nothing drains it | That stage has no container. `make up` |
| Grafana panels are empty | The shipper. `make logs-shipper` |
| A panel that groups by `stage` finds nothing | A field missing from `append_fields` in `filebeat.yml`, arriving as analysed text |
| An edit to `filebeat.yml` changed nothing | `setup.template.overwrite` is false, so Filebeat left the existing template alone |
| Grafana cannot read the database | `init.sh` has not run on this volume. Re-run it by hand |
| Streamlit refuses a file the API would accept | `server.maxUploadSize` is below `MAX_FILE_SIZE_MB` |
| The frontend will not start, naming a variable | It is in `.env` but not in the `streamlit` service's `environment` in `compose.yaml` |
| Argilla refuses the key | `ARGILLA_API_KEY` is under "My settings" in the UI. It is not `ARGILLA_PASSWORD` |
| A p99 climbing towards `LLM_TIMEOUT_SECONDS` | Healthy workers about to start looking abandoned. Raise the timeout, or the lease follows it |
| A scanned PDF is refused rather than parsed | OCR is not enabled. See the placeholder in `backend/preprocessing/parsing/pipelines/pdf.py` |

## Deleting things

```bash
make delete SHA=<sha256>          # a document and everything from it
make delete-derived SHA=<sha256>  # only its passages and facts
make wipe                         # every document, then the topics
make topics-delete                # every topic, membership and figure
```

Each of those is the **first of two deletions**. An AFTER DELETE trigger on
every table copies the row into `archived_rows`, and the removal paths move
the objects into the `archive` bucket instead of dropping them:

```bash
make archive                      # what is held, by table, with its age
make archive-purge DAYS=30        # the second deletion. This one is final
```

Nothing purges itself, so the archive grows — on ordinary runs too, because a
re-extraction and a re-chunk delete what they replace. `make archive` is how
you see it. See [`backend/archive/`](../backend/archive/README.md).

These two are still irreversible, and archive nothing — `TRUNCATE` and a
dropped volume fire no row trigger:

```bash
make down-volumes                 # the containers' data, all of it
make schema-reset                 # every table, rebuilt from the revisions
```

Deleting a document takes **everything derived from it**: its passages, their
topic memberships, the facts on them, and — through the trigger on
`question_facts` — the questions resting on those facts.

What it does **not** take is `topics`. A fit is over the corpus rather than
over a file, so deleting every document leaves the topics standing, describing
passages that are gone. Nothing is broken by that — the next fit replaces
every row — but it is why `make wipe` is one command that runs both.
