# Telemetry

Logging and tracing, configured identically in every process — the API, the
five workers, the frontend, the orchestrator and the host commands — and
called before anything else.

## Which store holds what

| Tool | Holds | Does not hold |
|---|---|---|
| **The application** | The **artefacts and the chain between them**: what one question was produced from, which gates read it, and a link into each of the four below | Anything about a run it did not store on a row |
| **Phoenix** | The **logic**: every stage's spans, every model call with its shape and prompt version, and every validation as an annotation of its own | Services. No HTTP, no SQL, no object store, no frontend |
| **Grafana** | **Every line every process writes**, per service, joined to its trace and its run | Container logs of the infrastructure |
| **Argilla** | The **artefacts a person judged**, one dataset per kind | |
| **Dagster** | The **workflow**: what ran, what it produced, in what order, whoever started it | Row-level lineage |

### One number, in all five

[`pipeline.py`](pipeline.py) holds the seven stages once, and the number in
front of a name is the stage's place among them:

| Stage | Phoenix project | Argilla dataset | Dagster group |
|---|---|---|---|
| 1 ingestion | — | — | — |
| 2 parsing | `2-parsing-<run>` | — | `stage_2_parsing` |
| 3 chunking | `3-chunking-<run>` | — | `stage_3_chunking` |
| 4 extraction | `4-extraction-<run>` | `4-facts` | `stage_4_extraction` |
| 5 topic modelling | `5-topic_modelling-<run>` | `5-topic-labels` | `stage_5_topic_modelling` |
| 6 question generation | `6-question_generation-<run>` | `6-questions` | `stage_6_question_generation` |
| 7 assessment | `7-assessment-<run>` | — | `stage_7_assessment` |

All four tools sort their own names alphabetically, which put Argilla's
`facts, questions, topic-labels` and Phoenix's `assessment-, extraction-,
questions-, topics-` in an order no corpus moves in. The number is what
makes each tool's own ordering the pipeline's, and it means the same thing
in every one of them.

Nothing migrates the projects and datasets written under the old names.
They stay until they age out.

`pipeline.py` also names each stage's **Dagster asset** and **Argilla
dataset**, because the two modules those really live in cannot be imported
from where they are needed: `orchestration` is a code location in an image
of its own, and no container carries the Argilla client.
[`tests/static/test_tool_names.py`](../tests/static/test_tool_names.py)
reads both as syntax and fails if a copy has gone stale — a renamed asset
would otherwise leave the application linking to a Dagster page that 404s,
which looks exactly like Dagster being down.

### Where a person starts

`GET /lineage/{kind}/{id}`, and the **How this was produced** fold on every
artefact page. It walks a question up to its facts, a fact to its passages,
a passage to its document, reports each step under the stage that produced
it, and links each artefact to its own span. That is the one question none
of the four tools answers on its own, because each holds one slice of it.

Two rules in [`traces.py`](traces.py) keep that line:

- **Only a run exports.** `configure(name, run=...)` builds an exporter only
  when `run` is given. A stage passes `settings.runs.run_id()`; the api, the
  frontend, the orchestrator and every host command pass nothing.
- **Only the logic is instrumented.** litellm, and the spans the stages open
  themselves. No requests, no botocore, no FastAPI — and the database only
  behind `TRACE_DATABASE`.

`RUN_ID` is spelled the same in all three stores: `questions.run_id` on the
rows, a project named `<stage>-<run id>` in Phoenix, `run.id` on every log
line. The **Run** dashboard is those three joined on one page.

## Logs

Each record is rendered **twice**: as a line of text on stdout, which is what
`make logs` shows, and as one JSON object per line in a file, which is what
reaches Elasticsearch.

```
api, 5 workers, streamlit, dagster  ──▶  logs volume  ─┐
                                                       ├─▶  filebeat  ──▶  elasticsearch  ──▶  grafana
a make target, dagster dev, by hand  ──▶  ./logs  ─────┘
```

Nothing is aggregated and nothing is dropped. Every level from `LOG_LEVEL`
upwards is shipped.

An exception is written **whole**: `error.type`, `error.message` and the
entire traceback in `error.stack_trace`, so a failure is readable in Grafana
without going back to the container.

### Field names are ECS

`log.level`, `service.name`, `log.logger` and `error.type` are names
[Elasticsearch's own template](https://www.elastic.co/guide/en/ecs/current/index.html)
already maps as keywords. `trace.id` is on every line, which ties a log line
to its span, and `run.id` ties it to the run that wrote it.

### What a line says it was working on

Each stage binds its unit of work where it claims it, and every line beneath
carries it — including the ones a library logs:

```python
with working(span, "extract", {"stage": self.name, "passage.id": passage.id}):
    log.info(
        "passage %d: %d fact(s)", passage.id, stored, extra={"facts.stored": stored}
    )
```

One call, because the span and the log line are one fact said twice. A stage
annotating its trace with `document.sha256` and its lines with something
spelled differently is a trace that cannot be joined to the logs explaining
it.

So `stage`, `document.sha256`, `passage.id`, `topic.id`, `llm.duration_ms`,
`facts.stored` and the rest are **fields** rather than prose inside
`message`. These are not ECS, so they are declared in
`setup.template.append_fields` in
[`configs/filebeat/filebeat.yml`](../configs/filebeat/filebeat.yml). Left to
dynamic mapping a string arrives as `text`, and a panel grouping by `stage`
finds nothing to group on.

### The file, and why it is per process

`{service}-{host}-{pid}.log`. The **writer** is what the name has to be
unique per, because two processes rotating one file take each other's lines
with them — and neither half alone is enough: a scaled stage is several
containers over one volume, and `./logs` is one machine over many runs.

The cost is a file per run, including the one-second ones. `make logs-prune`
is the only thing that sweeps them. Files rotate at 50 MB, three kept.

### Two directories, one data stream

| Where | `LOG_DIR` | Set by |
|---|---|---|
| A container | `/var/log/qa`, the `logs` volume | compose, over the value below |
| The host | `./logs`, bind-mounted into filebeat read-only | `.env` |
| Anywhere else | unset — stdout only | nobody |

One name, set over per container — the same arrangement
`OTEL_EXPORTER_OTLP_ENDPOINT` and `PHOENIX_BASE_URL` use, because the
Makefile sources `.env` and a host process must read the host's answer from
the name the code reads.

Filebeat reads both paths in one input, so a `make` target's lines land in
Grafana beside a worker's, on the same trace. `host.name` tells them apart.

`./logs` is made by `make up` rather than left to the container engine, which
creates a missing bind-mount source itself and can leave it owned by root.

## Traces

OpenTelemetry, to Phoenix. A span per unit of work of the **logic**,
annotated with the same names the log fields use.

A stage passes `settings.runs.run_id()` to `configure`, which sets
`openinference.project.name` on the resource. The project is the **stage**,
numbered so one pass over the pipeline sorts in the order a corpus moves.

**A run gets a project of its own only when somebody named it.**

```bash
make questions                      # 6-question_generation
make questions RUN_ID=a-gpt-4.1     # 6-question_generation-a-gpt-4.1
```

A project per run was the original design and it does not survive a
worker: `run_id` fell back to a uuid per PROCESS, `restart: unless-stopped`
makes a process per restart, and this deployment reached **five hundred
projects** — 497 of them holding one span, the model call a preflight made
before giving up. The comparison that design existed for is still there
and is now the thing you ask for by name.

Two runs sharing a project are still separable: `run.id` is a resource
attribute on every span, the same id the logs and the rows carry. Telling
them apart is a filter rather than a hunt through a sidebar.

**The id a row carries is the run that asked for it**, not the worker that
took it — a Dagster run, a Start button or a `make …-start`, whichever
queued the row. See [`settings/README.md`](../backend/settings/README.md).
The span attribute is the same value, so a fact, its log lines and its
model calls all filter on one id whoever set the stage going.

**The exporter is installed after the preflight, not before.** A project
is created by the first span filed under it, and a stage's first span is
the model call its preflight makes. A credential that has expired
therefore left a project per attempt — which under `restart:
unless-stopped` is a project per restart. `stages/cli.py` now configures
the logs first and tracing only once the process knows it is going to
claim rows.

```bash
make phoenix-projects   # what Phoenix holds
make phoenix-prune      # take back the ones in which no work happened
```

`prune` takes a project only when every span in it is a model call **and**
it could read every span. One too big to read in a page is left alone: the
first fifty spans of a real run are model calls too, and
`5-topic_modelling` reads exactly that way.

The api, the frontend, the orchestrator and every host command pass nothing,
and that is what turns their exporter off rather than filing them under
`default`.

Question generation also opens a span per question carrying the gate that
stopped it **and every gate that read it** — `question.gates_ran`, in order.
The checker returns on the first failure, so the gate alone could never say
how far a question got.

Two things the instrumentor cannot know go on beside its own attributes, from
`telemetry.asking`:

| On the span | Is |
|---|---|
| `tag.tags` | the Pydantic **shape** — `_Answered`, `_NamesItsSource`, `_Recovered`. This is `llm.shape` in the log, spelled the same |
| `llm.prompt_template.version` | the caller's **`PROMPT_VERSION`** |

Both are context rather than a span of our own. A caller that names no
version sets none.

The span carries the prompt **filled in**. What a version *asked for* — the
template — is the `prompts` table; see
[`backend/stages/prompts.py`](../backend/stages/prompts.py).

### How much reaches Phoenix, and when

Three channels, three mechanisms, and they fail differently.

| | Carries | Sent |
|---|---|---|
| **Spans** | a stage's unit of work, and one `completion` per model call | batched — every **5 s** or **512 spans**, whichever comes first |
| **Annotations** | the gate verdicts and the judge's, per span | batched at **100**, flushed when a queue empties and at the end of a drain |
| **Prompts, datasets** | what a version asked for, and the golden cases | only when `make prompts-publish` or `make eval-upload` is run |

The span batching is **OpenTelemetry's own**, configured by
`OTEL_BSP_MAX_QUEUE_SIZE`, `OTEL_BSP_SCHEDULE_DELAY` and
`OTEL_BSP_MAX_EXPORT_BATCH_SIZE`. Nothing is passed in code, because an
argument would override the environment and take that escape hatch away.

The defaults suit this pipeline by a wide margin: a queue of 2,048 drained
every 5 s is sized for a service answering requests, and the unit of work
here is a passage at a median 473 s. Measured against a real judging run —
5 artefacts judged, 5 `assess` spans and 18 `completion` spans in Phoenix,
nothing lost. A queue that did fill would log `Queue full, dropping spans`
per overflow, so that is a loud failure rather than a quiet one.

**A span is dropped and an annotation is not, when Phoenix is down.** The
exporter discards what it cannot send; `evaluations.py` warns once per run
and stops trying, and the verdict is in Postgres either way.

## Verdicts

A gate verdict is written three times. The **row** in Postgres is the truth.
The **span attribute** is what a trace is filtered by. The **annotation** —
[`evaluations.py`](evaluations.py), posted through `arize-phoenix-client` —
is what puts a label, a score and an explanation in Phoenix's Evaluations
view.

`annotator_kind` separates them: a gate is **CODE**, and the three phrasing
judgements are **LLM**.

The evaluation phase is the other writer of annotations here, and everything
it posts is **LLM**. One per metric, named as `arize-phoenix-evals` names it
— `hallucination`, `relevance`, `qa_correctness`, `summarization` — plus an
`assessment` summary per artefact. Same client, same batching, same bargain
with an unreachable Phoenix. See
[`backend/assessment/`](../backend/assessment/README.md), and note that
`hallucination` scores 1.0 for the **bad** label because its optimisation
direction is minimise; each annotation carries its `direction` as metadata
so a chart is read the right way up.

A question produces a **summary** — `gate`, labelled with whatever stopped it
or `accepted` — and then **one annotation per gate that read it**, named
`gate 1: structural` through `gate 6: round_trip`:

| Annotation | On an accepted question | On one refused at phrasing |
|---|---|---|
| `gate` | `accepted`, 1.0 | `leaks_source`, 0.0 |
| `gate 1: structural` | `passed`, 1.0 | `passed`, 1.0 |
| `gate 4: near_duplicate` | `passed`, 1.0 | `passed`, 1.0 |
| `gate 5: phrasing` | `passed`, 1.0 | **`refused`, 0.0** |
| `gate 6: round_trip` | `passed`, 1.0 | *absent* |

Only the gates that **ran**: a gate's rate is over the questions that reached
it, and scoring an unreached gate as passed would make a run that refused
everything at the first rule read as a round trip that passed everything.

The number is the gate's fixed **position**, not its order in that question,
because Phoenix sorts its columns by name.

Best-effort, like the exporter: a Phoenix that is down costs the annotation
and not the run, and warns once rather than once per batch.
`PHOENIX_BASE_URL` is where they are posted; the bearer is `PHOENIX_API_KEY`
where compose set it and `PHOENIX_ADMIN_SECRET` where `.env` did.

## Configuration

| Setting | Where | Default | What it does |
|---|---|---|---|
| `LOG_LEVEL` | `.env` | `INFO` | Every service, the frontend included |
| `LOG_DIR` | `.env`, set over in `compose.yaml` | `logs` on the host, `/var/log/qa` in a container | Where the JSON file goes. Unset means stdout only |
| `OTEL_CONTAINER_ENDPOINT` | `.env` | `http://phoenix:4317` | Trace collector, as the containers reach it |
| `TRACE_DATABASE` | unset | off | A span per SQL statement and per pool connect |

`service.name` is not a setting: each process passes its own name to
`telemetry.configure(...)`.

`LOG_DIR` is read through a variable rather than a literal, so the static
settings scan cannot see it. It is checked by hand.

## Retention

Two halves, because a line is in two places:

```bash
make logs-retention                        # 30 days in Elasticsearch
make logs-retention LOGS_RETENTION_DAYS=90
make logs-prune                            # 30 days of files on the volume
make logs-prune LOGS_KEEP_DAYS=7
```

`logs-retention` runs **once** against a running stack, after the shipper has
written something. Every backing index the stream rolls over to afterwards
inherits it. Until it has run, nothing is deleted.

`logs-prune` deletes by mtime, so a file a live process is appending to is
never old enough to take. It goes through the api, the one container that
mounts the volume writable.

The volume needs its own target because the file name carries the container:
`RotatingFileHandler` rotates only its own process's file, and filebeat's
`ignore_older` and `clean_inactive` age its **registry**, not the disk.

The retention belongs to the data stream, not to an ILM policy. The index is
named `qa-logs` and not `qa-logs-<date>`, because a date in the name creates
a second data stream every midnight. Naming an ILM policy in `filebeat.yml`
does not work either: Filebeat strips `index.lifecycle` out of the template
when `setup.ilm.enabled` is false.

`setup.template.overwrite: true` is what makes an edit to `filebeat.yml` take
effect at all.

## What is shipped, and what is not

Only this project's processes. Postgres, SeaweedFS, Redis and Elasticsearch
itself keep the `json-file` driver and are read with `make logs`. Collecting
those too would mean reading the engine's own log store, which is in a
different place under Docker and Podman and, on macOS, inside a virtual
machine a bind mount cannot see.

```bash
make logs-shipper    # why nothing is arriving, when nothing is arriving
```

## Tests

```sh
poetry run pytest tests/unit/telemetry
```

| File | Covers |
|---|---|
| [`test_logs.py`](../tests/unit/telemetry/test_logs.py) | The JSON line the shipper reads: the ECS names, the bound fields, and an exception written whole |
| [`test_evaluations.py`](../tests/unit/telemetry/test_evaluations.py) | The annotation a verdict becomes, the address it is posted to, and a run surviving a Phoenix that is down |
| [`test_dashboards.py`](../tests/static/test_dashboards.py) | The provisioned dashboards against the datasources and fields that serve them |
| [`test_log_fields.py`](../tests/static/test_log_fields.py) | Every field the code binds against what the shipper declares |

## Limits

- **Nothing is deleted until `make logs-retention` has run once.**
- **`make logs-retention` never deletes a file.** That is `make logs-prune`.
- **The api and the frontend produce no span at all.** Not a quieter project
  — no exporter. What they did is in the logs, in full.
- **An HTTP call and an S3 GET produce no span.**
- **SQL statements produce no span unless `TRACE_DATABASE` is set.**
- **`run.id` is empty on a line a run did not write.** Present and empty
  rather than absent, so there is one shape of line.
- **A host command leaves a file per invocation.**
- **`LOG_DIR` unset means stdout only** — a bare `python -m …` in a shell
  that sourced nothing, and pytest.
- **Editing `filebeat.yml` without `setup.template.overwrite` changes
  nothing.**
- **A new bound field needs adding to `append_fields`**, or it arrives as
  `text` and nothing can group on it.
- **A declared field reaches the template at once and the index at
  rollover.** A data stream's backing index keeps the mapping it was created
  with.
- **The Elasticsearch heap is shared with Argilla.** `ES_JAVA_OPTS` is where
  to raise it.
