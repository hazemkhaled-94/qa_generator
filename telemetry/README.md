# Telemetry

Logging and tracing, configured identically in every process.

One configuration, and **every process calls it before it does anything
else** — the API, the five workers, the frontend, the orchestrator and the
host commands alike.

## Logs

Each record is rendered **twice**: as a line of text on stdout, which is what
`make logs` shows, and as one JSON object per line in a file, which is what
reaches Elasticsearch. Same record, same fields; the JSON carries the ones a
text line has no room for.

```
api, 5 workers, streamlit, dagster  ──▶  logs volume  ─┐
                                                       ├─▶  filebeat  ──▶  elasticsearch  ──▶  grafana
a make target, dagster dev, by hand  ──▶  ./logs  ─────┘
```

Nothing is aggregated and nothing is dropped on the way. Every level from
`LOG_LEVEL` upwards is shipped, `DEBUG` included when it is set that low.

An exception is written **whole**: `error.type`, `error.message` and the
entire traceback in `error.stack_trace`, so a failure is readable in Grafana
without going back to the container. Every stage that records a failure
against a row also logs it with its traceback — the row's error column is one
line for a person reading the Documents page, not the whole story.

### Field names are ECS

`log.level`, `service.name`, `log.logger` and `error.type` are names
[Elasticsearch's own template](https://www.elastic.co/guide/en/ecs/current/index.html)
already maps as keywords, so Grafana can group on them out of the box.
`trace.id` is on every line too, which is what ties a log line to its span in
Phoenix.

### What a line says it was working on

A line carries what the pipeline was **doing** when it was written, not just
what happened. Each stage binds its unit of work where it claims it, and every
line beneath carries it — including the ones a library logs, which is the
point:

```python
with working(span, "extract", {"stage": self.name, "passage.id": passage.id}):
    log.info(
        "passage %d: %d fact(s)", passage.id, stored, extra={"facts.stored": stored}
    )
```

**One call, because the span and the log line are one fact said twice.** A
stage annotating its trace with `document.sha256` and its lines with something
spelled differently is a trace that cannot be joined to the logs explaining
it, and two calls drift into exactly that.

So `stage`, `document.sha256`, `passage.id`, `topic.id`, `llm.duration_ms`,
`facts.stored` and the rest are **fields** rather than prose inside `message`.
That is what makes "how many passages failed extraction today" and "show me
everything about this document" answerable at all — a document crosses five
processes over hours, and without the binding nothing in Elasticsearch says
two lines are about the same one.

These are not ECS, so they are declared in `setup.template.append_fields` in
[`configs/filebeat/filebeat.yml`](../configs/filebeat/filebeat.yml). Left to
dynamic mapping a string arrives as `text`, which is analysed and has no doc
values, and a panel grouping by `stage` finds nothing to group on.

### The file, and why it is per process

`{service}-{host}-{pid}.log`. The **writer** is what the name has to be
unique per, because **two processes rotating one file take each other's
lines with them** — and neither half alone is enough: a scaled stage is
several containers over one volume, and `./logs` is one machine over many
runs. So `extraction-3f15a9823823-1.log` from a worker sits beside
`extraction-hazems-mac-48213.log` from a drain somebody ran by hand.

The cost is a file per run, including the one-second ones. `make logs-prune`
is what sweeps them, and it is the only thing that does.

Files rotate at 50 MB, three kept; the shipper has read a line long before it
is deleted.

### Two directories, one data stream

`LOG_DIR` is what turns the file on, and it names a different directory
depending on where the process is:

| Where | `LOG_DIR` | Set by |
|---|---|---|
| A container | `/var/log/qa`, the `logs` volume | compose, over the value below |
| The host | `./logs`, bind-mounted into filebeat read-only | `.env` |
| Anywhere else | unset — stdout only | nobody |

One name, set over per container — the same arrangement
`OTEL_EXPORTER_OTLP_ENDPOINT` and `PHOENIX_BASE_URL` use, and for the same
reason: the Makefile sources `.env`, so a host process must be able to read
the host's answer from the name the code reads.

Filebeat reads both paths in one input, so **a `make` target's lines land in
Grafana beside a worker's**, joined to the same trace. `host.name` is what
tells them apart: a container id on one side, a machine name on the other.

`./logs` is made by `make up` rather than left to the container engine,
which creates a missing bind-mount source itself and can leave it owned by
root — and then the host process it exists for cannot write to it.

## Traces

OpenTelemetry, to Phoenix. A span per unit of work, annotated with the same
names the log fields use.

The API attaches tracing to the app instance **after it exists**, so a
caller's trace continues there rather than a new one beginning. The
orchestrator installs the same configuration, so its spans join the trace the
API continues.

`trace.id` on every log line is what joins the two: a slow extraction is a
span in Phoenix and a set of lines in Grafana, found from either end.

### A project per run

A stage passes `settings.runs.run_id()` to `configure`, which sets
`openinference.project.name` on the resource — the attribute Phoenix files
a span under. One run of one stage is then one project, `<stage>-<run id>`,
and two runs compare directly on the thing the database does not hold: the
call count, the tokens, the latency and the spend.

Named with the service as well as the run, so one pass over the pipeline is
five projects that sort together rather than one heap in which extraction's
calls and question generation's cannot be told apart.

The api and the frontend pass nothing and keep the default project. Neither
produces a run, and a project per API process is a project per restart.

Question generation also opens a span per question, carrying the gate that
stopped it. The model calls that question made are its children, so a run's
`leaks_source` rejections carry the price of the calls they wasted.

## Verdicts

A gate verdict is written three times, and each answers something the other
two cannot. The **row** in Postgres is the truth, and `make questions-runs`
counts it. The **span attribute** is what a trace is filtered by. The
**annotation** — [`evaluations.py`](evaluations.py), posted through
`arize-phoenix-client`, which the backend image carries for this — is what
puts a label, a score and an explanation in Phoenix's Evaluations view,
where it sorts, charts and compares across two projects with nobody writing
a query.

`annotator_kind` separates them: a gate is **CODE**, and the three phrasing
judgements, which are a model's opinion, are **LLM**.

Best-effort, like the exporter: a Phoenix that is down costs the annotation
and not the run, and warns once rather than once per batch.

`PHOENIX_BASE_URL` is where they are posted. One name for both — compose
sets the container's address over it, the way it does for
`OTEL_EXPORTER_OTLP_ENDPOINT` — because the Makefile sources `.env` for a
host command, so a second name read as a fallback is a host run posting to
`http://phoenix:6006`. The bearer is `PHOENIX_API_KEY` where compose set it
and `PHOENIX_ADMIN_SECRET` where `.env` did; Phoenix compares the token
against that value directly, so the two are one credential.

## Tools, and where each is used

| Tool | Where | Why this one |
|---|---|---|
| **logging** (stdlib) | [`logs.py`](logs.py) | A `LogRecord` factory is the one place trace ids can be added to every line, including a library's |
| **OpenTelemetry** | [`traces.py`](traces.py) | The spans, and the exporter Phoenix reads |
| **arize-phoenix-client** | [`evaluations.py`](evaluations.py) | An annotation is filed separately from the span it is about, which is what Phoenix's Evaluations view reads. An attribute cannot be one |
| **Filebeat** | [`configs/filebeat/`](../configs/filebeat/) | Reads the volume and writes the data stream. Runs as its own container |

The trace fields are read off the current span directly rather than through
`opentelemetry-instrumentation-logging`, which does not populate them in the
pinned version.

## Configuration

| Setting | Where | Default | What it does |
|---|---|---|---|
| `LOG_LEVEL` | `.env` | `INFO` | Every service, the frontend included. Everything at or above it reaches Grafana |
| `LOG_DIR` | `.env`, set over in `compose.yaml` | `logs` on the host, `/var/log/qa` in a container | Where the JSON file goes. Unset means stdout only |
| `OTEL_CONTAINER_ENDPOINT` | `.env` | `http://phoenix:4317` | Trace collector, as the containers reach it |

`service.name` is not a setting. Each process passes its own name to
`telemetry.configure(...)` — `"api"`, `"orchestration"`, or the stage's name
for a worker — so a process cannot be misconfigured into logging under
another's identity.

`LOG_DIR` is read through a variable rather than a literal, so the static
settings scan cannot see it. It is checked by hand.

## Retention

Two halves, because a line is in two places. One ages the **index**, the
other the **files** it was read out of, and neither does the other's:

```bash
make logs-retention                        # 30 days in Elasticsearch
make logs-retention LOGS_RETENTION_DAYS=90
make logs-prune                            # 30 days of files on the volume
make logs-prune LOGS_KEEP_DAYS=7
```

`logs-retention` runs **once** against a running stack, after the shipper has
written something. Elasticsearch remembers it, and every backing index the
stream rolls over to afterwards inherits it. Until it has run **nothing is
deleted**, which is the state the stack ships in.

`logs-prune` runs whenever the volume has grown, and deletes by mtime, so a
file a live process is appending to is never old enough to take. It goes
through the api, which is the one container that mounts the volume writable
— the shipper's mount is read-only on purpose.

The volume needs its own target because the file name carries the container.
`RotatingFileHandler` rotates the file its own process owns and nothing
else's, and filebeat's `ignore_older` and `clean_inactive` age its
**registry**, not the disk. So a recreated container starts a new file and
leaves the old one for ever: the volume had reached 78 files and 110 MB
before this existed.

The retention belongs to the data stream, not to an ILM policy. This version
of Filebeat writes to a data stream, which rolls its own backing indices over
by age and size — so the index is named `qa-logs` and not `qa-logs-<date>`: a
date in the name creates a second data stream every midnight, each of which
then has to be found and aged separately. Naming an ILM policy in
`filebeat.yml` does not work either, because Filebeat strips
`index.lifecycle` out of the template it installs when `setup.ilm.enabled` is
false.

`setup.template.overwrite: true` is what makes an edit to `filebeat.yml` take
effect at all. Without it Filebeat sees a template of that name already there
and leaves it alone, and a changed pattern silently stops matching the index
being written — which drops the whole thing to dynamic mapping, where `stage`
arrives as `text` and the panels grouping on it go quiet.

## What is shipped, and what is not

Only this project's processes. Postgres, SeaweedFS, Redis and Elasticsearch
itself keep the `json-file` driver and are read with `make logs`.

Collecting those too would mean reading the engine's own log store, which is
in a different place under Docker and Podman and, on macOS, inside a virtual
machine a bind mount cannot see. One more input in `filebeat.yml` is all it
takes once that path is known for a given machine.

```bash
make logs-shipper    # why nothing is arriving, when nothing is arriving
```

## Tests

```sh
poetry run pytest tests/unit/telemetry
```

| File | Covers |
|---|---|
| [`tests/unit/telemetry/test_logs.py`](../tests/unit/telemetry/test_logs.py) | The JSON line the log shipper reads: the ECS names, the bound fields, and an exception written whole |
| [`tests/unit/telemetry/test_evaluations.py`](../tests/unit/telemetry/test_evaluations.py) | The annotation a verdict becomes, the address it is posted to from either side, and a run surviving a Phoenix that is down |
| [`tests/static/test_dashboards.py`](../tests/static/test_dashboards.py) | The provisioned dashboards against the datasources and fields that serve them |
| [`tests/static/test_log_fields.py`](../tests/static/test_log_fields.py) | The other direction: every field the code binds against what the shipper declares |

## Known edges

Things that are true, are not bugs, and have surprised somebody.

- **Nothing is deleted until `make logs-retention` has run once.** That is the
  state the stack ships in, deliberately.
- **`make logs-retention` never deletes a file.** It ages the data stream.
  The files are `make logs-prune`, and until that runs every container the
  stack has ever recreated, and every host command ever run, still has its
  log file on disk.
- **A host command leaves a file per invocation.** `make parse-status` takes
  a second and writes `parsing-<host>-<pid>.log` for it. That is the price
  of a name no two writers share; `make logs-prune` is the sweep.
- **`LOG_DIR` unset means stdout only.** That is what a process given
  neither `.env` nor compose gets — a bare `python -m …` in a shell that
  sourced nothing, and pytest.
- **Editing `filebeat.yml` without `setup.template.overwrite` changes
  nothing.** Filebeat leaves an existing template alone.
- **A new bound field needs adding to `append_fields`.** Otherwise it arrives
  as `text` and nothing can group on it.
  [`test_log_fields.py`](../tests/static/test_log_fields.py) is what says so
  before the pipeline does.
- **A declared field reaches the template at once and the index at
  rollover.** A data stream's backing index keeps the mapping it was created
  with, so today's `llm.cost_usd` is the `float` dynamic mapping gave it and
  the `double` in `filebeat.yml` applies to the next backing index.
  Harmless here; not harmless for a field that arrived as `text`, which
  needs a rollover before a panel can group on it.
- **The Elasticsearch heap is shared with Argilla.** One node serves both;
  `ES_JAVA_OPTS` is where to raise it.
