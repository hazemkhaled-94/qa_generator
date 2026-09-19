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
api, 5 workers, streamlit  ──▶  logs volume  ──▶  filebeat  ──▶  elasticsearch  ──▶  grafana
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

### The file, and why it is per container

A worker writes to `{stage}-{container}.log` on a shared volume rather than to
one file per stage, because a scaled stage runs several containers over one
volume and **two processes rotating one file take each other's lines with
them**.

Files rotate at 50 MB, three kept; the shipper has read a line long before it
is deleted.

`LOG_DIR` is what turns the file on. It is set for the containers and unset
for every `make` target, so a host command logs to the terminal and nowhere
else.

## Traces

OpenTelemetry, to Phoenix. A span per unit of work, annotated with the same
names the log fields use.

The API attaches tracing to the app instance **after it exists**, so a
caller's trace continues there rather than a new one beginning. The
orchestrator installs the same configuration, so its spans join the trace the
API continues.

`trace.id` on every log line is what joins the two: a slow extraction is a
span in Phoenix and a set of lines in Grafana, found from either end.

## Tools, and where each is used

| Tool | Where | Why this one |
|---|---|---|
| **logging** (stdlib) | [`logs.py`](logs.py) | A `LogRecord` factory is the one place trace ids can be added to every line, including a library's |
| **OpenTelemetry** | [`traces.py`](traces.py) | The spans, and the exporter Phoenix reads |
| **Filebeat** | [`configs/filebeat/`](../configs/filebeat/) | Reads the volume and writes the data stream. Runs as its own container |

The trace fields are read off the current span directly rather than through
`opentelemetry-instrumentation-logging`, which does not populate them in the
pinned version.

## Configuration

| Setting | Where | Default | What it does |
|---|---|---|---|
| `LOG_LEVEL` | `.env` | `INFO` | Every service, the frontend included. Everything at or above it reaches Grafana |
| `LOG_DIR` | `compose.yaml` | unset on the host | Where the JSON file goes. Unset means stdout only |
| `OTEL_CONTAINER_ENDPOINT` | `.env` | `http://phoenix:4317` | Trace collector, as the containers reach it |

`service.name` is not a setting. Each process passes its own name to
`telemetry.configure(...)` — `"api"`, `"orchestration"`, or the stage's name
for a worker — so a process cannot be misconfigured into logging under
another's identity.

`LOG_DIR` is read through a variable rather than a literal, so the static
settings scan cannot see it. It is checked by hand.

## Retention

```bash
make logs-retention                        # 30 days
make logs-retention LOGS_RETENTION_DAYS=90
```

Run **once** against a running stack, after the shipper has written something.
Elasticsearch remembers it, and every backing index the stream rolls over to
afterwards inherits it. Until it has run **nothing is deleted**, which is the
state the stack ships in.

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
| [`tests/static/test_dashboards.py`](../tests/static/test_dashboards.py) | The provisioned dashboards against the datasources and fields that serve them |

## Known edges

Things that are true, are not bugs, and have surprised somebody.

- **Nothing is deleted until `make logs-retention` has run once.** That is the
  state the stack ships in, deliberately.
- **A `make` target logs to the terminal and nowhere else.** `LOG_DIR` is
  unset on the host, so a host drain leaves no line in Grafana.
- **Editing `filebeat.yml` without `setup.template.overwrite` changes
  nothing.** Filebeat leaves an existing template alone.
- **A new bound field needs adding to `append_fields`.** Otherwise it arrives
  as `text` and nothing can group on it.
- **The Elasticsearch heap is shared with Argilla.** One node serves both;
  `ES_JAVA_OPTS` is where to raise it.
