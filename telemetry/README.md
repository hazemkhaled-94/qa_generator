# Telemetry

Logging and tracing, configured identically in every process.

One configuration, and **every process calls it before it does anything
else** — the API, the five workers, the frontend, the orchestrator and the
host commands alike.

## Which store holds what

Four tools, and each answers a question the others cannot. The line
between the first two is the one this package draws:

| Tool | Holds | Does not hold |
|---|---|---|
| **Phoenix** | The **logic**: every stage's spans, every model call with its shape, prompt version and the work it was for, and every validation - the six question gates and extraction's eleven rules - as an annotation of its own | Services. No HTTP, no SQL, no object store, no frontend |
| **Grafana** | **Everything, literally.** Every line every process writes, per service, joined to its trace and its run | Container logs of the infrastructure, where the engine has no syslog driver; see below |
| **Argilla** | The **artefacts**, one dataset per kind. A sample by default and the whole corpus with `make review-all` | |
| **Dagster** | The **workflow**: what ran, what it produced, in what order - **whoever started it**, through the `progress` sensor | Row-level lineage. Assets are per stage |

Phoenix does not see services; it sees their inputs and outputs. A request
the frontend made, a statement the API issued and a file a worker fetched
are the system working, and the system is what the logs are for.

Two rules in [`traces.py`](traces.py) keep that line:

- **Only a run exports.** `configure(name, run=...)` builds an exporter
  only when `run` is given. A stage passes `settings.runs.run_id()`; the
  api, the frontend, the orchestrator and every host command pass nothing
  and send nothing. That is also why there is no project per API restart.
- **Only the logic is instrumented.** litellm, and the spans the stages
  open themselves. No requests, no botocore, no FastAPI — and the database
  only behind `TRACE_DATABASE`, for as long as somebody is reading it.

Every model call carries what it was **for**, not only what it was: the
stage, the passage, the topic and the run go on as metadata, read off the
same `bind` the log line beside it uses. Every **validation** is an
annotation — question generation's six gates and extraction's rejection
rules alike, each its own column, so a rule's mean over a project is its
refusal rate.

`RUN_ID` is spelled the same in all three stores, which is what lets one
run be followed across them: `questions.run_id` on the rows, a project
named `<stage>-<run id>` in Phoenix, `run.id` on every log line. The
**Run** dashboard is those three joined on one page.

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
Phoenix, and `run.id` is what ties it to the run that wrote it.

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

OpenTelemetry, to Phoenix. A span per unit of work of the **logic**,
annotated with the same names the log fields use.

`trace.id` on every log line is what joins the two: a slow extraction is a
span in Phoenix and a set of lines in Grafana, found from either end.
`run.id` joins them a level up — the whole run, rather than one call.

### A project per run

A stage passes `settings.runs.run_id()` to `configure`, which sets
`openinference.project.name` on the resource — the attribute Phoenix files
a span under. One run of one stage is then one project, `<stage>-<run id>`,
and two runs compare directly on the thing the database does not hold: the
call count, the tokens, the latency and the spend.

Named with the service as well as the run, so one pass over the pipeline is
five projects that sort together rather than one heap in which extraction's
calls and question generation's cannot be told apart.

The api, the frontend, the orchestrator and every host command pass
nothing — and **that is what turns their exporter off**, rather than
filing them under `default`. None of them is the logic Phoenix holds, and
a project per API process would be a project per restart.

Question generation also opens a span per question, carrying the gate that
stopped it **and every gate that read it** — `question.gates_ran`, in order.
The checker returns on the first failure, so the gate alone could never say
how far a question got, and two of the gates are conditional, so the
sequence cannot be derived from a verdict and a fixed order either. The
model calls that question made are its children, so a run's `leaks_source`
rejections carry the price of the calls they wasted.

### What a model call says it was

The instrumentor names every call `completion` and gives it the prompt, the
answer, the tokens and the price. Two things it cannot know go on beside
them, from `telemetry.asking`, which `llm/client.py` wraps every call in:

| On the span | Is | Why |
|---|---|---|
| `tag.tags` | the Pydantic **shape** — `_Answered`, `_NamesItsSource`, `_Recovered` | Otherwise a run is two hundred identical spans and nothing says which judgement cost what. This is `llm.shape` in the log, spelled the same |
| `llm.prompt_template.version` | the caller's **`PROMPT_VERSION`** | Two prompts are two datasets. Eight versions of the question prompt were declared before anything recorded which one a call had used |

Both are context, not a span of our own: the instrumentor's span is the one
carrying the prompt and the price, and wrapping it would be a second place
for the same call. A caller that names no version sets none — absent means
a prompt that has never been given one, which is not the same as version
zero.

The span carries the prompt **filled in**, with that call's passages in it.
What a version *asked for* — the template, and for a version the code has
moved past — is the `prompts` table; see
[`backend/stages/prompts.py`](../backend/stages/prompts.py). A span outlives
nothing: a Phoenix project is one run with a retention of its own, and the
row it explains is append-only.

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

### One annotation per gate, not one per question

A question produces a **summary** — `gate`, labelled with whatever stopped
it or `accepted` — and then **one annotation per gate that read it**, named
`gate 1: structural` through `gate 6: round_trip`:

| Annotation | On an accepted question | On one refused at phrasing |
|---|---|---|
| `gate` | `accepted`, 1.0 | `leaks_source`, 0.0 |
| `gate 1: structural` | `passed`, 1.0 | `passed`, 1.0 |
| `gate 4: near_duplicate` | `passed`, 1.0 | `passed`, 1.0 |
| `gate 5: phrasing` | `passed`, 1.0 | **`refused`, 0.0** |
| `gate 6: round_trip` | `passed`, 1.0 | *absent* |

That is what makes the Evaluations view a table of the pipeline rather than
one column: a gate's mean over a project **is** its pass rate, and two runs
compare gate by gate with nobody writing a query.

Only the gates that **ran**. A question refused at `structural` never
reached the round trip, and scoring it there either way would be untrue —
the absence is the fact, and a gate's rate is over the questions that
reached it. Scoring an unreached gate as passed would make a run that
refused everything at the first rule read as a round trip that passed
everything.

The number is the gate's fixed **position**, not its order in that
question: Phoenix sorts its columns by name, and `phrasing` alone would
sort beside `near_duplicate`, three gates earlier. `off_topic` is 2 whether
or not it ran, which for an answerable question it never does.

The last gate that ran is the one that refused it, where anything did.
That is what `check` guarantees by returning on the first failure, and it
is the only reason a rejection code can be turned back into a gate.

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
| `TRACE_DATABASE` | unset | off | A span per SQL statement and per pool connect |

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
- **The api and the frontend produce no span at all.** Not a quieter
  project — no exporter. They were 3.86M spans of which 1.2% were model
  calls. What they did is in the logs, in full.
- **An HTTP call and an S3 GET produce no span.** Neither instrumentor is
  applied any more; they were ~200k spans of a worker fetching a file.
- **SQL statements produce no span unless `TRACE_DATABASE` is set.** A
  `connect` per pooled checkout and a span per statement were 2.8M spans
  against 47k model calls. The one deliberate hole in the rule above, for
  as long as somebody is reading it.
- **`run.id` is empty on a line a run did not write.** Present and empty
  rather than absent, so there is one shape of line.
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
