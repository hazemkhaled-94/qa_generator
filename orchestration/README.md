# Orchestration

The Dagster code location: one asset per stage, a job, a schedule and two
sensors. It calls the API and nothing else.

Dagster is the "somebody" the queue model reserves a slot for. It **decides
when a stage should run and never runs one**: it posts to the stage routes and
polls `/status` until nothing claimable is left, which is the same surface the
Start button and the `make` targets use.

`orchestration/` is **additive**. Delete it and the pipeline runs exactly as
before, with nobody deciding for it.

## The asset graph

One asset per stage, chained in the order a document moves:

```
parsed_documents → passages → facts → topics → questions
```

Materialising one starts that stage and waits for the workers to drain it.

Failed rows are an **asset check** rather than an exception, because a failed
row is not a failed run — the rest of the corpus went through, the reason is
recorded against the row, and `retry` is what moves it. Raising would stop
every later stage because one document of four hundred was a scanned image.

`topics` is written out separately rather than built by the factory, because a
fit is asked for rather than begun: there are no `new` rows to start.

`questions` is the slowest asset by a distance, and the one most likely to
reach `ORCHESTRATION_DRAIN_TIMEOUT_SECONDS`.

## Three ways to set it going, all off until asked

| | |
|---|---|
| The `corpus` job | Every stage, end to end. Run it by hand from the UI |
| The `nightly_corpus` schedule | 02:00 UTC. A refit is corpus-wide and goes stale on every new document, which is the one thing worth a clock |
| The `arrivals` sensor | Polls parsing every minute and requests a run when documents are sitting `new`. This is what makes the pipeline unattended |

The schedule and the sensor both ship **stopped**, and are switched on in the
UI. A stack that starts running the pipeline the moment it comes up is one
nobody chose.

### A fourth that only watches, and ships running

| | |
|---|---|
| The `progress` sensor | Polls every stage's `/status` every 30 seconds and files an `AssetMaterialization` whenever a stage's worked count has risen |

Without it Dagster saw only its **own** runs. Every other way of setting a
stage going — `make extract`, `POST /extraction/start`, the Start button, a
worker draining a queue somebody else filled — moved rows and left the
asset graph saying "never materialised", which is the one reading of that
graph nobody should have to qualify.

So an asset is no longer only a trigger. What Dagster shows is the
pipeline, not the subset of it Dagster ran.

This one ships **running**, unlike the other three, and the difference is
the point: those three *decide that work should happen*, and a deployment
should opt into that. This one only watches, and **a watcher nobody
switched on is a graph that is quietly wrong**.

Its cursor holds the last count seen per stage, so a tick that finds
nothing new reports nothing. The first tick after an empty cursor records
every stage that has produced anything, which is how a graph that has been
running for weeks without this catches up in one poll.

`make up` starts the webserver and the daemon with everything else, and the UI
is at <http://localhost:3000>. Starting them costs nothing on its own: both
triggers ship stopped, so the webserver serves an asset graph and the daemon
ticks nothing until one is switched on.

## Why the stages did not move into Dagster

Deliberate. The watch loop, the lease sweep and `FOR UPDATE SKIP LOCKED` are
the execution model and they work; moving the stages into Dagster ops would
have thrown all three away in exchange for putting a 473-second model call
inside an op.

## It cannot reach the database

It runs in an image of its own — **372 MB** against the backend image's
3.02 GB — because it makes HTTP calls and needs neither torch nor spaCy nor a
model.

It is given **no `DATABASE_URL` and no object store credentials**, and
`tests/smoke/test_compose.py` pins that. A process that cannot reach the
application tables cannot grow a second way of moving a row that disagrees
with the other three.

## Running it

```bash
make logs-orchestration   # follow the webserver and the daemon
make dagster-dev          # the same code location on the host, against the
                          #   containerised PostgreSQL and API
```

## Tools, and where each is used

| Tool | Where | Why this one |
|---|---|---|
| **Dagster** | [`__init__.py`](__init__.py), [`stages.py`](stages.py) | Assets, checks, a schedule and two sensors, with a UI and a run history, in one dependency |
| **requests** | [`client.py`](client.py) | The stage routes. The only thing this package talks to |
| **telemetry** | [`__init__.py`](__init__.py) | The same logging and tracing every other process installs, so its lines land in the same index under the same field names and its spans join the trace the API continues |

## Configuration

From [`configs/env/orchestration.env`](../configs/env/orchestration.env),
which is in git:

| Setting | Default | What it does |
|---|---|---|
| `ORCHESTRATION_DRAIN_TIMEOUT_SECONDS` | 28800 | How long an asset waits for a stage to drain. Giving up is not failing the rows |
| `ORCHESTRATION_POLL_SECONDS` | 15 | How often it asks |

Plus `BACKEND_URL`, which is the only address it holds.

The instance and its one code location are configured in
[`configs/dagster/`](../configs/dagster/).

## Tests

```sh
poetry run pytest tests/unit/orchestration
```

| File | Covers |
|---|---|
| [`test_definitions.py`](../tests/unit/orchestration/test_definitions.py) | The code location, as Dagster loads it: the assets, the job, and that both triggers ship stopped |
| [`test_orchestration_client.py`](../tests/unit/orchestration/test_orchestration_client.py) | The stage routes as the orchestrator reads them |
| [`test_sensor.py`](../tests/unit/orchestration/test_sensor.py) | When the orchestrator decides the pipeline should start |
| [`tests/smoke/test_compose.py`](../tests/smoke/test_compose.py) | That the orchestrator holds no database credential |

None of these reaches a network.

## Known edges

Things that are true, are not bugs, and have surprised somebody.

- **A timeout is not a failure.** An asset that gives up waiting leaves every
  row exactly where it was; the workers carry on draining.
- **A failed row does not fail the run.** It is an asset check. That is what
  stops one scanned PDF halting the corpus.
- **The two triggers ship stopped and stay stopped across a restart** until
  somebody switches them on in the UI. The `progress` sensor is the
  exception and ships running: it starts nothing.
- **A materialisation no longer means Dagster ran it.** `progress` files
  one for work any surface set going, which is the point — but it does
  mean the graph is a record of the pipeline rather than of this
  orchestrator.
- **Deleting this directory changes nothing about the pipeline.** Every stage
  still answers its route, its `make` target and its Start button.
- **The orchestrator cannot fix a row.** It has no database access by design,
  so `retry` is a route call like everything else it does.
