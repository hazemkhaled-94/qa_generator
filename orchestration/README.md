# Orchestration

The Dagster code location: one asset per stage, a job, a schedule and two
sensors. It calls the API and nothing else.

Dagster **decides when a stage should run and never runs one**: it posts to
the stage routes and polls `/status` until nothing claimable is left, which
is the same surface the Start button and the `make` targets use.

`orchestration/` is **additive**. Delete it and the pipeline runs exactly as
before, with nobody deciding for it.

## The asset graph

```
parsed_documents → passages → facts → topics → questions
```

Materialising one starts that stage and waits for the workers to drain it.

Each asset is in a **group of its own, numbered by its stage**:
`stage_2_parsing` through `stage_7_assessment`. Dagster sorts both its
asset list and its group list by name, and the assets alphabetically are
`assessments, facts, parsed_documents, passages, questions, topics` — which
is not the order a corpus moves through them. The number is the same one
Phoenix's projects and Argilla's datasets carry;
[`telemetry/pipeline.py`](../telemetry/pipeline.py) holds it once.

The asset **keys** are unchanged, so every materialisation recorded so far
still belongs to the asset that produced it.

Failed rows are an **asset check** rather than an exception: a failed row is
not a failed run — the rest of the corpus went through, the reason is
recorded against the row, and `retry` is what moves it. Raising would stop
every later stage because one document of four hundred was a scanned image.

`topics` is written out separately rather than built by the factory, because
a fit is asked for rather than begun. `questions` is the slowest asset by a
distance, and the one most likely to reach
`ORCHESTRATION_DRAIN_TIMEOUT_SECONDS`.

## Three ways to set it going, all off until asked

| | |
|---|---|
| The `corpus` job | Every stage, end to end. Run it by hand from the UI |
| The `nightly_corpus` schedule | 02:00 UTC. A refit is corpus-wide and goes stale on every new document |
| The `arrivals` sensor | Polls parsing every minute and requests a run when documents are sitting `new` |

The schedule and the sensor both ship **stopped** and are switched on in the
UI. A stack that starts running the pipeline the moment it comes up is one
nobody chose.

### A fourth that only watches, and ships running

| | |
|---|---|
| The `progress` sensor | Polls every stage's `/status` every 30 seconds and files an `AssetMaterialization` whenever a stage's worked count has risen |

Without it Dagster saw only its **own** runs. Every other way of setting a
stage going — `make extract`, `POST /extraction/start`, the Start button, a
worker draining a queue somebody else filled — moved rows and left the asset
graph saying "never materialised".

It ships running, unlike the other three, because those three *decide that
work should happen* and this one only watches: a watcher nobody switched on
is a graph that is quietly wrong.

Its cursor holds the last count seen per stage, so a tick that finds nothing
new reports nothing. The first tick after an empty cursor records every stage
that has produced anything.

`make up` starts the webserver and the daemon with everything else, and the
UI is at <http://localhost:3000>. Both triggers ship stopped, so the daemon
ticks nothing until one is switched on.

## Why the stages did not move into Dagster

The watch loop, the lease sweep and `FOR UPDATE SKIP LOCKED` are the
execution model and they work. Moving the stages into Dagster ops would have
thrown all three away in exchange for putting a 473-second model call inside
an op.

## It cannot reach the database

It runs in an image of its own — 372 MB against the backend image's 3.02 GB —
because it makes HTTP calls and needs neither torch nor spaCy nor a model.

It is given **no `DATABASE_URL` and no object store credentials**, and
`tests/smoke/test_compose.py` pins that. A process that cannot reach the
application tables cannot grow a second way of moving a row.

## Running it

```bash
make logs-orchestration   # follow the webserver and the daemon
make dagster-dev          # the same code location on the host, against the
                          #   containerised PostgreSQL and API
```

## Configuration

From [`configs/env/orchestration.env`](../configs/env/orchestration.env):

| Setting | Default | What it does |
|---|---|---|
| `ORCHESTRATION_DRAIN_TIMEOUT_SECONDS` | 28800 | How long an asset waits for a stage to drain. Giving up is not failing the rows |
| `ORCHESTRATION_POLL_SECONDS` | 15 | How often it asks |

Plus `BACKEND_URL`, the only address it holds. The instance and its one code
location are configured in [`configs/dagster/`](../configs/dagster/).

## Tests

```sh
poetry run pytest tests/unit/orchestration
```

| File | Covers |
|---|---|
| [`test_definitions.py`](../tests/unit/orchestration/test_definitions.py) | The code location as Dagster loads it: the assets, the job, and that both triggers ship stopped |
| [`test_orchestration_client.py`](../tests/unit/orchestration/test_orchestration_client.py) | The stage routes as the orchestrator reads them |
| [`test_sensor.py`](../tests/unit/orchestration/test_sensor.py) | When the orchestrator decides the pipeline should start |
| [`tests/smoke/test_compose.py`](../tests/smoke/test_compose.py) | That the orchestrator holds no database credential |

None of these reaches a network.

## Limits

- **A timeout is not a failure.** An asset that gives up waiting leaves every
  row where it was; the workers carry on draining.
- **A failed row does not fail the run.** It is an asset check.
- **The two triggers ship stopped and stay stopped across a restart.** The
  `progress` sensor is the exception and ships running: it starts nothing.
- **A materialisation no longer means Dagster ran it.** `progress` files one
  for work any surface set going.
- **Deleting this directory changes nothing about the pipeline.**
- **The orchestrator cannot fix a row.** It has no database access by design.
