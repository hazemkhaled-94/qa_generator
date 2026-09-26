# Orchestration

The Dagster code location: one asset per stage, a job, a schedule and two
sensors. It calls the API and nothing else.

Dagster **decides when a stage should run and never runs one**: it posts to
the stage routes and polls `/status` until nothing claimable is left, which
is the same surface the Start button and the `make` targets use.

`orchestration/` is **additive**. Delete it and the pipeline runs exactly as
before, with nobody deciding for it.

## The one call that comes the other way

`POST /pipeline/run` — the Run button on the Documents page, and the same
thing over HTTP — asks the api to launch the `corpus` job here. So the api
now holds this package's address, `DAGSTER_URL`, and the graph is no longer
one-way.

It is worth being exact about what that does and does not change:

- **Nothing here calls the api differently.** The assets still post to the
  stage routes and poll `/status`. The new edge reaches the *webserver's*
  GraphQL, not this package's code, and asks for a job that already exists
  by name.
- **Deleting this directory still changes nothing about the pipeline.**
  `DAGSTER_URL` then points at nothing, `GET /pipeline` answers
  `available: false`, and the Run button is disabled with a line saying why.
  Every queue verb on every surface goes on working, including the
  corpus-wide `start`, `stop` and `retry`, because those are the backend's
  own queue and never came through here.
- **What is lost without it is the waiting.** Running the stages in order
  means waiting for each to drain, and a request cannot hold that wait. The
  shell holds it for `make corpus`; a run holds it for everything else.

The queries are pinned to Dagster 1.13 in
[`backend/api/orchestrator.py`](../backend/api/orchestrator.py) and proved
against a live webserver by `tests/smoke/test_orchestrator_queries.py`,
which sends every one of them against a name that does not exist — so it
validates the documents without launching, stopping or switching anything.

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
| The `corpus` job | Every stage, end to end. Run it by hand from the UI, from `make pipeline`, from the Run button, or from `POST /pipeline/run` |
| The `nightly_corpus` schedule | 02:00 UTC. A refit is corpus-wide and goes stale on every new document |
| The `arrivals` sensor | Polls parsing every minute and requests a run when documents are sitting `new` |

The schedule and the sensor both ship **stopped**. A stack that starts
running the pipeline the moment it comes up is one nobody chose. Either can
be switched on from Dagster's own UI, from `make auto`, or from
`PUT /pipeline/automation` — which is the same switch through the api, for a
deployment whose operators never open this UI at all.

### A fourth that only watches, and ships running

| | |
|---|---|
| The `progress` sensor | Polls every stage's `/status` every 30 seconds and files an `AssetMaterialization` when a stage's worked count has risen, and an `AssetCheckEvaluation` when its failed count has moved at all |

Without it Dagster saw only its **own** runs. Every other way of setting a
stage going — `make extract`, `POST /extraction/start`, the Start button, a
worker draining a queue somebody else filled — moved rows and left the asset
graph saying "never materialised".

It ships running, unlike the other three, because those three *decide that
work should happen* and this one only watches: a watcher nobody switched on
is a graph that is quietly wrong.

**The check is what makes the graph go red.** A materialisation alone cannot:
a stage that failed every row produced nothing, so counting only what was
produced reported silence for the one state worth interrupting somebody
over. The check is filed on a move in either direction, so a `retry` that
clears the failures turns the asset green again rather than leaving it red
until a Dagster run happens to evaluate it. It is filed under
`nothing_failed`, which is the check every asset already carries — named
once in [`stages.py`](stages.py) so a rename cannot file an evaluation
against a check Dagster has never heard of.

Its cursor holds the last worked and failed counts per stage, so a tick that
finds neither moved reports nothing. The first tick after an empty cursor
records every stage that has produced or failed anything. A cursor written
before failures were watched holds one number per stage and is read as the
worked count, so an upgrade does not re-report the whole corpus.

**The six `/status` reads go out at once.** Sequentially they are six GETs at
the client's 15s timeout and Dagster gives a sensor tick 60, so a backend
answering slowly used to switch the watcher off exactly when the graph
needed it. One `Backend` per thread: a requests `Session` is a connection
pool and is not documented thread-safe.

`make up` starts the webserver and the daemon with everything else, and the
UI is at <http://localhost:3000>. Both triggers ship stopped, so the daemon
ticks nothing until one is switched on.

## One corpus run at a time

`max_concurrent_runs: 1` on a `QueuedRunCoordinator`, in
[`configs/dagster/dagster.yaml`](../configs/dagster/dagster.yaml).

The `corpus` job drives the stage queues in order, so two of them racing
through the same queues interleave: one can ask for a topic fit while the
other is still draining extraction, and the fit then describes half a
corpus. `POST /pipeline/run` refuses a second run with a 409, but that
check reads the run list and then launches — two requests can pass it
together — and it is not on the path at all for `make pipeline`, for
`dagster job launch`, or for the Materialize button in Dagster's own UI.

The 409 stays, because a second run **queued behind** one that may take
hours is not what somebody pressing Run wants to be told nothing about. It
is now the affordance and the coordinator is the guarantee.

## The run that asked, on the rows it produced

Every asset passes `context.run_id` to the stage route it starts, the
backend writes it onto the rows that verb queues, and the worker that
claims one adopts it for as long as it holds it. So a fact produced by a
Dagster run carries that **run's** id, and `GET /lineage` leads from the
fact back to it.

It used to carry the worker's. `settings.runs.run_id` was one uuid per
PROCESS and a stage worker is long-lived, so a Dagster run, the Start
button and `make extract` all produced rows stamped with the same
container, and nothing in the data said which of them had asked.

The edge is the one already there — a `?run=` on the stage routes — so this
package still holds no credential and still reaches nothing but HTTP.
Surfaces that are not Dagster get the same treatment: the API mints an id
when a caller names none and answers with it, so pressing Start is also
followable. `RUN_ID` overrides both, which is what keeps the A/B in
[`evaluation/README.md`](../evaluation/README.md) working.

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

That still holds with the api calling in. `DAGSTER_URL` points the api at
this webserver; it gives this package no credential and no reach of its own,
and the only thing asked for through it is a run of a job defined here.

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
| [`test_definitions.py`](../tests/unit/orchestration/test_definitions.py) | The code location as Dagster loads it: the assets, the job, that both triggers ship stopped, and that the instance caps the corpus at one run |
| [`test_orchestration_client.py`](../tests/unit/orchestration/test_orchestration_client.py) | The stage routes as the orchestrator reads them, and the run it carries into them |
| [`test_sensor.py`](../tests/unit/orchestration/test_sensor.py) | When the orchestrator decides the pipeline should start, and what it records for the work it did not |
| [`tests/smoke/test_compose.py`](../tests/smoke/test_compose.py) | That the orchestrator holds no database credential |

None of these reaches a network.

## Limits

- **A timeout is not a failure.** An asset that gives up waiting leaves every
  row where it was; the workers carry on draining.
- **A failed row does not fail the run.** It is an asset check.
- **The two triggers ship stopped and stay stopped across a restart.** The
  `progress` sensor is the exception and ships running: it starts nothing.
- **A materialisation no longer means Dagster ran it.** `progress` files one
  for work any surface set going, and a check evaluation beside it when the
  failures move.
- **A runless event is not a run.** Work Dagster did not launch shows on the
  asset page and never in the run list: there is no timeline and no step log
  to open, because Dagster OSS has no run to attach them to. What joins that
  work to its logs is the run id on the rows, not a Dagster run.
- **A second corpus run waits rather than racing.** `max_concurrent_runs: 1`
  queues it; it is not refused, except at `POST /pipeline/run`, which still
  answers 409 rather than leaving somebody watching a queued run.
- **Deleting this directory changes nothing about the pipeline.** It takes
  away the Run button and `POST /pipeline/run`, both of which then report
  themselves unavailable; every queue verb on every surface is untouched.
- **The orchestrator cannot fix a row.** It has no database access by design.
- **The api holds this webserver's address, not this package's code.** The
  queries it sends are pinned to Dagster 1.13 and proved against a live
  server by `tests/smoke/test_orchestrator_queries.py`.
