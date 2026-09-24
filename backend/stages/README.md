# Stages — the queue, the worker and the command line

Five stages run off a queue, and all five run off *this* queue. A stage owns a
status column, an error column and the timestamp of its claim; claiming,
failing, sweeping an abandoned claim, requeueing, draining and the command
line that drives any of it are written here once, against those declarations.

A stage supplies the columns and the work. It supplies no queue mechanics, no
argument parser and no loop.

This is also **the CLI**: `make parse`, `make extract-start SHA=…` and the
rest are thin wrappers over the parser built here. See
[docs/make.md](../../docs/make.md) for the targets.

## The model

**Nothing starts by itself.** A row arrives `new`, which no worker looks at.
`start` moves it to `pending`, which is the only status a worker claims, and
`stop` moves it back. A stage never sets another stage going: the previous
stage finishing leaves a row `new`, and somebody — the Start button, the
route, a `make` target, the [orchestrator](../../orchestration/README.md) —
decides it should run.

A stage **selects on its own status column and on nothing else**. Chunking
never reads `parse_status`. That is what lets two stages own different columns
of the same `documents` row without either knowing the other exists.

```
new  ──start──▶  pending  ──claimed──▶  running  ──▶  done
 ▲                   ▲                     │
 └──────stop─────────┤                     ├──▶  failed  ──retry──▶  pending
                     │                     └──lease expired──▶ failed
                     └────────reclaim──────┘
```

`reclaim` is the only verb that moves a `running` row, and the only one
with no route. See [below](#reclaim-and-why-it-is-not-automatic).

### Claiming

`SELECT … FOR UPDATE SKIP LOCKED`, one row at a time. Two workers never take
the same row, which is the whole of the horizontal-scaling story:

```bash
podman compose up -d --scale extract-worker=4
```

### Leases, and why every failure is recoverable

Every claim is timestamped, and a stage's lease says how long one may go
unfinished before another run sweeps it. A row a worker died holding is
**failed by the next run of that stage**, with
`the worker did not finish; the run was interrupted` recorded against it,
rather than being left claimed and invisible. `retry` then returns it to the
queue.

There is no state a row can reach that nothing can move it out of.

A lease is derived from what the stage actually costs, never guessed.
Extraction derives its from `LLM_TIMEOUT_SECONDS` and `LLM_MAX_ATTEMPTS`, so
raising either never makes a healthy worker look abandoned. A topic's lease is
derived from `QUESTIONS_PER_TOPIC` candidates, each a writer call and a
verifier call — sized any smaller, a run would fail workers only halfway
through their first topic.

### Narrowing

Every queue operation takes an optional `within`, which narrows it to part of
the queue instead of all of it. A stage declares which narrowings it accepts
as `scopes`; the route and the command line both build theirs through
`StageQueue.narrow`, so neither learns a column name.

| Stage | Accepts |
|---|---|
| Parsing | `document` |
| Chunking | `document` |
| Extraction | `document`, `passage` |
| Question generation | `topic` |
| Topic modelling | none |

A scope a stage does not accept and a value its column cannot hold are one
exception, `Unnarrowable`, carrying both a `code` the API answers with and a
message the command line prints. Two surfaces, one refusal, spelled once.

## The command line

```
python -m <stage>.run [--status|--start|--stop|--retry|--reclaim|--rerun] [--only SCOPE=VALUE] [--watch]
```

The actions are **mutually exclusive** — two of them in one command silently
ran only the first.

| Flag | Does | Has a route |
|---|---|---|
| `--status` | Report the queue depth | yes |
| `--start` | Queue the rows never asked for | yes |
| `--stop` | Take back what has not begun | yes |
| `--retry` | Return failed rows to the queue | yes |
| `--rerun` | Queue every row again, finished ones included | yes |
| `--reclaim` | Return a row a dead worker still holds, without waiting out its lease | **no** |
| `--only SCOPE=VALUE` | Narrow the action to one item, as `document=<sha256>` or `passage=<id>` | — |
| `--watch` | Drain, and keep draining | — |
| no flag | Drain the queue once, in the foreground, and stop | — |

Five of the six mirror the route beside them. **`reclaim` is the one that
does not**, deliberately: it is the only verb that moves an `in_progress`
row, and it is safe only when a person knows the worker holding that row is
gone. A button is the wrong shape for that.

The five modules that answer it:

```bash
python -m preprocessing.parsing.run --status
python -m preprocessing.chunking.run --start --only document=<sha256>
python -m extraction.run --retry --only passage=42
python -m topic_modelling.run --status
python -m question_generation.run --only topic=7
```

A stage with an operation nobody else has declares it as one `Extra` and the
parser grows a flag for it — `--revocabulary` on chunking, `--revalidate`,
`--bridge`, `--recap` and `--embed` on extraction, `--reverify` and
`--balance` on question generation.

Topic modelling is the exception. It builds the parser from a set of actions
of its own rather than from the shared ones plus extras, because it has no
`start` and no `rerun` to extend: `--discover`, `--visualise` and `--delete`
sit in the same mutually exclusive group as `--status`, `--stop` and
`--retry`. It has no `--reclaim` either — a fit is one request row, and
`--retry` after the lease is the whole of recovering it.

### `reclaim`, and why it is not automatic

`abandon` is the automatic version and stays the normal one: it sweeps on a
lease, because **a row a live worker holds must not be given to a second**.
`reclaim` is the deliberate one, and it is worth having separately because
a person knows the worker is gone and the lease cannot.

It fills the gap between the other two verbs. `retry` takes the failed,
`rerun` explicitly skips what a worker holds, and `stop` takes back only
what has not begun — so a worker killed mid-row left that row unreachable
until its lease ran out. Question generation's lease is derived from what a
topic costs and **computes to 67 days** at 120 questions a topic and a
900-second timeout, which is not a wait.

**Narrow it.** Nothing here can tell a dead claim from a live one:

```bash
make questions-reclaim TOPIC=473   # one topic, while a worker may be up
make questions-reclaim             # every one the stage holds. Only when it is stopped
```

Two workers on one row is what the lease exists to prevent, and this is the
one verb that can cause it.

### Two things done once, before the first claim

`queue_main` takes two optional callables, and both run after the queue
verbs and before anything is drained.

**`preflight`** is what a stage proves before it claims anything — for the
three that call a model, that the model answers. It raises, and the process
stops with nothing taken. A worker with no usable credential would
otherwise claim a row every poll and fail it, and the queue would empty
into `failed` while the container reported itself up. A watching worker
retries instead of dying, backing off from 5 seconds towards 60, so a stack
that came up before its model did recovers on its own.

**`prompts`** records what this stage sends, so the version stamped on each
row it writes can be resolved to the text that produced it. It runs after
preflight, so a deployment that cannot work leaves no record of prompts it
never sent, and it **never fails a start** — see
[`prompts.py`](prompts.py) and [`backend/database/`](../database/README.md).

### Why a bare drain, on the host

A `make` target runs **one drain on the host, in the foreground, against the
same database** the containerised workers use. That is what you want while
developing a stage: the traceback is in your terminal, not in a container's
log, and `Ctrl-C` stops it.

It is the same code the worker runs. There is no second path.

### `--watch` and the worker container

`--watch` is what the five worker containers run: drain, sleep
`WORKER_POLL_SECONDS`, drain again. A poll rather than `LISTEN`/`NOTIFY` — at
this interval an idle worker costs one cheap query every few seconds, and
`NOTIFY` on the status column is the upgrade if that ever matters.

`SIGTERM` and `SIGINT` are caught: the worker **finishes the item it holds and
then stops**, rather than abandoning a claim a sweep would have to clean up.
The wait between polls is an `Event`, so a worker asleep between polls wakes at
once instead of serving out the rest of its interval.

A watching worker rebuilds its service when the settings change. The check
sits **between drains**, where nothing is claimed, so a change is picked up on
the row claimed next and never halfway through one.

## Tools, and where each is used

| Tool | Where | Why this one |
|---|---|---|
| **SQLAlchemy** | [`queue.py`](queue.py) | `FOR UPDATE SKIP LOCKED`, the lease sweep and the requeue, written against `InstrumentedAttribute` so a stage passes its own columns in |
| **argparse** | [`cli.py`](cli.py) | A mutually exclusive group is exactly the constraint the five verbs need, and it is in the standard library |
| **OpenTelemetry** | [`service.py`](service.py) | One span per item, annotated `stage.name` and `stage.outcome`, so one trace query finds everything that failed whichever stage failed it |
| **signal**, **threading** | [`worker.py`](worker.py) | A graceful stop, and a sleep that a signal can cut short |

## Configuration

| Setting | What it does |
|---|---|
| `WORKER_POLL_SECONDS` | How long a watching worker sleeps between drains |

Everything else a stage needs is the stage's own. Leases are derived rather
than configured, on purpose: a stage that could set its own timeout would be a
stage whose lease nobody could derive.

## Tests

```sh
poetry run pytest tests/unit/stages tests/integration/database/test_queue.py
```

| File | Covers |
|---|---|
| [`tests/unit/stages/test_cli.py`](../../tests/unit/stages/test_cli.py) | Which flag combinations are refused, and that no service is built until a flag needs one |
| [`tests/unit/stages/test_queue_narrowing.py`](../../tests/unit/stages/test_queue_narrowing.py) | Narrowing to one item, and both refusals |
| [`tests/unit/stages/test_worker.py`](../../tests/unit/stages/test_worker.py) | Ending a worker's loop |
| [`tests/integration/database/test_queue.py`](../../tests/integration/database/test_queue.py) | Claiming, the lease sweep and the queue verbs against a real PostgreSQL |
| [`tests/integration/database/test_prompt_store.py`](../../tests/integration/database/test_prompt_store.py) | Recording a prompt once per version, and reading back what a version asked for |
| [`tests/integration/database/test_question_queue.py`](../../tests/integration/database/test_question_queue.py) | Two queues on one table, against the database that has to keep them apart |
| [`tests/integration/database/test_reloading.py`](../../tests/integration/database/test_reloading.py) | A worker picking up a changed setting without being restarted |
| [`tests/integration/api/test_stages.py`](../../tests/integration/api/test_stages.py) | The five verbs that have a route, over HTTP |

## Known edges

Things that are true, are not bugs, and have surprised somebody.

- **`topics` carries two queues on one table.** A row with a NULL
  `topic_index` is a request to refit and belongs to topic modelling; a row
  that is a topic belongs to question generation. Every operation on the
  second carries `topic_index IS NOT NULL`, because without it `start` would
  queue the asking as though it were a subject and the two workers would
  fight over one row.
- **`rerun` skips what a worker holds right now.** It queues everything else,
  finished rows included. It is not a way to interrupt a running item.
  `reclaim` is, and it is the one verb that can hand a live worker's row to
  a second one. Narrow it.
- **`reclaim` has no route, on purpose.** It is safe only when a person
  knows the worker is gone, and the API cannot know that.
- **A worker that cannot call its model does not start.** It is not a
  silent failure and not a crash loop: a watching worker backs off from 5
  seconds towards 60 and waits for the model to come up.
- **A `make` target and a worker container can drain the same queue at the
  same time.** `SKIP LOCKED` makes that safe rather than forbidden, so a
  foreground drain during development quietly shares the work with whatever
  is running.
- **Scaling `topic-worker` past one achieves nothing.** A fit is one request
  row, and only one worker can claim it.
- **A stage's failure never reaches the API as a 500.** It is recorded
  against the row and read back through `/{stage}/status`.
