# Stages — the queue, the worker and the command line

Five stages run off a queue, and all five run off *this* queue. A stage owns a
status column, an error column and the timestamp of its claim; claiming,
failing, sweeping an abandoned claim, requeueing, draining and the command
line that drives any of it are written here once.

A stage supplies the columns and the work. It supplies no queue mechanics, no
argument parser and no loop.

This is also the CLI: `make parse`, `make extract-start SHA=…` and the rest
are thin wrappers over the parser built here. See
[docs/make.md](../../docs/make.md).

## The model

**Nothing starts by itself.** A row arrives `new`, which no worker looks at.
`start` moves it to `pending`, which is the only status a worker claims, and
`stop` moves it back. A stage never sets another stage going.

A stage **selects on its own status column and on nothing else**. Chunking
never reads `parse_status`, which is what lets two stages own different
columns of the same `documents` row.

```
new  ──start──▶  pending  ──claimed──▶  running  ──▶  done
 ▲                   ▲                     │
 └──────stop─────────┤                     ├──▶  failed  ──retry──▶  pending
                     │                     └──lease expired──▶ failed
                     └────────reclaim──────┘
```

### Claiming

`SELECT … FOR UPDATE SKIP LOCKED`, one row at a time, so two workers never
take the same row:

```bash
podman compose up -d --scale extract-worker=4
```

### Leases

Every claim is timestamped, and a stage's lease says how long one may go
unfinished before another run sweeps it. A row a worker died holding is
failed by the next run of that stage with
`the worker did not finish; the run was interrupted` recorded against it, and
`retry` returns it to the queue. There is no state a row can reach that
nothing can move it out of.

A lease is **derived** from what the stage costs, never guessed. Extraction
derives its from `LLM_TIMEOUT_SECONDS` and `LLM_MAX_ATTEMPTS`, so raising
either never makes a healthy worker look abandoned. A topic's lease is
derived from `QUESTIONS_PER_TOPIC` candidates, each a writer call and a
verifier call.

### Narrowing

Every queue operation takes an optional `within`. A stage declares which
narrowings it accepts as `scopes`; the route and the command line both build
theirs through `StageQueue.narrow`, so neither learns a column name.

| Stage | Accepts |
|---|---|
| Parsing | `document` |
| Chunking | `document` |
| Extraction | `document`, `passage` |
| Question generation | `topic` |
| Topic modelling | none |

A scope a stage does not accept and a value its column cannot hold are one
exception, `Unnarrowable`, carrying both a `code` the API answers with and a
message the command line prints.

## The command line

```
python -m <stage>.run [--status|--start|--stop|--retry|--reclaim|--rerun] [--only SCOPE=VALUE] [--watch]
```

The actions are **mutually exclusive**.

| Flag | Does | Has a route |
|---|---|---|
| `--status` | Report the queue depth | yes |
| `--start` | Queue the rows never asked for | yes |
| `--stop` | Take back what has not begun | yes |
| `--retry` | Return failed rows to the queue | yes |
| `--rerun` | Queue every row again, finished ones included | yes |
| `--reclaim` | Return a row a dead worker still holds, without waiting out its lease | **no** |
| `--only SCOPE=VALUE` | Narrow the action to one item | — |
| `--watch` | Drain, and keep draining | — |
| no flag | Drain the queue once, in the foreground, and stop | — |

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

Topic modelling builds its parser from actions of its own rather than the
shared ones plus extras, because it has no `start` and no `rerun` to extend.
It has no `--reclaim` either: a fit is one request row.

### `reclaim`

`abandon` is the automatic version and stays the normal one: it sweeps on a
lease, because **a row a live worker holds must not be given to a second**.
`reclaim` is the deliberate one, for when a person knows the worker is gone
and the lease cannot.

It fills the gap between the other verbs. `retry` takes the failed, `rerun`
skips what a worker holds, and `stop` takes back only what has not begun — so
a worker killed mid-row left that row unreachable until its lease ran out,
and question generation's lease computes to 67 days at 120 questions a topic.

**Narrow it.** Nothing here can tell a dead claim from a live one:

```bash
make questions-reclaim TOPIC=473   # one topic, while a worker may be up
make questions-reclaim             # every one the stage holds. Only when it is stopped
```

### Two things done once, before the first claim

`queue_main` takes two optional callables, both run after the queue verbs and
before anything is drained.

**`preflight`** is what a stage proves before it claims anything — for the
three that call a model, that the model answers. It raises, and the process
stops with nothing taken. A watching worker retries instead of dying, backing
off from 5 seconds towards 60.

**`prompts`** records what this stage sends, so the version stamped on each
row can be resolved to the text that produced it. It runs after preflight and
never fails a start. See [`prompts.py`](prompts.py) and
[`backend/database/`](../database/README.md).

### The worker

`--watch` is what the five worker containers run: drain, sleep
`WORKER_POLL_SECONDS`, drain again. A poll rather than `LISTEN`/`NOTIFY` — at
this interval an idle worker costs one cheap query every few seconds.

`SIGTERM` and `SIGINT` are caught: the worker **finishes the item it holds**
and then stops. The wait between polls is an `Event`, so a worker asleep
wakes at once instead of serving out its interval.

A watching worker rebuilds its service when the settings change. The check
sits between drains, where nothing is claimed.

A `make` target runs one drain on the host, in the foreground, against the
same database the containers use. It is the same code; there is no second
path.

## Configuration

| Setting | What it does |
|---|---|
| `WORKER_POLL_SECONDS` | How long a watching worker sleeps between drains |

Everything else a stage needs is the stage's own. Leases are derived rather
than configured: a stage that could set its own timeout would be a stage
whose lease nobody could derive.

## Tests

```sh
poetry run pytest tests/unit/stages tests/integration/database/test_queue.py
```

| File | Covers |
|---|---|
| [`test_cli.py`](../../tests/unit/stages/test_cli.py) | Which flag combinations are refused, and that no service is built until a flag needs one |
| [`test_queue_narrowing.py`](../../tests/unit/stages/test_queue_narrowing.py) | Narrowing to one item, and both refusals |
| [`test_worker.py`](../../tests/unit/stages/test_worker.py) | Ending a worker's loop |
| [`test_queue.py`](../../tests/integration/database/test_queue.py) | Claiming, the lease sweep and the queue verbs against a real PostgreSQL |
| [`test_prompt_store.py`](../../tests/integration/database/test_prompt_store.py) | Recording a prompt once per version, and reading back what a version asked for |
| [`test_question_queue.py`](../../tests/integration/database/test_question_queue.py) | Two queues on one table |
| [`test_reloading.py`](../../tests/integration/database/test_reloading.py) | A worker picking up a changed setting without being restarted |
| [`test_stages.py`](../../tests/integration/api/test_stages.py) | The five verbs that have a route, over HTTP |

## Limits

- **`topics` carries two queues on one table.** A row with a NULL
  `topic_index` is a request to refit; a row that is a topic belongs to
  question generation. Every operation on the second carries
  `topic_index IS NOT NULL`.
- **`rerun` skips what a worker holds right now.** It is not a way to
  interrupt a running item. `reclaim` is, and it is the one verb that can
  hand a live worker's row to a second one.
- **`reclaim` has no route, on purpose.** The API cannot know a worker is
  gone.
- **A worker that cannot call its model does not start.** A watching worker
  backs off and waits.
- **A `make` target and a worker container can drain the same queue at the
  same time.** `SKIP LOCKED` makes that safe rather than forbidden.
- **Scaling `topic-worker` past one achieves nothing.**
- **A stage's failure never reaches the API as a 500.**
