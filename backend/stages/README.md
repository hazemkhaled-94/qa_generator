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

Something still has to decide that a stage should run, and there are four
things that do: a `make` target, the Start button, `POST /{stage}/start`, and
the orchestrator. Only the last can run the stages **in order**, because only
it can wait for one to drain before starting the next — see
[`orchestration/`](../../orchestration/README.md) and `POST /pipeline/run`.

A stage **claims on its own status column and on nothing else**, which is
what lets two stages own different columns of the same `documents` row.

### The one thing read beside it

Queueing is not claiming, and one stage needs the difference. A passage
exists because chunking made it and a topic because a fit did, so for four
of the five stages "this row exists" already means "the stage before me
produced it". Chunking queues over `documents`, and a document exists from
the moment somebody uploaded one.

So `start` over a corpus half way through parsing used to queue the unparsed
half, and every row of it failed on the missing parsed object — a `Start all`
on the Passages page, pressed while parsing ran, turned into a column of
failures that then needed `retry`.

`RowQueue.ready` is the answer, and chunking is the only stage that declares
one:

```python
ready = Document.parse_status == Status.PARSED
```

It is carried by `start` and `reset` and by nothing else. **It says which
rows may enter the queue, not which a worker may take** — a row already
`pending` is a row somebody queued, and it is claimed, worked, failed and
swept exactly as before whatever `ready` says.

```
new  ──start──▶  pending  ──claimed──▶  running  ──▶  done
 ▲                   ▲                   │  ▲│
 └──────stop─────────┤                   │  └┘ heartbeat, every 30s
                     │                   ├──▶  failed  ──retry──▶  pending
                     │                   └──beat missed for 5 min──▶ failed
                     └────────reclaim────┘
```

A running row is held by the **beat**, not by a time limit. The worker
refreshes `claimed_at` every `HEARTBEAT_SECONDS` for as long as it holds the
row, so the work may take as long as it takes and the lease still answers
the only question a sweep needs answered: is anyone still on it?

### Claiming

`SELECT … FOR UPDATE SKIP LOCKED`, one row at a time, so two workers never
take the same row:

```bash
podman compose up -d --scale extract-worker=4
```

### Leases and the heartbeat

Every claim is timestamped, and a worker refreshes that timestamp every
`HEARTBEAT_SECONDS` for as long as it holds the row. The lease is how many
beats may be missed before another run sweeps the claim — five minutes, or
ten of them. A row a worker died holding is failed by the next drain of that
stage with `the worker did not finish; the run was interrupted` recorded
against it, and `retry` returns it to the queue.

**One lease for every stage, because it no longer measures the work.**
`claimed_at` used to record only when work started, so a sweep could ask
nothing better than "could this still be running?" — and the honest answer
is the worst case the stage might cost. Each stage therefore derived its own
from its settings, and question generation's came to 9 days. That bound the
wrong thing: it was also how long a killed worker's topic stayed unreachable,
since `stop` moves `pending`, `retry` moves `failed` and `rerun` skips what
is held.

Beating asks "is anyone still on it?" instead, which one small constant
answers for a stage whose unit is a passage and a stage whose unit is a whole
topic alike.

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
| `--reclaim` | Return a row a dead worker still holds, without waiting out its lease | yes |
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
a worker killed mid-row is the one case none of them reaches.

Before the heartbeat this was the *only* way out, and the wait was the whole
worst case the work might have taken: question generation derived a lease of
9 days at the settings it ran under, and 67 days at a 900-second timeout. A
killed worker now loses its rows to `abandon` in five minutes, so this verb
is for when five minutes is too long to wait.

**Narrow it.** Nothing here can tell a dead claim from a live one:

```bash
make questions-reclaim TOPIC=473   # one topic, while a worker may be up
make questions-reclaim             # every one the stage holds. Only when it is stopped
```

It **has a route now**, and used not to. The argument for leaving it out was
that the API cannot know a worker is gone — but neither can the command line,
which a person runs just as blindly, so the asymmetry protected nobody and
cost anyone driving this over HTTP the single stuck row that `retry`, `rerun`
and `stop` all step over. What the danger needs is the narrowing this section
already asks for, so that is what the route enforces:

```bash
curl -X POST localhost:8000/questions/topic/473/reclaim   # one topic
curl -X POST localhost:8000/questions/reclaim             # 400, confirm it
curl -X POST 'localhost:8000/questions/reclaim?confirm=true'
```

### Two things done once, before the first claim

`queue_main` takes two optional callables, both run after the queue verbs and
before anything is drained.

**`preflight`** is what a stage proves before it claims anything — for the
three that call a model, that the model answers. It raises, and the process
stops with nothing taken. A watching worker retries instead of dying, backing
off from 5 seconds towards 60.

**It says so once.** How often a worker tries and how often it says so used
to be the same number, and around each attempt the client wrote a
traceback, instructor wrote its attempts and a credential library listed
every identity it had tried: **836 lines in four minutes, from one worker,
about one unreachable address**. The check now silences those for the
length of its own call, the first refusal is reported in full with the
reason, and the repeats are counted every ten minutes. Coming back is a
line of its own, so a log going quiet is not mistaken for a worker that
started working. Same four minutes, after: **two lines.**

The retry interval is deliberately unchanged. How fast a worker recovers
is a different question from how loud it is while it waits.

**`prompts`** records what this stage sends, so the version stamped on each
row can be resolved to the text that produced it. It runs after preflight and
never fails a start. See [`prompts.py`](prompts.py) and
[`backend/database/`](../database/README.md).

What it writes it also **publishes to Phoenix**, so a span's
`llm.prompt_template.version` opens against a prompt without anybody
remembering a command. What it *wrote* and not the catalogue:
`prompts.create` posts a new Phoenix version every call and compares
nothing, so publishing everything each run would add an identical version
per prompt per run. A start-up that changed nothing publishes nothing. See
[`publish.py`](publish.py); `make prompts-publish` is the full republish,
for a Phoenix wiped while the rows stayed.

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

Everything else a stage needs is the stage's own. The lease is neither
configured nor derived: it counts missed heartbeats, and a worker that is
alive keeps its row however long the work runs.

## Tests

```sh
poetry run pytest tests/unit/stages tests/integration/database/test_queue.py
```

| File | Covers |
|---|---|
| [`test_cli.py`](../../tests/unit/stages/test_cli.py) | Which flag combinations are refused, and that no service is built until a flag needs one |
| [`test_queue_narrowing.py`](../../tests/unit/stages/test_queue_narrowing.py) | Narrowing to one item, and both refusals |
| [`test_worker.py`](../../tests/unit/stages/test_worker.py) | Ending a worker's loop |
| [`test_prompt_recording.py`](../../tests/unit/stages/test_prompt_recording.py) | That what reaches Phoenix is what the write moved, and nothing on a start-up that changed nothing |
| [`test_prompt_publishing.py`](../../tests/unit/stages/test_prompt_publishing.py) | What a published version carries beside its text |
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
- **`reclaim` over a whole stage needs `?confirm=true`.** Neither the route
  nor the command line can tell a dead claim from a live one; narrowed to
  one item it is the ordinary way out of a killed worker.
- **`ready` gates queueing, never claiming.** Chunking is the only stage
  that declares one, and a row already `pending` is worked whatever it says.
- **A worker that cannot call its model does not start.** A watching worker
  backs off and waits.
- **A `make` target and a worker container can drain the same queue at the
  same time.** `SKIP LOCKED` makes that safe rather than forbidden.
- **Scaling `topic-worker` past one achieves nothing.**
- **A stage's failure never reaches the API as a 500.**
