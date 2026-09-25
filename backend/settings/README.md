# Settings

Reading configuration: the environment, the overrides stored over it, what may
be configured, and what a change stales.

```
configs/env/*.env, .env   ──▶  the default, and required
        service_settings  ──▶  the override, if a row exists
```

The files stay required. A variable missing from both **stops the service at
start-up naming itself** — there are no defaults in code. An override is a
row in `service_settings`, and deleting that row returns a setting to
whatever the file says.

`tests/static/test_settings_documented.py` moves that start-up failure to the
pull request.

## When a change takes effect

A worker picks a change up **on the row it claims next**. The check sits
between drains, where nothing is claimed, so a change is never applied
halfway through an item.

Nothing is requeued. If the change means what a stage already produced was
made under the old value, whichever surface you used **says so and names the
stage**, and that stage's `-rerun` rebuilds it. Every fact, topic and
question records the configuration it was produced under.

That is what `invalidates` in the catalogue is for. It names the stages whose
stored output this setting produced — direct staleness only, since naming
`parsing` names the passages, facts and questions that cascade from it.

## The catalogue

[`catalog.py`](catalog.py) is the list of what may be configured: each
setting's service, its type, its bounds, the closed set of values where there
is one, one line of help, and what it stales.

One table rather than one per `config.py`, because the spaCy pipelines, the
worker poll and the pool sizes are read where they are used rather than by
one service.

Two directions are enforced by `tests/static/`:

| | |
|---|---|
| A setting the code reads and the catalogue does not describe | cannot be configured at all |
| A setting the catalogue describes and nothing reads | is a control that does nothing |

Neither is allowed to merge. That pair is why the Configuration panel on
every page holds no list of settings: adding one to `catalog.py` adds it to
the page.

### The seven services

| Service | Configured on | Holds |
|---|---|---|
| `ingestion` | Upload | The version stamp, the size limit, the allowlist |
| `parsing` | Documents | The converter's thresholds and modes |
| `chunking` | Passages | How a passage is cut, and what counts its tokens |
| `extraction` | Facts | What a fact may be, and the caps on each kind |
| `topics` | Topics | The fit: how many topics, how many passes, the vocabulary bounds |
| `questions` | Questions | The plan, the deal, the gates, the release |
| `platform` | System health | The model, the tokenizer and the language pipelines the six share |

The pool sizes and the addresses are read before a service could ask a
database for anything, so they are marked `fixed`: served read-only and shown
as the deployment's.

## The three surfaces

All three go through the **same validation** — the stage's own
`Settings.load` — so what you are refused with is the message the worker
would have failed at start-up with.

```bash
# The page that runs the service, under Configuration.

# The API, one service per request:
curl -s localhost:8000/settings/topics
curl -s -X PATCH localhost:8000/settings/topics \
  -H 'content-type: application/json' \
  -d '{"values": {"TOPIC_PASSES": "20"}}'

# A terminal:
make settings SERVICE=topics
make settings-set SERVICE=topics SET="TOPIC_PASSES=20"
make settings-unset SERVICE=topics UNSET="TOPIC_PASSES"
```

A `null` value returns one setting to what the files say. **The whole request
is refused if any one value is**, so a PATCH never lands half applied.

`GET /settings/{service}` answers, per setting: its value, what the files say
it would be, its type, its bounds, and whether somebody changed it.

## Precedence

Highest first:

1. A variable on a `make` command line — `make topics-discover TOPIC_PASSES=20`
2. A row in `service_settings`
3. `.env`
4. `configs/env/backend.env`, and the tool file for a tool that has one

The Makefile re-applies command-line overrides after sourcing the files,
because sourcing would otherwise overwrite them.

## A version is not a run

`settings_version` names a **configuration**: a digest of the stored
overrides, `environment` when nothing is overridden. It answers "what was
this row produced under".

It cannot answer "which run produced it". Run a stage twice without touching
a setting and both sets of rows carry the same version — which is the
position anybody comparing two runs is in, because an A/B usually changes a
model and nothing stored.

So [`runs.py`](runs.py) names the run, and `facts`, `questions` and `topics`
each carry a `run_id` beside their `settings_version`:

```bash
make questions RUN_ID=b-gemma4-12b      # name it
make questions                          # or let it mint a uuid
make questions-runs                     # what there is
make questions-diff RUNS="a b"          # the two, gate by gate
```

Per **process**, not per drain: a `--watch` worker drains whenever something
arrives, and an id per drain would be thousands of them in a week.

Unlike the version, it is not threaded through the factories. A version can
change while a process runs — that is what `reloading` in `stages/cli.py`
watches for — and a run cannot.

## Configuration

This package configures nothing of its own. See
[docs/configuration.md](../../docs/configuration.md) for the files and the
settings most likely to need changing.

[`env.py`](env.py) holds the readers — `required`, `optional`, `integer`,
`decimal`, `boolean`, `csv`, `mapping` — and their names are what the static
tests scan for. A setting read through a variable rather than a literal is
invisible to that scan.

## Tests

```sh
poetry run pytest tests/unit/settings tests/static/test_settings_catalogued.py
```

| File | Covers |
|---|---|
| [`test_env.py`](../../tests/unit/settings/test_env.py) | Reading configuration out of the environment, and every refusal |
| [`test_source.py`](../../tests/unit/settings/test_source.py) | Reading a setting from somewhere other than the process environment |
| [`test_configs.py`](../../tests/unit/settings/test_configs.py) | Each stage's settings, read out of the environment |
| [`test_settings_documented.py`](../../tests/static/test_settings_documented.py) | Every setting the code reads is named in a file that declares it |
| [`test_settings_catalogued.py`](../../tests/static/test_settings_catalogued.py) | Every setting the code reads is described, and every description is read |
| [`test_service_settings.py`](../../tests/integration/database/test_service_settings.py) | The settings a deployment changed, and what a stage then reads |
| [`test_reloading.py`](../../tests/integration/database/test_reloading.py) | A worker picks up a changed setting without being restarted |
| [`test_settings.py`](../../tests/integration/api/test_settings.py), [`test_settings_cli.py`](../../tests/integration/api/test_settings_cli.py) | The same change, over HTTP and from a terminal |

## Limits

- **Changing a setting does not requeue anything.** It says what went stale
  and names the stage; the `-rerun` is yours to run.
- **A `fixed` setting is visible and unwritable.**
- **A PATCH is all-or-nothing.** One invalid value refuses the whole request.
- **A setting read through a variable is invisible to the static scan.** The
  pool sizes and `LOG_DIR` are read that way and checked by hand.
- **`make settings SERVICE=…` reads the same store the UI writes.**
