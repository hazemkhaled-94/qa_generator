# Settings

Reading configuration: the environment, the overrides stored over it, what may
be configured, and what a change stales.

Every setting has two places it can come from. The **files** supply all of
them; a deployment can override one through the UI, the API or the command
line **without restarting anything**.

## The model

```
configs/env/*.env, .env   ──▶  the default, and required
        service_settings  ──▶  the override, if a row exists
```

The files stay required. A variable missing from both of them **stops the
service at start-up naming itself**, as it always did — there are no defaults
in code. An override is a row in `service_settings`, and deleting that row is
what returns a setting to whatever the file says. There is no stored copy of a
default, because the file is the copy.

`tests/static/test_settings_documented.py` moves that start-up failure to the
pull request, where the fix is one line in a file rather than a container that
will not come up.

### When a change takes effect

A worker picks a change up **on the row it claims next**. The check sits
between drains, where nothing is claimed, so a change is never applied halfway
through an item.

Nothing is requeued. If the change means what a stage already produced was
made under the old value, whichever surface you used **says so and names the
stage**, and that stage's own `-rerun` is what rebuilds it. Every fact, topic
and question records the configuration it was produced under, so a corpus
built across a change can still be read.

That is what `invalidates` in the catalogue is for. It names the stages whose
stored output this setting produced — direct staleness only: naming `parsing`
names the passages, facts and questions that cascade from it. A setting that
only shapes the next run names nothing.

## The catalogue

[`catalog.py`](catalog.py) is the list of what may be configured: each
setting's service, its type, its bounds, the closed set of values where there
is one, one line of help, and what it stales.

It is **one table rather than one per `config.py`**, because the spaCy
pipelines, the worker poll and the pool sizes are read where they are used
rather than by one service.

Two directions are enforced, both by `tests/static/`:

| | |
|---|---|
| A setting the code reads and the catalogue does not describe | cannot be configured at all |
| A setting the catalogue describes and nothing reads | is a control that does nothing |

Neither is allowed to merge. That pair is why the Configuration panel on every
page holds no list of settings: adding one to `catalog.py` adds it to the
page.

### The seven services

Each configures its own settings and no others:

| Service | Configured on | Holds |
|---|---|---|
| `ingestion` | Upload | The version stamp, the size limit, the allowlist |
| `parsing` | Documents | The converter's thresholds and modes |
| `chunking` | Passages | How a passage is cut, and what counts its tokens |
| `extraction` | Facts | What a fact may be, and the caps on each kind |
| `topics` | Topics | The fit: how many topics, how many passes, the vocabulary bounds |
| `questions` | Questions | The plan, the deal, the gates, the release |
| `platform` | System health | The model, the tokenizer and the language pipelines the six share |

### What is served but never written

The pool sizes and the addresses are read before a service could ask a
database for anything, so they are marked `fixed`: served read-only and shown
as the deployment's. Changing one needs a restart.

## The three surfaces

All three go through the **same validation** — the stage's own
`Settings.load` — so what you are refused with is the message the worker would
have failed at start-up with.

```bash
# The page that runs the service, under Configuration in its service panel.

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

1. A variable given on a `make` command line — `make topics-discover TOPIC_PASSES=20`
2. A row in `service_settings`
3. `.env`
4. `configs/env/backend.env` (and the tool file, for a tool that has one)

The Makefile re-applies command-line overrides after sourcing the files,
because sourcing would otherwise overwrite them — which is why
`make topics TOPIC_PASSES=20` once ran with the file's value and said nothing.

## Tools, and where each is used

| Tool | Where | Why this one |
|---|---|---|
| **SQLAlchemy** | [`store.py`](store.py) | The one table, and the read a stage does on the row it claims next |
| **dataclasses** | [`catalog.py`](catalog.py) | A frozen `Setting` is the whole declaration; no framework is needed to hold six fields |

[`env.py`](env.py) holds the readers — `required`, `optional`, `integer`,
`decimal`, `boolean`, `csv`, `mapping` — and their names are what the static
tests scan for. A setting read through a variable rather than a literal is
invisible to that scan, which is why the three stage-level readers that take a
name first are named explicitly in the test.

## Configuration

This package configures nothing of its own. Its subject is everything else.

See [docs/configuration.md](../../docs/configuration.md) for the files, what
each holds, and the settings most likely to need changing.

## Tests

```sh
poetry run pytest tests/unit/settings tests/static/test_settings_catalogued.py
```

| File | Covers |
|---|---|
| [`tests/unit/settings/test_env.py`](../../tests/unit/settings/test_env.py) | Reading configuration out of the environment, and every refusal |
| [`tests/unit/settings/test_source.py`](../../tests/unit/settings/test_source.py) | Reading a setting from somewhere other than the process environment |
| [`tests/unit/settings/test_configs.py`](../../tests/unit/settings/test_configs.py) | Each stage's settings, read out of the environment |
| [`tests/static/test_settings_documented.py`](../../tests/static/test_settings_documented.py) | Every setting the code reads is named in a file that declares it |
| [`tests/static/test_settings_catalogued.py`](../../tests/static/test_settings_catalogued.py) | Every setting the code reads is described, and every description is read |
| [`tests/integration/database/test_service_settings.py`](../../tests/integration/database/test_service_settings.py) | The settings a deployment changed, and what a stage then reads |
| [`tests/integration/database/test_reloading.py`](../../tests/integration/database/test_reloading.py) | A worker picks up a changed setting without being restarted |
| [`tests/integration/api/test_settings.py`](../../tests/integration/api/test_settings.py), [`test_settings_cli.py`](../../tests/integration/api/test_settings_cli.py) | The same change, over HTTP and from a terminal |

## Known edges

Things that are true, are not bugs, and have surprised somebody.

- **Changing a setting does not requeue anything.** It says what went stale
  and names the stage. The `-rerun` is yours to run.
- **A `fixed` setting is visible and unwritable.** The pool sizes and the
  addresses are read before the database exists to be asked.
- **A PATCH is all-or-nothing.** One invalid value refuses the whole request,
  so a partial application is not a state this can reach.
- **A setting read through a variable is invisible to the static scan.** The
  pool sizes and `LOG_DIR` are read that way and are checked by hand instead.
- **`make settings SERVICE=…` reads the same store the UI writes.** There is
  no separate file to keep in step.
