# Tests

Ten directories, seven markers. `make test` runs everything that gates a
merge; the two that do not are excluded from it.

```bash
make test           # everything that gates; starts containers of its own
make test-fast      # only the fast layers: no spaCy, no pyright, no containers
make test-unit      # everything but the integration, smoke and eval layers
make test-integration
make test-e2e
make test-smoke     # build both images and look inside them
make test-eval      # score the served model against the golden passages
make test-coverage  # the gating layers, with a coverage report
```

## The layers

| Directory | What it covers | Needs |
|---|---|---|
| [`static/`](static/) | The repository against itself: settings declared where they are read, the migration chain, the extensions the schema needs, the locks, the workflows, the provisioned dashboards against the datasources and fields that serve them, the pinned surface of three services, the frontend's gate list, and pyright at zero | nothing |
| [`unit/`](unit/) | One module at a time, no I/O. Includes the Dagster code location, the review round trip and the experiment evaluators, none of which reach a network | spaCy, for some |
| [`property/`](property/) | Invariants over generated input, with hypothesis | spaCy, for some |
| [`regression/`](regression/) | The verdicts the checks have always reached, pinned as a table | spaCy |
| [`contract/`](contract/) | The OpenAPI surface, the paths the frontend builds, and the refusals each route declares | a container |
| [`integration/`](integration/) | The database, the object store and the HTTP surface, against the images compose runs | a container |
| [`e2e/`](e2e/) | One document through every stage in this process, with the converter and the model stood in for | a container |
| [`frontend/`](frontend/) | Each Streamlit page against a scripted backend | nothing |
| [`smoke/`](smoke/) | Both images built and looked inside, and the compose file resolved: no service behind a profile, and the orchestrator holding no database credential | a container engine |
| [`eval/`](eval/) | How a real served model reads the golden passages, and whether the round-trip gate splits the golden questions. The cases are in [`evaluation/cases.py`](../evaluation/cases.py), read by this and by `make eval-score` | a served model |

## The markers

`--strict-markers` is on, so a typo is an error rather than a test that
silently never runs.

| Marker | Means |
|---|---|
| `nlp` | Loads a spaCy pipeline |
| `types` | Shells out to pyright |
| `integration` | Needs a container runtime |
| `frontend` | Runs a Streamlit view |
| `e2e` | Drives the whole pipeline in this process |
| `smoke` | Builds images and reads the compose file |
| `eval` | Scores a real served model; **never gates** |

## Containers

The integration layers start a PostgreSQL and a SeaweedFS of their own through
testcontainers and **skip, with a reason**, where no container engine answers.
Nothing they do touches a running stack.

## What never gates, and why

`tests/eval/` prints its numbers and asserts almost nothing. A model's answers
move between versions and between runs at the same temperature, so a threshold
there would fail on somebody else's Tuesday rather than on a regression.

The one thing it does assert is that the round-trip gate splits its golden
questions the right way round — not a measurement of the model's taste, but
whether the gate is wired up at all. **A gate that accepts everything cannot
be told from no gate.**

## Two tests that skip rather than download 2.2 GB

- [`unit/questions/test_embedding.py`](unit/questions/test_embedding.py) skips
  unless `EMBEDDING_MODEL` is already in the Hugging Face cache.
- [`e2e/test_pipeline.py`](e2e/test_pipeline.py) stands the embedder in for
  with a digest.

## How the tests are written

Non-UI tests follow the same **page-object** structure the UI tests do: one
*driver* per thing under test, exposing the operations a reader cares about
and holding the wiring out of the test body.

| File | Holds |
|---|---|
| [`conftest.py`](conftest.py) | The environment every test runs in, and the containers some need |
| [`factories.py`](factories.py) | Builders the tests share |
| [`drivers.py`](drivers.py) | The objects the facts tests drive the extraction service through |
| [`integration/seed.py`](integration/seed.py) | Rows to test against, with only the columns the schema insists on |
| [`frontend/pages.py`](frontend/pages.py) | One page object over every view, and the answers each view needs |

Only the database, the buckets and the served model are ever stood in for. The
checks, the readers and the orderings are the real code.

## Continuous integration

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs on every pull
request and on every push to `main`:

| Job | Runs |
|---|---|
| `setup` | Installs the dependencies and caches the virtualenv |
| `static` | The `static/` layer, pyright included |
| `unit` | Everything that needs neither a container nor a served model |
| `integration` | The container layers, against pulled images |
| `smoke` | Builds both images and looks inside them |
| `gate` | Waits for the rest. **This is the one a branch protection rule needs to require** |

`smoke` is the slow one. It is also the only layer that can see what an image
contains, and a lock that does not install is not worth finding out about
after the merge.

[`.github/workflows/nightly.yml`](../.github/workflows/nightly.yml) runs at
03:00 what is worth knowing but not worth blocking on: advisories against both
locks, every layer including the images, and the model evaluation — which
skips itself unless `LLM_MODEL` and `LLM_BASE_URL` are set as repository
variables.

The first run of any job installs the dependencies and caches the virtualenv
against `poetry.lock` and the Makefile. Later runs restore it.

## Known edges

Things that are true, are not bugs, and have surprised somebody.

- **A skipped integration test is not a passing one.** With no container
  engine the whole layer skips with a reason. Read the reason.
- **`make test` excludes `smoke` and `eval`.** Those are `make test-smoke` and
  `make test-eval`, and CI runs `smoke` as its own job.
- **`tests/contract/openapi.json` is a committed fixture.** A route added
  without updating it fails the contract test. That is the point.
- **`make test-fast` skips spaCy, pyright and every container.** It is a
  smoke-check while editing, not a substitute for `make test`.
- **Coverage figures quoted in a service README are a snapshot.** Nothing
  asserts them; `make test-coverage` is what re-measures.
