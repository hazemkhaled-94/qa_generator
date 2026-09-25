# Tests

Eleven directories, eight markers. `make test` runs everything that gates a
merge; the three that do not are excluded from it.

```bash
make test           # everything that gates; starts containers of its own
make test-fast      # only the fast layers: no spaCy, no pyright, no containers
make test-unit      # everything but the integration, smoke, eval and perf layers
make test-integration
make test-e2e
make test-smoke     # build both images and look inside them
make test-eval      # score the served model against the golden passages
make test-perf      # time the ceilings the code names in a comment
make test-coverage  # the gating layers, with a coverage report
make mutation       # change a gate and ask whether a test notices. Hours
```

## The layers

| Directory | What it covers | Needs |
|---|---|---|
| [`static/`](static/) | The repository against itself: settings declared where they are read, the migration chain, the schema extensions, the locks, the workflows, the dashboards, the pinned surface of three services, every prompt pinned by digest, the frontend's gate list, the two import contracts, every relative link and heading a README names, and pyright at zero | nothing |
| [`unit/`](unit/) | One module at a time, no I/O. Includes the Dagster code location, the review round trip and the experiment evaluators | spaCy, for some |
| [`property/`](property/) | Invariants over generated input, with hypothesis | spaCy, for some |
| [`regression/`](regression/) | The verdicts the checks have always reached, pinned as a table | spaCy |
| [`contract/`](contract/) | The OpenAPI surface, the paths the frontend builds, the refusals each route declares, and generated calls against every read route | a container |
| [`integration/`](integration/) | The database, the object store and the HTTP surface, against the images compose runs | a container |
| [`e2e/`](e2e/) | One document through every stage in this process, with the converter and the model stood in for | a container |
| [`frontend/`](frontend/) | Each Streamlit page against a scripted backend | nothing |
| [`smoke/`](smoke/) | Both images built and looked inside, and the compose file resolved | a container engine |
| [`eval/`](eval/) | How a real served model reads the golden passages, and whether the round-trip gate splits the golden questions | a served model |
| [`perf/`](perf/) | The ceilings the code names in a comment, as loose wall-clock budgets | nothing |

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
| `perf` | Times the ceilings the code names; **never gates** |

## A prompt is pinned, not just versioned

[`static/test_prompts_pinned.py`](static/test_prompts_pinned.py) pins every
prompt a stage sends by **digest**, under the `PROMPT_VERSION` its module
declares.

Changing a prompt fails this file, and the only way to make it pass is to
bump the version and write the new digest down — which is the decision the
version exists to record. The digest and not the text, because the prompts
run to hundreds of lines, spelled the way `stages/prompts.py` spells it.

## Two checks that are a tool rather than a test

Both run in `make test` through a thin wrapper in `static/`, and both have a
target of their own that prints the finding itself:

| | Runs | Covers |
|---|---|---|
| `make lint-imports` | import-linter, over [`.importlinter`](../.importlinter) | A stage sits above what it shares, and no backend service imports another |
| `make deps` | deptry | A declared dependency nothing imports, an import nothing declares, and a package reached only through somebody else's |

`.importlinter` does **not** replace
[`static/test_api_stays_light.py`](static/test_api_stays_light.py), and
cannot: grimp counts an import inside a function body, and deferring a heavy
import into one is exactly how the topic service keeps pyLDAvis out of the
api.

## Generated calls

[`contract/test_openapi_conformance.py`](contract/test_openapi_conformance.py)
reads the published document and sends calls derived from it, checking each
answer back against what was promised. Four checks only — a 5xx, an
undeclared status, a body that does not match its schema, and a media type
that does not.

`unsupported_method` is off: `POST /questions/{action}` is a real route, so a
POST to `/questions/quality` genuinely matches it and is refused 422.

It found two classes of real defect on its first run, now fixed in
[`backend/api/params.py`](../backend/api/params.py): an id or offset above
`bigint` reaching PostgreSQL, and a NUL byte in a search term reaching
psycopg. Both are now 422 from FastAPI's own validation.

## Containers

The integration layers start a PostgreSQL and a SeaweedFS of their own
through testcontainers and **skip, with a reason**, where no container engine
answers. Nothing they do touches a running stack.

## What never gates

`tests/eval/` prints its numbers and asserts almost nothing: a model's
answers move between versions and between runs at the same temperature.

`tests/perf/` does assert, but never on a pull request — a wall-clock budget
on a shared runner measures the runner. It runs nightly and by hand.

The one thing `eval` does assert is that the round-trip gate splits its
golden questions the right way round. That is not a measurement of the
model's taste; it is whether the gate is wired up at all, and **a gate that
accepts everything cannot be told from no gate**.

## Two tests that skip rather than download 2.2 GB

- [`unit/nlp/test_embedding.py`](unit/nlp/test_embedding.py) skips unless
  `EMBEDDING_MODEL` is already in the Hugging Face cache.
- [`e2e/test_pipeline.py`](e2e/test_pipeline.py) stands the embedder in for
  with a digest.

## How the tests are written

Non-UI tests follow the same **page-object** structure the UI tests do: one
*driver* per thing under test, holding the wiring out of the test body.

| File | Holds |
|---|---|
| [`conftest.py`](conftest.py) | The environment every test runs in, and the containers some need |
| [`factories.py`](factories.py) | Builders the tests share |
| [`drivers.py`](drivers.py) | The objects the facts tests drive the extraction service through |
| [`integration/seed.py`](integration/seed.py) | Rows to test against, with only the columns the schema insists on |
| [`frontend/pages.py`](frontend/pages.py) | One page object over every view, and the answers each view needs |

Only the database, the buckets and the served model are ever stood in for.

## Continuous integration

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs on every pull
request and every push to `main`:

| Job | Runs |
|---|---|
| `setup` | Installs the dependencies and caches the virtualenv |
| `static` | The `static/` layer, pyright included |
| `unit` | Everything that needs neither a container nor a served model |
| `integration` | The container layers, against pulled images |
| `smoke` | Builds both images and looks inside them |
| `coverage` | Combines the three jobs' shares into one figure |
| `gate` | Waits for the rest. **This is the one a branch protection rule needs to require** |

[`.github/workflows/nightly.yml`](../.github/workflows/nightly.yml) runs at
03:00:

| Job | Runs |
|---|---|
| `audit` | Known advisories against both locks |
| `everything` | Every layer, images included |
| `mutation` | `make mutation`, with the survivors in the job summary |
| `ceilings` | `make test-perf` |
| `evaluate` | The model evaluation — skips unless `LLM_MODEL` and `LLM_BASE_URL` are repository variables |

## Limits

- **A skipped integration test is not a passing one.** With no container
  engine the whole layer skips with a reason. Read the reason.
- **`make test` excludes `smoke`, `eval` and `perf`.**
- **`tests/contract/openapi.json` is a committed fixture.** A route added
  without updating it fails the contract test.
- **`make test-fast` skips spaCy, pyright and every container.** It is a
  smoke-check while editing.
- **Coverage figures quoted in a service README are a snapshot.**
  `make test-coverage` is what re-measures.
