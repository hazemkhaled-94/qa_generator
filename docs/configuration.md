# Configuration

Every setting comes from a file, and a deployment can override one through
the UI, the API or the command line without restarting anything. The
mechanics are in [`backend/settings/`](../backend/settings/README.md); this
is the map.

**This page names no defaults.** The files below hold the values, each one
exactly once, and a table here repeating them would be a copy free to
disagree with the file — which is what it was, and did: it described
`QUESTIONS_TYPE_MIX` as "eleven kinds" long after the file had thirteen.
What a setting *is* is here; what it is *set to* is in the file named
beside it.

## The files

| File | In git | Holds |
|---|---|---|
| `configs/env/backend.env` | yes | How the pipeline behaves: the parsing, chunking, extraction, topic, question and assessment settings |
| `configs/env/deployment.env` | yes | Where this deployment put things: the ports, the container-side addresses, the topology, the role names |
| `configs/env/elasticsearch.env` | yes | The node's certificate paths, security flags and heap |
| `configs/env/seaweedfs-filer.env` | yes | Which metadata store the filer uses |
| `configs/env/orchestration.env` | yes | How long Dagster waits on a stage |
| `configs/env/review.env` | yes | How a review sample is drawn |
| `configs/env/evaluation.env` | yes | What a golden-set run is called in Phoenix |
| `configs/env/provider.env` | **no** | Whatever env vars the model provider needs |
| `.env` | **no** | The credentials, and which model this deployment calls |

**A name appears in exactly one of them.** There is no file that overrides
another, and no name to look up in two places to find out which value wins.
`tests/static/test_env_files.py` refuses a second copy.

compose hands each file to the services that need it with `env_file`, and
the Makefile sources `backend.env`, `deployment.env` and `.env` for the host
commands — so one value reaches both.

`.env` is the only file with a secret in it, and `.env.example` stands in
for it in git, every value a `change_me_*` placeholder. `make setup` copies
one to the other and generates the passwords.

## What compose reads

compose interpolates `${...}` out of **two** files, in this order:

```bash
podman compose --env-file configs/env/deployment.env --env-file .env
```

The last one wins, which is what lets a secret sit over a name. Every
compose command in the Makefile goes through `$(COMPOSE_CMD)`, which carries
both flags; `COMPOSE` stays the engine alone, so
`make COMPOSE="docker compose" up` still carries the files rather than
replacing them. A bare `podman compose` outside make sees only `.env` and
fails on the first missing port; export `COMPOSE_ENV_FILES` to it instead:

```bash
export COMPOSE_ENV_FILES=configs/env/deployment.env,.env
```

It cannot be set inside `.env` — compose has to resolve the file list before
it reads one.

## What is derived, and so written nowhere

A port copied into an address is a copy free to disagree with it, and a
password copied into a connection string is a secret in two places. These
are built from the names above rather than declared:

| Value | Built from | By |
|---|---|---|
| `DATABASE_URL` | `APP_DB_USER`, `APP_DB_PASSWORD`, `APP_DB_NAME`, `POSTGRES_PORT` | the Makefile for a host command, compose for a container |
| `BACKEND_URL` | `BACKEND_PORT` | the Makefile. The container's is `BACKEND_CONTAINER_URL` |
| `S3_ENDPOINT` | `S3_PORT` | the Makefile. The container's is `S3_CONTAINER_ENDPOINT` |
| `PHOENIX_BASE_URL` | `PHOENIX_PORT` | the Makefile, and compose for the frontend's browser links. The container's is `PHOENIX_CONTAINER_BASE_URL` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `PHOENIX_GRPC_PORT` | the Makefile. The container's is `OTEL_CONTAINER_ENDPOINT` |
| `ARGILLA_API_URL` | `ARGILLA_PORT` | the Makefile |
| `SERVICE_URLS` | every published port | compose, for the System health page |
| `SPACY_MODELS` | `NLP_MODELS` | the Makefile, which exports it so the image bakes what the workers load |

### The host and the container see different addresses

Five things have two names, and the convention is the same for all of them:
the plain name is the **host's**, and `*_CONTAINER_*` is what compose sets
over it inside the network.

| The host's | The container's |
|---|---|
| `BACKEND_URL` | `BACKEND_CONTAINER_URL` |
| `S3_ENDPOINT` | `S3_CONTAINER_ENDPOINT` |
| `PHOENIX_BASE_URL` | `PHOENIX_CONTAINER_BASE_URL` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `OTEL_CONTAINER_ENDPOINT` |
| `LLM_MODEL`, `OLLAMA_BASE_URL` | `LLM_CONTAINER_MODEL`, `OLLAMA_CONTAINER_URL` |
| `QUESTIONS_VERIFIER_MODEL` | `QUESTIONS_VERIFIER_CONTAINER_MODEL` |
| `ASSESSMENT_JUDGE_MODEL` | `ASSESSMENT_JUDGE_CONTAINER_MODEL` |

The model pair is the one that does **not** fall back. A container holds no
cloud credential and must never reach a hosted provider, so
`LLM_CONTAINER_MODEL` is required, is checked by `make doctor`, and does not
inherit `LLM_MODEL` — inheriting it is how a stale container spent 2,019
restarts failing to authenticate to Azure.

## Precedence

Highest first:

1. A variable given on a `make` command line
2. A row in `service_settings` — set through the UI, the API or `make settings-set`
3. The file that declares it

The files stay required. A variable missing from its file stops the service
at start-up naming itself; there are no defaults in code, and deleting an
override row is what returns a setting to the file.

## Changing one

Three surfaces, all going through the stage's own `Settings.load`:

```bash
# On the page that runs the service, under Configuration.

# Or the API, one service per request:
curl -s localhost:8000/settings/topics
curl -s -X PATCH localhost:8000/settings/topics \
  -H 'content-type: application/json' \
  -d '{"values": {"TOPIC_PASSES": "20"}}'

# Or a terminal:
make settings SERVICE=topics
make settings-set SERVICE=topics SET="TOPIC_PASSES=20"
make settings-unset SERVICE=topics UNSET="TOPIC_PASSES"
```

A worker picks a change up on the row it claims next. Nothing is requeued:
if the change stales what a stage already produced, the surface you used says
so and names the stage, and that stage's `-rerun` rebuilds it.

Every fact, topic and question records the configuration it was produced
under.

## The eight services

`ingestion` on Upload, `parsing` on Documents, `chunking` on Passages,
`extraction` on Facts, `topics` on Topics, `questions` on Questions,
`assessment` on Assessment, and `platform` — the model, the tokenizer and the
language pipelines — on System health. Each configures its own settings and
no others.

A stage may name a different model of its own: `EXTRACTION_MODEL`,
`TOPIC_MODEL`, `QUESTIONS_MODEL`, each meaning `LLM_MODEL` when absent. Four
more name a model for one judgement: `EXTRACTION_DIGEST_MODEL`,
`QUESTIONS_PHRASING_MODEL`, `QUESTIONS_VERIFIER_MODEL` and
`ASSESSMENT_JUDGE_MODEL`.

Pool sizes and addresses are not configurable. They are read before a service
could ask a database for anything, so they are served read-only.

## Where to look for a setting

| Looking for | Read |
|---|---|
| Which model, and where it is served | `.env`, and `.env.example` for the provider table |
| How the model is called — mode, temperature, timeout, attempts, context window, reasoning | `configs/env/backend.env`, "Extraction" |
| The encoders, the tokenizer, the spaCy pipelines | `configs/env/backend.env`, "Embedding" and "Language pipelines" |
| What a fact may be, and the caps on each kind | `configs/env/backend.env`, "Extraction" |
| How many topics, and how they are fitted | `configs/env/backend.env`, "Topic modelling", and [`backend/topic_modelling/`](../backend/topic_modelling/README.md) |
| What a run costs, and what it writes | `configs/env/backend.env`, "Question generation", and [`backend/question_generation/`](../backend/question_generation/README.md) |
| Which of the accepted questions ship | `configs/env/backend.env`, "The balanced release" |
| Whether the judge runs, and who judges | `.env`; which artefacts and how many, `configs/env/backend.env` |
| A port, an address, a role name | `configs/env/deployment.env` |
| A password or a key | `.env`, listed in `.env.example` |

Each setting in `configs/env/backend.env` carries the paragraphs explaining
what it does and how its value was arrived at;
[`settings/catalog.py`](../backend/settings/catalog.py) carries the one-line
version the UI draws its controls from. How a default was measured is in
[measurements.md](measurements.md).

## Two settings with constraints

`PHOENIX_ADMIN_SECRET` needs at least 32 characters including a digit and a
lower-case letter; `make doctor` checks all three.
`PHOENIX_DEFAULT_ADMIN_INITIAL_PASSWORD` is applied only when Phoenix first
creates its admin user; changing it afterwards means dropping the `phoenix`
database.

`MAX_FILE_SIZE_MB` is the one setting written in two files, because
Streamlit reads a static TOML and no env var: `server.maxUploadSize` in
`frontend/.streamlit/config.toml` must match it, or the frontend rejects a
file before the API sees it and tells the person the wrong limit. A test
pins the pair.

## What is enforced

| Test | Checks |
|---|---|
| `tests/static/test_env_files.py` | One name, one file. Every `${...}` compose interpolates is declared, and one with no fallback is actually set. `.env.example` holds placeholders only. The image bakes the pipelines `NLP_MODELS` loads |
| `tests/static/test_settings_documented.py` | Every setting the code reads is named in a file that declares it |
| `tests/static/test_settings_catalogued.py` | Every setting the code reads is described in `settings/catalog.py`, and every description is read |
| `tests/static/test_doc_links.py` | Every relative link in a README resolves |
