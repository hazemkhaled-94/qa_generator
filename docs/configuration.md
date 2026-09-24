# Configuration

Every setting has two places it can come from. The files below supply all of
them, and a deployment can override one through the UI, the API or the command
line **without restarting anything**.

The mechanics — the store, the catalogue, what a change stales — are in
[`backend/settings/`](../backend/settings/README.md). This is the reference.

## The files

Split by what the value *is*, not by which service reads it.

| File | In git | Holds |
|---|---|---|
| `configs/env/backend.env` | yes | How the pipeline behaves: the parsing, chunking, extraction, topic and question settings the API and the five workers read |
| `configs/env/elasticsearch.env` | yes | The Elasticsearch node's certificate paths, security flags and heap. One node serves both Argilla and the logs |
| `configs/env/seaweedfs-filer.env` | yes | Which metadata store the filer uses |
| `configs/env/orchestration.env` | yes | How patiently Dagster watches a stage it started |
| `configs/env/review.env` | yes | How a review sample is drawn |
| `configs/env/evaluation.env` | yes | What a golden-set run is called in Phoenix |
| `configs/env/provider.env` | **no** | Whatever env vars the configured model provider needs, credentials included. Optional, and absent for a plain Ollama |
| `.env` | **no** | Credentials, ports, and the addresses a host reaches a service at. `.env.example` lists it |

compose hands each file to the services that need it with `env_file`, and the
Makefile sources `backend.env` and `.env` for the host commands, so one value
reaches both.

## Precedence

Highest first:

1. A variable given on a `make` command line
2. A row in `service_settings` — set through the UI, the API or `make settings-set`
3. `.env`
4. `configs/env/*.env`

**The files stay required.** A variable missing from both of them stops the
service at start-up naming itself. There are no defaults in code, and there is
no stored copy of a default — the file is the copy, and deleting the override
row is what returns a setting to it.

## Changing one

Three surfaces, all going through the same validation — the stage's own
`Settings.load`, so what you are refused with is the message the worker would
have failed at start-up with.

```bash
# On the page that runs the service, under Configuration in its service panel.

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

A worker picks a change up on the row it claims next. **Nothing is requeued**:
if the change means what a stage already produced was made under the old
value, whichever surface you used says so and names the stage, and that
stage's own `-rerun` is what rebuilds it.

Every fact, topic and question records the configuration it was produced
under, so a corpus built across a change can still be read.

## The seven services

Each configures its own settings and no others: `ingestion` on Upload,
`parsing` on Documents, `chunking` on Passages, `extraction` on Facts,
`topics` on Topics, `questions` on Questions, and `platform` — the model, the
tokenizer and the language pipelines the six of them share — on System health.

A stage that wants a different model from the rest names one of its own:
`EXTRACTION_MODEL`, `TOPIC_MODEL`, `QUESTIONS_MODEL`, each meaning
`LLM_MODEL` when it is absent. Three more name a model for one *judgement*
rather than for a stage — `EXTRACTION_DIGEST_MODEL`,
`QUESTIONS_PHRASING_MODEL` and `QUESTIONS_VERIFIER_MODEL` — because what
each judgement costs is not the same question as what a run costs. See
`make spend-by-shape`.

The pool sizes and the addresses are **not** configurable. They are read
before a service could ask a database for anything, so they are served
read-only and marked as the deployment's.

## The values most likely to need changing

### The platform

| Variable | Where | Default | Purpose |
|---|---|---|---|
| `LLM_MODEL` | `.env` | `ollama_chat/gemma4:31b` | LiteLLM model id; the prefix picks the provider |
| `LLM_BASE_URL` | `.env` | `http://localhost:11434` | Where that model is served |
| `LLM_CONTAINER_MODEL` | `.env` | — | **Required by compose.** The model the *containers* call, which is not always the host's: a container holds no interactive cloud credential, and inheriting `LLM_MODEL` is how a stale container spent 2,019 restarts failing to authenticate. Set it to the same value if one model serves both |
| `QUESTIONS_VERIFIER_CONTAINER_MODEL` | `.env` | unset | The verifier as the containers reach it. It follows `LLM_CONTAINER_MODEL`, because a verifier the containers cannot reach is a run of ungated questions |
| `OLLAMA_CONTAINER_URL` | `.env` | `http://host.docker.internal:11434` | Ollama's address as a container sees it. **Required by compose** too, and not overridable per stage: there is nowhere else a container may send a prompt. It also reaches Phoenix as `OLLAMA_BASE_URL`, which is what gives the playground a model it can use without an API key |
| `OLLAMA_BASE_URL` | `.env` | `http://localhost:11434` | Where a **self-hosted** model is served, for the case where one stage names `ollama_chat/…` while the shared model is somewhere else. The address follows the provider: a stage overriding only the model would otherwise send an Ollama request to Azure |
| `LLM_TIMEOUT_SECONDS` | `backend.env` | 120 | How long one call may take. The extraction lease is derived from it. Sized from 9,334 priced calls whose slowest was 15.5 s — **raise it for a larger model**, because everything below is derived from it. It was 900, for a 31B whose median passage took 473 s |
| `LLM_MAX_ATTEMPTS` | `backend.env` | 3 | Attempts per call. The lease is derived from this too |
| `LLM_TEMPERATURE` | `backend.env` | 0 | Zero, so a re-run is comparable to the last one |
| `LLM_STRUCTURED_MODE` | `backend.env` | `JSON_SCHEMA` | How a typed answer is asked for |
| `LLM_NUM_CTX`, `LLM_REASONING_EFFORT` | `backend.env` | unset | The context window to ask the runtime for, and the knob a reasoning model has. Both follow the provider |
| `EXTRACTION_MODEL`, `QUESTIONS_MODEL` | `backend.env` | unset | One stage calling a different model. The model only — where it is served and how patient to be stay `LLM_*`, because a stage that could set its own timeout would be a stage whose lease nobody could derive |
| `TOPIC_MODEL` | `backend.env` | `ollama_chat/gemma4:12b` | The same, for topic naming, which is a short prompt over a term list and does not need the writer's model |
| `EXTRACTION_DIGEST_MODEL`, `QUESTIONS_PHRASING_MODEL` | `backend.env` | `ollama_chat/gemma4:12b` | One judgement on a smaller model. Neither reads a passage the way the stage's own model does |
| `QUESTIONS_VERIFIER_MODEL` | `.env` | unset | The second model, which checks that a question's answer is in the passages it cites. Naming the writer's own model, or none at all, turns off the gates only an independent model may apply; the worker warns on every start |
| `EMBEDDING_MODEL` | `backend.env` | `intfloat/multilingual-e5-large` | The one embedding model, and the tokenizer chunking sizes a passage by. **Changing it is a migration** |
| `EMBEDDING_MAX_TOKENS` | `backend.env` | 512 | That model's context window, and so the longest passage |
| `NLI_MODEL` | `backend.env` | `MoritzLaurer/bge-m3-zeroshot-v2.0` | A local entailment encoder, asked whether the cited passages entail an answer recall did not find. Absent asks the verifier instead |
| `NLI_ENTAILMENT_THRESHOLD` | `backend.env` | 0.7 | How sure it must be before it rescues one |
| `QA_MODEL` | `backend.env` | unset | A local extractive reader, asked for the answer span before the verifier is. Absent asks the verifier for every question |
| `QA_ANSWER_CONFIDENCE` | `backend.env` | 0.9 | How sure that reader must be before its span is taken and no model is called |
| `ENCODER_MAX_TOKENS` | `backend.env` | 8192 | The longest pair either encoder reads. A **ceiling**: a checkpoint that cannot read that far is held to its own window |
| `NLP_MODELS` | `backend.env` | `de:de_core_news_md,en:en_core_web_md` | The spaCy pipeline per language, and the languages the detector may answer with. Must be in the image. **Medium, not small**: the small German model does not tag a modal as a finite verb |
| `NLP_DEFAULT_LANGUAGE` | `backend.env` | `en` | What a passage too short to judge is read as |
| `WORKER_POLL_SECONDS` | `backend.env` | 5 | How long a watching worker sleeps between drains |
| `DATABASE_POOL_SIZE`, `DATABASE_POOL_OVERFLOW` | `backend.env` | 5, 5 | Per process. The total across the API and every worker has to stay under PostgreSQL's `max_connections`. Fixed: a restart |

### Ingestion

| Variable | Where | Default | Purpose |
|---|---|---|---|
| `MAX_FILE_SIZE_MB` | `backend.env` | 100 | Largest upload accepted. Keep in step with `server.maxUploadSize` in `frontend/.streamlit/config.toml`, or Streamlit rejects the file before the API sees it |
| `ALLOWED_MIME_TYPES` | `backend.env` | `application/pdf` | Types with a parser behind them |
| `PIPELINE_VERSION` | — | — | Not a setting. Read from `version` in `pyproject.toml` and recorded on each object as it is stored: it describes the build, not the deployment |

### Topic modelling

The two that decide how many topics there are. The rest of the table is in
[`backend/topic_modelling/`](../backend/topic_modelling/README.md#configuration).

| Variable | Default | Purpose |
|---|---|---|
| `TOPIC_PASSAGES_PER_TOPIC` | 40 | How many passages one topic is worth. Above 0 this turns `TOPIC_NUM_TOPICS` into a **ceiling** and fits `passages / this` instead, floored at 2. One fit runs per language and two languages are rarely the same size, so a fixed count gives one side subjects and the other slivers |
| `TOPIC_NUM_TOPICS` | 40 | Topics per language, and the ceiling the line above works under. At least 2 |

A topic is also the unit question generation claims, so this and
`QUESTIONS_PER_TOPIC` multiply out to how many questions a run can write.

### Question generation

The settings most worth tuning, and the ones a run's cost is decided by. The
full table is in
[`backend/question_generation/`](../backend/question_generation/README.md#configuration).

| Variable | Default | Purpose |
|---|---|---|
| `QUESTIONS_PER_TOPIC` | 120 | How many to aim for per topic. This times the topic count is what a full run costs, twice over — each candidate is a writer call and a verifier call |
| `QUESTIONS_FACT_SAMPLE` | 2 | How many of a topic's facts are offered per call, divided between the passages the sample holds |
| `QUESTIONS_TYPE_MIX` | thirteen kinds named, eleven at weight 1 | Which kinds are written and in what proportion. A weight of 0, or a name left out, is never written — `entity` and `temporal` ship at 0 |
| `QUESTIONS_DIFFICULTY_MIX` | `easy:2,medium:3,hard:3` | Which bands the plan aims for. A request for a shape of sample; the band itself stays derived |
| `QUESTIONS_UNANSWERABLE_SHARE` | 0.30 | What share are **attempted** with no answer in the corpus. Three to four times the share a release should hold, because that is roughly how many survive their own gates |
| `QUESTIONS_FOLLOWUP_SHARE` | 0.5 | What share of the accepted, answerable roots get a follow-up thread |
| `QUESTIONS_ANSWER_CHARS` | `value:1:80,list:3:300,explanation:20:600` | The shortest and longest target answer per form |
| `QUESTIONS_DUPLICATE_COSINE` | 0.93 | How alike two questions must be before the later one is thrown away |
| `QUESTIONS_RELEASE_SIZE` | 0 | How many to draw. `0` is the largest the pool can fill without missing a quota |

### Observability, review and orchestration

| Variable | Where | Default | Purpose |
|---|---|---|---|
| `LOG_LEVEL` | `.env` | `INFO` | Every service, the frontend included. Everything at or above it reaches Grafana |
| `LOG_DIR` | `.env` | `logs` | Where a process writes the JSON lines the shipper reads. The host's directory; compose sets `/var/log/qa` over it per container, and filebeat reads both. Unset means stdout only |
| `OTEL_CONTAINER_ENDPOINT` | `.env` | `http://phoenix:4317` | Trace collector, as the containers reach it |
| `PHOENIX_BASE_URL` | `.env` | `http://localhost:6006` | Phoenix's HTTP API, where the golden-set datasets, the experiments and the gate verdicts go. The same service the line above sends spans to, read the other way. Compose sets the container's address over this name, as it does for the collector |
| `PHOENIX_CONTAINER_BASE_URL` | `.env` | `http://phoenix:6006` | What compose sets there. Read by compose only — a process reads `PHOENIX_BASE_URL` wherever it runs |
| `PHOENIX_BASE_URL` (frontend) | `.env` | `http://localhost:6006` | The same name on the `streamlit` service, taken as it stands rather than set over: it is written into a link a BROWSER follows, not called from the container. Unset shows a question's trace id as text instead of a link |
| `OLLAMA_CONTAINER_URL` (Phoenix) | `.env` | `http://host.docker.internal:11434` | Reaches the playground as `OLLAMA_BASE_URL`. Ollama is the one provider Phoenix offers that needs no API key, which an `az login` credential cannot give a container |
| `GRAFANA_DB_USER` | `.env` | `grafana_reader` | The role Grafana reads the application database as. `SELECT` and nothing else |
| `ARGILLA_API_URL` | `.env` | `http://localhost:6900` | Where the review tool reaches Argilla. The host's address: it is a `make` target, not a container |
| `ARGILLA_API_KEY` | `.env` | — | Argilla shows it under "My settings". **Not** `ARGILLA_PASSWORD` |
| `ARGILLA_WORKSPACE` | `.env` | `qa_generator` | One per deployment, so two people reviewing two corpora do not annotate each other's rows |
| `REVIEW_SAMPLE_SIZE` | `review.env` | 200 | How many records a push puts in front of a reviewer, split across the verdicts |
| `EVAL_RUN_NAME` | `evaluation.env` | unset | What a golden-set run is called. Unset names it after the model |
| `ORCHESTRATION_DRAIN_TIMEOUT_SECONDS` | `orchestration.env` | 28800 | How long a Dagster asset waits for a stage to drain. Giving up is not failing the rows |
| `ORCHESTRATION_POLL_SECONDS` | `orchestration.env` | 15 | How often it asks |
| `LOGS_RETENTION_DAYS` | `make` | 30 | How long the log index is kept, applied by `make logs-retention` |
| `LOGS_KEEP_DAYS` | `make` | 30 | How long a log FILE on the `logs` volume is kept, applied by `make logs-prune`. The other half: the target above deletes nothing from disk |

### The frontend

| Variable | Where | Default | Purpose |
|---|---|---|---|
| `BACKEND_URL` | `.env` | `http://api:8000` | The one address the frontend holds |
| `PAGE_SIZE` | `.env` | 50 | Rows per page in the listings |
| `SERVICE_URLS` | `compose.yaml` | built from the ports | Where a person opens each container, as `name=url` pairs, for the System health page. Built by compose rather than written in `.env`, because a browser needs the **published** port and the compose file is the only place both halves are written down — grafana serves 3000 and is published on 3001. A service left out simply gets no link |

The first two must **also** be listed in the `streamlit` service's
`environment` in `compose.yaml` — being in `.env` alone is not enough, and
the error says so.

## Two settings with constraints

`PHOENIX_ADMIN_SECRET` needs at least 32 characters including a digit and a
lower-case letter. `PHOENIX_DEFAULT_ADMIN_INITIAL_PASSWORD` is applied only
when Phoenix first creates its admin user — changing it afterwards means
dropping the `phoenix` database.

## What is declared where is enforced

Three static tests, all in `tests/static/`:

| | |
|---|---|
| `test_settings_documented.py` | Every setting the code reads is named in a file that declares it |
| `test_settings_catalogued.py` | Every setting the code reads is described in `settings/catalog.py`, and every description is read by something |
| `test_doc_links.py` | Every relative link in a README points at a file that is there |

A setting the code reads and the catalogue does not describe cannot be
configured at all; one the catalogue describes and nothing reads is a control
that does nothing. Neither is allowed to merge.

`configs/env/backend.env` is the long form — each setting carries the
paragraphs explaining how its default was arrived at.
`settings/catalog.py` carries the one-line version the UI draws its controls
from.
