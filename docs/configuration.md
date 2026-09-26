# Configuration

Every setting comes from a file, and a deployment can override one through
the UI, the API or the command line without restarting anything. The
mechanics are in [`backend/settings/`](../backend/settings/README.md); this
is the reference.

## The files

| File | In git | Holds |
|---|---|---|
| `configs/env/backend.env` | yes | How the pipeline behaves: the parsing, chunking, extraction, topic and question settings |
| `configs/env/elasticsearch.env` | yes | The node's certificate paths, security flags and heap |
| `configs/env/seaweedfs-filer.env` | yes | Which metadata store the filer uses |
| `configs/env/orchestration.env` | yes | How long Dagster waits on a stage |
| `configs/env/review.env` | yes | How a review sample is drawn |
| `configs/env/evaluation.env` | yes | What a golden-set run is called in Phoenix |
| `configs/env/provider.env` | **no** | Whatever env vars the model provider needs. Optional |
| `.env` | **no** | Credentials, ports, and the addresses a host reaches a service at |

compose hands each file to the services that need it with `env_file`, and
the Makefile sources `backend.env` and `.env` for host commands.

## Precedence

Highest first:

1. A variable given on a `make` command line
2. A row in `service_settings` — set through the UI, the API or `make settings-set`
3. `.env`
4. `configs/env/*.env`

The files stay required. A variable missing from both stops the service at
start-up naming itself; there are no defaults in code, and deleting an
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

A worker picks a change up on the row it claims next. Nothing is requeued: if
the change stales what a stage already produced, the surface you used says so
and names the stage, and that stage's `-rerun` rebuilds it.

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

## The platform

| Variable | Where | Default | Purpose |
|---|---|---|---|
| `LLM_MODEL` | `.env` | `ollama_chat/gemma4:31b` | LiteLLM model id; the prefix picks the provider |
| `LLM_BASE_URL` | `.env` | `http://localhost:11434` | Where that model is served |
| `LLM_CONTAINER_MODEL` | `.env` | — | **Required by compose.** The model the containers call |
| `QUESTIONS_VERIFIER_CONTAINER_MODEL` | `.env` | unset | The verifier as the containers reach it |
| `OLLAMA_CONTAINER_URL` | `.env` | `http://host.docker.internal:11434` | Ollama's address as a container sees it. **Required by compose** |
| `OLLAMA_BASE_URL` | `.env` | `http://localhost:11434` | Where a self-hosted model is served |
| `LLM_TIMEOUT_SECONDS` | `backend.env` | 300 | How long one call may take. The extraction lease derives from it, so **raise it for a larger model and everything below follows**. Sized off 9,334 priced calls whose slowest was 15.5 s, then raised from 120 alongside the assessment phase, whose judge reads a whole artefact back |
| `LLM_MAX_ATTEMPTS` | `backend.env` | 3 | Attempts per call. The lease derives from this too |
| `LLM_TEMPERATURE` | `backend.env` | 0 | Zero, so a re-run is comparable |
| `LLM_STRUCTURED_MODE` | `backend.env` | `JSON_SCHEMA` | How a typed answer is asked for |
| `LLM_NUM_CTX`, `LLM_REASONING_EFFORT` | `backend.env` | unset | Context window and reasoning knob. The window follows the provider; the reasoning knob is sent only when set, so a model keeps its own default |
| `EXTRACTION_MODEL`, `QUESTIONS_MODEL` | `backend.env` | unset | One stage calling a different model. The model only |
| `TOPIC_MODEL` | `backend.env` | `ollama_chat/gemma4:12b` | The model that names a topic |
| `EXTRACTION_DIGEST_MODEL`, `QUESTIONS_PHRASING_MODEL` | `backend.env` | `ollama_chat/gemma4:12b` | One judgement on a smaller model |
| `QUESTIONS_VERIFIER_MODEL` | `.env` | unset | The second model. Unset turns off the gates only an independent model may apply |
| `EMBEDDING_MODEL` | `backend.env` | `intfloat/multilingual-e5-large` | The embedding model, and the tokenizer chunking sizes a passage by. **Changing it is a migration** |
| `EMBEDDING_MAX_TOKENS` | `backend.env` | 512 | That model's window, and so the longest passage |
| `NLI_MODEL` | `backend.env` | `MoritzLaurer/bge-m3-zeroshot-v2.0` | The entailment encoder. Absent asks the verifier |
| `NLI_ENTAILMENT_THRESHOLD` | `backend.env` | 0.7 | How sure it must be before it rescues an answer |
| `QA_MODEL` | `backend.env` | unset | A local extractive reader. Absent asks the verifier for every question |
| `QA_ANSWER_CONFIDENCE` | `backend.env` | 0.9 | How sure that reader must be before its span is taken |
| `ENCODER_MAX_TOKENS` | `backend.env` | 8192 | The longest pair either encoder reads. A ceiling |
| `NLP_MODELS` | `backend.env` | `de:de_core_news_md,en:en_core_web_md` | The spaCy pipeline per language, and the languages the detector may answer with. **Medium, not small** |
| `NLP_DEFAULT_LANGUAGE` | `backend.env` | `en` | What a passage too short to judge is read as |
| `WORKER_POLL_SECONDS` | `backend.env` | 5 | How long a watching worker sleeps between drains |
| `DATABASE_POOL_SIZE`, `DATABASE_POOL_OVERFLOW` | `backend.env` | 5, 5 | Per process. The total must stay under `max_connections`. Fixed: a restart |

## Ingestion

| Variable | Where | Default | Purpose |
|---|---|---|---|
| `MAX_FILE_SIZE_MB` | `backend.env` | 100 | Largest upload accepted. Keep in step with `server.maxUploadSize` in `frontend/.streamlit/config.toml` |
| `ALLOWED_MIME_TYPES` | `backend.env` | `application/pdf` | Types with a parser behind them |
| `PIPELINE_VERSION` | — | — | Not a setting. Read from `version` in `pyproject.toml` |

## Topic modelling

The two that decide how many topics there are. The rest is in
[`backend/topic_modelling/`](../backend/topic_modelling/README.md).

| Variable | Default | Purpose |
|---|---|---|
| `TOPIC_PASSAGES_PER_TOPIC` | 40 | How many passages one topic is worth. Above 0 this turns `TOPIC_NUM_TOPICS` into a ceiling and fits `passages / this`, floored at 2 |
| `TOPIC_NUM_TOPICS` | 40 | Topics per language, and the ceiling above. At least 2 |

This and `QUESTIONS_PER_TOPIC` multiply out to how many questions a run can
write.

## Question generation

The ones a run's cost is decided by. The full table is in
[`backend/question_generation/`](../backend/question_generation/README.md).

| Variable | Default | Purpose |
|---|---|---|
| `QUESTIONS_PER_TOPIC` | 120 | How many to aim for per topic. Each candidate is a writer call and a verifier call |
| `QUESTIONS_FACT_SAMPLE` | 2 | How many of a topic's facts are offered per call |
| `QUESTIONS_TYPE_MIX` | eleven kinds at weight 1 | Which kinds are written and in what proportion. Weight 0 is never written |
| `QUESTIONS_DIFFICULTY_MIX` | `easy:2,medium:3,hard:3` | Which bands the plan aims for |
| `QUESTIONS_UNANSWERABLE_SHARE` | 0.30 | What share are attempted with no answer in the corpus |
| `QUESTIONS_FOLLOWUP_SHARE` | 0.5 | What share of accepted answerable roots get a thread |
| `QUESTIONS_ANSWER_CHARS` | `value:1:80,list:3:300,explanation:20:600` | Shortest and longest target answer per form |
| `QUESTIONS_DUPLICATE_COSINE` | 0.93 | How alike two questions must be before the later one is dropped |
| `QUESTIONS_RELEASE_SIZE` | 0 | How many to draw. `0` is the largest the pool can fill |

## The evaluation phase

An LLM judge over every fact, topic and question, run after the pipeline.
Off by default, and it changes nothing the pipeline keeps. The full table is
in [`backend/assessment/`](../backend/assessment/README.md).

| Variable | Where | Default | Purpose |
|---|---|---|---|
| `ASSESSMENT_ENABLED` | `.env` | `false` | Whether the phase runs at all |
| `ASSESSMENT_JUDGE_MODEL` | `.env` | — | Who judges. **Not** `LLM_MODEL`: a model asked whether its own facts follow from its own evidence says yes |
| `ASSESSMENT_KINDS` | `backend.env` | `fact,topic,question` | Which artefacts are judged |
| `ASSESSMENT_SAMPLE` | `backend.env` | 200 | How many of each kind one start enrols, newest first. `0` is all of them |

The two cost dials are the ones to set first. A fact and a topic are two
model calls each and a question is three, so a corpus of twenty thousand
facts judged whole is forty thousand calls.

## Observability, review and orchestration

| Variable | Where | Default | Purpose |
|---|---|---|---|
| `LOG_LEVEL` | `.env` | `INFO` | Every service, the frontend included |
| `LOG_DIR` | `.env` | `logs` | Where a process writes JSON lines. Unset means stdout only |
| `OTEL_CONTAINER_ENDPOINT` | `.env` | `http://phoenix:4317` | Trace collector, as the containers reach it |
| `PHOENIX_BASE_URL` | `.env` | `http://localhost:6006` | Phoenix's HTTP API |
| `PHOENIX_CONTAINER_BASE_URL` | `.env` | `http://phoenix:6006` | What compose sets over the name above. Read by compose only |
| `GRAFANA_DB_USER` | `.env` | `grafana_reader` | The role Grafana reads as. `SELECT` and nothing else |
| `ARGILLA_API_URL` | `.env` | `http://localhost:6900` | The host's address: review is a `make` target, not a container |
| `ARGILLA_API_KEY` | `.env` | — | Argilla shows it under "My settings". **Not** `ARGILLA_PASSWORD` |
| `ARGILLA_WORKSPACE` | `.env` | `qa_generator` | One per deployment |
| `REVIEW_SAMPLE_SIZE` | `review.env` | 200 | How many records a push puts in front of a reviewer |
| `EVAL_RUN_NAME` | `evaluation.env` | unset | What a golden-set run is called. Unset names it after the model |
| `ORCHESTRATION_DRAIN_TIMEOUT_SECONDS` | `orchestration.env` | 28800 | How long an asset waits for a stage to drain |
| `ORCHESTRATION_POLL_SECONDS` | `orchestration.env` | 15 | How often it asks |
| `DAGSTER_URL` | `.env` | `http://dagster-webserver:3000` in compose | Where the **API** reaches the orchestrator, for `POST /pipeline/run` and the Run button. The one address on that surface pointing outside the backend. Unset degrades rather than fails: the reads answer `available: false` and `run` is refused 503 `no_orchestrator`, while `start`, `stop` and `retry` go on working — those are queue verbs and the queue is the backend's own |
| `LOGS_RETENTION_DAYS` | `make` | 30 | How long the log index is kept, applied by `make logs-retention` |
| `LOGS_KEEP_DAYS` | `make` | 30 | How long a log file is kept, applied by `make logs-prune` |

## The frontend

| Variable | Where | Default | Purpose |
|---|---|---|---|
| `BACKEND_URL` | `.env` | `http://api:8000` | The one address the frontend holds |
| `PAGE_SIZE` | `.env` | 50 | Rows per page in the listings |
| `SERVICE_URLS` | `compose.yaml` | built from the ports | Where a person opens each container, for the System health page |

`BACKEND_URL` and `PAGE_SIZE` must **also** be listed in the `streamlit`
service's `environment` in `compose.yaml` — being in `.env` alone is not
enough.

## Two settings with constraints

`PHOENIX_ADMIN_SECRET` needs at least 32 characters including a digit and a
lower-case letter. `PHOENIX_DEFAULT_ADMIN_INITIAL_PASSWORD` is applied only
when Phoenix first creates its admin user; changing it afterwards means
dropping the `phoenix` database.

## What is enforced

| Test | Checks |
|---|---|
| `tests/static/test_settings_documented.py` | Every setting the code reads is named in a file that declares it |
| `tests/static/test_settings_catalogued.py` | Every setting the code reads is described in `settings/catalog.py`, and every description is read |
| `tests/static/test_doc_links.py` | Every relative link in a README resolves |

`configs/env/backend.env` is the long form; `settings/catalog.py` carries the
one-line version the UI draws its controls from. How a default was arrived at
is in [measurements.md](measurements.md).
