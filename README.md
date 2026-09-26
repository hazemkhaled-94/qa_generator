# Q&A Reference Dataset Generator

Generates a verified question-and-answer dataset from a corpus of documents,
so a retrieval-augmented chatbot can be measured against a fixed benchmark.

Documents are uploaded, parsed into passages, broken down into atomic facts
that each cite one sentence of their source, and turned into questions with
known answers. Some questions deliberately have no answer in the corpus, to
test whether a chatbot recognises the limits of its knowledge.

Everything runs on local infrastructure. **No document content leaves the
deployment.**

Nothing here is bound to a subject or an industry. German and English are the
two languages configured, in `NLP_MODELS`; adding a third is a handful of
edits listed in [`backend/nlp/`](backend/nlp/README.md).

## Prerequisites

| | |
|---|---|
| Python | 3.12 or 3.13 |
| [Poetry](https://python-poetry.org/docs/#installation) | 2.0 or later |
| [Podman](https://podman.io/docs/installation) or Docker | with Compose |

The Makefile invokes `podman compose`. For Docker, set
`COMPOSE=docker compose` in the environment.

## Install

```bash
git clone <repository-url> qa_generator && cd qa_generator
make setup
```

`setup` writes `.env` and `configs/env/provider.env` and fills in every
`change_me_*` placeholder. It is safe to run twice.

Then name the model. `LLM_MODEL` in `.env` is a
[litellm](https://docs.litellm.ai/docs/providers) identifier whose prefix
picks the provider:

```ini
LLM_MODEL=ollama_chat/gemma4:31b       # a local Ollama, no credentials
LLM_MODEL=openai/gpt-4o                # OPENAI_API_KEY in provider.env
LLM_MODEL=anthropic/claude-sonnet-4-5  # ANTHROPIC_API_KEY
LLM_MODEL=azure/<deployment>           # a key, or Entra ID
```

`LLM_CONTAINER_MODEL` is the model the containers call, and compose refuses
to start without it. Set it to the same value if one model serves both.

```bash
make install    # dependencies, and the spaCy pipelines NLP_MODELS names
make doctor     # checks the tools and the config, then asks the model one question
```

## Run

```bash
make dev
```

Generates TLS certificates, starts every service, waits for PostgreSQL and
creates the schema. The frontend is then at <http://localhost:8501> and the
API docs at <http://localhost:8000/docs>; the other services are listed in
[docs/operations.md](docs/operations.md).

```bash
make help        # every target, grouped
make services    # which containers are up, and where to open them
```

## Taking a corpus through

Upload a PDF on the Upload page, then press **Run the pipeline** on
Documents. That takes the corpus through every stage in order, waiting for
each to finish before starting the next, and only over what has not been done
already. Nothing runs until it is asked to.

Each page also runs its own stage, for when one is all you want: **Start all**
on Passages chunks, on Facts extracts, and so on.

The same from a terminal, in one command:

```bash
make corpus     # every stage in order, in the foreground
```

Or stage by stage, which is what `corpus` runs:

```bash
make parse-start     && make parse
make chunk-start     && make chunk
make extract-start   && make extract
make topics-discover
make questions-start && make questions
make questions-balance

make assess-start    && make assess        # optional; see below

make questions-export OUT=exam.xlsx FILTER="--status accepted"
```

Every stage also has a replay that skips the expensive part — re-applying a
changed setting without calling the model again. They are listed in
[docs/make.md](docs/make.md).

And over HTTP, for a deployment nobody drives from a browser or a terminal:

```bash
curl -s  localhost:8000/pipeline              # where the corpus has got to
curl -sX POST localhost:8000/pipeline/run     # take it through, in order
curl -sX POST localhost:8000/pipeline/retry   # every stage's failures, at once
```

`run` is the one call that needs the orchestrator, because running the stages
in order means waiting for each to drain and a request cannot hold that wait.
Without one it answers 503 and says which per-stage route to use instead. To
hand it over entirely — an upload starts a run by itself — `make auto`, or
`PUT /pipeline/automation`. See [`backend/api/`](backend/api/README.md).

### The evaluation phase

`assess` is the one step above that is optional and off by default. It puts
every fact, topic and question to an independent LLM judge and records what
it said **beside** the checker's verdict — never over it, because a judge
that could reject rows was measured at chance on this corpus's German half.

What it is for is the pairs where the two disagree: an artefact the pipeline
kept and the judge refused is either a check that let something through or a
judge that is wrong, and only a person settles which.

```ini
ASSESSMENT_ENABLED=true                      # .env
ASSESSMENT_JUDGE_MODEL=ollama_chat/qwen3:14b # not LLM_MODEL
```

```bash
make assess-enrol    # how many model calls it would cost, before paying
make assess-start && make assess
```

The verdicts land in the Assessment page, `/assessment`, the Dagster asset,
Phoenix's Evaluations view, the Argilla records and the exported workbook.
See [`backend/assessment/`](backend/assessment/README.md).

## Documentation

Each service documents itself beside its code.

**The pipeline, in the order a document moves:**

| | |
|---|---|
| [`backend/ingestion/`](backend/ingestion/README.md) | Upload validation, hashing and storage |
| [`backend/preprocessing/parsing/`](backend/preprocessing/parsing/README.md) | A stored file becomes a structured document |
| [`backend/preprocessing/chunking/`](backend/preprocessing/chunking/README.md) | That document becomes passages |
| [`backend/extraction/`](backend/extraction/README.md) | Those passages become facts citing a sentence |
| [`backend/topic_modelling/`](backend/topic_modelling/README.md) | Each language becomes topics over its own vocabulary |
| [`backend/question_generation/`](backend/question_generation/README.md) | Each topic's facts become questions with known answers |
| [`backend/assessment/`](backend/assessment/README.md) | An LLM judge reads all of it back, and decides nothing |

**The surfaces:**

| | |
|---|---|
| [`backend/api/`](backend/api/README.md) | The HTTP surface |
| [`backend/stages/`](backend/stages/README.md) | The queue, the worker and the command line every stage shares |
| [`frontend/`](frontend/README.md) | The Streamlit application |
| [`docs/make.md`](docs/make.md) | Every `make` target |

**What they stand on:**

| | |
|---|---|
| [`backend/database/`](backend/database/README.md) | The schema, the migrations and the triggers |
| [`backend/blob_store/`](backend/blob_store/README.md) | The four buckets |
| [`backend/archive/`](backend/archive/README.md) | What a deletion leaves behind |
| [`backend/nlp/`](backend/nlp/README.md) | Sentences, claims, vocabulary and language |
| [`backend/llm/`](backend/llm/README.md) | The served model |
| [`backend/settings/`](backend/settings/README.md) | Configuration, and what a change stales |
| [`telemetry/`](telemetry/README.md) | Logging and tracing |

**Around the edges:**

| | |
|---|---|
| [`orchestration/`](orchestration/README.md) | Dagster: one asset per stage |
| [`review/`](review/README.md) | Argilla: where a person overrules a model |
| [`evaluation/`](evaluation/README.md) | Phoenix: scoring a served model against the golden cases |
| [`tests/`](tests/README.md) | The layers, and what each needs |
| [`configs/`](configs/README.md) | Service configuration and init scripts |
| [`docs/architecture.md`](docs/architecture.md) | The four graphs |
| [`docs/configuration.md`](docs/configuration.md) | Every setting worth changing |
| [`docs/operations.md`](docs/operations.md) | Scaling, dashboards, and what goes wrong |
| [`docs/measurements.md`](docs/measurements.md) | The numbers the defaults were set from |

## How it is put together

No backend service imports another. Each owns a repository over the shared
models; `database/`, `blob_store/` and `nlp/` are the only packages that
build a connection or load a model. Stages hand work to each other through
their own status column and share only what is in `backend/stages/`.

A service package re-exports nothing, so a caller naming
`extraction.repository` pays for that submodule and no more. That is what
keeps litellm, Docling, gensim and spaCy out of the API process.

They are independent modules, not independent services: one PostgreSQL
schema, one `Status` enum, one container image.

## Limitations

- **Scanned documents are refused rather than parsed.** OCR is not enabled.
- **Figures become no passage of their own**, and key-value form regions are
  not modelled.
- **`EMBEDDING_MODEL` cannot be changed without a migration.**
  `questions.embedding` is `vector(1024)`.
- **The model is the bottleneck, not the pipeline.** A median passage
  measured at 473 s on a 31B model.

## Licence

[Apache 2.0](LICENSE).
