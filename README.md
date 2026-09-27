# Q&A Reference Dataset Generator

Builds a verified question-and-answer dataset from a corpus of documents, so
a retrieval-augmented chatbot can be measured against a fixed benchmark
instead of against an opinion.

Documents are parsed into passages, read into atomic facts that each cite one
sentence of their source, grouped into topics, and turned into questions with
known answers. Every question carries the facts it was written from, and every
fact carries the sentence it came from, so any answer in the set can be traced
back to a page. A deliberate share of questions have no answer in the corpus
at all, which is what measures whether a chatbot recognises the limits of its
knowledge.

The pipeline is subject-agnostic: no prompt names a domain, and topics are
discovered from the corpus's own vocabulary rather than supplied. German and
English ship configured in `NLP_MODELS`; adding a third language is a short
list of edits in [`backend/nlp/`](backend/nlp/README.md).

Everything runs on infrastructure you control. **No document content leaves
the deployment**, except to whichever model provider you configure — and with
a locally served model, not even that.

## What you get

An Excel workbook of four sheets, plus the same rows through the API and the
database:

| Sheet | One row per |
|---|---|
| Questions | Question, with its answer, difficulty, type, and what it rests on |
| Citations | Question × fact × passage, with the page number and the evidence |
| Assessment | Judgement of a question by an independent model |
| Summary | The counts for what the filter selected |

The run recorded in [docs/measurements.md](docs/measurements.md) produced
2,416 accepted questions from 16 documents, and a balanced release of 1,341
of them.

## Prerequisites

| | | |
|---|---|---|
| [Python](https://www.python.org/downloads/) | 3.12 or 3.13 | 3.14 is not supported: gensim does not build there |
| [Poetry](https://python-poetry.org/docs/#installation) | 2.0 or later | Dependency management |
| [Podman](https://podman.io/docs/installation) or [Docker](https://docs.docker.com/get-started/get-docker/) | with Compose | 24 containers |
| [Ollama](https://ollama.com/download) | any recent | **Required.** See below |
| Disk | ~15 GB | Images, model weights and volumes |
| RAM | 16 GB | 6 GB of it for the question worker |

**Ollama is not optional.** The containers hold no cloud credential and must
never reach a hosted provider, so `LLM_CONTAINER_MODEL` has to name a model
Ollama serves. The host may call a hosted provider as well, but the
containerised workers cannot. Pull at least one model before the first run:

```bash
ollama pull gemma4:31b
```

Pick a tag whose quantisation matches the runner serving it — an `-mlx` build
with nvfp4 weights will not load under Ollama's MLX runner.

## Install

```bash
git clone https://github.com/hazemkhaled-94/qa_generator.git && cd qa_generator
make setup      # writes .env and configs/env/provider.env, fills every placeholder
```

`setup` is safe to run twice: it replaces `change_me_*` placeholders and
nothing else.

Then name the model. `LLM_MODEL` in `.env` is a
[LiteLLM](https://docs.litellm.ai/docs/providers) identifier whose prefix
picks the provider:

```ini
LLM_MODEL=ollama_chat/gemma4:31b       # local, no credentials
LLM_MODEL=openai/gpt-4o                # OPENAI_API_KEY in provider.env
LLM_MODEL=anthropic/claude-sonnet-4-5  # ANTHROPIC_API_KEY
LLM_MODEL=azure/<deployment>           # a key, or Entra ID
```

`LLM_CONTAINER_MODEL` is what the containers call and compose refuses to start
without it. Set both to the same value if one model serves both.

```bash
make install    # dependencies, and the spaCy pipelines NLP_MODELS names
make doctor     # checks the tools and the config, then asks the model one question
```

`doctor` ends with a real call to the configured model, so a wrong model id,
an unreachable address or a stale credential fails there rather than on the
first passage of a run.

## Run

```bash
make dev
```

Generates TLS certificates, starts every service, waits for PostgreSQL and
creates the schema. The frontend is then at <http://localhost:8501> and the
API documentation at <http://localhost:8000/docs>.

```bash
make services   # which containers are up, and where to open them
make help       # every target, grouped
```

### Choosing a setup

Four choices, independent of each other.

**Container engine.** Podman is the default; the Makefile calls
`podman compose`.

```bash
make COMPOSE="docker compose" dev    # Docker instead
```

**Model provider.** Ollama alone is the simplest and needs no credentials.
A hosted provider is faster per call but applies only to host-side commands —
containerised workers always call Ollama. Credentials go in
`configs/env/provider.env`.

**Where the stages run.** The workers in their containers, `make extract` and
friends on the host, or both at once: they drain the same queue through
`SELECT … FOR UPDATE SKIP LOCKED`, so nothing is processed twice. The host
path exists for credentials that only exist where a person is.

**How the pipeline is driven.**

| | Start it with | Suits |
|---|---|---|
| Browser | **Run the pipeline** on the Documents page | Trying it out — the easiest |
| Terminal | `make corpus` | A scripted or reproducible run |
| HTTP | `POST /pipeline/run` | A deployment nobody drives interactively |
| Unattended | `make auto` | An upload starting a run by itself |

**The easiest path is `make setup` → `make install` → `make doctor` →
`make dev`, with Ollama for the model, Podman for the engine, and the
browser for the pipeline.** Nothing runs until asked; every stage is
incremental and skips what is already done.

## An example, end to end

Upload a PDF on the Upload page, then take it through:

```bash
make corpus                              # every stage in order, stopping at the first failure
make questions-balance                   # draw an evenly composed release
make questions-export OUT=exam.xlsx FILTER="--status accepted"
```

`corpus` runs the stages in order and waits for each to drain before starting
the next. Stage by stage, which is what it runs:

```bash
make parse-start     && make parse       # a stored file becomes a structured document
make chunk-start     && make chunk       # that document becomes passages
make extract-start   && make extract     # those passages become facts citing a sentence
make topics-discover                     # each language becomes topics
make questions-start && make questions   # each topic's facts become questions
```

Watch it with `make pipeline-status`, the Documents page, or Grafana's
pipeline dashboard.

The result is `exam.xlsx` — the four sheets above — and the same rows in
PostgreSQL behind `GET /questions`. A single row reads on its own:

| Column | Example |
|---|---|
| `question_text` | *Under what conditions may a test be re-run?* |
| `target_answer` | The key a chatbot is scored against |
| `answer_explanation` | The same answer at length, for a marker |
| `answerable` | `false` on the questions with no answer in the corpus |
| `difficulty` | `easy`, `medium` or `hard`, derived from how far the answer is spread |
| `question_type` | One of thirteen forms, never a subject |
| Citations | The facts it cites, their evidence, and the page each sits on |

Turning the evaluation phase on adds an independent model's opinion beside
each verdict — it decides nothing, and the pairs where the two disagree are a
queue for a person:

```ini
ASSESSMENT_ENABLED=true                 # .env
ASSESSMENT_JUDGE_MODEL=azure/gpt-4.1    # not LLM_MODEL
```

```bash
make assess-enrol                        # what it would cost, before paying
make assess-start && make assess
```

## Documentation

The [**project wiki**](https://github.com/hazemkhaled-94/qa_generator/wiki) is
the long form of everything below, on one site and cross-linked. In this
repository, each service documents itself beside its code.

**The pipeline, in the order a document moves:**

| | |
|---|---|
| [`backend/ingestion/`](backend/ingestion/README.md) | Upload validation, hashing and storage |
| [`backend/preprocessing/parsing/`](backend/preprocessing/parsing/README.md) | A stored file becomes a structured document |
| [`backend/preprocessing/chunking/`](backend/preprocessing/chunking/README.md) | That document becomes passages |
| [`backend/extraction/`](backend/extraction/README.md) | Those passages become facts citing a sentence |
| [`backend/topic_modelling/`](backend/topic_modelling/README.md) | Each language becomes topics over its own vocabulary |
| [`backend/question_generation/`](backend/question_generation/README.md) | Each topic's facts become questions with known answers |
| [`backend/assessment/`](backend/assessment/README.md) | An independent model reads it all back, and decides nothing |

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
| [`backend/blob_store/`](backend/blob_store/README.md) | The five buckets |
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
| [`docs/configuration.md`](docs/configuration.md) | Where every setting lives |
| [`docs/operations.md`](docs/operations.md) | Scaling, dashboards, and what goes wrong |
| [`docs/measurements.md`](docs/measurements.md) | What one run produced |
| [`docs/bugs.md`](docs/bugs.md) | What is wrong with what it produced, and the fix |

## How it is put together

No backend service imports another. Each owns a repository over the shared
models; `database/`, `blob_store/` and `nlp/` are the only packages that build
a connection or load a model. Stages hand work to each other through their own
status column and share only what is in `backend/stages/`.

A service package re-exports nothing, so a caller naming
`extraction.repository` pays for that submodule and no more — which is what
keeps LiteLLM, Docling, gensim and spaCy out of the API process.

They are independent modules, not independent services: one PostgreSQL schema,
one `Status` enum, one container image. The rules are enforced by
`make lint-imports`.

## Limitations

- **Scanned documents are refused rather than parsed.** OCR is not enabled.
- **Figures become no passage of their own**, and key-value form regions are
  not modelled.
- **`EMBEDDING_MODEL` cannot be changed without a migration.**
  `questions.embedding` is `vector(1024)`.
- **The model is the bottleneck, not the pipeline.** Question generation was
  two thirds of the wall clock and 71% of the spend on the measured run, and a
  locally served model is one to two orders slower per call than a hosted one.

## Licence

[Apache 2.0](LICENSE).
