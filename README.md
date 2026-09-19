# Q&A Reference Dataset Generator

Generates a verified question-and-answer dataset from a corpus of documents,
so that a retrieval-augmented chatbot can be measured against a fixed
benchmark rather than assessed by impression.

Documents are uploaded, parsed into passages, broken down into atomic facts
that each cite one sentence of their source, and turned into questions with
known answers. Questions that deliberately have no answer in the corpus are
included, to test whether a chatbot recognises the limits of its knowledge.

The pipeline runs entirely on local infrastructure. **No document content
leaves the deployment.**

Nothing here is bound to a subject or an industry. The parser, the chunker and
the topic model work over whatever the documents say; the extraction prompt's
worked example is deliberately about nothing in particular, because an example
drawn from the corpus at hand teaches the model to expect it. German and
English are the two languages configured, in `NLP_MODELS` — one list, naming
both the pipeline each language is read with and the languages the detector
may answer with.

## Prerequisites

| | |
|---|---|
| Python | 3.12 or 3.13 |
| [Poetry](https://python-poetry.org/docs/#installation) | 2.0 or later |
| [Podman](https://podman.io/docs/installation) or Docker | with Compose |

The Makefile invokes `podman compose`. For Docker, set
`COMPOSE=docker compose` in the environment or edit the variable at the top of
the Makefile.

## Install

```bash
git clone <repository-url> qa_generator && cd qa_generator
cp .env.example .env
```

That copies the credentials, ports and addresses. How the pipeline behaves is
in `configs/env/`, which comes with the clone and needs no copying.

Edit `.env` and replace every `change_me_*` value. Two have constraints:
`PHOENIX_ADMIN_SECRET` needs at least 32 characters including a digit and a
lower-case letter, and `PHOENIX_DEFAULT_ADMIN_INITIAL_PASSWORD` is applied
only when Phoenix first creates its admin user — changing it afterwards means
dropping the `phoenix` database.

```bash
make install
```

This installs the dependencies and downloads the spaCy pipelines named in
`NLP_MODELS`. They are also baked into the backend image, because the runtime
has no network.

`EMBEDDING_MODEL` is not baked in. One image serves the API and all five
workers, and its weights are 2.2 GB that four of those processes never load.
It is fetched on first use instead, into the `models` volume.

## Run

```bash
make dev
```

This generates TLS certificates, starts every service, waits for PostgreSQL,
and creates the database schema. First run pulls several images and takes a
few minutes.

Once it reports ready, the frontend is at <http://localhost:8501> and the API
docs at <http://localhost:8000/docs>. The other six services — Phoenix,
Grafana, Argilla, Adminer, Dagster and the object store — are listed in
[docs/operations.md](docs/operations.md).

## Example 1 — a corpus, through the pipeline

Upload a PDF on the Upload page, then go to Documents and press **Start
all**. **Nothing runs until it is asked to.**

Each page runs exactly one stage — Documents parses, Passages chunks, Facts
extracts, Topics fits, Questions writes — so the corpus moves one page at a
time, in that order. Pick a row in any table to see everything held about it
and to run that stage over that row alone, so one document can be re-run or
deleted while the rest of a corpus is mid-flight.

The same thing from a terminal, one stage at a time, then drawing a balanced
release out of everything that was accepted:

```bash
make parse-start     && make parse     # queue every new document, then drain
make chunk-start     && make chunk
make extract-start   && make extract
make topics-discover                   # a fit is asked for, not started
make questions-start && make questions
make questions-balance
```

## Example 2 — applying a change without re-reading the corpus

A setting that changes what a fact *is* does not mean calling the model again.
`extract-rerun` would do that over the whole corpus — hours — and return the
same statements when only a check changed. Re-derive instead:

```bash
make settings-set SERVICE=extraction SET="EXTRACTION_MIN_OTHER_SHARE=0.4"
make extract-recap SHA=<sha256>       # re-apply the cap. No model call
make extract-revalidate SHA=<sha256>  # judge the stored facts again
make questions-reverify               # carry it through to the questions
```

Every stage has a replay that skips the expensive part. They are listed in
[docs/make.md](docs/make.md#replaying-a-stage-without-redoing-it).

## Documentation

Each service documents itself, beside its code.

**The pipeline, in the order a document moves:**

| | |
|---|---|
| [`backend/ingestion/`](backend/ingestion/README.md) | Upload validation, hashing and storage |
| [`backend/preprocessing/parsing/`](backend/preprocessing/parsing/README.md) | A stored file becomes a structured document |
| [`backend/preprocessing/chunking/`](backend/preprocessing/chunking/README.md) | That document becomes passages, with their sentences and lemmas |
| [`backend/extraction/`](backend/extraction/README.md) | Those passages become facts citing a sentence |
| [`backend/topic_modelling/`](backend/topic_modelling/README.md) | Each language becomes topics over its own vocabulary |
| [`backend/question_generation/`](backend/question_generation/README.md) | Each topic's facts become questions with known answers |

**The surfaces:**

| | |
|---|---|
| [`backend/api/`](backend/api/README.md) | The HTTP surface the frontend and the orchestrator call |
| [`backend/stages/`](backend/stages/README.md) | The queue, the worker and the command line every stage shares |
| [`frontend/`](frontend/README.md) | The Streamlit application |
| [`docs/make.md`](docs/make.md) | Every `make` target |

**What they all stand on:**

| | |
|---|---|
| [`backend/database/`](backend/database/README.md) | The schema, the migrations and the one trigger |
| [`backend/blob_store/`](backend/blob_store/README.md) | The three buckets |
| [`backend/nlp/`](backend/nlp/README.md) | Sentences, claims, vocabulary and language |
| [`backend/llm/`](backend/llm/README.md) | The served model |
| [`backend/settings/`](backend/settings/README.md) | Configuration, and what a change stales |
| [`telemetry/`](telemetry/README.md) | Logging and tracing |

**Around the edges:**

| | |
|---|---|
| [`orchestration/`](orchestration/README.md) | Dagster: one asset per stage. Delete it and the pipeline is unchanged |
| [`review/`](review/README.md) | Argilla: where a person overrules a model |
| [`evaluation/`](evaluation/README.md) | Phoenix: scoring a served model against the golden cases |
| [`tests/`](tests/README.md) | Ten layers, and what each needs |
| [`configs/`](configs/README.md) | Service configuration and init scripts |
| [`docs/configuration.md`](docs/configuration.md) | Every setting worth changing |
| [`docs/operations.md`](docs/operations.md) | Scaling, dashboards, and the things that go wrong |

## How it is put together

No backend service imports another. Each owns a repository over the shared
models and passes plain dataclasses across its own boundaries; `database/`,
`blob_store/` and `nlp/` are the only packages that build a connection or load
a model, so a service can neither configure the infrastructure nor reach
around another service to it. Stages hand work to each other through their own
status column, and share only what is in `backend/stages/`.

A service package **re-exports nothing**. A caller names the submodule it
wants — `from extraction.repository import PassageQueue` — so it pays for that
submodule and no more. That is what keeps litellm, Docling, gensim and spaCy
out of the API process, which loads none of them.

They are independent modules, not independent services. They share one
PostgreSQL schema, one `Status` enum and one container image, and two stages
own different columns of the same `documents` row. That is the right shape at
this size; splitting them further would mean a schema and a migration history
each, and a contract between them that is not a table.

## Limitations

- **Scanned documents are refused rather than parsed.** OCR is not enabled;
  see the placeholder in `backend/preprocessing/parsing/pipelines/pdf.py`.
- **Figures become no passage of their own**, which would need a vision model
  to be worth anything, and key-value form regions are not modelled, because
  no document in the corpus has any.
- **`EMBEDDING_MODEL` cannot be changed without a migration.**
  `questions.embedding` is `vector(1024)`, and the worker refuses a model of
  another width at start-up.
- **The model is the bottleneck, not the pipeline.** A median passage measured
  at 473 s on a 31B model.

## Licence

[Apache 2.0](LICENSE).
