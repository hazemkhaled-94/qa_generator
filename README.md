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
may answer with. Adding a third is a handful of edits rather than a setting:
[`backend/nlp/`](backend/nlp/README.md#adding-a-language) is the list of them,
including the two readings a language has to support and the one that raises
without it.

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
make setup
```

`setup` writes the two files a clone does not come with — `.env` for the
credentials, ports and addresses, and `configs/env/provider.env` for whatever
credentials your model provider wants — and generates a password for every
`change_me_*` placeholder, including the two with constraints. It is safe to
run twice: it replaces placeholders and nothing else. How the pipeline
*behaves* is in `configs/env/backend.env`, which comes with the clone.

Two generated values are worth knowing where to find. `PHOENIX_ADMIN_SECRET`
is what the services authenticate their traces with, and
`PHOENIX_DEFAULT_ADMIN_INITIAL_PASSWORD` is what you log into Phoenix with —
it is applied only when Phoenix first creates its admin user, so changing it
afterwards means dropping the `phoenix` database. Both are in `.env`.

Then name the model. `LLM_MODEL` in `.env` is a
[litellm](https://docs.litellm.ai/docs/providers) identifier whose prefix
picks the provider, and that is the whole of changing provider:

```ini
LLM_MODEL=ollama_chat/gemma4:31b       # a local Ollama, no credentials
LLM_MODEL=openai/gpt-4o                # OPENAI_API_KEY in provider.env
LLM_MODEL=anthropic/claude-sonnet-4-5  # ANTHROPIC_API_KEY
LLM_MODEL=azure/<deployment>           # a key, or Entra ID
```

**Name the containers' model too.** `LLM_MODEL` is what a *host* command
calls; the containers get `LLM_CONTAINER_MODEL`, and compose refuses to
start without it:

```ini
LLM_CONTAINER_MODEL=ollama_chat/gemma4:12b
```

The two are separate because **some credentials exist only where a person
is** — an interactive cloud login, a key in a login keychain — and a
container holds none of them. Inheriting the host's model is how a stale
container spent 2,019 restarts failing to authenticate. So the containers
run a local model they can reach, the host runs whatever it can, and both
drain the same queues at once: claiming is `FOR UPDATE SKIP LOCKED`, so
that is two halves of one corpus rather than two runs of it.

Set them to the same value if one model serves both.

```bash
make install
make doctor
```

`install` fetches the dependencies and the spaCy pipelines named in
`NLP_MODELS`. Those are also baked into the backend image, because the
runtime does not download them.

`doctor` checks the tools, the configuration and then **asks the model one
question**, which is the only way to find out that a model id is wrong, an
address is unreachable or a credential is stale. Without it the first thing
to discover any of those is the first passage of a real run, hours in.

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

```bash
make help        # every target there is, grouped, with what each one does
make services    # which containers are up, and where to open them
```

`make` on its own prints that first list. It is read out of the Makefile
rather than written down twice, so it cannot fall behind the targets.

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
| [`backend/database/`](backend/database/README.md) | The schema, the migrations and the triggers |
| [`backend/blob_store/`](backend/blob_store/README.md) | The four buckets |
| [`backend/archive/`](backend/archive/README.md) | What a deletion left behind, and the second deletion |
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
| [`tests/`](tests/README.md) | Eleven layers, and what each needs |
| [`configs/`](configs/README.md) | Service configuration and init scripts |
| [`docs/architecture.md`](docs/architecture.md) | The four graphs: the pipeline, the queue, the imports, the containers |
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
