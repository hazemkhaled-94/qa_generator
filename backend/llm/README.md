# LLM

The served model, shared by the three stages that call one: extraction reads
passages with it, topic modelling names topics with it, and question
generation writes and verifies with it.

**One set of values for all three.** Importing [`client.py`](client.py) loads
litellm, and nothing outside a worker should name it —
`tests/static/test_api_stays_light.py` pins that the API loads none of it.

## What it does

### An answer in a shape, not prose to parse

Every call asks for a **typed object**. instructor wraps litellm and validates
the answer against a Pydantic model, so a stage receives a structure or an
exception.

`LLM_STRUCTURED_MODE` is how that shape is asked for. `JSON_SCHEMA` is the
default and is what a local runtime supports; a hosted provider may want
`TOOLS`. Getting it wrong is an error at the first call.

### Retries

Retried here with exponential backoff, up to `LLM_MAX_ATTEMPTS`: a connection
error, a timeout, an internal server error, a service-unavailable, and an
answer that would not validate. An authentication failure, an unknown model
or a malformed schema is raised at once.

**A rate limit is deliberately not in that list.** A hosted provider answers
429 with a `Retry-After` of twenty to sixty seconds, and tenacity cannot see
that header. It is handled a layer down: `num_retries` becomes litellm's
`max_retries`, which becomes `max_retries` on the provider's own SDK client,
and that one does read the header.

### What a stage may override

A stage may name a **different model** and nothing else:

```ini
EXTRACTION_MODEL=...
TOPIC_MODEL=...
QUESTIONS_MODEL=...
```

Each means `LLM_MODEL` when absent. Three more name a model for one
**judgement** rather than a stage — `EXTRACTION_DIGEST_MODEL`,
`QUESTIONS_PHRASING_MODEL` and `QUESTIONS_VERIFIER_MODEL` — and
`make spend-by-shape` is what each of them costs.

The address, the mode and the patience stay `LLM_*`: extraction's lease comes
from `LLM_TIMEOUT_SECONDS` and `LLM_MAX_ATTEMPTS`, so a stage that could
change either could have a healthy worker swept as abandoned.

`QUESTIONS_VERIFIER_MODEL` is a different thing: a **second** model that
checks the first one's work. See
[question generation](../question_generation/README.md).

### Changing provider

The model id's prefix picks the provider. Everything else is credentials, in
[`configs/env/provider.env`](../../configs/env/), which is **not in git**:

```ini
LLM_MODEL=ollama_chat/gemma4:31b       # a local Ollama
LLM_MODEL=azure/gpt-4o                 # Entra ID tokens, via azure-identity
LLM_MODEL=bedrock/anthropic.claude-... # AWS_ACCESS_KEY_ID and friends
```

`provider.env` is written by `make setup` from
[`provider.env.example`](../../configs/env/provider.env.example). It is
optional and absent for a plain Ollama. Compose hands it to the three
services that call a model and to nothing else, and the Makefile sources it
for a host command.

`make doctor` asks the configured model one question.

The one thing that does **not** move when you change providers is
`EMBEDDING_MODEL`, which runs locally in the worker.

### The containers call Ollama, and only Ollama

Some credentials exist only where a person is — an interactive cloud login, a
key in a login keychain — and a container has none of them.

```ini
LLM_MODEL=<what the host calls>                     # .env
LLM_CONTAINER_MODEL=ollama_chat/gemma4:12b          # what the containers call
QUESTIONS_VERIFIER_CONTAINER_MODEL=ollama_chat/granite4.2:8b
OLLAMA_CONTAINER_URL=http://host.docker.internal:11434
```

`LLM_CONTAINER_MODEL` and `OLLAMA_CONTAINER_URL` are **required**:
`compose.yaml` fails to start without them rather than falling back to
`LLM_MODEL`. That fallback is how a container created before
`LLM_CONTAINER_MODEL` existed kept a hosted model it had no credential for
and spent 2,019 restarts failing to authenticate — a compose variable is
resolved when a container is *created*, so a `.env` edit reaches a running
stack only through `up --force-recreate`.

There is no `LLM_CONTAINER_URL`: a container's address is Ollama's, because
there is nowhere else it may send a prompt. `make doctor` refuses a container
model that is not an `ollama*` one.

Both halves drain the **same queues at once**, because claiming is
`FOR UPDATE SKIP LOCKED`. That is throughput, and it is **not an A/B**: the
two cover different rows. Comparing two models is `RUN_ID` over the same
rows — see [settings](../settings/README.md).

### A worker proves the model before it claims anything

A watching worker whose model will not answer **stays up and asks again**,
backing off 5s, 10s, 20s to a minute. It claims nothing until the model
answers, and exits only when it is a one-off drain with a person holding the
exit code.

Exiting was the bug: under `restart: unless-stopped` a process is a restart,
a fresh `run_id` and a Phoenix project holding the one call that failed,
which made a pipeline that was down for a day read as one that had run two
thousand times.

`before_work` in [`check.py`](check.py) is the preflight
`stages.cli.queue_main` runs. It asks for one trivial structured answer,
before the watch loop — that loop logs an exception and polls again, which is
right for a drain that failed and wrong for a deployment that can never work.

```text
questions will not start: ollama_chat/no-such-model:1b at
http://host.docker.internal:11434 did not answer: ... model not found
```

Nothing in it names a provider: whatever authenticates a stage authenticates
the check.

## Cost and latency

Every call is logged with its duration, its token counts and, where the
provider prices it, what it cost:

```bash
make spend LOG=run.log                    # totals from a captured log
make spend LOG=run.log SINCE=2026-09-19   # one day of it
make spend-by-shape LOG=run.log           # split by model and judgement
```

`LOG` is **required**: it is the text log a drain wrote to a terminal, not
the shipped JSON. For a run happening now, Grafana's Pipeline throughput
dashboard and Phoenix are where the cost is.

The model is the bottleneck, not the pipeline — a median passage measured at
473 s on a 31B model. The panel worth watching is model latency p99: a p99
climbing towards `LLM_TIMEOUT_SECONDS` is healthy workers about to look
abandoned.

## Configuration

| Setting | Where | Default | What it does |
|---|---|---|---|
| `LLM_MODEL` | `.env` | `ollama_chat/gemma4:31b` | LiteLLM model id; the prefix picks the provider |
| `LLM_BASE_URL` | `.env` | `http://localhost:11434` | Where that model is served |
| `LLM_STRUCTURED_MODE` | `backend.env` | `JSON_SCHEMA` | How a typed answer is asked for |
| `LLM_TEMPERATURE` | `backend.env` | 0 | Zero, so a re-run is comparable to the last one |
| `LLM_TIMEOUT_SECONDS` | `backend.env` | 120 | How long one call may take. Extraction's lease derives from this |
| `LLM_MAX_ATTEMPTS` | `backend.env` | 3 | Attempts per call. The lease derives from this too |
| `LLM_NUM_CTX` | `backend.env` | unset | The context window to ask the runtime for |
| `LLM_REASONING_EFFORT` | `backend.env` | unset | For a model that has the knob. Unset sends nothing, so the model keeps its own default — a reasoning model thinks. Set `off` for a small model that cannot afford to |
| `OLLAMA_BASE_URL` | `.env` | `http://localhost:11434` | Where a self-hosted model is served. The address follows the provider |
| `LLM_CONTAINER_MODEL` | `.env` | unset | `LLM_MODEL` as the containers see it |
| `QUESTIONS_VERIFIER_CONTAINER_MODEL` | `.env` | unset | The verifier as the containers see it |
| `EXTRACTION_MODEL`, `QUESTIONS_MODEL` | `backend.env` | unset | One stage calling a different model. The model only |
| `TOPIC_MODEL` | `backend.env` | `ollama_chat/gemma4:12b` | The same, for topic naming |
| `EXTRACTION_DIGEST_MODEL`, `QUESTIONS_PHRASING_MODEL` | `backend.env` | `ollama_chat/gemma4:12b` | One judgement on a smaller model |

## Tests

```sh
poetry run pytest tests/unit/llm
```

| File | Covers |
|---|---|
| [`test_client.py`](../../tests/unit/llm/test_client.py) | Asking a served model, and what happens when it will not answer: which failures are retried, which are raised, and what the span carries |
| [`test_overridden.py`](../../tests/unit/llm/test_overridden.py) | One stage calling a different model, and that only the model moves |

No test here reaches a network. The evaluation layer is where a real served
model is asked anything — see [`evaluation/`](../../evaluation/README.md).

## Limits

- **A rate limit is not retried by this package.** Raising
  `LLM_MAX_ATTEMPTS` does not help a 429.
- **Raising `LLM_TIMEOUT_SECONDS` lengthens the extraction lease**, so a
  genuinely stuck worker is swept later too.
- **`LLM_TEMPERATURE=0` is not a guarantee of determinism.** A model's
  answers still move between versions and quantisations, which is why
  `tests/eval/` never gates.
- **`QUESTIONS_VERIFIER_MODEL` unset turns gates off rather than failing.**
  The worker warns on every start.
- **An unknown model id fails at the first call, not at start-up.** The only
  way to validate one is to call it; `make doctor` is that call.
