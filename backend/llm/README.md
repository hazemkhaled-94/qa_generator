# LLM

The served model, shared by the three stages that call one.

Extraction reads passages with it, topic modelling names topics with it, and
question generation writes and verifies with it. **One set of values for all
three**, because two settings for one served model is how the two come to
disagree.

Importing [`client.py`](client.py) loads litellm. Nothing outside a worker
should name it — the API process loads none of it, and
`tests/static/test_api_stays_light.py` pins that.

## What it does

### An answer in a shape, not prose to parse

Every call asks for a **typed object**. instructor wraps litellm and validates
the answer against a Pydantic model, so a stage receives a structure or an
exception, never a string it has to pick apart.

`LLM_STRUCTURED_MODE` is how that shape is asked for. `JSON_SCHEMA` is the
default and is what a local runtime supports; a hosted provider may want
`TOOLS`. Getting it wrong is an error at the first call, not a slow
degradation.

### Retries, and the one that is deliberately absent

Retried here, with exponential backoff, up to `LLM_MAX_ATTEMPTS`: a connection
error, a timeout, an internal server error, a service-unavailable, and an
answer that would not validate. An authentication failure, an unknown model or
a malformed schema is none of those and is raised at once.

**A rate limit is deliberately not in that list.** A hosted provider answers
429 with a `Retry-After` of twenty to sixty seconds; tenacity cannot see that
header, so the backoff here would retry twice inside three seconds and fail
the row for a condition that clears itself. It is handled a layer down
instead: `num_retries` on the call becomes litellm's `max_retries`, which
becomes `max_retries` on the provider's own SDK client, and that one does read
the header.

### What a stage may override, and what it may not

A stage may name a **different model** and nothing else:

```ini
EXTRACTION_MODEL=...
TOPIC_MODEL=...
QUESTIONS_MODEL=...
```

Each means `LLM_MODEL` when it is absent; `TOPIC_MODEL` ships naming a
smaller one, because naming a topic is a short prompt over a term list.
Three more name a model for one **judgement** rather than for a stage —
`EXTRACTION_DIGEST_MODEL`, `QUESTIONS_PHRASING_MODEL` and
`QUESTIONS_VERIFIER_MODEL` — and `make spend-by-shape` is what each of them
costs.

The address, the mode and the patience stay `LLM_*` — because a stage that could set its own timeout
would be a stage whose lease nobody could derive. Extraction's lease comes from
`LLM_TIMEOUT_SECONDS` and `LLM_MAX_ATTEMPTS`; if a stage could change either,
a healthy worker could be swept as abandoned.

`QUESTIONS_VERIFIER_MODEL` is the fourth, and it is a different thing: a
**second** model that checks the first one's work. See
[question generation](../question_generation/README.md#the-round-trip).

### Changing provider

The model id's prefix picks the provider — litellm's convention. Everything
else is credentials, and those go in
[`configs/env/provider.env`](../../configs/env/), which is **not in git**:

```ini
LLM_MODEL=ollama_chat/gemma4:31b       # a local Ollama
LLM_MODEL=azure/gpt-4o                 # Entra ID tokens, via azure-identity
LLM_MODEL=bedrock/anthropic.claude-... # AWS_ACCESS_KEY_ID and friends
```

`provider.env` is written by `make setup` from
[`provider.env.example`](../../configs/env/provider.env.example), which lists
the variables each provider wants. It is optional and absent for a plain
Ollama, which needs none of it.

Compose hands it to the three services that call a model and to nothing else.
The Makefile sources it too, for a host command that calls one — `make
extract` against a hosted provider used to read `backend.env` and `.env`,
find no key in either, and fail to authenticate.

`make doctor` asks the configured model one question, which is how a wrong
id, an unreachable address or a stale credential is found before a run pays
for it.

The one thing that does **not** move when you change providers is
`EMBEDDING_MODEL`, which runs locally in the worker whatever `LLM_MODEL` names.

### Entra ID, and why a stage is run from the host

Two-factor happens when a token is **issued**, not when it is used. So a
container never does the 2FA itself — it is either given an identity that
has none, or lent the result of a 2FA somebody already did.

| Where it runs | Credential | 2FA |
|---|---|---|
| A container, anywhere | A **service principal**: tenant, client, secret | None. An app identity is not a user |
| A container on Azure | Managed or workload identity | None. Nothing to set |
| **The host** | `az login`, found by `DefaultAzureCredential` | Once, in a browser, and the CLI keeps the refresh token |
| A container, lent the host's | `AZURE_OPENAI_AD_TOKEN`, minted host-side | Once, and again every hour when it expires |

The third row is why `make extract`, `make topics` and `make questions` are
run from the host against an Entra ID deployment: they inherit the `az login`
session, and the Makefile sources `provider.env` for them. A container could
use that session too, but only by being handed both halves — bind-mount
`~/.azure` *and* put the `az` CLI in the image, because `AzureCliCredential`
works by shelling out to it. The image carries no az, so the host is the
path that works today.

The fourth row is the one that goes stale. `make doctor` reads the token's
`exp` before it calls anything, because an expired token is taken in
preference to a working credential and the provider's refusal names four
possible causes without saying which.

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

The model is the bottleneck, not the pipeline: a median passage measured at
**473 s** on a 31B model. The panel worth watching during a long run is the
model latency p99 on the Pipeline throughput dashboard — a p99 climbing
towards `LLM_TIMEOUT_SECONDS` is healthy workers about to start looking
abandoned.

## Tools, and where each is used

| Tool | Where | Why this one |
|---|---|---|
| **litellm** | [`client.py`](client.py) | One model id names the provider, so changing provider is one line and no code |
| **instructor** | [`client.py`](client.py) | Validates the answer into a Pydantic model, so a stage gets a structure or an exception |
| **tenacity** | [`client.py`](client.py) | The backoff, over exactly the failures worth another attempt |
| **azure-identity** | declared, imported by litellm | Entra ID tokens on an `azure/*` model. Named in `pyproject.toml` because litellm imports it on that path |
| **OpenTelemetry** | [`client.py`](client.py) | One span per call carrying `llm.duration_ms`, so the latency panel and the log line are the same fact |

## Configuration

| Setting | Where | Default | What it does |
|---|---|---|---|
| `LLM_MODEL` | `.env` | `ollama_chat/gemma4:31b` | LiteLLM model id; the prefix picks the provider |
| `LLM_BASE_URL` | `.env` | `http://localhost:11434` | Where that model is served |
| `LLM_STRUCTURED_MODE` | `backend.env` | `JSON_SCHEMA` | How a typed answer is asked for |
| `LLM_TEMPERATURE` | `backend.env` | 0 | Zero, so a re-run of a stage is comparable to the last one |
| `LLM_TIMEOUT_SECONDS` | `backend.env` | 900 | How long one call may take. Extraction's lease is derived from this |
| `LLM_MAX_ATTEMPTS` | `backend.env` | 3 | How many attempts one call gets. The lease is derived from this too |
| `LLM_NUM_CTX` | `backend.env` | unset | The context window to ask the runtime for. Unset takes its default |
| `LLM_REASONING_EFFORT` | `backend.env` | unset | For a model that has the knob |
| `OLLAMA_BASE_URL` | `.env` | `http://localhost:11434` | Where a self-hosted model is served, for a stage naming `ollama_chat/…` while the shared model is somewhere else. The address follows the provider |
| `EXTRACTION_MODEL`, `QUESTIONS_MODEL` | `backend.env` | unset | One stage calling a different model. The model only |
| `TOPIC_MODEL` | `backend.env` | `ollama_chat/gemma4:12b` | The same, for topic naming |
| `EXTRACTION_DIGEST_MODEL`, `QUESTIONS_PHRASING_MODEL` | `backend.env` | `ollama_chat/gemma4:12b` | One judgement on a smaller model. Neither reads a passage the way the stage's own model does |

## Tests

```sh
poetry run pytest tests/unit/llm
```

| File | Covers |
|---|---|
| [`test_client.py`](../../tests/unit/llm/test_client.py) | Asking a served model, and what happens when it will not answer: which failures are retried, which are raised at once, and what the span carries |
| [`test_overridden.py`](../../tests/unit/llm/test_overridden.py) | One stage calling a different model from the rest, and that only the model moves |

No test here reaches a network. The evaluation layer is where a real served
model is asked anything — see [`evaluation/`](../../evaluation/README.md) and
`tests/eval/`.

## Known edges

Things that are true, are not bugs, and have surprised somebody.

- **A rate limit is not retried by this package.** It is handled by the
  provider's own SDK, which reads `Retry-After`. Raising `LLM_MAX_ATTEMPTS`
  does not help a 429.
- **Raising `LLM_TIMEOUT_SECONDS` lengthens the extraction lease.** That is
  the intent, and it means a genuinely stuck worker is swept later too.
- **`LLM_TEMPERATURE=0` is not a guarantee of determinism.** A model's answers
  still move between versions and between quantisations, which is why
  `tests/eval/` never gates.
- **`QUESTIONS_VERIFIER_MODEL` unset turns gates off rather than failing.**
  The worker warns on every start, and the verdicts are logged instead of
  applied.
- **An unknown model id fails at the first call, not at start-up.** Nothing
  here validates the id against the provider before work is claimed, because
  the only way to validate one is to call it. `make doctor` is that call,
  made deliberately and once, and it is not run for you.
