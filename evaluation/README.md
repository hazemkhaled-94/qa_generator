# Evaluation

The golden cases, and scoring a served model against them in Phoenix.

`make test-eval` scores a model against the cases in [`cases.py`](cases.py)
and **prints** the numbers. This package records each run in Phoenix instead,
so two can be compared by more than their scrollback.

The numbers themselves are in [docs/measurements.md](../docs/measurements.md).

## Using it

```bash
make eval-upload                                 # the cases become a Phoenix dataset
make eval-score                                  # score the model, and record it
make eval-score EVAL_RUN_NAME=extraction-prompt-v7
make eval-phrasing                               # the two phrasing judgements, against their floor
make second-opinion RUN=<run id>                 # an independent judge, and where it disagrees
make prompts-publish                             # the recorded prompts
```

Underneath:

```bash
python -m evaluation.run --upload extraction-golden
python -m evaluation.run --score extraction-golden
python -m evaluation.run --second-opinion <run id> --limit 50
python -m evaluation.run --publish-prompts
```

The four are mutually exclusive and one is required. Results are at
<http://localhost:6006>.

## Publishing the prompts

A span already carries the prompt it sent, filled in with that call's
passages. What it cannot show is the **template**, and
`llm.prompt_template.version` on the span names a version with nothing in
Phoenix under it.

Each recorded prompt becomes a Phoenix prompt, so a span's version opens
beside the trace and the playground can select one. A published version
carries the whole call:

| What Phoenix shows | Where it comes from |
|---|---|
| System message | `prompts.text`, composed by the stage that sends it |
| User message | `prompts.user_text`, the template with `{{name}}` where a call substitutes |
| Invocation parameters | `llm.config.Settings` — temperature, timeout, and `num_ctx` and `reasoning_effort` where sent |
| Response format | `prompts.response_schema`, the JSON schema of the Pydantic shape |

All four go through `PromptVersion.from_openai`. The plain `PromptVersion`
constructor takes a message list, a model name and a template format and
nothing else, so a version built that way carries an empty
invocation-parameters block and no response format.

Versions are published **oldest first**, because Phoenix keeps the one
created last as the prompt's current version.

The version and the digest go in the **version's** description, not the
prompt's: `prompt_description` is set when Phoenix creates a prompt and
ignored on every version after it. The prompt's description carries only
what never changes.

**A stage publishes what it records when it starts**, so this target is the
full republish rather than the only way in. The publisher lives in
[`backend/stages/publish.py`](../backend/stages/publish.py), beside the
write, and goes wherever that goes — a host command, a worker container,
the stage a Dagster run set going. `record` hands it the prompts it
actually wrote, which is none on every start-up after the first, and that
is what stops an identical Phoenix version piling up per prompt per run.

What this target covers is the one case the table cannot detect: a Phoenix
wiped while the rows stayed. It reads from the **database**, not from the
code, which is also what makes it work for a version the source has moved
past — a publisher that asked `question_generation.prompts` would load
litellm to read a string.

**The source is the code.** Phoenix's UI allows an edit and an edit there
reaches nothing: the next publish writes over whatever the UI did.

## Comparing two runs

```bash
make questions-runs                              # which runs there are
make questions-diff RUNS="a-gpt-4.1 b-gemma4-12b"
```

Two things made that possible, and neither is in this package:

- **`questions.run_id`**, from
  [`settings/runs.py`](../backend/settings/runs.py). `settings_version` names
  a *configuration*, so two runs under one configuration were
  indistinguishable — which is every A/B where only the model moved.
- **`archived_rows`**. `questions-rerun` deletes what it replaces, so
  producing run B used to destroy run A. The diff reads live rows **and**
  archived ones.

Each run's spans also go to a Phoenix project of its own,
`<stage>-<run id>`, which is the half SQL cannot answer: the gate counts are
columns, and the calls, tokens, latency and spend are in the spans.

## A second opinion, which is a queue and not a verdict

`make second-opinion RUN=<id>` puts a run's **accepted** answers to an
independent judge — `phoenix.evals`' hallucination classifier, through
litellm, against `QUESTIONS_VERIFIER_MODEL` — and prints where it disagrees
with the gates.

It settles nothing. What it produces is a list of ids worth a person's time,
which [`review/`](../review/README.md) takes directly:

```bash
make review-push-questions IDS=12,34,56
```

Only one direction is reported. A question the gates kept and the judge calls
unsupported is either a gate that let something through or a judge that is
wrong; a question the gates *rejected* and the judge calls factual is usually
neither, because `leaks_source` and `compound` say nothing about whether the
answer is in the passages.

This is the one place in the repository where an LLM judge is the right tool,
because its output is a queue rather than a column.

## What makes two runs comparable

Phoenix **versions the dataset**, so a case added today does not invalidate
yesterday's experiments. Each run records the model, the temperature and the
structured mode as metadata.

`EVAL_RUN_NAME` is optional, and its absence means something: unset, the
runner names the experiment after the **model** it scored. Set it when the
model is the same and something else changed.

## Scored by the pipeline's own checker

Not by an LLM judge. The checker is what decides whether a fact is kept in
production, so a golden set scored by anything else measures something this
pipeline does not use.

### Two of the three sets are scored

| Set | Scored | Because |
|---|---|---|
| `extraction` | yes | Its numbers are pure measurement, asserted against nothing |
| `phrasing` | yes | The same, and each judgement is scored beside the floor a judge that ignores its input reaches |
| `questions` | no | It already **asserts**, so it is a gate that fails a pull request rather than a trend |

The questions set is uploaded anyway, so the cases are browsable from the
Phoenix UI.

`make eval-phrasing` records four numbers, not two: `names_its_source` and
`self_contained`, and a `_floor` for each. The floor is what a judge that
answers the same thing every time would score on the same cases, computed
from `cases.py` rather than written down. A fifth, `answered`, counts how
much was judged at all, so a run where the model was unreachable does not
read as a run where the judge got everything wrong.

## Never a gate

A model's answers move between versions, between quantisations, and between
two runs at the same temperature. A threshold here would fail on somebody
else's Tuesday rather than on a regression.

The one thing `tests/eval/` asserts is that the round-trip gate splits its
golden questions the right way round — whether the gate is wired up at all.

## Layout

| File | Holds |
|---|---|
| [`cases.py`](cases.py) | The golden cases, and nothing that runs them. Read by this package **and** by `tests/eval/` |
| [`experiments.py`](experiments.py) | Scoring the cases into Phoenix, and the floor each score is read against |
| [`second_opinion.py`](second_opinion.py) | An independent judge over one run |
| [`config.py`](config.py) | Where Phoenix is |
| [`run.py`](run.py) | The command line |

`cases.py` holding no runner is what lets one set of cases serve both the
pytest layer and the Phoenix layer without either importing the other.

Nothing here runs in an image. Both this and `review/`'s Argilla client run
on the host: a worker has no business holding a judge. The backend image does
carry `arize-phoenix-client`, for the one thing a worker posts itself — the
gate verdicts in [`telemetry/evaluations.py`](../telemetry/evaluations.py).

## Configuration

| Setting | Where | What it does |
| --- | --- | --- |
| `EVAL_RUN_NAME` | [`configs/env/evaluation.env`](../configs/env/evaluation.env) | What a run is called. Unset names it after the model |
| `PHOENIX_BASE_URL` | `.env` | Phoenix's HTTP API, as the host reaches it |
| `PHOENIX_ADMIN_SECRET` | `.env` | At least 32 characters, including a digit and a lower-case letter |
| `LLM_MODEL`, `LLM_BASE_URL` | `.env` | The model being scored |

`PHOENIX_DEFAULT_ADMIN_INITIAL_PASSWORD` is applied only when Phoenix first
creates its admin user.

## Tests

```sh
poetry run pytest tests/unit/evaluation   # no network
poetry run pytest tests/eval              # needs a served model
```

| File | Covers |
|---|---|
| [`test_experiments.py`](../tests/unit/evaluation/test_experiments.py) | The scores an experiment records, and the shape it records them in |
| [`test_extraction_quality.py`](../tests/eval/test_extraction_quality.py) | How well a real served model reads facts out of a passage |
| [`test_question_quality.py`](../tests/eval/test_question_quality.py) | How a real served model writes and checks questions |

`tests/eval/` is excluded from `make test`. The nightly workflow runs it and
skips itself unless `LLM_MODEL` and `LLM_BASE_URL` are repository variables.

## Limits

- **Nothing here gates a merge, ever.**
- **`--score` needs the set uploaded first.** `--upload` is not implied.
- **A case added today does not invalidate yesterday's numbers.**
- **`EVAL_RUN_NAME` unset is the useful default.** Setting it to something
  static makes two runs of two models look like one experiment.
- **Changing `PHOENIX_DEFAULT_ADMIN_INITIAL_PASSWORD` after first boot does
  nothing.**
- **A phrasing score without its floor means nothing.**
- **`make second-opinion` cannot read a deleted run.** The gate counts can —
  `make questions-diff` reads the archive — but the judge needs the text the
  answer rests on.
- **Rows written before `run_id` existed carry NULL** and cannot be told
  apart. Nothing is backfilled.
