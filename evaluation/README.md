# Evaluation

The golden cases, and scoring a served model against them in Phoenix.

`make test-eval` scores a model against the cases in [`cases.py`](cases.py)
and **prints** the numbers. Printing is the problem: a prompt change, a model
change or a quantisation change moves precision and recall, and the question
anybody has is whether it moved them *up*.

This package is the answer to that question. Phoenix records each run so two
can be compared by more than their scrollback.

## Using it

```bash
make eval-upload                                 # the cases become a Phoenix dataset
make eval-score                                  # score the model, and record it
make eval-score EVAL_RUN_NAME=extraction-prompt-v7
```

Underneath:

```bash
python -m evaluation.run --upload extraction
python -m evaluation.run --score extraction
```

The two are mutually exclusive and one is required. Results are at
<http://localhost:6006>, signed in as `admin@localhost` with
`PHOENIX_DEFAULT_ADMIN_INITIAL_PASSWORD`.

## What makes two runs comparable

**Phoenix versions the dataset**, so a case added today does not invalidate
yesterday's experiments — they stay attached to the version they scored.

Each run records the model, the temperature and the structured mode as
metadata, so two runs are comparable by more than their timestamps.

`EVAL_RUN_NAME` is optional, and its absence means something: unset, the
runner names the experiment after the **model** it scored, which is what makes
two runs of two models comparable without anybody typing anything. Set it when
the model is the same and something else changed — a prompt, a temperature, a
quantisation — because then the model name is not what tells them apart.

## Scored by the pipeline's own checker

Not by an LLM judge. The checker is what decides whether a fact is kept in
production, so a golden set scored by anything else measures something this
pipeline does not use.

## Only the extraction set is scored

That is a decision rather than a gap.

| Set | Why |
|---|---|
| `extraction` | Its numbers are **pure measurement**, asserted against nothing, which is what is worth comparing between two prompts |
| `questions` | Already **asserts** — the recoverable cases must pass and the rest must be stopped — so it is a gate that fails a pull request rather than a trend that draws a line |

The questions set is uploaded anyway, so the cases are browsable and an
experiment can be run against them from the Phoenix UI.

## Never a gate

For the same reason `make test-eval` is not: a model's answers move between
versions, between quantisations, and between two runs at the same temperature.
A threshold here would fail on somebody else's Tuesday rather than on a
regression.

The one thing `tests/eval/` does assert is that the round-trip gate splits its
golden questions the right way round — which is not a measurement of the
model's taste. It is whether the gate is wired up at all, and **a gate that
accepts everything cannot be told from no gate**.

## The baseline, 2026-09-20

What the models named in `.env` scored before anything was moved off them —
`azure/gpt-5.4` writing and `azure/gpt-4.1` judging. Recorded because every
swap below it is measured against these numbers and not against an opinion.

The three phrasing judgements, each scored on its own over the nineteen
labelled cases, against the floor a judge that ignores its input would reach:

| Judgement | Score | Answering the same way every time |
|---|---|---|
| `names_its_source` | 15/19 — 78.9% | 63.2% |
| `self_contained` | 16/19 — 84.2% | **78.9%** |
| `subject` | 16/19 — 84.2% | — |

**`self_contained` beats a constant answer by one case.** That is the measured
version of the README's observation that it fired zero times in 3,131
questions: it is not reading the question.

The split that says what is actually wrong is by language:

| | Both judgements right |
|---|---|
| English | 7/7 — 100% |
| German | 6/12 — **50%** |

Every one of the seven misses is German. gpt-4.1 answers these perfectly in
English and at chance in German, in a call whose other half — recovering the
answer — it does well in both. A judgement sharing a call with a harder task
is answered in the language the harder task is thinking in.

Four of the misses are `names_its_source` on `Kapitel 5`, `[R22]` and
`Beck 2003` — the three a pattern matches without a model at all.

Two golden question cases fail at this baseline, and they are different in
kind:

- **the LMT case** — `Was ist ein Liquiditätsmanagementtool?` answered
  `eine einjährige Rückgabefrist` is accepted, because the entailment rescue
  in `_backed` says the passages support it. The case is in the set precisely
  because recoverability used to catch what an NLI model and an LLM judge
  both missed, so this is a regression and not a mislabelling.
- **`a plausible question the passage does not cover`** — its `target` is
  `None`, so the checker reads it as an unanswerable question, and a passage
  failing to answer one of those is the point rather than a fault. The case
  asserts the opposite. That one is the harness being wrong.

## What moved, same day

The same command after the judgements were split out of the reading call.
The set is **larger and harder** than the baseline's — 25 cases against 19,
with six added for the phrasing the old measurement could not see — so these
are not the same nineteen scored again:

| Judgement | Baseline | After | Answered by |
|---|---|---|---|
| `names_its_source` | 15/19 — 78.9% | **25/25** | `gates.cites_source`, a model for the residue |
| `self_contained` | 16/19 — 84.2% | **25/25** | `nlp.pointing`, then a model on what it found |
| `subject` | 16/19 — 84.2% | **25/25** | `gates.subject`, no model |

| | Baseline | After |
|---|---|---|
| English | 7/7 — 100% | 9/9 — 100% |
| German | 6/12 — **50%** | **16/16 — 100%** |

Both golden question cases pass: the LMT rescue is refused again, and the
mislabelled unanswerable case now carries the answer somebody would wrongly
give it.

**Where to distrust this.** Six of the 25 cases were written in the same
change as the rules and the widened measurement, and a rule written against
a case it is then scored on has not been tested by it. The four the baseline
got wrong — `Kapitel 5`, `[R22]`, `Beck 2003`, `in diesem Lehrplan` — are
the honest evidence here, because those were labelled before anything was
built to catch them. Treat the pointing cases as a regression test rather
than as a measurement until a run produces some of its own.

### What the entailment encoder actually does with a fragment

Measured against the cached `mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`,
because it decides how `_backed` passes its hypothesis:

| Premise | Hypothesis | Entailment |
|---|---|---|
| the LMT passage | `Ein Liquiditätsmanagementtool ist eine einjährige Rückgabefrist.` | 0.262 |
| the LMT passage | `eine einjährige Rückgabefrist` | **0.970** |
| `…within 48 hours…` | `48 hours` | 0.957 |
| `…within 48 hours…` | `4 hours` | 0.076 (0.896 contradiction) |

The encoder refuses the LMT claim when it is put as a **proposition** and
accepts it as a **fragment** — and a fragment is what a target answer is. So
the encoder does not catch that case and was never going to; `gates.about`
is what does, by reading the question instead of the answer.

Prefixing the question to the answer was tried and is worse: it separates
the LMT pair correctly and drops a plainly good English case to 0.486, which
is under the threshold. The fragment stays, because the pass can only ever
accept and a guard that refuses good rescues costs more than it saves.

## What the encoders do to real questions, 2026-09-20

Measured against **120 accepted answerable questions drawn from this
corpus**, not against a golden set — the question is what would happen to
the run, so the input is the run's own output.

### The QA encoder is the wrong comparison, and it is off

`timpal0l/mdeberta-v3-base-squad2`, 0.19 s a question:

| Answer form | n | span at ≥0.9 | agrees with the target | net usable |
|---|---|---|---|---|
| `value` | 37 | 70% | 54% | **38%** |
| `explanation` | 53 | 72% | 21% | **15%** |
| `list` | 30 | 57% | 6% | **3%** |

It finds a confident span most of the time and **the span is not the
answer**: `750 Minuten` comes back as `mindestens 22,75 Unterrichtsstunden`,
`36 K2-LOs` as `36`. Both are in the passage; neither passes `agrees`.

The cause is structural, not a bad checkpoint. A target answer was *written*
by the question writer — normalised, in prose, in the form its type asked
for — and an extractive span is a literal substring. Comparing the two on
content lemmas fails by construction. **Turning this on would flip roughly
half the accepted set to `not_recoverable`**, so `QA_MODEL` is empty.

Worth recording separately: `deepset/xlm-roberta-large-squad2`, the obvious
choice and a company's, **cannot be loaded under transformers 5** — it ships
no `tokenizer.json` and its sentencepiece model is mis-read as a tiktoken
file. `transformers` 5 also removed the `question-answering` pipeline
entirely, so [`nlp/qa.py`](../backend/nlp/qa.py) does the span decode itself.

### The NLI encoder separates, and it is on

`bge-m3-zeroshot-v2.0`, 1.72 s a question. Each target answer against its
own passages, and — as a negative control — against another question's:

| Threshold | Backs its own answer | Backs a stranger's |
|---|---|---|
| 0.5 | 67% | 6% |
| 0.7 | **59%** | **3%** |
| 0.9 | 40% | 1% |

That is a usable separation, and the failure direction is the safe one. The
pass can only ever *accept*, so the number to keep low is the 3%; the 41% of
genuine answers it does not back are questions that were already refused and
stay refused. `NLI_ENTAILMENT_THRESHOLD=0.7`.

What this does **not** say is whether gpt-4.1 rescued more of the genuine
ones. Replacing it is a saving of ~250 calls and possibly a loss of some
rescues, and nothing here has measured that side.

### The phrasing judgements run well locally

The residue each judgement actually asks, against models already on the
host, scored on the labelled cases:

| Model | `names_its_source` | `self_contained` | Per call |
|---|---|---|---|
| `granite4.2:3b` | 19/20 | **2/9** | 0.3 s |
| `granite4.2:8b` | 19/20 | 6/9 | 0.9 s |
| `gemma4:12b` | 19/20 | **9/9** | 3.2 s |
| `gemma4:31b` | 19/20 | **9/9** | 7.8 s |

Two findings. **The 31B buys nothing over the 12B here** and costs 2.4× the
time. And the two judgements are not the same difficulty: naming a source is
easy enough for a 3B, while deciding whether a pointing word lands needs 12B
— which is the kind of thing that stays hidden while both ride in one call.

For reference, gpt-4.1 scored 15/19 and 16/19 on these when they shared the
reading call.

## One topic run end to end, 2026-09-20

`make questions-rerun TOPIC=439`, with `NLI_MODEL` on and `QA_MODEL` off.
67 questions, 27 accepted, no errors.

| Gate | This topic | Baseline run |
|---|---|---|
| accepted | 40% | 66% |
| `duplicate` | 25% | 14% |
| `compound` | 10% | 10% |
| **`leaks_source`** | **10%** | **0.3%** |
| `not_recoverable` | 10% | 5% |
| **`unanchored`** | **1.5%** | **0%** |

**Read the acceptance rate with care.** A rerun is offered the facts the
first run did not reach, which are the leftovers rather than a sample, so
`duplicate` is inflated by design and the rate is not comparable. What the
fact selection cannot explain is `leaks_source` going from 9 in 3,131 to 7
in 67 — a 35-fold rise — and `unanchored` firing at all.

Four of the seven `leaks_source` rejections came from `cites_source` and
**all four are right**: three name `Kapitel 3` and one says `in diesem
Lehrplan`. The other three came from the model residue, and one of those is
arguably wrong — *"Welche Unterlagen nennen eine Methodik …"* asks WHICH
documents say something rather than citing one.

The single `unanchored` rejection was **wrong, and it found a real defect**.
*"…die im Korrekturverzeichnis genannten Abschnitte…"* says inside the
question where to look, so `genannt` points at nothing outside it — but
bare `genannt` was in the anaphoric trigger list, and the model agreed with
the measurement, so both halves of the two-holder rule failed together. The
trigger is now the unambiguous compounds only (`vorgenannt`, `obengenannt`,
`besagt`), and a trigger must be tagged as a modifier, which keeps English
`said` from matching the past tense of `say`.

`not_recoverable` doubling is the one to watch on the next run. It is
consistent with the NLI encoder rescuing less than gpt-4.1 did — measured
above at 59% recall on genuine answers — but the leftover facts are a
confound, so this is a hypothesis rather than a finding.

## Layout

| File | Holds |
|---|---|
| [`cases.py`](cases.py) | The golden cases, and nothing that runs them. Read by this package **and** by `tests/eval/` |
| [`experiments.py`](experiments.py) | Scoring the cases into Phoenix |
| [`config.py`](config.py) | Where Phoenix is |
| [`run.py`](run.py) | The command line |

`cases.py` holding no runner is what lets one set of cases serve both the
pytest layer and the Phoenix layer without either importing the other.

## Tools, and where each is used

| Tool | Where | Why this one |
|---|---|---|
| **arize-phoenix** | [`experiments.py`](experiments.py) | Versioned datasets and experiment history, in the same service that already collects the traces |
| The pipeline's own checker | [`experiments.py`](experiments.py) | The evaluator. Anything else would measure something production does not use |

Phoenix is the same service `OTEL_CONTAINER_ENDPOINT` sends spans to, read the
other way round.

## Configuration

| Setting | Where | Default | What it does |
|---|---|---|---|
| `EVAL_RUN_NAME` | [`configs/env/evaluation.env`](../configs/env/evaluation.env) | unset | What a run is called. Unset names it after the model |
| `PHOENIX_BASE_URL` | `.env` | `http://localhost:6006` | Phoenix's HTTP API, as the host reaches it |
| `PHOENIX_ADMIN_SECRET` | `.env` | — | At least 32 characters, including a digit and a lower-case letter |
| `LLM_MODEL`, `LLM_BASE_URL` | `.env` | — | The model being scored |

`PHOENIX_DEFAULT_ADMIN_INITIAL_PASSWORD` is applied only when Phoenix first
creates its admin user — changing it afterwards means dropping the `phoenix`
database.

## Tests

```sh
poetry run pytest tests/unit/evaluation   # no network
poetry run pytest tests/eval              # needs a served model
```

| File | Covers |
|---|---|
| [`tests/unit/evaluation/test_experiments.py`](../tests/unit/evaluation/test_experiments.py) | The scores an experiment records, and the shape it records them in |
| [`tests/eval/test_extraction_quality.py`](../tests/eval/test_extraction_quality.py) | How well a real served model reads facts out of a passage |
| [`tests/eval/test_question_quality.py`](../tests/eval/test_question_quality.py) | How a real served model writes and checks questions |

`tests/eval/` is excluded from `make test`. The nightly workflow runs it, and
skips itself unless `LLM_MODEL` and `LLM_BASE_URL` are set as repository
variables.

## Known edges

Things that are true, are not bugs, and have surprised somebody.

- **Nothing here gates a merge, ever.** By design. A model's answers move
  between two runs at the same temperature.
- **`--score` needs the set uploaded first.** `--upload` is not implied.
- **A case added today does not invalidate yesterday's numbers.** Phoenix
  versions the dataset and the old experiments stay attached to the old
  version.
- **`EVAL_RUN_NAME` unset is the useful default.** Setting it to something
  static makes two runs of two models look like one experiment.
- **Changing `PHOENIX_DEFAULT_ADMIN_INITIAL_PASSWORD` after first boot does
  nothing.** It is applied once, when Phoenix creates the admin user.
