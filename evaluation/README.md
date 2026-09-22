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
make eval-phrasing                               # the two phrasing judgements, against their floor
make second-opinion RUN=<run id>                 # an independent judge, and where it disagrees
```

Underneath:

```bash
python -m evaluation.run --upload extraction-golden
python -m evaluation.run --score extraction-golden
python -m evaluation.run --second-opinion <run id> --limit 50
```

The three are mutually exclusive and one is required. Results are at
<http://localhost:6006>, signed in as `admin@localhost` with
`PHOENIX_DEFAULT_ADMIN_INITIAL_PASSWORD`.

## Comparing two runs

The tables further down this file were typed by hand off two terminals.
They are queries now.

```bash
make questions-runs                              # which runs there are
make questions-diff RUNS="a-gpt-4.1 b-gemma4-12b"
```

Two things made that possible, and neither is in this package:

- **`questions.run_id`**, from [`settings/runs.py`](../backend/settings/runs.py).
  `settings_version` names a *configuration*, so two runs under one
  configuration were indistinguishable — which is every A/B where only the
  model moved. `RUN_ID=<name>` names a run; unset mints a uuid.
- **`archived_rows`**. `questions-rerun` deletes what it replaces, so
  producing run B used to destroy run A. The A/B below says so outright:
  *"the first run's questions were deleted so the second was offered
  exactly what the first saw."* The diff reads live rows **and** archived
  ones, so a run deleted an hour ago still answers.

Each run's spans also go to a Phoenix project of their own,
`<stage>-<run id>`, which is the half SQL cannot answer: the gate counts
are columns, and the calls, tokens, latency and spend are in the spans.

## A second opinion, which is a queue and not a verdict

`make second-opinion RUN=<id>` puts a run's **accepted** answers to an
independent judge — `phoenix.evals`' hallucination classifier, through
litellm, against `QUESTIONS_VERIFIER_MODEL` — and prints where it disagrees
with the gates.

It settles nothing, by the same argument as everything else here. What it
produces is a list of ids worth a person's time, which
[`review/`](../review/README.md) takes directly:

```bash
make review-push-questions IDS=12,34,56
```

Only one direction is reported. A question the gates kept and the judge
calls unsupported is either a gate that let something through or a judge
that is wrong; a question the gates *rejected* and the judge calls factual
is usually neither, because `leaks_source` and `compound` say nothing about
whether the answer is in the passages.

This is the one place in the repository where an LLM judge is the right
tool, and the reason is that its output is a queue rather than a column.

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

## Two of the three sets are scored

That is a decision rather than a gap.

| Set | Why |
|---|---|
| `extraction` | Its numbers are **pure measurement**, asserted against nothing, which is what is worth comparing between two prompts |
| `phrasing` | The same, and each judgement is scored **beside the floor a judge that ignores its input reaches** — see below |
| `questions` | Already **asserts** — the recoverable cases must pass and the rest must be stopped — so it is a gate that fails a pull request rather than a trend that draws a line |

The questions set is uploaded anyway, so the cases are browsable and an
experiment can be run against them from the Phoenix UI.

### The floor is scored with the score

`make eval-phrasing` records four numbers, not two: `names_its_source` and
`self_contained`, and a `_floor` for each. The floor is what a judge that
answers the same thing every time would score on the same cases.

That column is the whole reason this set is scored here rather than
printed. The baseline below reads `self_contained` at 84.2% — which looks
like competence until the floor beside it says **78.9%**, one case in
nineteen. It is computed from `cases.py` rather than written down, so
adding a case moves it instead of leaving a stale number in a docstring.

A fifth, `answered`, counts how much was judged at all: `names_its_source`
abstains where no rule settles it and no model answered, and an abstention
is not a wrong answer. Without it, a run where the model was unreachable
reads as a run where the judge got everything wrong.

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

## The duplicate gate, measured 2026-09-20

`duplicate` is the largest single loss in the pipeline — 433 of 3,131 in the
baseline run, 25% of one topic's rerun, more than every model-backed gate
combined. It had never been looked at, because it costs nothing and fires
quietly.

Every one of the 454 duplicates was paired with the accepted question it was
refused against, and the pairs read wrong:

> *"Wofür ist der Lehrplan für Advanced Level Test Management gedacht?"*
> refused against
> *"Welche Stellen und Personengruppen dürfen den Lehrplan … verwenden?"*

Those are a purpose and an audience. What they share is a subject, and **a
sentence embedding encodes what a question is ABOUT** — two questions about
one subject sit on top of each other whatever they ask for.

Read off the question words, which `compound` already uses:

| | |
|---|---|
| duplicates examined | 454 |
| share a question word with their twin | 136 — 30% |
| **share none** | **318 — 70%** |

`warum` against `welche`, `wie` against `warum`. So `asks_the_same` is now a
veto on the probe, in the shape the phrasing gates already use.

**Graded rather than absolute**, because a different question word is
evidence and not proof: *"How heavy is the device?"* and *"What is the
weight?"* want the same answer. A question asking something else has to
reach `threshold + (1 - threshold) / 2` — 0.965 at the shipped 0.93 —
before it counts as a duplicate anyway. The measured false positives sit at
0.937 to 0.948; a real paraphrase sits above.

Over the stored corpus that keeps **270 of 454 duplicates, 59%** — about
8.6% of the whole run — and sends them on to the gates that read their
answers, which is where they should have been refused or kept on the merits.

## Four things measured and decided, 2026-09-20

### `compound` is right 89% of the time — and stays as it is

All 309 compound rejections, read against the question words that refused
them: 276 are also joined by `und`/`sowie`/`oder`, and **33 carry no
conjunction at all**. Every one of those 33 is the same shape — a question
word inside a subordinate clause:

> *"Wovon hängt ab, **wie** der Teststatus am besten kommuniziert wird?"*

That is one question. The fix looks obvious — count only question words
outside a subordinate clause, as `_predicates` counts only independent
finite verbs — and **it was tried and reverted**. Two parses defeat it.
German labels `Wie` and `Warum` `mo`, the same label a subordinate clause
carries, so testing the word discards every adverbial question; testing the
nearest finite verb above it instead works on 8 of 9 cases and still lets
*"Welche Änderung gab es bei den Reviews, **und wie** werden Testabläufe
formuliert?"* through, because the comma makes spaCy read the coordinated
clause as subordinate.

Letting a compound question into the benchmark is the costly direction — a
chatbot answering half of one is neither right nor wrong — so an 89%-correct
gate keeps its 11% error rather than trading it for an unknown one.

### The reranker moves no call and costs 7 seconds each

Measured, not assumed, because the first version of this note assumed:
`Qwen/Qwen3-Reranker-0.6B` on CPU takes **6.92 s per candidate**. It ranks
correctly — 0.996 for the passage that answers, 0.014 and 0.002 for two that
do not — and it is a 0.6B causal model scored on one token, so that is what
it costs here.

Selection reranks up to eight candidates per wide sample: ~2,300 samples ×
8 × 6.92 s is **about 44 hours**. For `answerable_elsewhere` alone it is ~1.5
hours, for a gate that fired 8 times in 3,131 questions. And it changes **no
Azure call count at all** — reordering is its whole effect.

`RERANKER_MODEL` is gone, and so is `nlp/reranking.py`. The setting was
empty and was always going to be; the numbers above are what is worth
keeping, and they are here.

### Answer equivalence does not rescue the QA route

The idea was that `agrees()` — lemma containment — is what rejected `750
Minuten` against `mindestens 22,75 Unterrichtsstunden`, and that scoring
answer *equivalence* instead would fix both the QA encoder and
`not_recoverable`.

Tested the cheapest version: mutual NLI entailment, each answer as premise
for the other, over the 81 confident QA spans.

| Threshold | Called equivalent | `agrees()` accepts |
|---|---|---|
| 0.5 | 18 — 22% | **23 — 28%** |
| 0.7 | 14 — 17% | **23 — 28%** |
| 0.9 | 11 — 14% | **23 — 28%** |

**Worse than what it would replace, at every threshold.** Same root cause as
before: NLI wants propositions and an answer is a fragment, so comparing two
fragments bidirectionally is further outside what XNLI trained on, not
closer.

The purpose-built option is BEM — `kortukov/answer-equivalence-bem`, 1.4k
downloads a month — and it is **BERT-base English**. Every answer-equivalence
checkpoint on the hub is. For a German corpus there is nothing to use.

### The NLI control: the encoder rescues MORE, not less

The pilot's `not_recoverable` doubling suggested the encoder was rescuing
less than gpt-4.1. Put to 30 real `not_recoverable` questions, with both
judges asked the same pairs:

| | |
|---|---|
| stopped by `asserted` and `about` before any judge | 10 |
| reaching a judge | 20 |
| both rescue | 1 |
| **only gpt-4.1** | **3** |
| **only the encoder** | **9** |
| neither | 7 |

So the hypothesis was **wrong**: the encoder rescues 10 where gpt-4.1
rescues 4. The pilot's doubling is the leftover-fact confound, not the
encoder.

That is the permissive direction on a gate that can only accept, which is
the one to watch — but it is bounded by the earlier control, where the same
encoder at the same threshold backed 3% of answers against passages that had
nothing to do with them. And a third of these never reach it: the `about`
guard stopped 10 of 30.

## The phrasing swap, A against B on one topic

Same topic, same fact pool — the first run's questions were deleted so the
second was offered exactly what the first saw. The only differences are the
phrasing model and the `genannt` fix.

| | A: `azure/gpt-4.1` | B: `ollama_chat/gemma4:12b` |
|---|---|---|
| questions written | 67 | 59 |
| **accepted** | **40%** | **39%** |
| `duplicate` | 25% | 27% |
| **`leaks_source`** | **10%** | **17%** |
| `not_recoverable` | 10% | 10% |
| `compound` | 10% | 5% |
| `unanchored` | 1 | 0 |
| **phrasing calls to Azure** | **~50** | **0** |
| phrasing calls to Ollama | 0 | 39 |

**Acceptance is unchanged** — 40% against 39% — which is the result that
matters: moving a third of the run's calls to a local model did not cost
questions.

`leaks_source` rising from 10% to 17% is **not a clean win**, and the
rejections say why. Four came from the rules, which are the same rules in
both runs. Of the six gemma4:12b refused: `in dem Foundation Level
Lehrplan`, `des Lehrplans` and `ISO 26262` are right, and `Welche K3-Themen
…` twice and `in den Angaben zur Testdurchführung` are a taxonomy code and a
vague back-reference rather than a source. So the local model catches more
true positives **and** over-fires where gpt-4.1 did not. The labelled set
scored it 19/20 and contains nothing of that shape, which is the gap to
close before trusting the number.

`unanchored` going 1 → 0 is the `genannt` fix removing the false positive
the pilot found, not the model.

## The full run under all of it, 2026-09-21

Every change above, over the whole corpus. 3,094 questions written, **1,744
accepted**, against the baseline's 3,131 and 2,077.

| Gate | Baseline | This run |
|---|---|---|
| **accepted** | **2,077 — 66.3%** | **1,744 — 56.4%** |
| `not_recoverable` | 147 | **512** |
| `compound` | 298 | 265 |
| `duplicate` | 433 | **180** |
| `leaks_source` | 9 | **89** |
| `answerable_after_all` | 93 | 83 |
| `unanchored` | 0 | **82** |
| `answerable_elsewhere` | 8 | **69** |

**Fewer questions, and that is not the same as worse.** Where the 333 went:

- `duplicate` fell by 253 and `not_recoverable` rose by 365. Those are
  mostly the same questions: the dedup gate ran *before* the round trip, so
  a question it refused never reached recall. Sampled, only 12% of the new
  `not_recoverable` is the `about` guard and 66% is a judge saying no. **The
  duplicates were largely questions that would have failed recall anyway** —
  which is worth knowing, because it means the 70%-different-question-word
  finding was right about the gate and wrong about the prize.
- `leaks_source`, `unanchored` and `answerable_elsewhere` refuse 232
  questions the baseline accepted.

Whether that is a gain turns on whether those 232 are bad, so they were
read. `leaks_source` splits 35 by rule and 54 by the model. Of ten
model-driven ones sampled, seven are right — `TM-2.3.1`, `FL-5.2.3`,
`ISO 25010`, `im Material`, `R01`, and *"Auf welcher Seite steht der
Indexeintrag?"*, which is a question about the document's pagination and
has no business in a benchmark — and three are wrong, including *"laut den
Angaben"*, which the labelled set says is False and which the rules
correctly abstain on. **So roughly a 30% false-positive rate on the model
half, and none measured on the rule half.**

`answerable_elsewhere` going 8 → 69 is not the extractive reader being
loose: it only ever *saves* a call there, and the verifier still confirms
every rejection. It is the probe retrieving on `passages.embedding` instead
of on shared lemmas and actually finding the passage that answers.

### What came out

`make questions-balance` drew **610 questions**, a 35% yield, balanced to
8–9% per type and 32–35% per band, 94% answerable. The pool came up short on
medium and hard, which is what limits the draw.

## Layout

| File | Holds |
|---|---|
| [`cases.py`](cases.py) | The golden cases, and nothing that runs them. Read by this package **and** by `tests/eval/` |
| [`experiments.py`](experiments.py) | Scoring the cases into Phoenix, and the floor each score is read against |
| [`second_opinion.py`](second_opinion.py) | An independent judge over one run, and the disagreements it found |
| [`config.py`](config.py) | Where Phoenix is |
| [`run.py`](run.py) | The command line |

`cases.py` holding no runner is what lets one set of cases serve both the
pytest layer and the Phoenix layer without either importing the other.

## Tools, and where each is used

| Tool | Where | Why this one |
|---|---|---|
| **arize-phoenix-client** | [`experiments.py`](experiments.py) | Versioned datasets and experiment history, in the same service that already collects the traces |
| **arize-phoenix-evals** | [`second_opinion.py`](second_opinion.py) | A tested hallucination template, a constrained answer, retries and concurrency — which is what `phrasing.py` hand-rolls per judgement. Used for the queue only, never for a verdict |
| The pipeline's own checker | [`experiments.py`](experiments.py) | The evaluator. Anything else would measure something production does not use |

Neither Phoenix package is in any image. Both run on the host, like
`review/`'s Argilla client and for the same reason: a worker has no
business holding a judge.

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
- **A phrasing score without its floor means nothing.** `make eval-phrasing`
  records both; reading one column is how 84.2% looked like competence.
- **`make second-opinion` needs the passages, so it cannot read a deleted
  run.** The gate counts can — `make questions-diff` reads the archive —
  but the judge needs the text the answer rests on, and that is gone.
- **Rows written before this commit have `run_id` NULL** and cannot be told
  apart. Nothing is backfilled, because inventing a name for them would
  make two runs look like one.
