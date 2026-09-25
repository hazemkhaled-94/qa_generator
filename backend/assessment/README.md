# Assessment — the evaluation phase

An LLM judge over every fact, topic and question, run **after** the whole
pipeline and never during it. It records an opinion beside the checker's
verdict and **decides nothing**.

```bash
make assess-enrol            # what it would cost, before paying for it
make assess-start && make assess
make assess-status
```

Off unless `ASSESSMENT_ENABLED` is true in `.env`.

## Why it is called assessment

Because `evaluation/` is taken. That package is the golden-set runner — it
scores a served model against `cases.py` on the host. Two packages of one
name on two import roots is a `make` target that picks whichever `sys.path`
put first, so this is `assessment` everywhere the code says it and "the
evaluation phase" everywhere a person does.

## What it never does

**It is not a gate, and it cannot become one.** `facts.validated` is the
checker's and `questions.status` is the gates'. This phase writes neither.
It writes `assessments.approved`, which no stage reads.

That is not caution for its own sake. [`evaluation/README.md`](../../evaluation/README.md)
records the measurement: over nineteen labelled cases gpt-4.1 answered a
phrasing judgement **7/7 in English and 6/12 in German**, which is chance.
Half of this corpus is German. A phase that could reject rows would have
thrown away good German questions on the strength of a coin.

What a judge is good for is the **disagreement** — and that is what this
produces, at corpus scale, where `make second-opinion` produced it for one
run into a terminal.

## What is judged, and how

Three kinds, which are the three [`review/`](../../review/README.md)
already puts in front of a person. A passage and a parsed document are
absent: judging a conversion needs the source page rendered beside it, and
nothing here keeps one.

| Artefact | Judged on | Asked | Calls |
|---|---|---|---|
| `fact` | `hallucination` | Do the cited sentences support every part of the statement? | 3 |
| | `relevance` | Are those sentences *about* what the statement is about? | |
| | `toxicity` | Did the pipeline write something offensive of its own? | |
| `topic` | `summarization` | Is the label a good short name for these terms? | 2 |
| | `relevance` | Are these terms a subject, or the document's furniture? | |
| `question` | `hallucination` | Do the facts support every part of the answer? | 6 |
| | `qa_correctness` | Does the answer answer the question that was asked? | |
| | `relevance` | Do the facts contain what the question needs? | |
| | `refusal` | Is the answer an answer, or a refusal written down as one? | |
| | `conciseness` | Does the answer give what was asked for and stop? | |
| | `toxicity` | Did the pipeline write something offensive of its own? | |

### `refusal` is the one no gate makes

A question marked answerable whose stored answer is *"the document does not
specify"* is well formed, the right length, cites its facts and leaks no
source. **Every gate passes it**, and a benchmark built from it scores a
chatbot against a non-answer. Nothing else in this repository looks for it.

`conciseness` is the same shape one step down: `QUESTIONS_ANSWER_CHARS`
refuses an answer that is too long, and nothing refused one that spent its
allowance apologising.

`toxicity` is about what the pipeline **wrote**, not what the corpus is
about. A corpus may document harassment, and a neutral question about it is
not toxic — the templates say so at length, because a judge that conflates
the two would flag the most careful part of a difficult corpus.

### What is deliberately not asked

phoenix-evals ships 26 evaluators. The seven above are the ones that apply.

| Not asked | Why |
|---|---|
| `faithfulness`, `correctness` | Near-duplicates of `hallucination` and `qa_correctness` — the same judgement with different rails. A second call asking the same question. |
| `hallucination_span_level`, `qa_span_level` | Span-level variants of two already asked. |
| `code_readability`, `code_functionality`, `sql_gen_eval` | No code and no SQL is produced. |
| `tool_calling`, `tool_selection`, `tool_parameter_extraction`, `tool_invocation`, `tool_response_handling` | Nothing here calls a tool. |
| `reference_link_correctness` | Artefacts carry no links. |
| `human_vs_ai` | Needs a human-written reference answer. `review/` stores a human **verdict**, not a human answer. |
| `user_frustration` | Needs a user conversation. |
| `document_relevance` | This *is* `relevance` above. |

**An artefact is approved only when every one of its metrics approves.** Not
a majority: an answer that is well-sourced and answers the wrong question is
worth somebody's time even though two of its three passed.

### Unanswerable questions are not judged

Two of the three question metrics ask whether the facts support the answer,
and a deliberately unanswerable question has none. Scoring it anyway would
report the pipeline's most careful output as its worst. `second_opinion.py`
samples the same way for the same reason.

## The names are phoenix-evals'. The prompts are not.

This is the one decision in the package worth arguing about, so:

**The names, the labels, the scores and the optimisation direction** are
read off `arize-phoenix-evals`' own evaluator configs and pinned against
them in `tests/unit/assessment/test_templates.py`. A `hallucination`
annotation here sits in the same Phoenix Evaluations view as one posted by
anything else built on that package, and a reader sorts and charts them
together. Renaming it, or scoring it the other way round, would produce a
column that looks like the one they know and behaves differently.

> `hallucination` is the confusable one. Its direction is **minimise**, so
> `hallucinated` scores 1.0 and `factual` scores 0.0 — the opposite of the
> other three. That is why `assessment_metrics.approved` is a stored column
> rather than derived from the score: a reader summing scores across
> metrics would be adding a hallucination rate to a relevance rate.

**The prompts are this repository's**, for three reasons the shipped ones
cannot meet:

- **Half of this corpus is German.** The shipped templates instruct in
  English about an English answer. Every template here says the material may
  be in any language and that the judgement is about what it does, not what
  it is written in.
- **The artefacts are not a RAG answer.** phoenix-evals asks about a query,
  a context and a response. A fact has a statement and the sentences it
  cites; a topic has terms and a name. Substituting those into a three-slot
  template about retrieval asks the model to judge something it is not
  looking at.
- **An unexplained verdict is one a reviewer cannot act on.** Every template
  asks for the reason in the same call as the label.

`arize-phoenix-evals` is also host-side only — no image carries it — so
using it here would put a second LLM client, a second rate limiter and a
pandas dependency into a worker container to ask a question the client
already in that container can ask. The judging goes through
[`llm/client.py`](../llm/README.md) like every other model call: one retry
policy, one timeout, one place the reasoning tags are stripped, one place
the tokens and the spend are recorded.

**One call per judgement.** Three in one call is what
[`question_generation/phrasing.py`](../question_generation/README.md) exists
to undo.

## Where a verdict shows up

Six places, which is the point of it being a phase rather than a script.

| Where | What it shows |
|---|---|
| **Postgres** | `assessments` and `assessment_metrics`. The truth. |
| **Phoenix** | One annotation per metric, named as above, plus an `assessment` summary. `annotator_kind` is **LLM**, against **CODE** for the gates. |
| **The API** | `/assessment`, `/assessment/quality`, `/assessment/plan`, and the five queue verbs. |
| **The UI** | The Assessment page — a tab per artefact kind, a search over the judge's own reasoning, and a checkbox per kind deciding what the queue verbs act on. |
| **Dagster** | The `assessments` asset, last in the graph, with a `judge_agrees_with_the_checker` check that never fails. |
| **Argilla** | A `judge` field and two metadata terms on all three datasets. |
| **The export** | Two columns on the Questions sheet and an Assessment sheet of its own. |

### In Phoenix

Each artefact opens an `assess` span. The annotations hang off it:

| Annotation | On an approved fact | On one the evidence does not support |
|---|---|---|
| `assessment` | `approved`, 1.0 | `refused`, 0.0 |
| `hallucination` | `factual`, **0.0** | `hallucinated`, **1.0** |
| `relevance` | `relevant`, 1.0 | `relevant`, 1.0 |

The summary scores 1.0 for approved, so a project's mean under that name
*is* the approval rate. The per-metric scores keep phoenix-evals' own
direction, which is why they cannot be averaged together and why each
annotation carries its `direction` as metadata.

Best-effort, like the gate verdicts: a Phoenix that is down costs the
annotation and not the run.

### In Argilla

The judge's verdict and its reasoning ride along on the records
[`review/`](../../review/README.md) already pushes — `judge` as a field, and
`judge` and `judge_disagrees` as metadata a reviewer can filter on. The
guidelines say in as many words that it is not an answer key.

Anchoring is the risk and it is the lesser one, for the reason that package
already gives: a sample nobody can interpret gets abandoned.

## The queue, which nothing upstream fills

Every other stage claims rows something before it created. Nothing creates
an assessment: a fact arrives from extraction with nowhere to record a
judgement.

So **`start` enrols and then queues**. `ON CONFLICT DO NOTHING` per kind, so
running it twice enrols only what arrived since, and a re-run over a judged
corpus costs one statement per kind rather than a model call per row. That
is topic modelling's arrangement — asking is what creates the work — reached
from the other direction.

`--enrol` does the first half alone, which is the dry run: how many
artefacts this would cost a model call each, before anybody pays for them.

Enrolment ignores `--only`. A row sitting `new` costs nothing and no worker
looks at one, so enrolling a kind nobody asked to judge yet is free; the
alternative was reading a value back out of a SQLAlchemy condition, which
this codebase does nowhere else.

## Cost

A model call per metric: **three per fact, two per topic, six per question.**
A corpus of twenty thousand facts is sixty thousand calls, which is why the
default is a **sample of 200 per kind, newest first** and why the phase ships
off.

`ASSESSMENT_KINDS=question` is the setting that buys the most: the questions
are the deliverable, and they are the artefact whose `qa_correctness`
judgement has no counterpart anywhere upstream.

This is the only worker that loads **no** encoder — no spaCy pipeline, no
embedder, no NLI model — so it is also the cheapest stage to scale
horizontally.

## Configuration

From `.env`, because both are one deployment's decision:

| Setting | Default | What it does |
|---|---|---|
| `ASSESSMENT_ENABLED` | `false` | Whether the phase runs at all |
| `ASSESSMENT_JUDGE_MODEL` | — | Who judges. **Not** `LLM_MODEL`: a model asked whether its own facts follow from its own evidence says yes, and the worker warns on every start when it is unset |

From [`configs/env/backend.env`](../../configs/env/backend.env), in git:

| Setting | Default | What it does |
|---|---|---|
| `ASSESSMENT_KINDS` | `fact,topic,question` | Which artefacts are judged |
| `ASSESSMENT_SAMPLE` | `200` | How many of each kind one `start` enrols, newest first. 0 is all of them |

## Layout

| File | Holds |
|---|---|
| [`templates.py`](templates.py) | The seven judging templates, and the names, labels and scores borrowed from phoenix-evals |
| [`judge.py`](judge.py) | Asking one model one judgement, and abstaining when it will not answer |
| [`repository.py`](repository.py) | The queue, the enrolment, and reading the verdicts back |
| [`service.py`](service.py) | The drain loop, and the annotations it posts |
| [`config.py`](config.py) | The flag, the judge and the two cost dials |
| [`factory.py`](factory.py) | Wiring |
| [`run.py`](run.py) | The command line |

Reads `database.qa_generator` directly and no other service's repository —
this stage is about facts, topics and questions at once, and a package
importing all three services to read them would weld the four together.

## Tests

```sh
poetry run pytest tests/unit/assessment
```

| File | Covers |
|---|---|
| [`test_templates.py`](../../tests/unit/assessment/test_templates.py) | The metric names and scores against phoenix-evals' own, and every placeholder against the fields that fill it |
| [`test_judge.py`](../../tests/unit/assessment/test_judge.py) | One call per metric, and an abstention against a refusal |
| [`test_assessment_service.py`](../../tests/unit/assessment/test_assessment_service.py) | The row, the annotations, and the columns this phase does not write |
| [`test_assessment_config.py`](../../tests/unit/assessment/test_assessment_config.py) | The flag, the lease and the judge |

## Limits

- **Nothing here gates anything, ever.** No stage reads `approved`.
- **A judge that cannot be reached fails a row; it does not refuse an
  artefact.** `assess-retry` is what moves it.
- **One metric abstaining leaves a partial verdict.** The artefact is still
  `assessed`, judged on the metrics that answered.
- **Unanswerable questions are not enrolled.** They have no answer to judge.
- **`hallucination` scores 1.0 for the bad label.** Read the direction.
- **A sample is the default.** `ASSESSMENT_SAMPLE=0` judges the corpus, and
  on a large one that is tens of thousands of model calls.
- **A question costs six calls and a fact three.** Judging questions alone
  is the setting that buys the most.
- **A row judged under template version 1 carries three metrics fewer.**
  The version is on the row; `make assess-rerun` re-judges it.
- **Enrolment ignores `--only`.** The requeue after it does not.
- **A judge that is `LLM_MODEL` is marking its own work.** Warned, not
  refused.
- **Deleting a fact, topic or question deletes its assessment.** Three
  cascading foreign keys, so a verdict never outlives what it is about.
- **A re-run overwrites.** A corpus never holds two opinions from two
  template versions with nothing saying which is current.
