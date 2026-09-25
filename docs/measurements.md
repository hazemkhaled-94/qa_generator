# Measurements

Numbers taken off real runs of this pipeline. They are recorded here so the
service READMEs can stay short and so a default can be changed against
evidence rather than an opinion.

All of it is one corpus unless stated: eight documents, 1,495 passages,
German and English.

## Model cost

| | |
|---|---|
| Median passage, 31B model | 473 s |
| `LLM_TIMEOUT_SECONDS` sized from | 9,334 priced calls, slowest 15.5 s |
| PDF conversion, 56 pages on two cores | 990 s |
| Conversion inside the API process | held one request 16 minutes |
| Question generation lease at 120 q/topic | 67 days |

A container inheriting the host's hosted model spent **2,019 restarts**
failing to authenticate, producing 2,116 Phoenix projects.

## Parsing

| | |
|---|---|
| Lower-bound confidence, this corpus | 0.72 – 0.94 |
| `enforce_same_font` left on | 397 text splits on one 73-page document |
| `enforce_same_font` off | none |

## Chunking

| | |
|---|---|
| Reading `row_header` as a header row | left 82% of cells with nothing quotable |
| German passages inside English-labelled files | 45 |
| `de_core_news_sm` vs `_md`, modal as finite verb | 3 of 8 vs 8 of 8 sentences |
| German facts refused for that parser limitation | 11% |

## Extraction

### The table identifier share

| | Mean identifier share | Over 0.5 |
|---|---|---|
| Tables carrying prose | 0.155 | none |
| Matrices, release notes, abbreviation lists | 0.424 | 28 |

Threshold is 0.5, set where nothing good is refused.

### The atomic cap

Read without a cap this corpus gave **18.4 atomic facts a passage**, and one
author list became fifty facts of the form *X wrote the original edition*.

The dedup HNSW index answers a corpus of this size in about **2 ms**, against
the minutes the model call that wrote the fact took.

## Topic modelling

### Why factorisation rather than LDA

| | c_v coherence |
|---|---|
| LDA on tf-idf | 0.514 |
| LDA on counts | 0.522 |

Weighting the input, 501 German passages at twelve topics: inter-topic term
overlap **32% → 11%**, coherence **0.52 → 0.66**.

### Why a passage rather than a sentence

| Unit | c_v (German, 12 topics) |
|---|---|
| Passage | 0.522 |
| Sentence | 0.396 |

A sentence averages 5.7 content tokens, and 6% of sentences hold none of the
vocabulary.

### How many topics

501 German and 240 English passages, three seeds each:

| k | de coherence | biggest topic | en coherence | biggest topic |
|---|---|---|---|---|
| 8 | 0.704 | 71% | 0.561 | 44% |
| 12 | 0.656 | 46% | 0.553 | 37% |
| 16 | 0.599 | 40% | 0.591 | 29% |
| 20 | 0.613 | 30% | 0.582 | 26% |

Coherence alone always prefers fewer topics: at four, one German topic held
71% of the corpus. Read beside that share and inter-topic overlap, twelve was
the best German point — lowest overlap (11%), most stable across seeds.

`TOPIC_PASSAGES_PER_TOPIC=40` replaced the fixed count. This corpus carries
1,434 passages in one language and 51 in the other; twelve topics over each
gave twelve subjects on one side and twelve slivers of four passages on the
other, which produced thirteen questions between them.

### `TOPIC_NO_ABOVE`

Swept 0.02 to 1.0. Below 0.5 the widespread *real* terms go first, promoting
noise into the top terms; at 0.02 coherence collapses and 120 passages fall
out of every topic. Above 0.5 nothing changes — no term reaches half the
corpus.

### Known limits of the output

| | |
|---|---|
| German topic #2, an English topic | 9.3% of the German corpus |
| Three German apparatus topics (table headers, captions) | ≈19% of the German corpus |
| English model | 12 nameable subjects, evenly 5.5% – 11.4%, no apparatus |

Title-cased English headings defeat the capitalisation rule that drops
foreign lower-case nouns, which is how English function words reached the
German vocabulary. `tests/unit/topics/test_vocabulary.py` carries a strict
`xfail` reproducing it.

Front-matter topics by share of dominant passages carrying a validated fact:
**0.00 and 0.11**, where the next topic up was **0.56**. That is what
`TOPIC_LABEL_MIN_FACT_SHARE` reads.

## Question generation

### Pairing the second passage

Choosing the second side for what it **meets** rather than for its own rank,
over 120 real cross-document pairs, halved the pairs offered with nothing in
common: **73 → 38**.

34% of multi-passage questions still rest on a pair no more alike than a
well-ranked random draw.

### `QUESTIONS_MEETS_FLOOR`, and why it ships at 0

Swept over 1,853 accepted multi-passage questions:

| Floor | Narrowed | Of those, welded | Weld rate left |
|---|---|---|---|
| 0.80 | 5.0% | 55.4% | 31.1% |
| 0.84 | 37.3% | 45.7% | 24.3% |
| 0.88 | 75.8% | 39.0% | 11.1% |

There is no knee. At 0.84 it throws away a third of every multi-passage
question to remove a minority of welds. Only 4% of questions carry a
coordinated noun pair at all, so there is nothing to build a gate on either.

Offering condensed facts first was measured and does not reach the weld: the
share of offered facts that are condensed moves **25.6% → 27.8%**, because a
passage is dealt every fact it holds either way. Questions citing a condensed
fact weld less often — 19.7% against 35.8% for an `explanation`, 32.0%
against 40.6% for a `list` — but that is narrowing after the fact, not
prevention.

### Boilerplate

| | |
|---|---|
| Accepted questions resting on a recurring passage | 14.1% |
| The same, with `QUESTIONS_BOILERPLATE_COSINE` on | 2.5% |
| Reading document *words* instead | fired on 58% at 26% precision |
| Of 234 passages with a cross-document twin, in a document's first or last twelve pages | 71%, against 17% of everything else |

Counting questions resting on those pages directly gives 20.6% and is an
over-count: that range holds a glossary, which is subject matter.

### What repetition misses

| | Passages | Mean twin | Caught |
|---|---|---|---|
| Learning-objective traceability matrices | 65 | 0.957 | 52 |
| Copyright and changelog | 28 | 0.937 | 9 |
| Acknowledgements | 20 | 0.932 | 2 |

Every syllabus thanks different people, so acknowledgements are structurally
identical and lexically different. `QUESTIONS_PARTY_DENSITY` reads them
instead:

| | Median | p90 | Above 0.25 |
|---|---|---|---|
| Acknowledgements (20) | 0.794 | 0.944 | 65% |
| Everything else (250) | 0.000 | 0.031 | none |

A floor of 0.15 reaches 75% of them and starts costing 1% of the others.

### The writer

| | |
|---|---|
| Accepted `reason` questions opening with the same word | 92% |
| `application` | 68% |
| `comparison` | 66% |
| `condition` | 59% |

A second worked example is what moved it; changing the rule alone did
nothing.

Before the writer's own citations were read, 91 of the first 140 rows carried
a `cross_document` label earned by a fact the question never used.

### The answer form

Measured against six realistic answers under the old single verb rule, five
were refused as malformed — every why, how, what-happens-if and which-things
answer. Only `EUR 15,000` survived.

A floor of 15 characters on `value` answers refused **41%** of the answers
this corpus had accepted, `70%`, `2025` and `Bafin` among them.

Explanations before the column existed sat at a **median of 157 characters
against a ceiling of 600**, and 32% were under 120. Six of 1,551 came near
the limit, so the ceiling was never what held them short — the prompt's "one
or two sentences" and fragment-shaped examples were.

### The gates

| | |
|---|---|
| `compound` read off interrogatives, over 17 real questions | separated 16 |
| The model's `wrong_type`, over 71 questions | fired 0 times, kind wrong on 6 of 27 accepted |
| `entity` slots refused as `wrong_type` in one run | 43 of 73 |
| `unanchored`, over a full run of 3,131 questions | fired 0 times |
| `leaks_source` in that run | 9 |
| `not_recoverable` in that run | 147 |

A circularity gate was written in two formulations and measured against 61
real rows. Both refused questions like *"Wie hoch war die Arbeitslosenquote
im August 2025?"* → `6,4 Prozent`, which is as good as a benchmark question
gets.

The `READS` rule against pointing at what the asker cannot see took dangling
references in accepted root questions from **9.4% to 2.3%** on its own.

### Where the phrasing judgements went

The reading call asked five things at once. Split by language:

| | Both phrasing judgements right |
|---|---|
| English | 7/7 — 100% |
| German | 6/12 — 50% |

Every miss was German, in a call whose recall half is good in both languages.
After the split:

| Judgement | Answered by | Measured |
|---|---|---|
| `subject` | spaCy noun chunks | 19/19, against 16/19 for the model |
| `names_its_source` | a pattern, a model for the residue | rules settle 5 cases with 0 errors, including all four the model got wrong |

### Follow-up threads

| | |
|---|---|
| Accepted follow-ups citing nothing new, before `Deal.widen` | 739 of 1,047 — 70.6% |
| Accepted follow-ups sharing no passage with their root | 147 |
| `comparison` turns written when every thread started at the first kind | 0 across 3,119 questions |
| Facts cited by 759 follow-ups that no root cited | 76 — a quarter of the writing, 1.2% of the coverage |

### Unanswerable questions

At `QUESTIONS_UNANSWERABLE_SHARE=0.10`: **79 attempts, 18 accepted** — a
22.8% survival rate and 4.4% of the accepted set, against a release ceiling
of 10%. The refusals were 31 `answerable_after_all`, 13
`answerable_elsewhere` and 11 `off_topic`. The default is 0.30 because of
this.

### Coverage

| | |
|---|---|
| passages | 1,495 |
| passages a validated fact rests on | 1,167 |
| passages an accepted question rests on | 880 — 75% of askable, 59% of all |
| validated facts | 6,242 |
| facts an accepted question cites | 2,114 — 34% |
| questions per passage asked about | 1.98 |
| questions per fact cited | 0.82 |

The 328 passages missing from 1,495 are ones the extractor skipped or whose
every fact a check refused.

Modelled over this corpus's spread (37 topics, 9 to 426 facts each), a topic
runs out of samples before the cap binds:

| Cap | Limited by | Facts reached |
|---|---|---|
| 60 | 17 topics by the cap, 20 out of facts | 26% |
| 120 | 1 by the cap, 36 out of facts | 31% |
| 180 | none by the cap | 31% |
| 360 | none by the cap | 31% |

The model reads 26% where 60 actually measured 34%, so its absolute numbers
are a floor and the shape is the finding. What raises coverage instead:

- **Running again.** Selection skips only the facts an *accepted* question
  rests on, so runs compound: roughly 34%, 54%, 68%, 78% over four.
- **`QUESTIONS_FACT_SAMPLE`.** At a cap of 240 it models 31% at 3, 46% at 2
  and 86% at 1. Below 2 the five two-passage types cannot be written.

### The balanced release

One measured run came out **45% unanswerable and 96.5% easy** with every gate
doing its job, which is why the composition is chosen after the fact rather
than aimed at.

A later full run drew **610 questions, a 35% yield**, balanced to 8–9% per
type and 32–35% per band, 94% answerable. The pool came up short on medium
and hard.

### The export bug worth remembering

Built as a WHERE subquery, SQLAlchemy correlated the four joined tables to
the outer query, emptied the subquery's FROM and turned **1,229 rows into six
million**, which Excel refused outright. It is a JOIN against a derived table
with `correlate(None)` now.

## The golden cases

### Baseline, 2026-09-20

`azure/gpt-5.4` writing and `azure/gpt-4.1` judging, before anything was
moved off them. Each phrasing judgement scored against the floor a judge that
ignores its input would reach:

| Judgement | Score | Answering the same way every time |
|---|---|---|
| `names_its_source` | 15/19 — 78.9% | 63.2% |
| `self_contained` | 16/19 — 84.2% | 78.9% |
| `subject` | 16/19 — 84.2% | — |

`self_contained` beats a constant answer by one case in nineteen. Four of the
misses are `names_its_source` on `Kapitel 5`, `[R22]` and `Beck 2003` — the
three a pattern matches without a model at all.

### After the split, same day

A larger, harder set — 25 cases against 19, with six added for phrasing the
old measurement could not see:

| Judgement | Baseline | After |
|---|---|---|
| `names_its_source` | 15/19 — 78.9% | 25/25 |
| `self_contained` | 16/19 — 84.2% | 25/25 |
| `subject` | 16/19 — 84.2% | 25/25 |

| | Baseline | After |
|---|---|---|
| English | 7/7 — 100% | 9/9 — 100% |
| German | 6/12 — 50% | 16/16 — 100% |

Six of the 25 cases were written in the same change as the rules, so treat
the pointing cases as a regression test rather than a measurement. The four
the baseline got wrong were labelled beforehand and are the honest evidence.

## The encoders

### Entailment on a fragment

Against `mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`:

| Premise | Hypothesis | Entailment |
|---|---|---|
| the LMT passage | `Ein Liquiditätsmanagementtool ist eine einjährige Rückgabefrist.` | 0.262 |
| the LMT passage | `eine einjährige Rückgabefrist` | 0.970 |
| `…within 48 hours…` | `48 hours` | 0.957 |
| `…within 48 hours…` | `4 hours` | 0.076 |

The encoder refuses the claim as a **proposition** and accepts it as a
**fragment**, and a fragment is what a target answer is.

### The QA encoder, and why `QA_MODEL` is empty

`timpal0l/mdeberta-v3-base-squad2`, 0.19 s a question, over 120 accepted
answerable questions:

| Answer form | n | span at ≥0.9 | agrees with the target | net usable |
|---|---|---|---|---|
| `value` | 37 | 70% | 54% | 38% |
| `explanation` | 53 | 72% | 21% | 15% |
| `list` | 30 | 57% | 6% | 3% |

It finds a confident span and the span is not the answer: `750 Minuten` comes
back as `mindestens 22,75 Unterrichtsstunden`. Turning it on would flip
roughly half the accepted set to `not_recoverable`.

`deepset/xlm-roberta-large-squad2` cannot be loaded under transformers 5 at
all — no `tokenizer.json`, and its sentencepiece model is mis-read as a
tiktoken file.

### The NLI encoder, and why it is on

`bge-m3-zeroshot-v2.0`, 1.72 s a question. Each target answer against its own
passages, and against another question's as a negative control:

| Threshold | Backs its own answer | Backs a stranger's |
|---|---|---|
| 0.5 | 67% | 6% |
| 0.7 | 59% | 3% |
| 0.9 | 40% | 1% |

Put to 30 real `not_recoverable` questions against gpt-4.1 on the same pairs:

| | |
|---|---|
| stopped by `asserted` and `about` before any judge | 10 |
| both rescue | 1 |
| only gpt-4.1 | 3 |
| only the encoder | 9 |
| neither | 7 |

The encoder rescues more, not less.

### Answer equivalence does not rescue the QA route

Mutual NLI entailment over the 81 confident QA spans:

| Threshold | Called equivalent | `agrees()` accepts |
|---|---|---|
| 0.5 | 18 — 22% | 23 — 28% |
| 0.7 | 14 — 17% | 23 — 28% |
| 0.9 | 11 — 14% | 23 — 28% |

Worse than what it would replace at every threshold. The purpose-built
option, `kortukov/answer-equivalence-bem`, is BERT-base **English** — every
answer-equivalence checkpoint on the hub is.

### The reranker, removed

`Qwen/Qwen3-Reranker-0.6B` on CPU: **6.92 s per candidate**. It ranks
correctly — 0.996 for the passage that answers, 0.014 and 0.002 for two that
do not — and selection reranks up to eight candidates per wide sample, so
~2,300 samples × 8 × 6.92 s is about **44 hours**. It changes no model call
count at all; reordering is its whole effect.

### Phrasing on local models

The residue each judgement actually asks, scored on the labelled cases:

| Model | `names_its_source` | `self_contained` | Per call |
|---|---|---|---|
| `granite4.2:3b` | 19/20 | 2/9 | 0.3 s |
| `granite4.2:8b` | 19/20 | 6/9 | 0.9 s |
| `gemma4:12b` | 19/20 | 9/9 | 3.2 s |
| `gemma4:31b` | 19/20 | 9/9 | 7.8 s |

The 31B buys nothing over the 12B and costs 2.4× the time. gpt-4.1 scored
15/19 and 16/19 on these when they shared the reading call.

## The duplicate gate, 2026-09-20

`duplicate` is the largest single loss in the pipeline — 433 of 3,131 in the
baseline run, more than every model-backed gate combined. All 454 duplicates
were paired with the accepted question they were refused against:

| | |
|---|---|
| duplicates examined | 454 |
| share a question word with their twin | 136 — 30% |
| share none | 318 — 70% |

A sentence embedding encodes what a question is *about*, so a purpose and an
audience sit on top of each other. `asks_the_same` is now a graded veto: a
question asking something else has to reach `threshold + (1 - threshold) / 2`
— 0.965 at the shipped 0.93. Measured false positives sit at 0.937 to 0.948.

That keeps **270 of 454 duplicates, 59%** — about 8.6% of the whole run — and
sends them to the gates that read their answers.

## `compound` stays at 89%

All 309 compound rejections read against the question words that refused
them: 276 are also joined by `und`/`sowie`/`oder`, and **33 carry no
conjunction** — every one a question word inside a subordinate clause.

Counting only question words outside a subordinate clause was tried and
reverted. German labels `Wie` and `Warum` `mo`, the same label a subordinate
clause carries, so testing the word discards every adverbial question;
testing the nearest finite verb above it works on 8 of 9 and still lets a
comma-joined coordinated clause through.

Letting a compound question into the benchmark is the costly direction, so an
89%-correct gate keeps its 11% error rather than trading it for an unknown
one.

## The phrasing swap, A against B on one topic

Same topic, same fact pool. The only differences are the phrasing model and
the `genannt` fix.

| | A: `azure/gpt-4.1` | B: `ollama_chat/gemma4:12b` |
|---|---|---|
| questions written | 67 | 59 |
| accepted | 40% | 39% |
| `duplicate` | 25% | 27% |
| `leaks_source` | 10% | 17% |
| `not_recoverable` | 10% | 10% |
| `compound` | 10% | 5% |
| phrasing calls to Azure | ~50 | 0 |
| phrasing calls to Ollama | 0 | 39 |

Acceptance is unchanged, so moving a third of the run's calls to a local
model cost no questions. `leaks_source` rising is not a clean win: the local
model catches more true positives *and* over-fires where gpt-4.1 did not.

## The full run, 2026-09-21

Every change above, over the whole corpus. 3,094 questions written, **1,744
accepted**, against the baseline's 3,131 and 2,077.

| Gate | Baseline | This run |
|---|---|---|
| accepted | 2,077 — 66.3% | 1,744 — 56.4% |
| `not_recoverable` | 147 | 512 |
| `compound` | 298 | 265 |
| `duplicate` | 433 | 180 |
| `leaks_source` | 9 | 89 |
| `answerable_after_all` | 93 | 83 |
| `unanchored` | 0 | 82 |
| `answerable_elsewhere` | 8 | 69 |

Where the 333 went:

- `duplicate` fell by 253 and `not_recoverable` rose by 365, and those are
  mostly the same questions — the dedup gate ran before the round trip, so a
  question it refused never reached recall. Sampled, 12% of the new
  `not_recoverable` is the `about` guard and 66% is a judge saying no.
- `leaks_source`, `unanchored` and `answerable_elsewhere` refuse 232
  questions the baseline accepted.

`leaks_source` splits 35 by rule and 54 by the model. Of ten model-driven
ones sampled, seven are right, giving roughly a **30% false-positive rate on
the model half** and none measured on the rule half.

`answerable_elsewhere` going 8 → 69 is the probe retrieving on
`passages.embedding` instead of shared lemmas and finding the passage that
answers.

## Telemetry volume

What the instrumentors produced before they were removed:

| | Spans |
|---|---|
| api and frontend | 3.86M, of which 1.2% were model calls |
| HTTP and S3 | ~200k |
| SQL statements and pool connects | 2.8M, against 47k model calls |

The `logs` volume reached **78 files and 110 MB** before `make logs-prune`
existed.

## Sizes

| | |
|---|---|
| Backend image | 3.02 GB |
| Orchestration image | 372 MB |
| `EMBEDDING_MODEL` weights | 2.2 GB |
| A question embedding as archived JSON | ~13 kB a row — 90 MB against 8.6 MB for everything else |

## Test coverage

Snapshots, not assertions. `make test-coverage` re-measures.

| Package | Statements covered |
|---|---|
| `backend/topic_modelling` | 99% — 538 of 539 |
| `backend/extraction` | 99% |
| `backend/ingestion` | 99% |

`make deps` found twelve declared-but-unused dependencies on its first run.
The OpenAPI conformance layer found two real 500s on its first run: an id or
offset above `bigint`, and a NUL byte in a search term.
