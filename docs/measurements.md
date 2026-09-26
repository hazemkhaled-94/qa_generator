# Measurements

Every number on this page comes from one run of the pipeline, on
**2026-09-26**. Nothing here is averaged over runs or carried forward from an
earlier one: re-measuring means taking a corpus through again and rewriting
the page.

This is the only place in the repository that holds run figures. A README
that needs one links here.

## The run

| | |
|---|---|
| Corpus | 16 PDF certification syllabi, German and English |
| Extractor, topic labeller, question writer | `azure/gpt-5.4` |
| Verifier and judge | `azure/gpt-4.1` |
| Wall clock | 7 h 31 min |
| Model calls | 20,218 |
| Tokens | 30,366,704 in, 1,471,085 out |
| Cost | USD 56.63 |

| Stage | Wall clock | Calls | Cost (USD) |
|---|---|---|---|
| Parsing | 5.6 min | — | — |
| Chunking | 1.2 min | — | — |
| Extraction | 116.2 min | 3,380 | 13.62 |
| Topic modelling | 1.6 min | 74 | 0.21 |
| Question generation | 296.5 min | 14,812 | 40.01 |
| Evaluation phase | 25.5 min | 1,952 | 2.79 |

Question generation is 66% of the wall clock and 71% of the spend.

## Corpus

| | German | English | Total |
|---|---|---|---|
| Documents | 8 | 8 | 16 |
| Pages | 645 | 582 | 1,227 |
| Characters | 1,648,644 | 1,527,068 | 3,175,712 |

All 16 parsed and chunked. None exceeded `MAX_FILE_SIZE_MB`.

## Parsing

| | |
|---|---|
| Mean confidence | 0.944 |
| Lower bound, across documents | 0.852 – 0.920 |

## Chunking

3,080 passages — 192.5 per document, from 92 to 294. Mean length 1,267
characters in German and 1,250 in English.

| Passage language | Count |
|---|---|
| German | 1,499 |
| English | 1,581 |

54 passages sit in a document of the other language: 50 English passages in
German documents, 4 the other way.

| Block type | Passages |
|---|---|
| `text` | 2,022 |
| `list_item` | 570 |
| `table` | 329 |
| `document_index` | 154 |
| `caption` | 3 |
| `section_header` | 1 |
| `footnote` | 1 |

## Extraction

12,154 statements written, **9,056 validated — 74.5%**.

| Kind | Written | Validated |
|---|---|---|
| `atomic` | 9,846 | 7,425 |
| `summary` | 1,154 | 1,037 |
| `outline` | 1,154 | 594 |

| Refused as | Count | Share of refusals |
|---|---|---|
| `duplicate` | 2,292 | 74.0% |
| `unsupported_addition` | 491 | 15.8% |
| `unresolved_reference` | 132 | 4.3% |
| `asserts_nothing` | 107 | 3.5% |
| `not_atomic` | 61 | 2.0% |
| `copied` | 8 | 0.3% |
| `not_condensed` | 7 | 0.2% |

2,205 of 3,080 passages carry a validated fact — 4.11 each on average, median
4, most 45.

| Block type | Passages | Carrying a validated fact |
|---|---|---|
| `text` | 2,022 | 1,679 |
| `list_item` | 570 | 374 |
| `table` | 329 | 207 |
| `document_index` | 154 | 0 |

`document_index` yields nothing: a contents page states no claim a citation
can rest on.

## Topic modelling

76 topics — 37 German, 39 English.

| | German | English |
|---|---|---|
| Vocabulary after filtering | 2,870 terms | 2,445 terms |
| Passages in no topic | 10 | 2 |
| Largest topic, share of its language | 4.3% | 4.3% |

Topics hold 42 to 308 dominant passages, 182.8 on average. 9 of the 76 went
unlabelled, below `TOPIC_LABEL_MIN_FACT_SHARE`.

## Question generation

5,873 questions written, **2,416 accepted — 41.1%**. 4,186 candidates were
first attempts and 1,687 were retries.

| Gate | Refused | Share of written |
|---|---|---|
| `compound` | 881 | 15.0% |
| `not_recoverable` | 520 | 8.9% |
| `answerable_after_all` | 504 | 8.6% |
| `answerable_elsewhere` | 264 | 4.5% |
| `asks_nothing_new` | 231 | 3.9% |
| `answer_incomplete` | 200 | 3.4% |
| `explanation_unusable` | 135 | 2.3% |
| `restates_question` | 126 | 2.1% |
| `leaks_source` | 120 | 2.0% |
| `duplicate` | 100 | 1.7% |
| `off_thread` | 76 | 1.3% |
| `unanchored` | 71 | 1.2% |
| `off_topic` | 69 | 1.2% |
| `wrong_type` | 62 | 1.1% |
| `wrong_form` | 49 | 0.8% |
| `answer_too_long` | 31 | 0.5% |
| `answer_too_short` | 12 | 0.2% |
| `malformed` | 6 | 0.1% |

How far each gate was reached:

| Gate group | Questions reaching it |
|---|---|
| `structural` | 5,873 |
| `kind_and_thread` | 4,823 |
| `near_duplicate` | 4,193 |
| `phrasing` | 4,055 |
| `round_trip` | 3,866 |
| `off_topic` | 1,349 |

### What was accepted

| Kind | Accepted | Share |
|---|---|---|
| `reason` | 387 | 16.0% |
| `condition` | 323 | 13.4% |
| `factoid` | 311 | 12.9% |
| `definition` | 268 | 11.1% |
| `application` | 208 | 8.6% |
| `comparison` | 188 | 7.8% |
| `procedure` | 179 | 7.4% |
| `consequence` | 159 | 6.6% |
| `enumeration` | 151 | 6.3% |
| `implication` | 131 | 5.4% |
| `aggregation` | 111 | 4.6% |

`entity` and `temporal` carry a weight of 0 and were not written.

| | | |
|---|---|---|
| Language | German 1,304 | English 1,112 |
| Answerable | 1,945 — 80.5% | Unanswerable 471 — 19.5% |
| Passage scope | `single_passage` 1,453 | `multi_passage` 963 |
| Document scope | `single_document` 1,645 | `cross_document` 771 |
| Topic scope | `single_topic` 1,976 | `multi_topic` 440 |

| Difficulty | Accepted | Share |
|---|---|---|
| `easy` | 1,211 | 50.1% |
| `hard` | 666 | 27.6% |
| `medium` | 539 | 22.3% |

| Cognitive level | Accepted |
|---|---|
| `analyse` | 976 |
| `apply` | 710 |
| `understand` | 419 |
| `recall` | 311 |

An accepted question cites 1.53 facts on average, at most 3.

### Answer length

| Form | Accepted | Median | Mean | Range |
|---|---|---|---|---|
| `explanation` | 1,157 | 123 | 139 | 23 – 594 |
| `list` | 399 | 104 | 113 | 9 – 293 |
| `value` | 389 | 29 | 32 | 1 – 80 |

### Follow-up threads

| Position | Written | Accepted |
|---|---|---|
| 1 — root | 4,879 | 2,039 |
| 2 | 717 | 277 |
| 3 | 277 | 100 |

Acceptance falls from 41.8% at the root to 38.6% and 36.1%.

## The balanced release

1,341 questions drawn from the 2,416 accepted — a 55.5% yield. Every quota
was filled but one.

| Held at | Result |
|---|---|
| Kind | 10 kinds at 123 — 9.2% each; `aggregation` 111 — 8.3%, the pool's limit |
| Difficulty | `easy` 449, `medium` 445, `hard` 447 — 33.5 / 33.2 / 33.3% |
| Unanswerable | 135 — 10.1%, against a ceiling of 10% |

The accepted pool was 50.1% easy and 19.5% unanswerable; the release is
evenly banded and at its unanswerable ceiling. That gap is what balancing
does.

## Coverage

| | |
|---|---|
| Passages | 3,080 |
| Passages a validated fact rests on | 2,205 — 71.6% |
| Passages an accepted question rests on | 1,547 — 70.2% of askable, 50.2% of all |
| Validated facts | 9,056 |
| Facts an accepted question cites | 3,293 — 36.4% |
| Questions per passage asked about | 1.56 |
| Questions per fact cited | 0.73 |
| Accepted questions per topic | 196.5, from 1 to 395 |

The 875 passages carrying no validated fact are ones the extractor skipped or
whose every statement a check refused.

## Confidence

How close a row came to the verdict that would have refused it.

| | Rows | Mean | Range |
|---|---|---|---|
| Validated facts | 9,055 | 0.191 | 0.000 – 0.915 |
| Accepted questions | 2,416 | 0.230 | 0.000 – 1.000 |

## The evaluation phase

`azure/gpt-4.1` over a sample of 200 facts, 200 questions and all 76 topics.
It records an opinion and changes no verdict.

| Artefact | Judged | Approved | Refused |
|---|---|---|---|
| `fact` | 200 | 186 | 14 |
| `question` | 200 | 144 | 56 |
| `topic` | 76 | 56 | 20 |

| Metric | Judgements | Mean score | Approved |
|---|---|---|---|
| `toxicity` | 400 | 0.000 | 400 |
| `conciseness` | 200 | 1.000 | 200 |
| `refusal` | 200 | 0.040 | 192 |
| `relevance` | 476 | 0.935 | 445 |
| `qa_correctness` | 200 | 0.870 | 174 |
| `hallucination` | 400 | 0.105 | 358 |
| `summarization` | 76 | 0.803 | 61 |

`hallucination` is scored to minimise: 0.0 is the good label.

### Where the judge and the checker disagree

Facts:

| | Judge approved | Judge refused |
|---|---|---|
| Validated | 100 | 4 |
| Refused | 86 | 10 |

Questions:

| | Judge approved | Judge refused |
|---|---|---|
| Accepted | 71 | 20 |
| Rejected | 73 | 36 |

The judge approves 86 of the 96 facts the checker refused and 73 of the 109
questions the gates rejected. The two do not measure the same thing: the
checker asks whether a citation resolves and a gate holds, the judge whether
an artefact reads as sound. The disagreeing pairs are the queue for a person.

## Model latency

Per call, over the run:

| Stage | Median | p95 | Slowest |
|---|---|---|---|
| Extraction | 1.6 s | 2.8 s | 10.0 s |
| Topic modelling | 0.8 s | 1.0 s | 1.3 s |
| Question generation | 0.8 s | 2.4 s | 12.6 s |
| Evaluation phase | 0.7 s | 1.0 s | 2.2 s |

`LLM_TIMEOUT_SECONDS` and the leases derived from it are sized against the
slowest column. A locally served model of the same size is one to two orders
slower per call, and the wall clock above is the figure that moves.

## Test coverage

Not a run figure. `make test-coverage` re-measures.

| Package | Statements covered |
|---|---|
| `backend/topic_modelling` | 99% |
| `backend/extraction` | 99% |
| `backend/ingestion` | 99% |

## Reproducing this page

```bash
make corpus                       # every stage in order
make questions-balance            # draw the release
make assess-start && make assess  # the evaluation phase
make pipeline-status              # the counts
make spend LOG=run.log            # the calls and the cost
```

Per-stage wall clock comes from the run's log files in `./logs`, and the
counts from the tables. Nothing on this page is computed anywhere else.
