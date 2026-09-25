# Review

Human review of what the models decided, through Argilla.

| Dataset | The decision | Lands in |
|---|---|---|
| `facts` | Does the statement follow from the evidence, and stand on its own? | `facts.reviewed_verdict` |
| `topic-labels` | Is this a good name for these terms, and is it a subject worth asking about? | `topics.label`, `topics.include_in_coverage` |
| `questions` | Would somebody ask this, and is the answer right? | `questions.status` |

Argilla holds a **copy** of the rows put in front of somebody and the answers
they gave. A pull brings the answers home and the copy is disposable —
delete the Argilla dataset and the pipeline has lost nothing.

Nothing in the pipeline reads Argilla, and no container carries the client:
this runs on the host, like `make schema`.

## Using it

```bash
make review-push-facts       # a sample, spread over the verdicts
make review-pull-facts       # write the submitted verdicts back
make review-push-topics
make review-pull-topics
make review-push-questions   # a sample, spread over the gates
make review-pull-questions
make review-status           # how much has been looked at, and who agreed

make review                  # push all three samples in one go
make review-all              # every row of every kind, not a sample
make pull                    # pull all three back
```

Push, review in the UI at `ARGILLA_API_URL`, then pull.

Underneath, one command line:

```bash
python -m review.run --push facts
python -m review.run --push facts --all
python -m review.run --pull questions
python -m review.run --status
```

`--push`, `--pull` and `--status` are mutually exclusive and one is required.
`--all` and `--ids` modify a push.

## What a review produces

`review-status` reports, per dataset, how many rows carry a verdict and **how
often the person and the model reached the same one**:

```text
facts: 120 of 6687 reviewed (98 accepted, 22 rejected); agreed with the model on 104 of them, 87%
questions: 0 of 6997 reviewed; nobody has looked, so there is no agreement to report
```

It is a query because the two verdicts are kept in different columns and
neither overwrites the other. A fact holds the checker's in `validated` and
the person's in `reviewed_verdict`. A question holds the gates' in
`rejected_reason` and the person's in `reviewed_verdict`, with `status`
saying what the question now is.

Accepting a question a gate refused does **not** clear `rejected_reason`: it
is the other half of the disagreement, and cleared it would make an overruled
rejection indistinguishable from a question no gate ever stopped.

## A sample, or the corpus

`REVIEW_SAMPLE_SIZE` rows spread over the verdicts is the default, and what
`make review` sends. `--all` sends every row of that kind instead, so Argilla
holds the corpus and the reviewer filters there.

`make review-all` is that over all three kinds. It is the right shape when
somebody is working through a corpus; the sample is the right shape when the
question is "how are we doing".

**Stratified over the verdicts, not in id order.** A review answers "is the
checker right", and a sample of only what it accepted cannot answer that —
nor can a sample in id order, which for a corpus is a sample of whichever
document was extracted first. At least one from each group, because a
rejection code that fired twice in a whole corpus is the interesting one.

## Reviewing a queue somebody else built

```bash
make review-push-questions IDS=12,34,56
```

Exactly those rows instead of a sample. What produces the list is
`make second-opinion RUN=<id>`: the questions the gates **kept** and an
independent judge calls unsupported. A stratified sample cannot find them —
they are rare in every band and every verdict.

## Where each verdict lands

Two of the three write through routes that already existed, so a verdict
given here and one given on the Questions or Topics page are the same write.

Facts had nowhere. `validated` and `rejection_code` are the checker's, and
`make extract-revalidate` rewrites both from scratch, so a human decision
recorded there would last until the next re-judgement.
`reviewed_verdict` is its own column.

A pull takes only **submitted** answers. Argilla saves a draft the moment a
record is touched, and writing one back would record an opinion nobody has
finished having.

## Configuration

From [`configs/env/review.env`](../configs/env/review.env), in git:

| Setting | Default | What it does |
|---|---|---|
| `REVIEW_SAMPLE_SIZE` | 200 | How many records one push puts in front of a reviewer, split across the groups |
| `REVIEW_REVIEWER` | unset | What to call the reviewer in a row's provenance |

From `.env`, not in git:

| Setting | Default | What it does |
|---|---|---|
| `ARGILLA_API_URL` | `http://localhost:6900` | The **host's** address: this is a `make` target, not a container |
| `ARGILLA_API_KEY` | — | Argilla shows it under "My settings". **Not** `ARGILLA_PASSWORD` |
| `ARGILLA_WORKSPACE` | `qa_generator` | One per deployment |

Argilla's Elasticsearch is the one the logs already use — a shared heap, and
[`configs/env/elasticsearch.env`](../configs/env/elasticsearch.env) is where
`ES_JAVA_OPTS` raises it.

## Tests

```sh
poetry run pytest tests/unit/review
```

| File | Covers |
|---|---|
| [`test_records.py`](../tests/unit/review/test_records.py) | How a review sample is drawn, and what a record carries |
| [`test_service.py`](../tests/unit/review/test_service.py) | Bringing verdicts home |

Neither reaches a network.

## Limits

- **`ARGILLA_API_KEY` is not `ARGILLA_PASSWORD`.**
- **A draft is not pulled.** Only a submitted answer comes home.
- **A push replaces what a previous push left.** The Argilla dataset is a
  disposable copy, not an accumulating record.
- **`extract-revalidate` does not touch `reviewed_verdict`.**
- **This never runs in a container.** No image carries the Argilla client.
