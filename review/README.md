# Review

Human review of what the models decided, through Argilla.

Three things in this pipeline are a model's judgement, and each has somewhere
for a person to disagree. Argilla is where a hundred of those decisions get
made in a row instead of one at a time through a table.

| Dataset | The decision | Lands in |
|---|---|---|
| `facts` | Does the statement follow from the evidence, and stand on its own? | `facts.reviewed_verdict` |
| `topic-labels` | Is this a good name for these terms, and is it a subject worth asking about? | `topics.label`, `topics.include_in_coverage` |
| `questions` | Would somebody ask this, and is the answer right? | `questions.status` |

## The database decides

Argilla holds a **copy** of the rows put in front of somebody and the answers
they gave. A pull brings the answers home and the copy is disposable — delete
the Argilla dataset and the pipeline has lost nothing.

That is the whole design. Nothing in the pipeline reads Argilla, and no
container carries the client: this runs on the host, like `make schema`.

## Using it

```bash
make review-push-facts       # a sample, spread over the verdicts
make review-pull-facts       # write the submitted verdicts back
make review-push-topics
make review-pull-topics
make review-push-questions   # a sample, spread over the gates
make review-pull-questions
make review-status           # how much has been looked at
```

Push, review in the UI at `ARGILLA_API_URL`, then pull.

Underneath, one command line:

```bash
python -m review.run --push facts
python -m review.run --pull questions
python -m review.run --status
```

The three are mutually exclusive and one is required.

## Why the samples are stratified

**Over the verdicts, not in id order.** A review answers "is the checker
right", and a sample of only what it accepted cannot answer that — nor can a
sample in id order, which for a corpus is a sample of whichever document was
extracted first.

At least one from each group, because **a rejection code that fired twice in a
whole corpus is the interesting one**, and a proportional sample would never
show it.

## Where each verdict lands, and why facts needed a column

Two of the three write through routes that already existed: a verdict given
here and one given on the Questions or Topics page are the **same write**,
which is what keeps `labelled_by` honest about who named a topic.

Facts had nowhere. `validated` and `rejection_code` are the checker's, and
`make extract-revalidate` rewrites both from scratch — so a human decision
recorded there would last until the next re-judgement and then be gone with
nothing saying it had been. `reviewed_verdict` is its own column, the way
`questions.status` is its own beside `rejected_reason`.

## A pull takes only submitted answers

Argilla saves a **draft** the moment a record is touched, and writing one back
would record an opinion nobody has finished having.

## Tools, and where each is used

| Tool | Where | Why this one |
|---|---|---|
| **argilla** | [`datasets.py`](datasets.py), [`service.py`](service.py) | The annotation UI, its workspace model and its record format in one client |
| **SQLAlchemy** | [`records.py`](records.py) | Drawing the stratified sample, and writing the verdicts home |

Argilla's Elasticsearch is the one the logs already use. That is a deliberate
reuse rather than a second node, and it is a shared heap — `ES_JAVA_OPTS` in
[`configs/env/elasticsearch.env`](../configs/env/elasticsearch.env) is where to
raise it if a long run makes either slow.

## Configuration

Tuning, from [`configs/env/review.env`](../configs/env/review.env), in git:

| Setting | Default | What it does |
|---|---|---|
| `REVIEW_SAMPLE_SIZE` | 200 | How many records one push puts in front of a reviewer, split across the groups |
| `REVIEW_REVIEWER` | unset | What to call the reviewer in a row's provenance. Its absence means something: a deployment with one annotator does not need to say who each time |

Where Argilla is and how to sign in, from `.env`, not in git:

| Setting | Default | What it does |
|---|---|---|
| `ARGILLA_API_URL` | `http://localhost:6900` | The **host's** address: this is a `make` target, not a container |
| `ARGILLA_API_KEY` | — | Argilla shows it under "My settings". **Not** `ARGILLA_PASSWORD` |
| `ARGILLA_WORKSPACE` | `qa_generator` | One per deployment, so two people reviewing two corpora do not annotate each other's rows |

## Tests

```sh
poetry run pytest tests/unit/review
```

| File | Covers |
|---|---|
| [`test_records.py`](../tests/unit/review/test_records.py) | How a review sample is drawn, and what a record carries |
| [`test_service.py`](../tests/unit/review/test_service.py) | Bringing verdicts home |

Neither reaches a network.

## Known edges

Things that are true, are not bugs, and have surprised somebody.

- **`ARGILLA_API_KEY` is not `ARGILLA_PASSWORD`.** The key is under "My
  settings" in the UI. This has caught everybody once.
- **A draft is not pulled.** Argilla saves one the moment a record is touched;
  only a submitted answer comes home.
- **A push replaces what a previous push left.** The Argilla dataset is a
  disposable copy, not an accumulating record.
- **`extract-revalidate` does not touch `reviewed_verdict`.** That is why it
  is a separate column.
- **This never runs in a container.** No image carries the Argilla client, so
  the address it needs is the host's.
