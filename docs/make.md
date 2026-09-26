# Make commands

Every target, grouped. `make` sources the same configuration the containers
read, so one value reaches both.

```bash
make help            # or just `make`
make help | grep -i topic
```

`help` reads the Makefile itself — a target's group is the section banner it
sits below and its description is the comment above it — so it cannot fall
behind. This page is the long form.

## First run

| Target | Does |
|---|---|
| `make setup` | Write `.env` and `configs/env/provider.env`, and fill in every placeholder |
| `make install` | The Python dependencies and the spaCy pipelines |
| `make doctor` | Check the tools and the configuration, then ask the model one question |
| `make dev` | Certificates, every service, and the schema |

`setup` is safe to run twice: it replaces `change_me_*` placeholders and
nothing else. It gives one secret per distinct placeholder and substitutes it
everywhere that placeholder appears, which keeps `APP_DB_PASSWORD` and the
password inside `DATABASE_URL` the same string.

`doctor` ends with a real call to the configured model, so an unknown model
id, an unreachable address or a stale credential fails here rather than on
the first passage of a run:

```text
  ok       podman
  ok       poetry
  ok       .env
  ok       no placeholder passwords
  ok       PHOENIX_ADMIN_SECRET
  ok       LLM_BASE_URL is not host-local

Asking azure/gpt-5.4 one question...
```

## The short list

| Target | Does |
|---|---|
| `make all` | Bring the stack up, then take the corpus end to end |
| `make corpus` | Every stage in order, stopping at the first failure. Incremental |
| `make pipeline-status` | Where the corpus has got to, every stage in one call |
| `make services` | Every container, whether it is listening, and where to open it |
| `make open` | The same, and open the application |
| `make review` | Push a sample of facts, topic labels and questions to Argilla |
| `make pull` | Pull every decision made there back into the database |

`make auto` hands the whole thing to Dagster — see
[Orchestration](#orchestration).

## Configuration and the container engine

Targets read `configs/env/backend.env` then `.env`, in the order the
containers do. A variable given on the command line beats both:

```bash
make topics-discover TOPIC_PASSES=20
```

The Makefile re-applies command-line overrides after sourcing the files,
because sourcing would otherwise overwrite them.

The Makefile invokes `podman compose`. For Docker:

```bash
make COMPOSE="docker compose" up
```

`CONTAINER` is the engine itself, used by `lock`, which needs a build and a
run compose has no equivalent of.

## Starting and stopping

| Target | Does |
|---|---|
| `make dev` | Certificates, every service, wait for PostgreSQL, create the schema |
| `make install` | Dependencies and the spaCy pipelines `NLP_MODELS` names |
| `make up` | Start services |
| `make down` | Stop services, keep data |
| `make down-volumes` | Stop services and **delete all data** |
| `make certs` | Generate the TLS certificates Elasticsearch needs |
| `make prune` | Remove dangling container images |

## Logs

| Target | Follows |
|---|---|
| `make logs` | All logs |
| `make logs-api` | The API |
| `make logs-frontend` | The frontend |
| `make logs-orchestration` | The Dagster webserver and daemon |
| `make logs-shipper` | The log shipper, when Grafana shows nothing |
| `make logs-retention` | Not a follow: how long Elasticsearch keeps them. **Run once** |
| `make logs-prune` | Not a follow: delete the files on the `logs` volume |

```bash
make logs-retention LOGS_RETENTION_DAYS=90
make logs-prune LOGS_KEEP_DAYS=7
```

## The corpus as one thing

The same three verbs as below, over every stage at once, through the API.
After a model outage the failures are spread over five queues, and visiting
five of them is five chances to miss one.

| Target | Does |
|---|---|
| `make pipeline-status` | Every stage's queue, the run in flight, and what is set to start one |
| `make pipeline-stop` | Take back everything queued, in every stage, and stop the run behind it |
| `make pipeline-retry` | Return every stage's failures to the queue |

There is no `pipeline-start`: `make corpus` is the better one for a terminal,
because it starts each stage **and waits for it to drain** before the next,
which is the whole difficulty. The Run button and `make pipeline` ask Dagster
to hold that wait instead.

## The five verbs every stage answers

Each is the same operation as the route beside it.

| | Parsing | Chunking | Extraction | Questions | Assessment |
|---|---|---|---|---|---|
| queue what was never asked for | `parse-start` | `chunk-start` | `extract-start` | `questions-start` | `assess-start` |
| drain here, in the foreground | `parse` | `chunk` | `extract` | `questions` | `assess` |
| show the queue | `parse-status` | `chunk-status` | `extract-status` | `questions-status` | `assess-status` |
| take back what has not begun | `parse-stop` | `chunk-stop` | `extract-stop` | `questions-stop` | `assess-stop` |
| return failures to the queue | `parse-retry` | `chunk-retry` | `extract-retry` | `questions-retry` | `assess-retry` |
| queue everything again | `parse-rerun` | `chunk-rerun` | `extract-rerun` | `questions-rerun` | `assess-rerun` |

`assess-start` also **enrols**: nothing upstream creates an assessment, so it
creates the rows as well as queuing them.

`chunk-start` and `chunk-rerun` queue **only the documents parsing has
finished**. A document exists from the moment it is uploaded, so without that
guard a start over a corpus half way through parsing queued the other half
and failed every row of it on a parsed form that was not there yet. It is the
only stage that needs it: a passage exists because chunking made it, and a
topic because a fit did.

Topic modelling has no `start` and no `rerun`: a fit is all-or-nothing over
one vocabulary.

| Target | Does |
|---|---|
| `make topics-discover` | Ask for a fit over the whole corpus, and run it |
| `make topics` | Run any queued fit here, in the foreground |
| `make topics-status` | The topics held, and any queued fit |
| `make topics-stop` | Withdraw a queued fit |
| `make topics-retry` | Return a failed fit to the queue |
| `make topics-visualise` | Write each language's stored pyLDAvis page to `./topics/` |
| `make topics-render` | Redraw those pages from the stored models, without re-fitting. `CODE=de` for one |
| `make topics-delete` | Delete every topic and membership |

### Narrowing to one item

Any verb narrows with `SHA`, `PASSAGE` or `TOPIC`:

```bash
make parse-start SHA=<sha256>     # queue one document for parsing
make chunk-rerun SHA=<sha256>     # rebuild one document's passages
make extract-start SHA=<sha256>   # queue every passage of one document
make extract-retry PASSAGE=<id>   # return one failed passage to the queue
make questions-start TOPIC=<id>   # write the questions for one topic
```

Parsing and chunking narrow to a document, extraction to a document or a
passage, question generation to a topic, topic modelling to nothing.

## Replaying a stage without redoing it

Re-derive what a stage computed over rows already stored, without the
expensive part:

| Target | Does |
|---|---|
| `make extract-revalidate` | Judge every stored fact again. No model call |
| `make chunk-revocabulary` | Re-read language and vocabulary. Only `passages.language` and `passages.lemmas` change |
| `make questions-reverify` | Put every stored question through the gates that need no model |
| `make extract-recap` | Re-apply the atomic cap `EXTRACTION_MIN_OTHER_SHARE` works out to |
| `make extract-embed` | Write vectors onto passages and facts already stored. Does not apply the dedup gate |
| `make extract-bridge` | Read each topic's passage groups for claims spanning passages |

`chunk-revocabulary` and `extract-revalidate` take `SHA`;
`extract-revalidate`, `extract-recap` and `extract-embed` take `PASSAGE`;
`questions-reverify` takes `TOPIC`.

The alternative is destructive: `extract-rerun` calls the model again over
the whole corpus, and `chunk-rerun` deletes every passage of a document along
with the facts drawn from them.

## Questions

| Target | Does |
|---|---|
| `make questions-balance` | Draw the balanced release out of what was accepted |
| `make questions-export` | Write the questions a filter selects to an `.xlsx` |
| `make questions-reverify` | Re-check stored questions; no model call |
| `make questions-runs` | Which runs there are, newest first |
| `make questions-diff` | Two runs side by side, on the gate that stopped each |
| `make questions-reclaim` | Return a topic a dead worker still holds. **Narrow it** |

```bash
make questions RUN_ID=b-gemma4-12b
make questions-diff RUNS="a-gpt-4.1 b-gemma4-12b"
```

### Getting the dataset out

`OUT` names the file and `FILTER` takes the same narrowing the Questions page
and `GET /questions` do. **There is no default scope** — ask for nothing and
you get everything, rejected rows included.

```bash
make questions-export OUT=exam.xlsx FILTER="--status accepted --cognitive-level analyse"
make questions-export FILTER="--status accepted --unanswerable"
make questions-export FILTER="--no-citations"   # much faster, questions only

python -m question_generation.export --help     # every filter there is
```

Four sheets: the questions, the facts each cites — one row per fact per
passage — what an LLM judge made of each answer, and the counts. Argilla is
not this: it holds a disposable copy of a sample pushed for review.

A question's own row carries what it rests on, so it reads on its own: the
facts it cites and the evidence under them, its subjects, its documents **by
title**, the answer at length, and why the judge said what it did. The
Citations sheet is still the grain that carries the page number and the
passage ordinal, which is what checking an answer against the source needs.

The Assessment sheet and the three `Judge` columns are empty unless
`make assess` has run.

`questions-diff` reads live questions **and** archived ones, so a run
`questions-rerun` replaced still answers. A run listed `(deleted)` is being
read out of `archived_rows`.

```bash
make questions-reclaim TOPIC=473   # one topic, while a worker may be up
make questions-reclaim             # every one the stage holds
```

`reclaim` fills the gap between `retry`, which takes the failed, and `rerun`,
which skips what a worker holds. Narrow it while a worker may be up: nothing
can tell a dead claim from a live one.

It has a route now as well — `POST /{stage}/{scope}/{value}/reclaim`, with
`?confirm=true` needed for the unnarrowed form. See
[`backend/stages/`](../backend/stages/README.md).

## The evaluation phase

An LLM judge over every fact, topic and question, run after the pipeline and
never during it. Off unless `ASSESSMENT_ENABLED` is true in `.env`, and every
target says so and stops rather than quietly doing nothing.

**It decides nothing.** The verdict goes in a column of its own beside the
checker's, and no stage reads it. See
[`backend/assessment/`](../backend/assessment/README.md) for why it cannot
be a gate.

| Target | Does |
|---|---|
| `make assess-enrol` | Create the rows without queuing any. The dry run: how many model calls this would cost |
| `make assess-start` | Enrol and queue everything that has no verdict |
| `make assess` | Drain the judging queue here, in the foreground |
| `make assess-status` | The queue, and what the judge has said so far |
| `make assess-stop` | Take back what has not begun |
| `make assess-retry` | Return the artefacts the judge could not be reached for |
| `make assess-rerun` | Judge everything again, under the current templates |
| `make assess-reclaim` | Return an artefact a dead worker holds. **Narrow it** |

```bash
make assess-enrol                              # count the cost first
make assess-start ONLY=--only\ kind=question   # the deliverable only
make assess
```

A fact and a topic cost two model calls each; a question costs three.
`ASSESSMENT_KINDS` and `ASSESSMENT_SAMPLE` in `configs/env/backend.env` are
the dials, and the default is a sample of 200 per kind rather than a corpus.

A failure here is the **model** being unreachable, not an artefact being bad:
a judgement that came back as a refusal is recorded as one and is not a
failed row.

## Documents

Documents are ingestion's, not a stage's. **All of these are irreversible.**

| Target | Does |
|---|---|
| `make documents` | List every document with its parse state |
| `make delete SHA=<sha256>` | Delete a document and everything derived from it |
| `make delete-derived SHA=<sha256>` | Delete only its passages and facts |
| `make wipe` | Every document, then the topics. Waits 5 seconds first |

`wipe` runs both deletions because deleting every document leaves the topics
standing. It also drops the upload history, which a single deletion keeps.

## The archive

Every deletion above is the first of two.

| Target | Does |
|---|---|
| `make archive` | What is held, by table, with its age |
| `make archive-purge` | The second deletion. **This one is final** |

```bash
make archive-purge TABLE=questions   # one table's rows
make archive-purge DAYS=30           # everything archived before then
make archive-purge ALL=1             # the whole thing, objects included
```

`TABLE` is about rows and leaves the bucket alone. `DAYS` and `ALL` take the
objects too. See [`backend/archive/`](../backend/archive/README.md).

## Settings

```bash
make settings SERVICE=topics
make settings-set SERVICE=topics SET="TOPIC_PASSES=20"
make settings-unset SERVICE=topics UNSET="TOPIC_PASSES"
```

The services are `ingestion`, `parsing`, `chunking`, `extraction`, `topics`,
`questions` and `platform`. See [configuration.md](configuration.md).

## Schema

| Target | Does |
|---|---|
| `make migration m="add the dropped counts"` | Write a revision from the models |
| `make schema` | Apply everything outstanding |
| `make schema-status` | Where the database is |
| `make schema-down` | Take the newest revision back off |
| `make schema-reset` | Drop every table and rebuild. **Irreversible** |
| `make schema-stamp` | Adopt a database that already holds the tables |

**Read the generated revision before applying it.** Autogenerate does not see
a trigger, a data backfill, or anything ordered.

## Review and evaluation

| Target | Does |
|---|---|
| `make review-push-facts` / `review-pull-facts` | Send a sample of facts to Argilla; bring the verdicts back |
| `make review-push-topics` / `review-pull-topics` | The same, for topic labels |
| `make review-push-questions` / `review-pull-questions` | The same, for questions |
| `make review-status` | How much has been looked at |
| `make review-all` | Every artefact of every kind, not a sample |
| `make eval-upload` | Put the golden cases in Phoenix |
| `make eval-score` | Score the served model against them, and record it |
| `make prompts-publish` | Send the recorded prompts to Phoenix |
| `make eval-phrasing` | Score the two phrasing judgements, each against its floor |
| `make second-opinion` | An independent judge over one run, and where it disagrees |

```bash
make eval-score EVAL_RUN_NAME=extraction-prompt-v7
make second-opinion RUN=<run id> LIMIT=50
make review-push-questions IDS=12,34,56
```

`second-opinion` settles nothing; it produces a queue for a person.

## Phoenix's projects

| Target | Does |
|---|---|
| `make phoenix-projects` | What Phoenix holds, changing nothing |
| `make phoenix-prune` | Delete the projects in which no work happened. **Irreversible**, and it says what it is taking before it takes it |

A Phoenix project is created by whatever sends the first span to it and
removed by nothing. While a project meant a run somebody asked for that was
fine — but `run_id` mints a uuid **per process**, a `--watch` worker is a
process per restart, and a credential that has expired restarts it every
minute. This deployment reached **five hundred projects, 497 of them
holding one span**: the model call a preflight made before giving up.

Two changes stop it growing — an unnamed run appends to its stage's own
project, and the exporter is not installed until the preflight passes — and
`prune` takes back what the old arrangement left.

`prune` takes a project only when **every span in it is a model call and it
could read every span**. A project too big to read in one page is left
alone, because the first fifty spans of a real run are model calls too.

## Tests and checks

| Target | Runs |
|---|---|
| `make test` | Everything that gates; starts containers of its own |
| `make check` | An alias for `make test` |
| `make test-fast` | Only the fast layers: no spaCy, no pyright, no containers |
| `make test-unit` | Everything but integration, smoke, eval and perf |
| `make test-integration` | The container layers |
| `make test-e2e` | One document through every stage, in this process |
| `make test-smoke` | Build both images and look inside them |
| `make test-eval` | Score the served model against the golden passages. Never gates |
| `make test-perf` | Time the ceilings the code names. Never gates |
| `make test-coverage` | The gating layers, with a coverage report |
| `make mutation` | Whether the tests would notice a gate changing. Hours |
| `make lint` | ruff check and format check |
| `make format` | Apply every fix ruff can make |
| `make typecheck` | pyright, at zero |
| `make audit` | Known advisories against the two locks |
| `make lint-imports` | The two architecture rules, as import-linter contracts |
| `make deps` | A declared dependency nothing imports, and an import nothing declares |
| `make lock` | Rewrite `backend/api/requirements.lock` |

`lint`, `format` and `typecheck` cover `backend`, `frontend`, `telemetry`,
`orchestration`, `review`, `evaluation` and `tests`.

`test-perf` and `mutation` are the two nightly ones. Which modules and layers
mutation covers is `[tool.mutmut]` in `pyproject.toml`; `mutmut browse` reads
the survivors.

## Cost

```bash
make spend LOG=run.log                    # totals from a captured log
make spend LOG=run.log SINCE=2026-09-19   # one day of it
make spend-by-shape LOG=run.log           # split by model and judgement
```

`LOG` is **required** and names the text log a drain wrote to a terminal —
`make questions | tee run.log` — not the shipped JSON. For a run happening
now, use Grafana's **Pipeline throughput** dashboard or Phoenix.

`spend-by-shape` splits by the model that answered and the Pydantic class the
call had to return, which is what each judgement costs.

## Orchestration

The asset graph already chains the stages. These decide **when**.

| Target | Does |
|---|---|
| `make runs` | The sensor, the schedule and the last five runs |
| `make pipeline` | Every stage once, as a recorded Dagster run |
| `make auto` | Start the `arrivals` sensor: an upload starts a run on its own |
| `make manual` | Stop it again |
| `make dagster-dev` | Run the code location on the host |

`auto` is the unattended mode and ships stopped. `pipeline` and `make corpus`
do the same work; the difference is that `pipeline` survives the terminal
closing.

All four have an equivalent through the api, for a deployment nobody drives
from a terminal: `POST /pipeline/run` is `make pipeline`, and
`PUT /pipeline/automation` is `auto` and `manual`. The Run button on the
Documents page is the first of those.
