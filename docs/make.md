# Make commands

Every target, grouped. `make` is the only entry point a person needs: it
sources the same configuration the containers read, so one value reaches both.

## How a target reads its configuration

In the order the containers do: the tuning values from
`configs/env/backend.env`, then `.env` for the credentials, the ports and the
addresses a host reaches services at.

**A variable given on the command line beats both:**

```bash
make topics-discover TOPIC_PASSES=20
```

That re-application is deliberate. Sourcing the files would otherwise
overwrite it — which is why `make topics TOPIC_PASSES=20` once ran with the
file's value and said nothing.

A tool with a tuning file of its own gets that too, through `$(call WITH,…)`.

## Using Docker instead of Podman

The Makefile invokes `podman compose`. For Docker:

```bash
make COMPOSE="docker compose" up
```

Or set `COMPOSE=docker compose` in the environment, or edit the variable at
the top of the Makefile. `CONTAINER` is the engine itself, used by `lock`,
which needs a build and a run that compose has no equivalent of.

## Starting and stopping

| Target | Does |
|---|---|
| `make dev` | Generate certificates, start every service, wait for PostgreSQL, create the schema. The one command a first run needs |
| `make install` | Install the dependencies and download the spaCy pipelines `NLP_MODELS` names, retrying three times |
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
| `make logs-retention` | Not a follow: sets how long a day's logs are kept. **Run once**, against a running stack |

```bash
make logs-retention                        # 30 days
make logs-retention LOGS_RETENTION_DAYS=90
```

## The five verbs every stage answers

Each is the same operation as the route beside it. Run any of them with
`--help` for the flags.

| | Parsing | Chunking | Extraction | Questions |
|---|---|---|---|---|
| queue what was never asked for | `parse-start` | `chunk-start` | `extract-start` | `questions-start` |
| drain here, in the foreground | `parse` | `chunk` | `extract` | `questions` |
| show the queue | `parse-status` | `chunk-status` | `extract-status` | `questions-status` |
| take back what has not begun | `parse-stop` | `chunk-stop` | `extract-stop` | `questions-stop` |
| return failures to the queue | `parse-retry` | `chunk-retry` | `extract-retry` | `questions-retry` |
| queue everything again | `parse-rerun` | `chunk-rerun` | `extract-rerun` | `questions-rerun` |

Topic modelling has no `start` and no `rerun`: a fit is all-or-nothing over
one vocabulary, so there is no single topic to start, stop or refit.

| Target | Does |
|---|---|
| `make topics-discover` | Ask for a fit over the whole corpus, and run it |
| `make topics` | Run any queued fit here, in the foreground |
| `make topics-status` | The topics held, and any queued fit |
| `make topics-stop` | Withdraw a queued fit |
| `make topics-retry` | Return a failed fit to the queue |
| `make topics-visualise` | Write each language's pyLDAvis page to `./topics/` |
| `make topics-delete` | Delete every topic and membership |

### Narrowing to one item

Any of the five verbs narrows with `SHA`, `PASSAGE` or `TOPIC`, which is the
same operation against fewer rows:

```bash
make parse-start SHA=<sha256>     # queue one document for parsing
make chunk-rerun SHA=<sha256>     # rebuild one document's passages
make extract-start SHA=<sha256>   # queue every passage of one document
make extract-retry PASSAGE=<id>   # return one failed passage to the queue
make extract-status SHA=<sha256>  # that document's passages by extract state
make questions-start TOPIC=<id>   # write the questions for one topic
```

Parsing and chunking narrow to a document, extraction to a document or a
passage, question generation to a topic, topic modelling to nothing. The first
one given wins.

## Replaying a stage without redoing it

Four operations re-derive what a stage computed, over rows already stored,
**without the expensive part**:

| Target | Does |
|---|---|
| `make extract-revalidate` | Judges every stored fact again. The model is not called and no statement changes — only what the checks read off one |
| `make chunk-revocabulary` | Reads every stored passage's language and vocabulary again. Only `passages.language` and `passages.lemmas` change |
| `make questions-reverify` | Puts every stored question through the gates that need no model |
| `make extract-recap` | Re-applies the atomic cap `EXTRACTION_MIN_OTHER_SHARE` works out to. A corpus extracted before the cap existed is re-balanced in seconds rather than re-read over hours |
| `make extract-bridge` | Reads each topic's passage groups for claims spanning passages |

`chunk-revocabulary` and `extract-revalidate` take `SHA`, `extract-revalidate`
and `extract-recap` take `PASSAGE`, and `questions-reverify` takes `TOPIC`.

These exist because the obvious way to apply a change is the destructive one.
`extract-rerun` calls the model again over the whole corpus, which costs hours
and returns the same statements when only a check changed. `chunk-rerun`
deletes every passage of a document, and the facts drawn from them go with it.

See [`backend/extraction/`](../backend/extraction/README.md) and
[`backend/question_generation/`](../backend/question_generation/README.md) for
what each leaves alone and why.

## Questions

| Target | Does |
|---|---|
| `make questions-balance` | Draw the balanced release out of what was accepted |
| `make questions-reverify` | Re-check stored questions; no model is called |

## Documents

Documents are ingestion's, not a stage's. **All of these are irreversible.**

| Target | Does |
|---|---|
| `make documents` | List every document with its parse state |
| `make delete SHA=<sha256>` | Delete a document and everything derived from it |
| `make delete-derived SHA=<sha256>` | Delete only its passages and facts |
| `make wipe` | Empty the corpus: every document, then the topics. Waits 5 seconds first |

The first two have a route: `DELETE /documents/{sha}` and
`DELETE /documents/{sha}/derived`.

`wipe` is one command that runs both deletions in order, because deleting
every document leaves the topics standing — a topic has no foreign key to a
document. It also drops the upload history, which a single deletion
deliberately keeps.

## Settings

```bash
make settings SERVICE=topics
make settings-set SERVICE=topics SET="TOPIC_PASSES=20"
make settings-unset SERVICE=topics UNSET="TOPIC_PASSES"
```

The services are `ingestion`, `parsing`, `chunking`, `extraction`, `topics`,
`questions` and `platform`. See
[configuration.md](configuration.md).

## Schema

| Target | Does |
|---|---|
| `make migration m="add the dropped counts"` | Write a revision from the models |
| `make schema` | Apply everything outstanding |
| `make schema-status` | Where the database is |
| `make schema-down` | Take the newest revision back off |
| `make schema-reset` | Drop every table and rebuild from the revisions. **Irreversible** |
| `make schema-stamp` | Adopt a database that already holds the tables |

**Read the generated revision before applying it.** Autogenerate does not see
a trigger, a data backfill, or anything that has to happen in a particular
order.

## Review and evaluation

| Target | Does |
|---|---|
| `make review-push-facts` / `review-pull-facts` | Send a sample of facts to Argilla; bring the verdicts back |
| `make review-push-topics` / `review-pull-topics` | The same, for topic labels |
| `make review-push-questions` / `review-pull-questions` | The same, for questions |
| `make review-status` | How much has been looked at |
| `make eval-upload` | Put the golden cases in Phoenix |
| `make eval-score` | Score the served model against them, and record it |

```bash
make eval-score EVAL_RUN_NAME=extraction-prompt-v7
```

## Tests and checks

| Target | Runs |
|---|---|
| `make test` | Everything that gates; starts containers of its own |
| `make check` | An alias for `make test` |
| `make test-fast` | Only the fast layers: no spaCy, no pyright, no containers |
| `make test-unit` | Everything but the integration, smoke and eval layers |
| `make test-integration` | The container layers |
| `make test-e2e` | One document through every stage, in this process |
| `make test-smoke` | Build both images and look inside them |
| `make test-eval` | Score the served model against the golden passages. Never gates |
| `make test-coverage` | The gating layers, with a coverage report |
| `make lint` | ruff check and format check |
| `make format` | Apply every fix ruff can make |
| `make typecheck` | pyright, at zero |
| `make audit` | Known advisories against the two locks |
| `make lock` | Rewrite `backend/api/requirements.lock` |

`lint`, `format` and `typecheck` all cover `backend`, `frontend`, `telemetry`,
`orchestration`, `review`, `evaluation` and `tests`.

## Cost

```bash
make spend                    # totals from the workers' logs
make spend SINCE=2026-09-19   # one day
make spend LOG=/path/to.log   # one file
```

Reads the priced model calls out of the logs and reports the call count, the
token counts and the cost. Reads `/var/log/qa/*.log` unless `LOG` says
otherwise, so it is run against a machine that has the logs volume mounted.

## Orchestration

| Target | Does |
|---|---|
| `make dagster-dev` | Run the code location on the host, against the containerised PostgreSQL and API |

## Known edges

- **`make dev` is `install` plus `certs` plus `up` plus `schema`.** On a first
  run it pulls several images and takes a few minutes.
- **`make down-volumes` and `make wipe` are both irreversible** and delete
  different things: the first takes the volumes, the second takes the rows.
- **`make lock` needs the container engine, not compose.** Compose has no
  equivalent of `--target`.
- **A `make` target logs to the terminal and nowhere else.** `LOG_DIR` is
  unset on the host, so a host drain leaves no line in Grafana.
- **`make spend` finds nothing on a machine with no logs volume.** Point it at
  a file with `LOG=`.
