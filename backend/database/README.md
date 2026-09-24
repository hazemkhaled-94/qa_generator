# Database

One PostgreSQL schema, one module per table, and the engine every process
opens its own pool against.

This package builds connections. It is one of only three that do —
[`blob_store/`](../blob_store/README.md) and [`nlp/`](../nlp/README.md) are
the others — which is what stops a service configuring the infrastructure or
reaching around another service to it.

It holds **no business logic**. A stage owns a repository over these models;
this package owns the tables, the enums, the connection and the migrations.

## The tables

| Table | Holds | Written by |
|---|---|---|
| `ingest_events` | Every upload attempt, refused ones included | [ingestion](../ingestion/README.md) |
| `documents` | One row per kept document, keyed by the SHA-256 of its bytes, carrying every stage's status | ingestion, then each stage's own column |
| `passages` | The unit facts are drawn from: text, numbered sentences, lemmas, language, table cells, bounding box, heading trail | [chunking](../preprocessing/chunking/README.md) |
| `facts` | One claim each, with its verdict, its rejection code and the configuration it was produced under | [extraction](../extraction/README.md) |
| `fact_passages` | Which passages a fact rests on, and the span it cited in each | extraction |
| `topics` | Both a fitted topic and a request to refit — see below | [topic modelling](../topic_modelling/README.md), [question generation](../question_generation/README.md) |
| `passage_topics` | How much of each topic a passage holds | topic modelling |
| `questions` | The question, its answer, its derived difficulty, its thread, its embedding and its release | question generation |
| `question_facts` | Which facts each question cites | question generation |
| `service_settings` | The settings a deployment changed — see [`settings/`](../settings/README.md) | the API, the CLI, the UI |
| `prompts` | One prompt per version per service, stored **as composed** — what a `prompt_version` on a row actually asked for | each stage, once before its first claim; see [`stages/`](../stages/README.md) |
| `archived_rows` | Every row deleted from any of the ten above, as JSON — see below | an AFTER DELETE trigger, and [`archive/`](../archive/README.md) |

### `topics` carries two queues

A row with a **NULL `topic_index`** is a request to refit, and belongs to
topic modelling. A row that **is** a topic belongs to question generation,
which queues over it by `question_status`. The table is both the queue and the
result.

Every operation on the second carries `topic_index IS NOT NULL`, because
without it `start` would queue the asking as though it were a subject and the
two workers would fight over one row.

### How a row reaches its topics

A fact or a question reaches its topics by **joining through its passages** —
`fact_passages` → `passage_topics` — rather than holding a topic of its own,
so no two rows can disagree about which topic a passage is in.

### Cascades, and the triggers

Deleting a document takes everything derived from it: its passages, their
topic memberships, the facts on them, and — through a trigger on
`question_facts` — the questions resting on those facts.

The trigger exists because a cascade cannot express it. A question rests on
*several* facts, so `ON DELETE CASCADE` from `facts` would take the question
when its **first** citation went rather than its last. The trigger deletes a
question once its last citation is gone, and no sooner.

`ingest_events.document_sha256` is `ON DELETE SET NULL`, so the record that an
upload happened outlives the document.

What a deletion does **not** take is `topics`. A topic has no foreign key to a
document, because a fit is over the corpus rather than over a file. Deleting
every document leaves the topics standing, describing passages that are gone.
Nothing is broken by that — the next fit replaces every row — but it is why
`make wipe` runs both in order.

### Nothing is deleted without a copy

Every table carries an `AFTER DELETE ... FOR EACH ROW` trigger that writes
the row into [`archived_rows`](qa_generator/archived_rows.py) as
`to_jsonb(OLD) - 'embedding'`. One shared function, because `to_jsonb` names
no column.

A trigger and not the services, for the reason the paragraphs above describe:
one `make delete` reaches six tables, and only the first row of it passes
through any Python. The cascades, the orphan triggers, a re-chunk, a
re-extraction and a statement typed into Adminer are all archived the same
way.

It is not a soft delete. There is no `deleted_at` on any table, no predicate
for a repository to remember, and the live tables still lose the row.
`make archive-purge` is what finishes a deletion; see
[`archive/`](../archive/README.md).

## Extensions

Two, installed by the initial revision and by
[`configs/postgres/init.sh`](../../configs/postgres/init.sh) alike:

| Extension | For |
|---|---|
| `vector` | `questions.embedding`, and the HNSW index over it the dedup gate probes |
| `pg_trgm` | The substring search behind every page's `q` box |

`tests/static/test_schema_extensions.py` checks that the migrations install
every extension the bootstrap does, so a database built by either route is the
same database.

## Migrations

Alembic owns the schema. The models say what the tables should be; a revision
under [`migrations/versions/`](migrations/versions/) says how to get an
existing database there.

```bash
make migration m="add the dropped counts"  # write a revision from the models
make schema                                # apply everything outstanding
make schema-status                         # where the database is
make schema-down                           # take the newest revision back off
make schema-reset                          # drop every table and rebuild. Irreversible
make schema-stamp                          # adopt a database that already holds the tables
```

**Read the generated revision before applying it.** Autogenerate compares
tables, columns, indexes and constraints; it does not see a trigger, a data
backfill, or anything that has to happen in a particular order. Both such
cases in this history are written out by hand: every trigger, and the deletion
of every fact in the revision that changed what a citation is.

Every revision is reversible, and
`tests/integration/database/test_migration_rollback.py` applies and takes back
out every one of them on a real PostgreSQL. A revision whose `downgrade` does
not work is caught before it is merged, not during an incident.

### Revisions as documentation

A revision's docstring says **why**, not what — the diff says what. The one
that removed `wrong_type` records the measurement that killed it: the gate
fired zero times over 71 questions while the kind was plainly wrong on six of
the 27 accepted. That is the only place that finding is written down where it
cannot drift.

## Column comments

Column comments live on the models and are carried into the database by the
migrations, so `\d+ passages` in `psql` and Adminer's column list both say
what a column is for. That is what makes the database browsable by somebody
who has not read this repository — including anybody writing SQL into a
Grafana panel.

## Connections

Each process opens its **own** pool, sized by `DATABASE_POOL_SIZE` and
`DATABASE_POOL_OVERFLOW`. The total across the API and every worker has to
stay under PostgreSQL's `max_connections`.

Both are read before a service could ask a database for anything, so they are
the deployment's: served read-only through `/settings/platform` and marked
`fixed`. Changing one needs a restart.

| Setting | Default |
|---|---|
| `DATABASE_POOL_SIZE` | 5 |
| `DATABASE_POOL_OVERFLOW` | 5 |

The URL is assembled from the credentials in `.env` and is never a setting.

## Grafana reads as a different role

`grafana_reader` holds `SELECT` and nothing else. A dashboard is a place
people paste SQL into, and the application role can `DROP`.

It is granted on **two** databases: the application's and Phoenix's, which
the Run dashboard reads together. Phoenix owns its own schema, so the
default privilege there is declared for the Phoenix role rather than the
superuser — otherwise a table Phoenix creates on a later migration would be
invisible to Grafana.

The role is created by [`configs/postgres/init.sh`](../../configs/postgres/init.sh),
which only runs on the first boot of an empty volume. On a stack that already
has one, re-run it by hand:

```bash
podman compose exec postgres bash /docker-entrypoint-initdb.d/init.sh
```

## Tools, and where each is used

| Tool | Where | Why this one |
|---|---|---|
| **SQLAlchemy** | [`qa_generator/`](qa_generator/) | Declarative models are the single statement of what a table is, and the migrations are generated from them |
| **Alembic** | [`migrations/`](migrations/) | One revision per schema change, each reversible |
| **pgvector** | [`questions.py`](qa_generator/questions.py) | An HNSW index over 1024-wide vectors, in the database that already holds the rows |
| **psycopg** | [`engine.py`](qa_generator/engine.py) | The driver. Nothing else in the project imports it |

## Tests

```sh
poetry run pytest tests/integration/database
```

| File | Covers |
|---|---|
| [`test_schema.py`](../../tests/integration/database/test_schema.py) | The schema alembic builds on an empty database |
| [`test_migration_rollback.py`](../../tests/integration/database/test_migration_rollback.py) | Every revision, applied and taken back out |
| [`test_constraints.py`](../../tests/integration/database/test_constraints.py), [`test_fact_constraints.py`](../../tests/integration/database/test_fact_constraints.py) | What the database refuses to hold |
| [`test_indexes.py`](../../tests/integration/database/test_indexes.py) | The indexes and column types the queries are written against |
| [`test_trigger.py`](../../tests/integration/database/test_trigger.py) | The trigger that deletes a question once its last fact is gone |
| [`test_queue.py`](../../tests/integration/database/test_queue.py), [`test_question_queue.py`](../../tests/integration/database/test_question_queue.py) | Claiming, the lease sweep, and two queues on one table |
| [`test_orphans.py`](../../tests/integration/test_orphans.py) | What survives deleting every document |
| [`tests/static/test_migrations.py`](../../tests/static/test_migrations.py) | The revision chain: one head, no gaps |
| [`tests/static/test_schema_extensions.py`](../../tests/static/test_schema_extensions.py) | The migrations install every extension the bootstrap does |

These start a PostgreSQL of their own through testcontainers and skip, with a
reason, where no container engine answers. Nothing they do touches a running
stack.

## Known edges

Things that are true, are not bugs, and have surprised somebody.

- **A CHECK constraint duplicates a check the code already makes.** That is
  deliberate: `unsupported_addition` and `unresolved_reference` are both
  constraints as well as checks, so a bug in the checker cannot store a fact
  that contradicts them.
- **`make schema-reset` is irreversible and says so.** It drops every table
  and rebuilds from the revisions. The data is gone; the objects in the
  buckets are not.
- **Autogenerate will not notice a trigger.** It compares tables, columns,
  indexes and constraints. Anything else is written by hand.
- **`documents` is the one row two stages both write.** Parsing owns
  `parse_status`, chunking owns `chunk_status`, and neither reads the other's.
  That is the whole contract between them.
- **Deleting every document does not empty the corpus.** The topics survive.
  `make wipe` is the command that does both.
