# Configs

Service configuration and init scripts — everything that is a **decision**
rather than a credential, and so lives in git.

The credentials, and which model this deployment calls, are in `.env`, which
is not. The ports and addresses used to be there too and are now in
[`env/deployment.env`](env/deployment.env), because neither is a secret and
both are worth reviewing in a diff. See
[docs/configuration.md](../docs/configuration.md) for the full picture.

| Path | Holds |
|---|---|
| [`env/`](env/) | The tuning files, one per concern |
| [`dagster/`](dagster/) | The Dagster instance and its one code location |
| [`filebeat/`](filebeat/) | What the log shipper reads, and where it puts it |
| [`grafana/`](grafana/) | Three datasources and four dashboards, provisioned |
| [`postgres/`](postgres/) | The init script: the databases, the roles and the two extensions |
| [`seaweedfs/`](seaweedfs/) | The S3 gateway's Caddyfile, the bucket init and the entrypoint |

## `env/`

| File | In git | Holds |
|---|---|---|
| `backend.env` | yes | How the pipeline behaves: the parsing, chunking, extraction, topic, question and assessment settings |
| `deployment.env` | yes | Where this deployment put things: the ports, the container-side addresses, the topology, the role names |
| `elasticsearch.env` | yes | The node's certificate paths, security flags and heap. One node serves both Argilla and the logs |
| `seaweedfs-filer.env` | yes | Which metadata store the filer uses |
| `orchestration.env` | yes | How long Dagster waits on a stage it started |
| `review.env` | yes | How a review sample is drawn |
| `evaluation.env` | yes | What a golden-set run is called in Phoenix |
| `provider.env` | **no** | Whatever env vars the model provider needs. Optional, absent for a plain Ollama, read by the three services that call a model |

**A name appears in exactly one of these, and in `.env` or here but never
both.** No file overrides another, so there is no order to know;
`tests/static/test_env_files.py` refuses a second copy. A value that could
be written twice — a port inside an address, a password inside a connection
string — is derived instead and written nowhere. See
[docs/configuration.md](../docs/configuration.md) for the list.

`backend.env` is the long form: each setting carries the paragraphs
explaining what it does. [`settings/catalog.py`](../backend/settings/catalog.py)
carries the one-line version, and `tests/static/` refuses a setting that
appears in one and not the other.

compose hands each file to the services that need it with `env_file`, and the
Makefile sources `backend.env`, `deployment.env` and `.env` for host
commands. `deployment.env` is also one of the two files compose interpolates
`${...}` out of, which is why every compose command goes through
`$(COMPOSE)`:

```bash
podman compose --env-file configs/env/deployment.env --env-file .env
```

## `postgres/`

[`init.sh`](postgres/init.sh) creates the five databases, the application
role, the `grafana_reader` role, and the `vector` and `pg_trgm` extensions.

It **only runs on the first boot of an empty volume**. On a stack that
already has one:

```bash
podman compose exec postgres bash /docker-entrypoint-initdb.d/init.sh
```

`grafana_reader` holds `SELECT` and nothing else, and is granted on **two**
databases — the application's and Phoenix's, which the Run dashboard reads
together. Phoenix owns its own schema, so the default privilege there is
declared for the Phoenix role rather than the superuser.

## `grafana/`

Four dashboards over three datasources, all provisioned. A dashboard edited
in the UI is lost on the next restart.

| Dashboard | Reads | Shows |
|---|---|---|
| Pipeline state | `Pipeline` | Queue depth per stage, failures with reasons, fact and question acceptance, topic coverage |
| Pipeline throughput | `Logs` | Units finished per interval, model latency at p50/p95/p99, facts and questions accepted against refused |
| Pipeline logs | `Logs` | Lines per level, what failed and where |
| Run | all three | One run on one page: what it produced, what it cost, its Phoenix projects, the gate verdicts, every line it wrote |

| Datasource | Points at |
|---|---|
| `Pipeline` | The application database |
| `Logs` | Elasticsearch, the `qa-logs` data stream |
| `Phoenix` | Phoenix's own database — the same PostgreSQL server, a different database, the same read-only role |

Logs say what happened once; the tables say what is true now. **Run** crosses
that split deliberately, which is why `Phoenix` exists as a datasource: a run
is `questions.run_id` on one side and a project named `<stage>-<run id>` on
the other.

`tests/static/test_dashboards.py` checks each panel against the datasource
and the fields that serve it.

## `filebeat/`

[`filebeat.yml`](filebeat/filebeat.yml) reads the shared logs volume and
writes the `qa-logs` data stream.

Two settings in it are load-bearing and are explained in
[`telemetry/README.md`](../telemetry/README.md):
`setup.template.append_fields`, which keeps `stage` a keyword rather than
analysed text, and `setup.template.overwrite`, without which an edit to this
file does nothing.

## `seaweedfs/`

[`bucket-init.sh`](seaweedfs/bucket-init.sh) creates the buckets `S3_BUCKETS`
names, at first boot. [`Caddyfile`](seaweedfs/Caddyfile) fronts the S3
gateway.

## `dagster/`

[`dagster.yaml`](dagster/dagster.yaml) is the instance, with its run storage
in the `dagster` database, and [`workspace.yaml`](dagster/workspace.yaml)
names the one code location, [`orchestration/`](../orchestration/README.md).

## Certificates

`certs/` is **not** in git and is generated by `make certs`, run for you by
`make dev`. Elasticsearch needs them; nothing else does.

## Limits

- **`init.sh` runs once, on an empty volume.** Every change to it needs
  re-running by hand on a stack that already exists.
- **A Grafana dashboard edited in the UI is lost on restart.** Edit the JSON.
- **`provider.env` is optional and absent by default.** Its absence is not an
  error.
- **An edit to `filebeat.yml` does nothing without
  `setup.template.overwrite`.**
- **Everything here is in git and `.env` is not**, and all of them are
  sourced by the Makefile. A value in the wrong one either leaks or goes
  missing; the rule is that only a secret is in `.env`, and a test checks
  that `.env.example` holds nothing but placeholders.
- **A bare `podman compose` does not read `deployment.env`.** Use `make`, or
  export `COMPOSE_ENV_FILES=configs/env/deployment.env,.env`. It cannot be
  set inside `.env` — compose resolves the file list before reading one.
