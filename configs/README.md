# Configs

Service configuration and init scripts — everything that is a **decision**
rather than a credential, and so lives in git.

The split is by *what the value is*, not by which service reads it. Tuning is
here; credentials, ports and the addresses a host reaches a service at are in
`.env`, which is not in git. See
[docs/configuration.md](../docs/configuration.md) for the full picture.

## Layout

| Path | Holds |
|---|---|
| [`env/`](env/) | The tuning files, one per concern — see below |
| [`dagster/`](dagster/) | The Dagster instance and its one code location |
| [`filebeat/`](filebeat/) | What the log shipper reads, and where it puts it |
| [`grafana/`](grafana/) | Three datasources and four dashboards, provisioned |
| [`postgres/`](postgres/) | The init script: the databases, the roles and the two extensions |
| [`seaweedfs/`](seaweedfs/) | The S3 gateway's Caddyfile, the bucket init and the entrypoint |

## `env/`

| File | In git | Holds |
|---|---|---|
| `backend.env` | yes | How the pipeline behaves: the parsing, chunking, extraction, topic and question settings the API and the five workers read |
| `elasticsearch.env` | yes | The node's certificate paths, security flags and heap. One node serves both Argilla and the logs |
| `seaweedfs-filer.env` | yes | Which metadata store the filer uses |
| `orchestration.env` | yes | How patiently Dagster watches a stage it started |
| `review.env` | yes | How a review sample is drawn |
| `evaluation.env` | yes | What a golden-set run is called in Phoenix |
| `provider.env` | **no** | Whatever env vars the configured model provider needs, credentials included. Optional, absent for a plain Ollama, read by the three services that call a model and by nothing else |

`backend.env` is the long form: each setting carries the paragraphs explaining
what it does and how its default was arrived at.
[`settings/catalog.py`](../backend/settings/catalog.py) carries the one-line
version, and `tests/static/` refuses a setting that appears in one and not the
other.

compose hands each file to the services that need it with `env_file`, and the
Makefile sources `backend.env` and `.env` for the host commands, so one value
reaches both.

## `postgres/`

[`init.sh`](postgres/init.sh) creates the five databases, the application
role, the `grafana_reader` role, and the `vector` and `pg_trgm` extensions.

It **only runs on the first boot of an empty volume**. On a stack that already
has one, re-run it by hand:

```bash
podman compose exec postgres bash /docker-entrypoint-initdb.d/init.sh
```

`grafana_reader` holds `SELECT` and nothing else: a dashboard is a place
people paste SQL into, and the application role can `DROP`.

It is granted on **two** databases — the application's and Phoenix's, which
the Run dashboard reads together. Phoenix owns its own schema, so the
default privilege there is declared for the Phoenix role rather than the
superuser, or a table Phoenix creates on a later migration would be
invisible to Grafana.

## `grafana/`

Four dashboards over three datasources, all provisioned — nothing is clicked
into existence, and a dashboard edited in the UI is lost on the next restart.

| Dashboard | Reads | Shows |
|---|---|---|
| Pipeline state | `Pipeline` | Queue depth per stage, failures with reasons, fact acceptance by rejection code, question acceptance by gate, topics and their coverage |
| Pipeline throughput | `Logs` | Units finished per interval, model latency at p50/p95/p99, facts and questions accepted against refused, and which queue verb was asked for over HTTP |
| Pipeline logs | `Logs` | Lines per level, what failed and where, every line |
| Run | all three | One run on one page: what it produced, what it cost, where the questions went, what the calls were for, its Phoenix projects, the gate verdicts recorded there, and every line it wrote |

| Datasource | Points at |
|---|---|
| `Pipeline` | The application database |
| `Logs` | Elasticsearch, the `qa-logs` data stream |
| `Phoenix` | Phoenix's own database — the same PostgreSQL server as `Pipeline`, a different database, the same read-only role |

**The split is the point.** Logs say what happened once; the tables say what
is true now. A row a worker died holding logged nothing and is still counted
in Pipeline state, which is the difference that matters when a stage has gone
quiet.

**Run crosses that split deliberately**, and it is why `Phoenix` exists as a
datasource. A run is `questions.run_id` on the application side and a project
named `<stage>-<run id>` on the other; the gate counts are rows and the
calls, tokens, latency and spend are in the spans.

`tests/static/test_dashboards.py` checks each panel against the datasource and
the fields that serve it, so a renamed column fails a pull request rather than
emptying a panel nobody is looking at.

## `filebeat/`

[`filebeat.yml`](filebeat/filebeat.yml) reads the shared logs volume and
writes the `qa-logs` data stream.

Two settings in it are load-bearing and are explained in
[`telemetry/README.md`](../telemetry/README.md): `setup.template.append_fields`,
which is what keeps `stage` a keyword rather than analysed text, and
`setup.template.overwrite`, without which an edit to this file does nothing at
all.

## `seaweedfs/`

[`bucket-init.sh`](seaweedfs/bucket-init.sh) creates the buckets `S3_BUCKETS`
names, at first boot. [`Caddyfile`](seaweedfs/Caddyfile) fronts the S3
gateway.

## `dagster/`

[`dagster.yaml`](dagster/dagster.yaml) is the instance — its run storage, in
the `dagster` database — and [`workspace.yaml`](dagster/workspace.yaml) names
the one code location, which is [`orchestration/`](../orchestration/README.md).

## Certificates

`certs/` is **not** in git and is generated by `make certs`, which is run for
you by `make dev`. Elasticsearch needs them; nothing else does.

## Known edges

Things that are true, are not bugs, and have surprised somebody.

- **`init.sh` runs once, on an empty volume.** Every change to it needs
  re-running by hand on a stack that already exists.
- **A Grafana dashboard edited in the UI is lost on restart.** Provisioning
  is one-way. Edit the JSON.
- **`provider.env` is optional and absent by default.** A plain Ollama needs
  nothing in it. Its absence is not an error.
- **An edit to `filebeat.yml` does nothing without
  `setup.template.overwrite`.** Filebeat leaves an existing template alone.
- **`backend.env` is in git and `.env` is not**, and both are sourced by the
  Makefile. A value in the wrong one either leaks or goes missing.
