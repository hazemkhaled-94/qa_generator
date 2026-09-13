# `certs` uses process substitution, which /bin/sh does not have.
SHELL := /bin/bash

# Override for Docker: make COMPOSE="docker compose" up
COMPOSE ?= podman compose

# The engine itself, for the build and run that `lock` needs; compose has no
# equivalent of --target.
CONTAINER ?= podman

LOCK_TMP := backend/api/requirements.lock.new

# What every host command reads, and in the same order the containers do: the
# tuning values from configs/env/backend.env, then .env for the credentials,
# the ports and the addresses a host reaches services at.
#
# OVERRIDE re-applies whatever was given on the command line, because sourcing
# the files above would otherwise overwrite it - which is why
# `make topics TOPIC_PASSES=20` used to run with the file's value and say
# nothing.
OVERRIDE = $(if $(MAKEOVERRIDES),&& export $(MAKEOVERRIDES),)
LOADENV = set -a && . ./configs/env/backend.env && . ./.env && set +a $(OVERRIDE)

# Narrows a stage target to one item instead of the whole queue, mirroring
# the route's /{scope}/{value} segment:
#
#   make parse-start   SHA=<sha256>     one document
#   make chunk-rerun   SHA=<sha256>     one document
#   make extract-start SHA=<sha256>     every passage of one document
#   make extract-retry PASSAGE=<id>     one passage
#
# Which scopes a stage accepts is the stage's own; parsing and chunking take
# a document, extraction takes either, topic modelling takes neither. SHA
# wins if both are given.
ONLY = $(if $(SHA),--only document=$(SHA),$(if $(PASSAGE),--only passage=$(PASSAGE)))

.PHONY: dev install up down down-volumes logs logs-frontend logs-api \
        schema schema-reset schema-status schema-down schema-stamp migration \
        parse parse-status parse-start parse-stop parse-retry parse-rerun \
        chunk chunk-status chunk-start chunk-stop chunk-retry chunk-rerun \
        extract extract-status extract-start extract-stop extract-retry \
        extract-rerun \
        topics topics-status topics-discover topics-stop topics-delete \
        topics-retry topics-visualise \
        documents delete delete-derived \
        check lint format lock certs dagster-dev

# ── Bootstrap ──────────────────────────────────────────────────────────────

# Install dependencies, start every service, and create the schema.
dev: install certs
	$(COMPOSE) up -d
	@echo "Waiting for PostgreSQL..."
	@$(LOADENV) && \
	  until $(COMPOSE) exec -T postgres pg_isready -U "$$POSTGRES_SUPERUSER" -q; \
	  do sleep 2; done
	$(MAKE) schema

# The spaCy pipelines are downloaded, not resolved: they are not on PyPI
# under a version range. The same names go into the image; see
# SPACY_MODELS in backend/api/Dockerfile.
install:
	poetry install --with llm,nlp,data,storage,api,pipeline,viz,observability,dev
	poetry run python -m spacy download de_core_news_md
	poetry run python -m spacy download en_core_web_md

# ── Services ───────────────────────────────────────────────────────────────

up: certs
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

# Deletes every volume. Irreversible.
down-volumes:
	@echo "WARNING: this deletes all persistent data. Ctrl-C within 5s to abort."
	@sleep 5
	$(COMPOSE) down -v

logs:
	$(COMPOSE) logs -f

logs-frontend:
	$(COMPOSE) logs -f streamlit

# Upload failures surface here; the frontend only sees the status code.
logs-api:
	$(COMPOSE) logs -f api

# ── Database ───────────────────────────────────────────────────────────────
#
# Alembic owns the schema. The models say what the tables should be; a
# revision says how to get an existing database there, which `create_all`
# could never do. Runs on the host against the mapped port, as the
# application role, so tables are not owned by the superuser.
#
# After changing a model:  make migration m="what changed"  then  make schema
# Review the generated revision before applying it: autogenerate sees tables,
# columns, indexes and constraints, and never a trigger or a data backfill.

# Bring the database up to the newest revision.
schema:
	$(LOADENV) && poetry run alembic upgrade head

# Write a revision from the difference between the models and the database.
#   make migration m="add the dropped counts"
migration:
	@[ -n "$(m)" ] || { echo 'usage: make migration m="what changed"'; exit 2; }
	$(LOADENV) && poetry run alembic revision --autogenerate -m "$(m)"

# What the database is at, and everything it could be at.
schema-status:
	$(LOADENV) && poetry run alembic current
	$(LOADENV) && poetry run alembic history --indicate-current

# Take the newest revision back off.
schema-down:
	$(LOADENV) && poetry run alembic downgrade -1

# Adopt a database that already holds the tables, without running anything:
# records it as being at the newest revision. For a deployment that predates
# migrations.
schema-stamp:
	$(LOADENV) && poetry run alembic stamp head

# Drops every table and rebuilds from the revisions. Irreversible.
schema-reset:
	@echo "WARNING: this drops every table in qa_generator. Ctrl-C within 5s to abort."
	@sleep 5
	$(LOADENV) && poetry run alembic downgrade base
	$(LOADENV) && poetry run alembic upgrade head

# ── Pipeline stages ────────────────────────────────────────────────────────
#
# Nothing starts by itself. A row arrives `new` and no worker looks at it;
# `-start` is what makes it claimable and `-stop` is what takes it back. A
# stage never sets another stage going.
#
# The bare target runs one drain on the host, in the foreground, against the
# same database the workers use - which is what you want while developing a
# stage. Every other target is the same operation as the route beside it:
#
#   make parse-status      GET /parsing/status
#   make parse-start      POST /parsing/start
#   make parse-stop       POST /parsing/stop
#   make parse-retry      POST /parsing/retry
#   make parse-rerun      POST /parsing/rerun
#   make chunk-status      GET /chunking/status
#   make chunk-start      POST /chunking/start
#   make chunk-stop       POST /chunking/stop
#   make chunk-retry      POST /chunking/retry
#   make chunk-rerun      POST /chunking/rerun
#   make extract-status    GET /extraction/status
#   make extract-start    POST /extraction/start
#   make extract-stop     POST /extraction/stop
#   make extract-retry    POST /extraction/retry
#   make extract-rerun    POST /extraction/rerun
#
# Add SHA or PASSAGE to narrow any of them to one item, which mirrors the
# route's own /{scope}/{value} segment:
#
#   make parse-start SHA=abc…       POST /parsing/document/abc…/start
#   make extract-rerun SHA=abc…     POST /extraction/document/abc…/rerun
#   make extract-retry PASSAGE=41   POST /extraction/passage/41/retry
#   make extract-status SHA=abc…     GET /extraction/document/abc…/status
#
# None of them runs anything: they move rows between statuses, and whichever
# worker is watching picks up what is claimable.
#
# The API has no /run. It cannot: the work happens in the workers, and the
# api container is sized to serve JSON.

# Drain the parsing queue here. The first run downloads Docling's weights.
parse:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m preprocessing.parsing.run

# Report how many documents are in each parse state.
parse-status:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m preprocessing.parsing.run --status $(ONLY)

# Queue every document this stage has not been asked to do yet. Nothing
# reaches a worker until this runs.
parse-start:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m preprocessing.parsing.run --start $(ONLY)

# Take back whatever has not begun. The one in hand finishes.
parse-stop:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m preprocessing.parsing.run --stop $(ONLY)

# Return every failed document to the pending queue. A document a worker died
# holding is failed by the next run, so this covers that too.
parse-retry:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m preprocessing.parsing.run --retry $(ONLY)

# Run this stage again over every document, finished ones included. For when
# the code behind it changed and its output needs rebuilding.
parse-rerun:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m preprocessing.parsing.run --rerun $(ONLY)

# Split every parsed document into passages. Replaces the passages a document
# already had, which cascades to the facts and questions drawn from them.
chunk:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m preprocessing.chunking.run

# Report how many documents are in each chunking state.
chunk-status:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m preprocessing.chunking.run --status $(ONLY)

# Queue every document this stage has not been asked to do yet. Nothing
# reaches a worker until this runs.
chunk-start:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m preprocessing.chunking.run --start $(ONLY)

# Take back whatever has not begun. The one in hand finishes.
chunk-stop:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m preprocessing.chunking.run --stop $(ONLY)

# Return every failed document to the pending queue.
chunk-retry:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m preprocessing.chunking.run --retry $(ONLY)

# Run this stage again over every document, finished ones included. For when
# the code behind it changed and its output needs rebuilding.
chunk-rerun:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m preprocessing.chunking.run --rerun $(ONLY)

# Drain the extraction queue here. Needs the model in LLM_MODEL to be
# served at LLM_BASE_URL.
extract:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m extraction.run

# Report the extraction queue, and how many facts passed every check.
extract-status:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m extraction.run --status $(ONLY)

# Queue every passage this stage has not been asked to do yet. Nothing
# reaches a worker until this runs.
extract-start:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m extraction.run --start $(ONLY)

# Take back whatever has not begun. The one in hand finishes.
extract-stop:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m extraction.run --stop $(ONLY)

# Return every failed passage to the pending queue.
extract-retry:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m extraction.run --retry $(ONLY)

# Run this stage again over every passage, finished ones included. For when
# the code behind it changed and its output needs rebuilding.
extract-rerun:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m extraction.run --rerun $(ONLY)

# ── Topic modelling ────────────────────────────────────────────────────────
#
# The one stage nothing triggers on its own. Every topic is fitted jointly
# over one vocabulary, so a topic cannot be rediscovered by itself and a fit
# is asked for: topics-discover queues a run and the worker picks it up.
#
#   make topics-status       GET /topics/status
#   make topics-discover    POST /topics/discover
#   make topics-visualise    GET /topics/visualisation/{language}
#   make topics-stop        POST /topics/stop
#   make topics-retry       POST /topics/retry
#   make topics-delete    DELETE /topics

# Run any queued fit now, in the foreground.
topics:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m topic_modelling.run

# Report the run queue, and how many topics and memberships are stored.
topics-status:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m topic_modelling.run --status

# Queue a fit and run it. Replaces every existing topic and membership;
# labels are carried over where the top terms still match.
topics-discover:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m topic_modelling.run --discover

# Write each language's pyLDAvis page to ./topics/<language>.html. Drawn by a
# fit, so run topics-discover first if there is none.
topics-visualise:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m topic_modelling.run --visualise

# Delete every topic and membership. Passages, facts and questions stay.
# Irreversible for any label a person assigned: nothing else stores one.
topics-delete:
	@echo "WARNING: this deletes every topic, and any label on one. Ctrl-C within 5s to abort."
	@sleep 5
	$(LOADENV) && PYTHONPATH=backend poetry run python -m topic_modelling.run --delete

# Withdraw a fit that has been asked for but not started.
topics-stop:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m topic_modelling.run --stop

# Return every failed fit to the queue.
topics-retry:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m topic_modelling.run --retry

# ── Documents ──────────────────────────────────────────────────────────────

# List every stored document with its digest and parse state.
documents:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m ingestion.run --list

# Delete a document, its file, its converted form and everything derived from
# it. Irreversible. The upload record in ingest_events is kept.
#   make delete SHA=<sha256>
delete:
	@[ -n "$(SHA)" ] || { echo "usage: make delete SHA=<sha256>"; exit 2; }
	$(LOADENV) && PYTHONPATH=backend poetry run python -m ingestion.run --delete $(SHA)

# Delete only what the pipeline built: passages and facts. The document and
# its file stay, and chunking goes back to `new`, so the next run rebuilds
# them once somebody starts it.
#   make delete-derived SHA=<sha256>
delete-derived:
	@[ -n "$(SHA)" ] || { echo "usage: make delete-derived SHA=<sha256>"; exit 2; }
	$(LOADENV) && PYTHONPATH=backend poetry run python -m ingestion.run --delete-derived $(SHA)

# ── Static checks ──────────────────────────────────────────────────────────

# The logic that would fail silently: the fact checks, the sentence numbering
# a citation resolves against, the chunker's size filter, the table reader's
# labelling, the parser's word repair, the worker's shutdown and the topic
# model's weights and vocabulary. No database and no served model, so it runs
# before anything is started; it does load the spaCy pipelines.
check:
	PYTHONPATH=backend poetry run python -m checks

# Configuration lives in [tool.ruff] in pyproject.toml.
lint:
	poetry run ruff check backend frontend telemetry
	poetry run ruff format --check backend frontend telemetry

format:
	poetry run ruff check --fix backend frontend telemetry
	poetry run ruff format backend frontend telemetry

# ── Dependency locks ───────────────────────────────────────────────────────

# Rewrites backend/api/requirements.lock from the ranges in
# requirements.txt. Run it after changing requirements.txt; the image build
# refuses a lock that does not pin everything requirements.txt declares.
#
# The resolving happens in the Dockerfile's `resolve` stage, in the same
# base image the runtime uses, because a lock resolved on macOS pins wheels
# that do not exist for linux/aarch64. The file's comment header is carried
# over; everything below it is replaced.
lock:
	$(CONTAINER) build --target resolve -t qa_generator-resolve \
	  -f backend/api/Dockerfile .
	@awk '/^#/ {print; next} {exit}' backend/api/requirements.lock > $(LOCK_TMP)
	set -o pipefail; $(CONTAINER) run --rm qa_generator-resolve | sort >> $(LOCK_TMP)
	@mv $(LOCK_TMP) backend/api/requirements.lock
	@echo "backend/api/requirements.lock rewritten; rebuild with: $(COMPOSE) build api"

# ── TLS certificates ───────────────────────────────────────────────────────

# Generates a self-signed CA and an Elasticsearch node certificate into certs/,
# for the Elasticsearch instance Argilla connects to. Idempotent; delete certs/
# to rotate.
#
# Two details are load-bearing:
#
#   elasticsearch.key is mode 644. Elasticsearch runs as uid 1000, but under
#   rootless Podman the host user maps to container uid 0 and the host group
#   does not map at all, so the key arrives as 0:65534. At 600 the node cannot
#   read it and exits with AccessDeniedException. ca.key stays 600; no
#   container reads it.
#
#   The -addext flags on the CA are required. `openssl req -x509` emits no
#   keyUsage extension, and OpenSSL 3 clients reject such a CA even when it is
#   passed explicitly as the trust anchor.
#
# For a real deployment these come from a secret store.
certs:
	@mkdir -p certs
	@[ -f certs/ca.key ] && echo "certs/ already exists, skipping generation" && exit 0; \
	openssl genrsa -out certs/ca.key 4096; \
	openssl req -new -x509 -days 3650 -key certs/ca.key -out certs/ca.crt \
	  -subj "/CN=qa-generator-ca" \
	  -addext "basicConstraints=critical,CA:TRUE" \
	  -addext "keyUsage=critical,keyCertSign,cRLSign"; \
	openssl genrsa -out certs/elasticsearch.key 4096; \
	openssl req -new -key certs/elasticsearch.key -out certs/elasticsearch.csr \
	  -subj "/CN=elasticsearch"; \
	openssl x509 -req -days 3650 \
	  -in certs/elasticsearch.csr \
	  -CA certs/ca.crt -CAkey certs/ca.key -CAcreateserial \
	  -extfile <(printf "%s\n" \
	    "subjectAltName=DNS:elasticsearch,DNS:localhost,IP:127.0.0.1" \
	    "basicConstraints=critical,CA:FALSE" \
	    "keyUsage=critical,digitalSignature,keyEncipherment" \
	    "extendedKeyUsage=serverAuth") \
	  -out certs/elasticsearch.crt; \
	rm certs/elasticsearch.csr certs/ca.srl; \
	echo "Certificates written to certs/"
	@chmod 600 certs/ca.key
	@chmod 644 certs/elasticsearch.key

# ── Pipeline ───────────────────────────────────────────────────────────────

# Runs the Dagster UI and daemon on the host against the containerised
# PostgreSQL and object store. Requires a code location in
# configs/dagster/workspace.yaml, which does not exist yet.
dagster-dev:
	$(LOADENV) && DAGSTER_HOME=$$PWD/configs/dagster dagster dev
