# `certs` uses process substitution, which /bin/sh does not have.
SHELL := /bin/bash

# Override for Docker: make COMPOSE="docker compose" up
COMPOSE ?= podman compose

# The engine itself, for the build and run that `lock` needs; compose has no
# equivalent of --target.
CONTAINER ?= podman

LOCK_TMP := backend/api/requirements.lock.new
AUDIT_TMP := backend/api/requirements.lock.audit

# Every directory ruff reads, so `lint` and `format` cannot cover different
# ones. orchestration, review and evaluation are here and not in pyright's
# `include`: they are checked by both, but pyright is configured in
# pyproject.toml rather than on this command line.
SOURCES := backend frontend telemetry orchestration review evaluation tests

# What every host command reads, and in the same order the containers do: the
# tuning values from configs/env/backend.env, then .env for the credentials,
# the ports and the addresses a host reaches services at.
#
# OVERRIDE re-applies whatever was given on the command line, because sourcing
# the files above would otherwise overwrite it - which is why
# `make topics TOPIC_PASSES=20` used to run with the file's value and say
# nothing.
OVERRIDE = $(if $(MAKEOVERRIDES),&& export $(MAKEOVERRIDES),)

# The provider's credentials, for whichever host command calls a model.
# Optional and absent for a plain Ollama, which needs none - hence the test
# rather than a plain `.`, which would stop every target on a clone that has
# no such file.
#
# Here rather than beside each model-calling target because compose hands
# this file to the three services that call a model and the host was given
# it nowhere: `make extract` against a hosted provider read backend.env and
# .env, found no key in either, and failed to authenticate.
PROVIDER = { [ ! -f ./configs/env/provider.env ] || . ./configs/env/provider.env; }

LOADENV = set -a && . ./configs/env/backend.env && $(PROVIDER) && . ./.env \
          && set +a $(OVERRIDE)

# The same, for a tool with a tuning file of its own:
#
#     $(call WITH,review.env) && python -m review.run --status
#
# Sourced inside the `set -a` region, which is the whole point of having
# this rather than appending `&& . ./configs/env/x.env` to LOADENV: after
# `set +a` a sourced file's values are set in the shell and not exported,
# so the process below saw none of them and stopped naming the first.
WITH = set -a && . ./configs/env/backend.env && . ./configs/env/$(1) && \
       $(PROVIDER) && . ./.env && set +a $(OVERRIDE)

# Narrows a stage target to one item instead of the whole queue, mirroring
# the route's /{scope}/{value} segment:
#
#   make parse-start     SHA=<sha256>     one document
#   make chunk-rerun     SHA=<sha256>     one document
#   make extract-start   SHA=<sha256>     every passage of one document
#   make extract-retry   PASSAGE=<id>     one passage
#   make questions-start TOPIC=<id>       one topic
#
# Which scopes a stage accepts is the stage's own; parsing and chunking take
# a document, extraction takes either, question generation takes a topic, and
# topic modelling takes none. The first one given wins.
ONLY = $(if $(SHA),--only document=$(SHA),\
       $(if $(PASSAGE),--only passage=$(PASSAGE),\
       $(if $(TOPIC),--only topic=$(TOPIC))))

# ── Help ───────────────────────────────────────────────────────────────────

# Listed before anything runs, because `make` on its own used to install
# dependencies and start nine containers.
.DEFAULT_GOAL := help

# Every target there is, grouped, with what each one does.
#
# Read out of this file rather than written out again here: the section
# banner a target sits below is its group, and the first sentence of the
# comment above it is its description. A target cannot be added without
# appearing, and a description cannot drift from the target it names.
#
# Colour only when a terminal is reading, so `make help | grep` is clean.
help:
	@if [ -t 1 ]; then b=$$(printf '\033[1m'); c=$$(printf '\033[36m'); r=$$(printf '\033[0m'); fi; \
	printf '%sQ&A Reference Dataset Generator%s\n\n' "$$b" "$$r"; \
	printf '  usage:      make <target> [VAR=value]\n'; \
	printf '  first run:  %smake setup%s, then %smake doctor%s, then %smake dev%s\n' \
	  "$$c" "$$r" "$$c" "$$r" "$$c" "$$r"; \
	printf '  detail:     docs/make.md, docs/configuration.md\n'; \
	awk -v b="$$b" -v c="$$c" -v r="$$r" ' \
	  function reset() { p = ""; done = 0 } \
	  /^# / && index($$0, "──") { s = $$0; gsub(/[─#]/, "", s); \
	    sub(/^ +/, "", s); sub(/ +$$/, "", s); reset(); next } \
	  /^#/ { \
	    line = $$0; sub(/^# ?/, "", line); \
	    if (line ~ /^[ \t]*$$/) { done = (p != ""); next } \
	    if (!done) p = (p == "" ? line : p " " line); \
	    next \
	  } \
	  /^[a-z][a-zA-Z0-9_-]*:/ { \
	    t = $$1; sub(/:.*/, "", t); \
	    if (p != "") { \
	      if (match(p, /\. /)) p = substr(p, 1, RSTART); \
	      if (length(p) > 66) p = substr(p, 1, 63) "..."; \
	      if (s != shown) { printf "\n%s%s%s\n", b, s, r; shown = s } \
	      printf "  %s%-20s%s %s\n", c, t, r, p \
	    } \
	    reset(); next \
	  } \
	  { reset() } \
	' $(MAKEFILE_LIST)

.PHONY: help setup doctor \
        dev install up down down-volumes logs logs-frontend logs-api \
        logs-shipper logs-retention logs-prune logs-dir logs-orchestration \
        prune spend spend-by-shape \
        schema schema-reset schema-status schema-down schema-stamp migration \
        parse parse-status parse-start parse-stop parse-retry parse-rerun \
        chunk chunk-status chunk-start chunk-stop chunk-retry chunk-rerun \
        chunk-revocabulary \
        extract extract-status extract-start extract-stop extract-retry \
        extract-rerun extract-revalidate extract-bridge extract-recap \
        extract-embed \
        topics topics-status topics-discover topics-stop topics-delete \
        topics-retry topics-visualise \
        settings settings-set settings-unset \
        questions questions-status questions-start questions-stop wipe \
        questions-retry questions-reclaim questions-rerun questions-reverify \
        questions-balance questions-export \
        questions-runs questions-diff \
        documents delete delete-derived \
        archive archive-purge \
        test test-fast test-unit test-integration test-e2e test-smoke \
        test-eval test-coverage test-perf mutation \
        check typecheck audit lint lint-imports deps \
        eval-phrasing second-opinion \
        format lock \
        certs dagster-dev \
        review-status review-push-facts review-pull-facts \
        review-push-topics review-pull-topics \
        review-push-questions review-pull-questions \
        eval-upload eval-score prompts-publish \
        all corpus open services review pull auto manual pipeline runs

# ── Bootstrap ──────────────────────────────────────────────────────────────

# The assignments in .env, which is where a placeholder counts. The file's
# own first line says to replace every change_me_* value and is not one.
ASSIGNED = grep -E '^[A-Z][A-Z0-9_]*=' .env

# Write the two configuration files a clone does not come with, and fill in
# every password. Safe to run twice: it replaces the placeholders and
# nothing else, so a file already edited by hand keeps what it says.
#
# One secret per distinct placeholder, replaced everywhere it appears -
# which is what keeps APP_DB_PASSWORD and the password inside DATABASE_URL
# the same string without this needing to know that they are related.
#
# Longest placeholder first, because `change_me_phoenix` is a prefix of
# `change_me_phoenix_ui_password` and replacing the short one first would
# leave the long one half rewritten.
#
# Hex is lower-case and a digit is appended, which is what
# PHOENIX_ADMIN_SECRET requires of the deployment: 32 characters or more,
# with a digit and a lower-case letter.
setup:
	@test -f .env || { cp .env.example .env; echo "wrote .env"; }
	@test -f configs/env/provider.env || { \
	  cp configs/env/provider.env.example configs/env/provider.env; \
	  echo "wrote configs/env/provider.env"; }
	@for token in $$($(ASSIGNED) | grep -oE 'change_me_[a-z0-9_]*' | sort -u \
	                 | awk '{ print length, $$0 }' | sort -rn | cut -d' ' -f2-); do \
	  secret="$$(openssl rand -hex 24)1"; \
	  sed -i.bak "s|$$token|$$secret|g" .env && rm -f .env.bak; \
	  echo "generated $$token"; \
	done
	@echo
	@echo "Now name the model in .env (LLM_MODEL, LLM_BASE_URL) and put"
	@echo "whatever credentials it needs in configs/env/provider.env."
	@echo "Then: make doctor"

# What a first run gets wrong, before a corpus pays for it.
#
# Every check runs and the failures are counted, rather than stopping at the
# first: somebody setting this up wants the whole list, not one item of it
# six times.
#
# The last check is a real call to the configured model, which is the one
# thing no amount of reading the files can tell you. An unknown model id, a
# wrong address and a rejected credential all fail at the first call, and
# without this that call is the first passage of a real run.
doctor:
	@fail=0; \
	say() { printf '  %-8s %s\n' "$$1" "$$2"; }; \
	command -v $(CONTAINER) >/dev/null && say ok "$(CONTAINER)" \
	  || { say MISSING "$(CONTAINER) - install it, or set CONTAINER= and COMPOSE="; fail=1; }; \
	command -v poetry >/dev/null && say ok "poetry" \
	  || { say MISSING "poetry - https://python-poetry.org/docs/#installation"; fail=1; }; \
	if test -f configs/env/provider.env; then say ok "configs/env/provider.env"; \
	  else say "-" "no configs/env/provider.env, which a plain Ollama does not need"; fi; \
	if ! test -f .env; then say MISSING ".env - run: make setup"; \
	  echo; echo "Fix the above, then run make doctor again."; exit 1; fi; \
	say ok ".env"; \
	if $(ASSIGNED) | grep -q change_me; then \
	  say FAIL "$$($(ASSIGNED) | grep -c change_me) placeholder password(s) left - run: make setup"; fail=1; \
	  else say ok "no placeholder passwords"; fi; \
	$(LOADENV); \
	if printf '%s' "$$PHOENIX_ADMIN_SECRET" | grep -qE '^.{32,}$$' \
	   && printf '%s' "$$PHOENIX_ADMIN_SECRET" | grep -q '[0-9]' \
	   && printf '%s' "$$PHOENIX_ADMIN_SECRET" | grep -q '[a-z]'; then \
	  say ok "PHOENIX_ADMIN_SECRET"; \
	  else say FAIL "PHOENIX_ADMIN_SECRET needs 32+ characters, a digit and a lower-case letter"; fail=1; fi; \
	if test -z "$$OLLAMA_CONTAINER_URL"; then \
	  say FAIL "OLLAMA_CONTAINER_URL is unset, and it is the only address a container may call"; fail=1; \
	elif test -z "$$LLM_CONTAINER_MODEL"; then \
	  say FAIL "LLM_CONTAINER_MODEL is unset: containers run Ollama and never inherit the host's model"; fail=1; \
	elif case "$$LLM_CONTAINER_MODEL" in ollama*) false;; *) true;; esac; then \
	  say FAIL "LLM_CONTAINER_MODEL is $$LLM_CONTAINER_MODEL; containers reach Ollama only"; fail=1; \
	elif test -n "$$QUESTIONS_VERIFIER_CONTAINER_MODEL" && \
	     case "$$QUESTIONS_VERIFIER_CONTAINER_MODEL" in ollama*) false;; *) true;; esac; then \
	  say FAIL "QUESTIONS_VERIFIER_CONTAINER_MODEL is $$QUESTIONS_VERIFIER_CONTAINER_MODEL; containers reach Ollama only"; fail=1; \
	else \
	  say ok "containers call $$LLM_CONTAINER_MODEL at $$OLLAMA_CONTAINER_URL"; \
	fi; \
	echo; echo "Asking $$LLM_MODEL one question, as the host calls it..."; \
	PYTHONPATH=backend poetry run python -m llm.check || fail=1; \
	echo; \
	if test $$fail -eq 0; then echo "Ready. Next: make dev"; \
	  else echo "Fix the above, then run make doctor again."; exit 1; fi

# Install dependencies, start every service, and create the schema.
dev: install certs logs-dir
	$(COMPOSE) up -d
	@echo "Waiting for PostgreSQL..."
	@$(LOADENV) && \
	  until $(COMPOSE) exec -T postgres pg_isready -U "$$POSTGRES_SUPERUSER" -q; \
	  do sleep 2; done
	$(MAKE) schema

# The spaCy pipelines are downloaded, not resolved: they are not on PyPI
# under a version range. The same names go into the image; see
# SPACY_MODELS in backend/api/Dockerfile.
#
# Retried, because they come from GitHub's release downloads rather than
# from an index: a 504 there fails the whole install, and did.
SPACY_MODELS := de_core_news_md en_core_web_md

# Install the Python dependencies and the spaCy pipelines.
install:
	poetry install --with llm,nlp,data,storage,api,pipeline,viz,observability,dev
	@for model in $(SPACY_MODELS); do \
	  attempt=1; \
	  until poetry run python -m spacy download $$model; do \
	    attempt=$$((attempt + 1)); \
	    if [ $$attempt -gt 3 ]; then \
	      echo "could not download $$model after 3 attempts" >&2; exit 1; \
	    fi; \
	    echo "spacy download $$model failed; retrying in 15s"; sleep 15; \
	  done; \
	done

# ── Services ───────────────────────────────────────────────────────────────

# The directory a host command writes its JSON log lines to, which compose
# bind-mounts into the shipper read-only. Made here rather than left to the
# container engine: a missing bind-mount source is created by the engine
# itself, which can leave it owned by root, and then the host process the
# directory exists for cannot write to it.
logs-dir:
	@mkdir -p logs

# Start every service, generating the certificates first if they are missing.
up: certs logs-dir
	$(COMPOSE) up -d

# Stop every service. The volumes are kept.
down:
	$(COMPOSE) down
	@$(MAKE) --no-print-directory prune

# Reclaim the images a rebuild orphaned. Neither podman nor docker collects
# them: an untagged layer may still be the cache the next build reuses, so
# the engine keeps it until asked. A tag rebuilt six times leaves five
# untagged copies of itself, and this project's backend image is 3 GB - one
# run of this recovered 30 GB, after a build had already failed with `no
# space left on device` inside the podman VM.
#
# Dangling only. `-a` would take images no container is running right now,
# which on a stopped stack is all of them.
prune:
	-$(CONTAINER) image prune -f

# Deletes every volume. Irreversible.
down-volumes:
	@echo "WARNING: this deletes all persistent data. Ctrl-C within 5s to abort."
	@sleep 5
	$(COMPOSE) down -v

# The container logs, as the engine holds them. Every service, this project's
# and the infrastructure alike. What the api, the four workers and the
# frontend log also goes to Elasticsearch as JSON; Grafana is where you read
# it back across services, filtered and over time.
logs:
	$(COMPOSE) logs -f

# The Streamlit application's log.
logs-frontend:
	$(COMPOSE) logs -f streamlit

# Upload failures surface here; the frontend only sees the status code.
logs-api:
	$(COMPOSE) logs -f api

# The shipper's own log. Where to look when Grafana shows nothing: a
# connection refused, a certificate it will not trust, or a rejected mapping
# is reported here and nowhere else.
logs-shipper:
	$(COMPOSE) logs -f filebeat

# The webserver and the daemon together. A code location that will not load
# is reported by the webserver, and a schedule or sensor that fired and
# failed by the daemon.
logs-orchestration:
	$(COMPOSE) logs -f dagster-webserver dagster-daemon

# How long the logs are kept. Run once against a running stack, after the
# shipper has written something: the retention belongs to the data stream,
# Elasticsearch remembers it, and every backing index the stream rolls over
# to afterwards inherits it. Nothing deletes anything until this has run,
# which is the state the stack ships in.
#
# The data stream's own lifecycle, not an ILM policy. A shipper cannot
# create an ILM policy, and naming one in filebeat.yml does not work either:
# Filebeat strips `index.lifecycle` out of the template it installs when
# `setup.ilm.enabled` is false, so the setting arrived nowhere and the
# indices were aged by nothing.
#
# The wildcard covers the dated streams an earlier version of filebeat.yml
# created alongside today's single one.
#
# Idempotent - PUT replaces. LOGS_RETENTION_DAYS overrides the default:
#
#     make logs-retention LOGS_RETENTION_DAYS=90
LOGS_RETENTION_DAYS ?= 30
# Age the shipped logs out after LOGS_RETENTION_DAYS. Until this has run,
# nothing is deleted.
#
# Run through `sh -c`, with the credential read from the container's own
# environment rather than passed in, so it stays out of the process list
# and out of make's echo of the command.
#
# No `--fail-with-body`, which is how this would normally report a refusal:
# the curl in the Elasticsearch image predates it and exits with a usage
# error instead. So the answer is read directly, which is also the only way
# to see which streams it matched.
logs-retention:
	@$(COMPOSE) exec -T elasticsearch sh -c 'answer=$$(curl -sS \
	  --cacert /usr/share/elasticsearch/config/certs/ca.crt \
	  -u "elastic:$$ELASTIC_PASSWORD" \
	  -X PUT "https://localhost:9200/_data_stream/qa-logs*/_lifecycle" \
	  -H "Content-Type: application/json" \
	  -d "{\"data_retention\":\"$(LOGS_RETENTION_DAYS)d\"}"); \
	  echo "$$answer"; \
	  case "$$answer" in *\"acknowledged\":true*) ;; *) exit 1;; esac'
	@echo "qa-logs is kept for $(LOGS_RETENTION_DAYS) days."

# The other half of the retention, and the one `logs-retention` cannot do.
# That target ages what Elasticsearch holds; this ages the FILES the shipper
# read them out of, and nothing else does.
#
# Each process writes {service}-{host}-{pid}.log, because the WRITER is what
# the name has to be unique per: two processes rotating one file take each
# other's lines with them. The cost is a file per container and a file per
# host run, and nothing else deletes one. `RotatingFileHandler` only rotates
# the file its own live process owns, filebeat mounts both directories
# read-only on purpose, and filebeat's `ignore_older` and `clean_inactive`
# age its REGISTRY and not the disk. The volume had reached 78 files and
# 110 MB before this target existed.
#
# Both places a line is written: the volume, through the api, which is the
# one container that mounts it writable, and ./logs on the host, which make
# can delete from itself. Deletes by mtime, so a file a live process is
# still appending to is never old enough to take.
#
#     make logs-prune LOGS_KEEP_DAYS=7
LOGS_KEEP_DAYS ?= 30
# Delete the log files no process has written to for LOGS_KEEP_DAYS.
logs-prune:
	@{ $(COMPOSE) exec -T api sh -c \
	     'find /var/log/qa -type f -mtime +$(LOGS_KEEP_DAYS) -print -delete'; \
	   find logs -type f -mtime +$(LOGS_KEEP_DAYS) -print -delete 2>/dev/null; \
	 } | wc -l | xargs printf '%s file(s) deleted.\n'

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

# Read every stored passage's language and vocabulary again, in place. Only
# those two change; passages, sentence offsets and facts all stay, which is
# what chunk-rerun cannot promise - it deletes the passages and the facts go
# with them. Fit the topics afterwards to see the difference.
chunk-revocabulary:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m preprocessing.chunking.run --revocabulary $(ONLY)

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

# Judge every stored fact again with today's checks. The model is not called
# and no statement changes: only what the checks read off one. This is what
# applies a change to the checks without re-extracting the corpus.
extract-revalidate:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m extraction.run --revalidate $(ONLY)

# Read every topic's passage groups for the claims that need more than one of
# them. A pass of its own rather than part of the queue: the unit is a group
# of passages the topic model put together, so topics must be fitted first.
# Re-running replaces the bridges it wrote last time instead of adding to
# them.
extract-bridge:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m extraction.run --bridge $(ONLY)

# Refuse the atomic facts a passage yielded above the cap
# EXTRACTION_MIN_OTHER_SHARE works out to, keeping the ones that assert a
# number, a date or a name. No model is called, so a corpus extracted before
# the cap existed is re-balanced in seconds rather than re-read over hours.
# Only the verdict moves: nothing is deleted, and the facts refused carry
# `over_cap` like any other refusal.
extract-recap:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m extraction.run --recap $(ONLY)

# Write the vectors onto passages and facts already stored. No model is
# called - a vector is read off a statement that is already written - so a
# corpus extracted before the embedding columns existed is filled in at the
# speed of the embedding model rather than re-read over hours.
#
# It does not apply the dedup gate. A fact stored before the gate existed was
# accepted, and refusing it now would rewrite a verdict the corpus was
# measured under; re-extract to have it judged.
extract-embed:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m extraction.run --embed $(ONLY)

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
# A label a person assigned is stored nowhere else, and the archived row is
# now the only copy of one.
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

# ── Question generation ────────────────────────────────────────────────────
#
# The same five verbs as every other stage, over topics. A question's subject
# is a topic, so that is what the queue is over, and TOPIC=<id> narrows any
# of them to one:
#
#   make questions-status                 GET /questions/status
#   make questions-start                 POST /questions/start
#   make questions-start TOPIC=7         POST /questions/topic/7/start
#
# Needs the model in LLM_MODEL served at LLM_BASE_URL, the second model in
# QUESTIONS_VERIFIER_MODEL served beside it, and EMBEDDING_MODEL downloaded -
# the first run fetches it into the Hugging Face cache and later runs read it
# from there.
#
# What is written is QUESTIONS_TYPE_MIX and QUESTIONS_DIFFICULTY_MIX, both in
# configs/env/backend.env, and both proportions rather than counts. A kind
# with a weight of 0 is never written. The Questions page puts the mix that
# was asked for beside the mix that came out.
#
# Both models are served locally here, so LLM_NUM_CTX and
# LLM_REASONING_EFFORT matter: a thinking model asked for a structured answer
# spends its window reasoning and returns an empty string.

# Drain the question queue here, in the foreground.
questions:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m question_generation.run

# Report the queue by topic state, and how many questions are held.
questions-status:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m question_generation.run --status $(ONLY)

# Queue every topic this stage has not been asked to do yet. Nothing reaches
# a worker until this runs.
questions-start:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m question_generation.run --start $(ONLY)

# Take back whatever has not begun. The one in hand finishes.
questions-stop:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m question_generation.run --stop $(ONLY)

# Return every failed topic to the pending queue.
questions-retry:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m question_generation.run --retry $(ONLY)

# Return a topic a dead worker still holds, without waiting out its lease.
#
# The gap between `retry`, which takes the failed, and `rerun`, which skips
# what a worker holds. An interrupted topic is `in_progress` and neither of
# those touches it, so until this existed the only thing that moved one was
# the lease - derived from what a topic costs, and 67 days at 120 questions
# a topic and a 900-second timeout.
#
#   make questions-reclaim TOPIC=473    one topic
#   make questions-reclaim              every one the stage holds
#
# NARROW IT while a worker may be up. Nothing here can tell a dead claim
# from a live one, and handing a live worker's row to a second worker is
# what the lease exists to prevent. Over the whole stage only when it is
# stopped.
questions-reclaim:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m question_generation.run --reclaim $(ONLY)

# Run this stage again over every topic, finished ones included. Cheaper than
# it looks: facts an accepted question already rests on are skipped, so this
# writes about what the last run did not reach rather than starting over.
questions-rerun:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m question_generation.run --rerun $(ONLY)

# Put every stored question through the gates that need no model: its facts
# still pass their own checks, its evidence is still spread the way its
# difficulty says, it is still well formed, and no earlier question already
# asks it. Rejects what no longer holds and never un-rejects, because
# accepting is a person's. This is what carries extract-revalidate, or a
# re-extraction of one document, through to the questions resting on it.
questions-reverify:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m question_generation.run --reverify $(ONLY)

# Draw the balanced release out of every accepted question, and report what
# it came out as. Accepting a question says it is sound; this says what the
# SET looks like, and the two are different problems - every gate can do its
# job and still leave a set that is 45% unanswerable and 96.5% easy, because
# what survives a filter is whatever the material happened to offer.
#
# Holds three shares at once: how much of the release has no answer, how it
# spreads over the difficulty bands, and how it spreads over the kinds. A
# column on the rows already there, so nothing is deleted, the questions left
# out stay queryable, and running it again replaces the draw. Reports the
# yield - how much of the accepted pool made it in - and names any quota the
# pool could not fill.
questions-balance:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m question_generation.run --balance $(ONLY)

# The questions as a spreadsheet: one sheet of questions, one of the facts
# each cites, one of counts. This is the way the dataset leaves the database
# - everything else that reads these rows is a service, and Argilla is NOT
# this: it holds a disposable copy of a stratified SAMPLE pushed for review,
# so exporting from there gives back the hundred rows somebody was asked to
# look at rather than the set.
#
# OUT names the file. FILTER is passed through to the exporter, and takes
# the same narrowing the Questions page and the API do - so what lands in
# the workbook is what the filter says, with no default scope quietly
# applied. Ask for nothing and you get everything, rejected rows included,
# which is what the same request to /questions returns.
#
#   make questions-export
#   make questions-export OUT=exam.xlsx FILTER="--status accepted --cognitive-level analyse"
#   make questions-export FILTER="--status accepted --unanswerable"
#   make questions-export FILTER="--no-citations"      # much faster, questions only
#
#   python -m question_generation.export --help        # every filter there is
OUT ?= questions.xlsx
questions-export:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m question_generation.export --out $(OUT) $(FILTER)

# Which runs there are, newest first, and how many questions each wrote.
# A run is named by RUN_ID where one was given and by a uuid otherwise; see
# backend/settings/runs.py. A run marked (deleted) has had its questions
# removed and is being read out of archived_rows.
questions-runs:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m question_generation.runs --list

# Two runs side by side, on the gate that stopped each question. Ordered by
# how much the two disagree, because that is the question being asked.
#
#   make questions-diff RUNS="a-gpt-4.1 b-gemma4-12b"
#
# Reads the live questions AND the archived ones, which is what makes an A/B
# possible at all: `questions-rerun` deletes what it replaces, so before the
# archive existed, producing run B destroyed run A. Nothing here gates - a
# model's answers move between two runs at the same temperature.
questions-diff:
	@test -n "$(RUNS)" || { echo 'usage: make questions-diff RUNS="<older> <newer>"'; exit 2; }
	$(LOADENV) && PYTHONPATH=backend poetry run python -m question_generation.runs $(RUNS)

# What the models have cost, read off a CAPTURED log - the text one a drain
# wrote to a terminal, as `make questions ... | tee run.log` leaves behind.
# Every call logs its tokens and its price; litellm prices the response
# rather than this counting it, because the provider is the only thing that
# knows what it billed for. A self-hosted model has no published price and
# logs no cost at all, which is the truth about it rather than a zero.
#
# LOG is required, and that is the fix for what these used to do. They
# defaulted to /var/log/qa/*.log, which is a path inside the containers and
# not on the host they run on - and holds JSON, which the pattern below
# cannot match anyway. Both printed "no priced calls in the log", which
# reads as "the pipeline cost nothing" rather than "I read nothing".
#
# For the LIVE numbers, neither of these: Grafana's Pipeline throughput
# dashboard sums llm.cost_usd off Elasticsearch, and Phoenix has the same
# figures per span, per run. What these two have that neither does is a log
# from a run that is over, on a machine with no stack up - which is what the
# measurements in evaluation/README.md were taken from.
#
#   make spend LOG=run.log
#   make spend LOG=run.log SINCE=2026-09-18
# The same log, split by which model answered and what it was asked for.
#
# The shape is the Pydantic class the call had to return, and every stage
# has its own: _Facts and _Digest read a passage, _Answered writes a
# question, _Recovered gets its answer back out, _NamesItsSource and
# _SelfContained judge the wording. So this is what each JUDGEMENT costs,
# which `make spend` cannot say and which is the number to look at after
# moving one of them to another model.
#
#   make spend-by-shape LOG=run.log
#   make spend-by-shape LOG=run.log SINCE=2026-09-21
#
# Ordered by call count, because the shape at the top is the one worth
# moving somewhere cheaper.
spend-by-shape:
	@test -n "$(LOG)" || { echo 'usage: make spend-by-shape LOG=<a captured log> [SINCE=<date>]'; exit 2; }
	@printf '%-24s %-18s %8s %12s %12s %10s\n' \
	   MODEL SHAPE CALLS 'TOKENS IN' 'TOKENS OUT' COST
	@cat $(LOG) \
	 | $(if $(SINCE),grep "$(SINCE)",cat) \
	 | grep -hoE '[^ ]+ answered _[A-Za-z]+ in [0-9.]+s \([0-9,]+ in, [0-9,]+ out, \$$[0-9.]+\)' \
	 | sed -E 's/^([^ ]+) answered (_[A-Za-z]+) in [0-9.]+s \(([0-9,]+) in, ([0-9,]+) out, \$$([0-9.]+)\)/\1 \2 \3 \4 \5/' \
	 | tr -d ',' \
	 | awk '{k=$$1" "$$2; n[k]++; i[k]+=$$3; o[k]+=$$4; c[k]+=$$5} \
	     END {for (k in n) {split(k,p," "); \
	       printf "%-24s %-18s %8d %12d %12d %9.2f\n", p[1],p[2],n[k],i[k],o[k],c[k]}}' \
	 | sort -k3 -rn
	@cat $(LOG) \
	 | $(if $(SINCE),grep "$(SINCE)",cat) \
	 | grep -hoE 'answered _[A-Za-z]+ in [0-9.]+s \([0-9,]+ in, [0-9,]+ out, \$$[0-9.]+\)' \
	 | sed -E 's/.*\(([0-9,]+) in, ([0-9,]+) out, \$$([0-9.]+)\)/\1 \2 \3/' \
	 | tr -d ',' \
	 | awk '{i+=$$1; o+=$$2; c+=$$3; n++} END {if (n==0) {print "\nno priced calls in the log"; exit} \
	     printf "%-24s %-18s %8d %12d %12d %9.2f\n", "", "TOTAL", n, i, o, c}'

# What the models have cost in total, over the same log.
spend:
	@test -n "$(LOG)" || { echo 'usage: make spend LOG=<a captured log> [SINCE=<date>]'; exit 2; }
	@grep -hoE '^[0-9T:-]+ .*answered _[A-Za-z]+ in [0-9.]+s \([0-9,]+ in, [0-9,]+ out, \$$[0-9.]+\)' \
	   $(LOG) \
	 | $(if $(SINCE),grep "^$(SINCE)",cat) \
	 | sed -E 's/.*\(([0-9,]+) in, ([0-9,]+) out, \$$([0-9.]+)\)/\1 \2 \3/' \
	 | tr -d ',' \
	 | awk '{i+=$$1; o+=$$2; c+=$$3; n++} END {if (n==0) {print "no priced calls in the log"; exit} \
	     printf "calls   %d\ntokens  %d in, %d out\ncost    $$%.2f  (mean $$%.5f a call)\n", n, i, o, c, c/n}'

# ── Settings ───────────────────────────────────────────────────────────────
#
# What each service is configured to do, and changing it. The same operation
# as the route beside it, and as the Configuration panel on that service's
# page:
#
#   make settings SERVICE=topics                GET  /settings/topics
#   make settings-set SERVICE=topics SET=...   PATCH /settings/topics
#   make settings-unset SERVICE=topics UNSET=…  PATCH, with a null
#
# A value written here reaches a worker when it next claims a row. Nothing is
# requeued: the command names what it staled, and that stage's -rerun is what
# rebuilds it.
#
# The environment still supplies every setting; these write an override over
# it, and -unset deletes that override.

# Report one service's settings, beside what the files say.
#   make settings SERVICE=extraction
settings:
	@[ -n "$(SERVICE)" ] || { echo "usage: make settings SERVICE=<name>"; exit 2; }
	$(LOADENV) && PYTHONPATH=backend poetry run python -m settings.run --service $(SERVICE)

# Change one or more of them. Repeat SET for each, space separated.
#   make settings-set SERVICE=topics SET="TOPIC_PASSES=20"
#   make settings-set SERVICE=platform SET="LLM_MODEL=ollama_chat/qwen3:14b"
settings-set:
	@[ -n "$(SERVICE)" ] && [ -n "$(SET)" ] || { echo 'usage: make settings-set SERVICE=<name> SET="NAME=VALUE"'; exit 2; }
	$(LOADENV) && PYTHONPATH=backend poetry run python -m settings.run --service $(SERVICE) $(foreach one,$(SET),--set $(one))

# Return one or more to what the files say.
#   make settings-unset SERVICE=topics UNSET="TOPIC_PASSES"
settings-unset:
	@[ -n "$(SERVICE)" ] && [ -n "$(UNSET)" ] || { echo 'usage: make settings-unset SERVICE=<name> UNSET="NAME"'; exit 2; }
	$(LOADENV) && PYTHONPATH=backend poetry run python -m settings.run --service $(SERVICE) $(foreach one,$(UNSET),--unset $(one))

# ── Documents ──────────────────────────────────────────────────────────────

# List every stored document with its digest and parse state.
documents:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m ingestion.run --list

# Delete a document, its file, its converted form and everything derived from
# it. The rows are archived and the objects are moved to the `archive` bucket,
# so `make archive-purge` is what makes it final. The upload record in
# ingest_events is kept.
#   make delete SHA=<sha256>
delete:
	@[ -n "$(SHA)" ] || { echo "usage: make delete SHA=<sha256>"; exit 2; }
	$(LOADENV) && PYTHONPATH=backend poetry run python -m ingestion.run --delete $(SHA)

# Empty the corpus: every document, every file, every converted form, and
# with them by cascade every passage, membership, fact and question. Then the
# topics, which are none of those. All of it is archived on the way out; see
# `make archive`.
#
# Two commands and not one because `topics` has no foreign key to a document:
# a fit is over the corpus rather than over a file, so deleting every document
# leaves the topics behind, holding their labels, their coverage flags and
# their question-generation queue state, all describing passages that are
# gone. Deleting the topics takes their figures out of the export bucket too.
#
# The upload history goes with the documents here, though a single deletion
# keeps it: an empty corpus reporting nine upload attempts is a status panel
# describing documents nothing has.
wipe:
	@echo "WARNING: this deletes every document, topic, fact and question. Ctrl-C within 5s to abort."
	@sleep 5
	$(LOADENV) && PYTHONPATH=backend poetry run python -m ingestion.run --delete-all
	$(LOADENV) && PYTHONPATH=backend poetry run python -m topic_modelling.run --delete

# Delete only what the pipeline built: passages and facts. The document and
# its file stay, and chunking goes back to `new`, so the next run rebuilds
# them once somebody starts it.
#   make delete-derived SHA=<sha256>
delete-derived:
	@[ -n "$(SHA)" ] || { echo "usage: make delete-derived SHA=<sha256>"; exit 2; }
	$(LOADENV) && PYTHONPATH=backend poetry run python -m ingestion.run --delete-derived $(SHA)

# ── Archive ────────────────────────────────────────────────────────────────
#
# Every deletion above is the first of two. An AFTER DELETE trigger on every
# table copies the row into `archived_rows` as it goes - which catches the
# cascades and the orphan triggers, where most of the deleting actually
# happens - and the removal paths move the objects into the `archive` bucket
# instead of dropping them. Nothing reads any of it; it is there so the
# second deletion is a separate decision.

# What is held, by table, with its age and what it costs.
archive:
	$(LOADENV) && PYTHONPATH=backend poetry run python -m archive.run --status

# The second deletion. This one is final.
#
#   make archive-purge TABLE=questions    one table's rows
#   make archive-purge DAYS=30            everything archived before then
#   make archive-purge ALL=1              the whole thing, objects included
#
# TABLE is about rows, so it leaves the bucket alone: an archived upload
# belongs to no table. DAYS and ALL take the objects too.
archive-purge:
	@[ -n "$(TABLE)$(DAYS)$(ALL)" ] || { echo "usage: make archive-purge [TABLE=<table>] [DAYS=<n>] [ALL=1]"; exit 2; }
	@[ -z "$(ALL)" ] || { echo "WARNING: this deletes the whole archive, rows and objects. Ctrl-C within 5s to abort."; sleep 5; }
	$(LOADENV) && PYTHONPATH=backend poetry run python -m archive.run --purge \
	   $(if $(TABLE),--table $(TABLE)) $(if $(DAYS),--older-than $(DAYS)) $(if $(ALL),--all)

# ── Tests ──────────────────────────────────────────────────────────────────

# Everything that gates: the unit layers, the static gates, and the
# integration layers, which start a Postgres and a SeaweedFS of their own
# and are skipped where no container engine answers. Configuration lives in
# [tool.pytest.ini_options] in pyproject.toml.
#
# The order is shuffled every run, and the seed is printed at the top. A
# failure that only happens in one order is a test depending on another,
# and `-p randomly --randomly-seed=<n>` reproduces it.
test:
	poetry run pytest -m "not smoke and not eval and not perf"

# Without the slow ones: no spaCy pipelines, no pyright, no containers.
# Over every core, which the unit layer can be: it shares no container.
test-fast:
	poetry run pytest -n auto -m "not nlp and not types and not integration and not perf"

# The layers on their own, for working on one of them.
test-unit:
	poetry run pytest -n auto -m "not integration and not smoke and not eval and not perf"

# Everything needing a Postgres and a SeaweedFS, short of end to end.
test-integration:
	poetry run pytest -m "integration and not e2e"

# One corpus, every stage, against a stack of its own.
test-e2e:
	poetry run pytest -m e2e

# Builds both images and reads the compose file. Minutes, not seconds, and
# it starts no stack: compose.yaml binds its ports from .env, so a second
# copy would collide with a running one rather than run beside it.
test-smoke:
	poetry run pytest -m smoke

# Scores the served model LLM_MODEL names against the golden passages.
# Never a gate: a model's answers move between versions and between runs.
test-eval:
	$(LOADENV) && poetry run pytest -m eval -s

# Branch coverage over every package a deployment runs, both layers
# combined. Combined because neither covers the other's half: the
# repositories are the integration layer's and the gates are the unit
# layer's, so a threshold against either alone is against the wrong number.
#
# What it is measured over, what is left out and the floor it has to clear
# are [tool.coverage.*] in pyproject.toml.
# --cov-fail-under=0 on each half: the floor is against the combination,
# and enforced per run it fails the first one for not being the second.
test-coverage:
	poetry run coverage erase
	poetry run pytest -m "not integration and not smoke and not eval and not perf" \
	  --cov --cov-append --cov-report= --cov-fail-under=0
	poetry run pytest -m integration --cov --cov-append --cov-report= --cov-fail-under=0
	poetry run coverage report

# Times the ceilings the code names in a comment: the combination search in
# `adds_up`, the release draw, the composition report. Never a gate - a
# budget on a shared runner measures the runner.
test-perf:
	poetry run pytest -m perf

# Whether the tests would notice a gate changing. Line coverage says the
# line ran; this changes it and asks whether anything fails.
#
# Which modules and which layers are [tool.mutmut] in pyproject.toml.
# Hours, so nightly rather than per change; `mutmut browse` reads the
# survivors afterwards.
mutation:
	poetry run mutmut run
	poetry run mutmut results

# Everything that gates. Another name for `test`.
check: test

# ── Static checks ──────────────────────────────────────────────────────────

# Every type error, not just the ones above the recorded baseline. What the
# suite enforces is tests/static/pyright_baseline.json; this is the list to
# work from when bringing a file's count down. Configuration lives in
# [tool.pyright] in pyproject.toml.
typecheck:
	poetry run pyright

# Known advisories against what the images install. Reported, not gated:
# an advisory with no fixed version yet is not a reason to stop merging.
#
# --no-deps because both locks already pin every transitive dependency.
#
# torch and torchvision are held out: they are pinned to +cpu builds, which
# is a local version PyPI does not carry and PyTorch's own index publishes
# only for linux. Resolving them anywhere else fails, so their advisories
# are not covered here - `pip-audit -r` inside the built image is what
# would cover them.
audit:
	@grep -viE "^(torch|torchvision)==" backend/api/requirements.lock > $(AUDIT_TMP)
	poetry run pip-audit --no-deps --progress-spinner off -r $(AUDIT_TMP) || true
	@rm -f $(AUDIT_TMP)
	poetry run pip-audit --no-deps --progress-spinner off \
	  -r frontend/requirements.lock || true

# Configuration lives in [tool.ruff] in pyproject.toml.
lint:
	poetry run ruff check $(SOURCES)
	poetry run ruff format --check $(SOURCES)

# The two architecture rules the root README states. Configuration, and the
# reasoning, live in .importlinter.
#
# PYTHONPATH=backend because the image puts each backend package at the top
# level: `extraction` is the name a container resolves, not
# `backend.extraction`, and contracts written against the other name would
# pass by matching nothing. tests/static/test_import_contracts.py runs this
# same check under `make test`.
lint-imports:
	PYTHONPATH=backend:. poetry run lint-imports --verbose

# A declared dependency nothing imports, an import nothing declares, and a
# package reached only through somebody else's. Configuration is
# [tool.deptry] in pyproject.toml, which is also where each deliberate
# exception is argued for.
deps:
	poetry run deptry .

# Fix what ruff can fix, then format every source directory.
format:
	poetry run ruff check --fix $(SOURCES)
	poetry run ruff format $(SOURCES)

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

# ── Golden-set experiments ─────────────────────────────────────────────────
#
# `make test-eval` scores a served model against the cases in
# evaluation/cases.py and prints the numbers. These record the same scores
# in Phoenix instead, so two runs are comparable by more than scrollback:
#
#   make eval-upload    put the cases in Phoenix, as a new dataset version
#   make eval-score     run the model against them and record the scores
#
# Never a gate, for the reason test-eval is not: a model's answers move
# between versions, between quantisations and between two runs at the same
# temperature.
#
# Name a run when the model is not what changed:
#
#     make eval-score EVAL_RUN_NAME=extraction-prompt-v7
EVAL = $(call WITH,evaluation.env) && \
       PYTHONPATH=backend poetry run python -m evaluation.run
EVAL_DATASET ?= extraction-golden

# Put the golden cases in Phoenix, as a new dataset version.
eval-upload:
	$(EVAL) --upload $(EVAL_DATASET)

# Run the served model against them and record the scores in Phoenix.
eval-score:
	$(EVAL) --score $(EVAL_DATASET)

# Send the recorded prompts to Phoenix, so a span opens against one.
#
# The span already carries the prompt it sent, filled in with that call's
# passages. What it cannot show is the TEMPLATE, and
# `llm.prompt_template.version` on it names a version Phoenix knows nothing
# about until this has run. One Phoenix prompt per prompt, one Phoenix
# version per PROMPT_VERSION, so two are diffable there.
#
# Read from the `prompts` table rather than from the code, so a version the
# source has moved past goes too - which is the reason that table exists.
# The stages write it when they start, so run one of them first.
#
# The source is still the code. An edit in Phoenix's UI reaches nothing and
# the next run of this writes over it.
prompts-publish:
	$(EVAL) --publish-prompts

# The two phrasing judgements, each against the floor a judge that ignores
# its input reaches. That floor is the point: `self_contained` scored 84.2%
# against a constant-answer 78.9% - one case in nineteen - which is the
# measured version of a judgement that fired zero times in 3,131 questions.
#
#     make eval-phrasing                       # upload, then score
#     make eval-phrasing EVAL_RUN_NAME=gemma4-12b
#
# Scores QUESTIONS_PHRASING_MODEL, which is the model the pipeline asks, so
# swapping a 31B for a 12B and running this again is what decides it.
eval-phrasing:
	$(EVAL) --upload phrasing-golden
	$(EVAL) --score phrasing-golden

# An independent judge over one run's ACCEPTED answers, and the
# disagreements it found. Never a verdict - the checker decides what is
# kept, and this is measured at chance on German in evaluation/README.md.
# What it produces is a queue: a question the gates kept and a judge calls
# unsupported is either a gate that let something through or a judge that
# is wrong, and only a person settles which.
#
#     make second-opinion RUN=<run id>          # 200 answers
#     make second-opinion RUN=<run id> LIMIT=50
#
# It prints the ids. Push them with:
#     make review-push-questions IDS=12,34,56
second-opinion:
	@test -n "$(RUN)" || { echo 'usage: make second-opinion RUN=<run id>  (make questions-runs lists them)'; exit 2; }
	$(EVAL) --second-opinion $(RUN) $(if $(LIMIT),--limit $(LIMIT))

# ── Review ─────────────────────────────────────────────────────────────────
#
# Human review of what the models decided, through Argilla. Three things
# have somewhere for a person to disagree, and each already had a
# human-owned column before this: a fact's verdict, a topic's name and
# whether it is a subject, and whether a question is any good.
#
#   make review-push-facts      a sample, spread over the checker's verdicts
#   make review-pull-facts      write the submitted verdicts back
#
# Push, review in the UI at ARGILLA_API_URL, then pull. A pull takes only
# submitted answers: Argilla saves a draft the moment a record is touched,
# and a draft is somebody part-way through thinking.
#
# Runs on the host against the same database the workers use. Nothing in
# the pipeline calls it, and no container carries it.
REVIEW = $(call WITH,review.env) && \
         PYTHONPATH=backend poetry run python -m review.run

# What is in Argilla now, and how much of it has been submitted.
review-status:
	$(REVIEW) --status

# Push a sample of facts, spread over the checker's verdicts.
review-push-facts:
	$(REVIEW) --push facts $(if $(ALL),--all)

# Write the submitted fact verdicts back to the database.
review-pull-facts:
	$(REVIEW) --pull facts

# Push each topic's name and whether it is a subject.
review-push-topics:
	$(REVIEW) --push topic-labels

# Write the submitted topic labels back to the database.
review-pull-topics:
	$(REVIEW) --pull topic-labels

# A stratified sample, or exactly the ids IDS names - which is how the
# disagreements `make second-opinion` found reach a reviewer. A sample
# cannot find those: they are rare in every band and every verdict, which
# is the shape a proportional draw misses.
review-push-questions:
	$(REVIEW) --push questions $(if $(IDS),--ids $(IDS)) $(if $(ALL),--all)

# Write the submitted question judgements back to the database.
review-pull-questions:
	$(REVIEW) --pull questions

# ── Pipeline ───────────────────────────────────────────────────────────────

# The orchestrator decides when a stage should run; it never runs one. It
# posts to the stage routes and polls /status, which is the same surface
# the Start button and the targets above use, so nothing here can move a
# row in a way they could not.
#
# `make up` starts both, like every other service. There is no target to
# start them on their own and nothing to switch on here: the schedule and
# the sensor both ship STOPPED, so the webserver serves an asset graph and
# the daemon ticks nothing until somebody enables one in the UI.

# Runs the UI and the daemon on the host instead, against the
# containerised PostgreSQL and the containerised api.
#
# Two things differ from the containers and are given here rather than in
# .env, because .env holds what both use. The database is reached on its
# published port rather than over the compose network, and the api is too.
dagster-dev:
	$(call WITH,orchestration.env) && \
	  DAGSTER_HOME=$$PWD/configs/dagster \
	  DAGSTER_DB_HOST=localhost DAGSTER_DB_PORT=$$POSTGRES_PORT \
	  BACKEND_URL=http://localhost:$$BACKEND_PORT \
	  poetry run dagster dev


# ── Journeys ───────────────────────────────────────────────────────────────
#
# Fewer commands, each of which is several of the ones above in the order
# somebody runs them. Nothing here does anything the fine-grained targets
# cannot, and nothing here is required: every stage still has its own six.
#
# They exist because the common paths were six commands long and the order
# mattered, which is a thing to get wrong rather than a thing to decide.

# The whole of it: bring the stack up, then take the corpus end to end.
all: up corpus

# Every stage, in the order a document moves through them, stopping at the
# first failure.
#
# Incremental, not a rebuild. Each `-start` queues only what has not been
# asked for yet, so a corpus already through a stage costs one HTTP call
# there and a run after one upload does that document alone.
#
# One `$(MAKE)` per line rather than a prerequisite list: prerequisites are
# ordered only while nothing runs in parallel, and this order is the point.
corpus:
	$(MAKE) parse-start     && $(MAKE) parse
	$(MAKE) chunk-start     && $(MAKE) chunk
	$(MAKE) extract-start   && $(MAKE) extract
	$(MAKE) topics-discover && $(MAKE) topics
	$(MAKE) questions-start && $(MAKE) questions
	$(MAKE) questions-balance

# Every container, whether it is listening, and where to open it.
#
# Asked of the api rather than read out of .env, so this and the System
# health page cannot come to disagree about what is running. A worker
# serves no port and is shown as `-`: it reads a queue, and its own stage
# page is what says whether it is doing that.
#
# localhost and BACKEND_PORT, not BACKEND_URL: that one is the container's
# view - `http://api:8000` - and this runs on the host.
services:
	@$(LOADENV) && answer=$$(curl -sf http://localhost:$$BACKEND_PORT/services) \
	  && printf '%s' "$$answer" | python3 -c 'import sys, json; [print("{:>4}  {:<20}  {}".format({True: "up", False: "DOWN", None: "-"}[r["ok"]], r["name"], r["url"] or "")) for r in json.load(sys.stdin)]' \
	  || echo "The api is not answering. Try: make up"

# The same, and open the application.
open: services
	@$(LOADENV) && url=http://localhost:$$STREAMLIT_PORT; \
	  if command -v open >/dev/null; then open $$url; \
	  elif command -v xdg-open >/dev/null; then xdg-open $$url; \
	  else echo "Open $$url"; fi

# Everything a person reviews, pushed to Argilla in one go.
review: review-push-facts review-push-topics review-push-questions

# Every artefact of every kind, each in its own dataset. Argilla then holds
# the corpus rather than a draw from it; `make review` is the sample.
review-all:
	$(MAKE) review ALL=1
	@$(LOADENV) && echo "Review at http://localhost:$$ARGILLA_PORT"

# Every decision made there, pulled back into the database.
pull: review-pull-facts review-pull-topics review-pull-questions

# ── Dagster, from here rather than from its own UI ─────────────────────────
#
# The asset graph already chains the stages: each waits for the one before
# it to drain, which is the order `corpus` above runs them in. What these
# decide is WHEN, which is the only thing an orchestrator was ever for.
#
# Run inside the webserver container, which is where DAGSTER_HOME and the
# workspace already are. Doing it from the host would need both again, and
# a second definition of either is a second thing to keep in step.
#
# A function rather than a prefix, because `-w` belongs after the
# subcommand and not before it:
#
#     $(call DAGSTER,sensor list)
#
# Its output is filtered at each call site rather than here. The compose
# shim hands back the container's two streams merged, so the interesting
# line arrives among the code server's startup logging and there is no
# redirection that separates them.
DAGSTER = $(COMPOSE) exec -T dagster-webserver dagster $(1) \
          -w /opt/dagster/dagster_home/workspace.yaml 2>&1

# What it is doing: the sensor, the schedule, and the last few runs.
#
# `run list` reads the instance and takes no `-w`, so it is not written
# through DAGSTER above. It asks the run storage what happened; the other
# two ask the code location what is defined.
#
# Matched without anchoring: the CLI colours its output, so a line starts
# with an escape sequence rather than with its own first word.
runs:
	-@$(call DAGSTER,sensor list)   | grep -E "Sensor: "
	-@$(call DAGSTER,schedule list) | grep -E "Schedule: "
	-@$(COMPOSE) exec -T dagster-webserver dagster run list --limit 5 2>&1 \
	  | grep -E "Run: |Job: |Status: "

# Every stage once, through Dagster instead of in the foreground. The same
# work `corpus` does; the difference is that it is recorded as a run, with
# a materialisation and a check per stage, and it survives this terminal
# closing.
pipeline:
	@$(call DAGSTER,job launch -j corpus) | grep -E "Launched run|Error" 

# Hand it over: an upload starts a run on its own from now on.
#
# The sensor watches parsing for documents nobody has asked for, and the
# asset graph carries them the rest of the way. Stopped until this runs,
# because a pipeline that starts the moment the stack comes up is one
# nobody chose.
auto:
	@$(call DAGSTER,sensor start arrivals) | grep -E "sensor arrivals|Error" || true
	@echo "An upload now starts the pipeline. 'make manual' hands it back."

# Take it back.
manual:
	@$(call DAGSTER,sensor stop arrivals) | grep -E "sensor arrivals|Error" || true
