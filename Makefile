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
LOADENV = set -a && . ./configs/env/backend.env && . ./.env && set +a $(OVERRIDE)

# The same, for a tool with a tuning file of its own:
#
#     $(call WITH,review.env) && python -m review.run --status
#
# Sourced inside the `set -a` region, which is the whole point of having
# this rather than appending `&& . ./configs/env/x.env` to LOADENV: after
# `set +a` a sourced file's values are set in the shell and not exported,
# so the process below saw none of them and stopped naming the first.
WITH = set -a && . ./configs/env/backend.env && . ./configs/env/$(1) && \
       . ./.env && set +a $(OVERRIDE)

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

.PHONY: dev install up down down-volumes logs logs-frontend logs-api \
        logs-shipper logs-retention logs-orchestration \
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
        questions-retry questions-rerun questions-reverify questions-balance \
        documents delete delete-derived \
        test test-fast test-unit test-integration test-e2e test-smoke \
        test-eval test-coverage check typecheck audit lint format lock \
        certs dagster-dev \
        review-status review-push-facts review-pull-facts \
        review-push-topics review-pull-topics \
        review-push-questions review-pull-questions \
        eval-upload eval-score

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
#
# Retried, because they come from GitHub's release downloads rather than
# from an index: a 504 there fails the whole install, and did.
SPACY_MODELS := de_core_news_md en_core_web_md

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

up: certs
	$(COMPOSE) up -d

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

# What the models have cost, read off the logs the client already writes.
# Every call logs its tokens and its price - litellm prices the response
# rather than this counting it, because the provider is the only thing that
# knows what it billed for. A self-hosted model has no published price and
# logs no cost at all, which is the truth about it rather than a zero.
#
#   make spend                 everything the log holds
#   make spend SINCE=2026-09-18  from a date
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
#   make spend-by-shape SINCE=2026-09-21
#
# Ordered by call count, because the shape at the top is the one worth
# moving somewhere cheaper.
spend-by-shape:
	@printf '%-24s %-18s %8s %12s %12s %10s\n' \
	   MODEL SHAPE CALLS 'TOKENS IN' 'TOKENS OUT' COST
	@cat $(if $(LOG),$(LOG),/var/log/qa/*.log) 2>/dev/null \
	 | $(if $(SINCE),grep "$(SINCE)",cat) \
	 | grep -hoE '[^ ]+ answered _[A-Za-z]+ in [0-9.]+s \([0-9,]+ in, [0-9,]+ out, \$$[0-9.]+\)' \
	 | sed -E 's/^([^ ]+) answered (_[A-Za-z]+) in [0-9.]+s \(([0-9,]+) in, ([0-9,]+) out, \$$([0-9.]+)\)/\1 \2 \3 \4 \5/' \
	 | tr -d ',' \
	 | awk '{k=$$1" "$$2; n[k]++; i[k]+=$$3; o[k]+=$$4; c[k]+=$$5} \
	     END {for (k in n) {split(k,p," "); \
	       printf "%-24s %-18s %8d %12d %12d %9.2f\n", p[1],p[2],n[k],i[k],o[k],c[k]}}' \
	 | sort -k3 -rn
	@cat $(if $(LOG),$(LOG),/var/log/qa/*.log) 2>/dev/null \
	 | $(if $(SINCE),grep "$(SINCE)",cat) \
	 | grep -hoE 'answered _[A-Za-z]+ in [0-9.]+s \([0-9,]+ in, [0-9,]+ out, \$$[0-9.]+\)' \
	 | sed -E 's/.*\(([0-9,]+) in, ([0-9,]+) out, \$$([0-9.]+)\)/\1 \2 \3/' \
	 | tr -d ',' \
	 | awk '{i+=$$1; o+=$$2; c+=$$3; n++} END {if (n==0) {print "\nno priced calls in the log"; exit} \
	     printf "%-24s %-18s %8d %12d %12d %9.2f\n", "", "TOTAL", n, i, o, c}'

spend:
	@grep -hoE '^[0-9T:-]+ .*answered _[A-Za-z]+ in [0-9.]+s \([0-9,]+ in, [0-9,]+ out, \$$[0-9.]+\)' \
	   $(if $(LOG),$(LOG),/var/log/qa/*.log) 2>/dev/null \
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
# it. Irreversible. The upload record in ingest_events is kept.
#   make delete SHA=<sha256>
delete:
	@[ -n "$(SHA)" ] || { echo "usage: make delete SHA=<sha256>"; exit 2; }
	$(LOADENV) && PYTHONPATH=backend poetry run python -m ingestion.run --delete $(SHA)

# Empty the corpus: every document, every file, every converted form, and
# with them by cascade every passage, membership, fact and question. Then the
# topics, which are none of those. Irreversible.
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

# ── Tests ──────────────────────────────────────────────────────────────────

# Everything that gates: the unit layers, the static gates, and the
# integration layers, which start a Postgres and a SeaweedFS of their own
# and are skipped where no container engine answers. Configuration lives in
# [tool.pytest.ini_options] in pyproject.toml.
test:
	poetry run pytest -m "not smoke and not eval"

# Without the slow ones: no spaCy pipelines, no pyright, no containers.
test-fast:
	poetry run pytest -m "not nlp and not types and not integration"

# The layers on their own, for working on one of them.
test-unit:
	poetry run pytest -m "not integration and not smoke and not eval"

test-integration:
	poetry run pytest -m "integration and not e2e"

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

test-coverage:
	poetry run pytest -m "not smoke and not eval" \
	  --cov=backend --cov=telemetry --cov-report=term-missing:skip-covered

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

eval-upload:
	$(EVAL) --upload $(EVAL_DATASET)

eval-score:
	$(EVAL) --score $(EVAL_DATASET)

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

review-status:
	$(REVIEW) --status

review-push-facts:
	$(REVIEW) --push facts

review-pull-facts:
	$(REVIEW) --pull facts

review-push-topics:
	$(REVIEW) --push topic-labels

review-pull-topics:
	$(REVIEW) --pull topic-labels

review-push-questions:
	$(REVIEW) --push questions

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
