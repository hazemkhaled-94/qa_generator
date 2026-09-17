# Build context is the REPO ROOT: the code location imports `telemetry`
# from beside itself. See compose.yaml.
#
# An image of its own rather than the backend one, which serves the api and
# all five workers. That image carries torch, spaCy, Docling and litellm
# because some process in it needs each; this process makes HTTP calls and
# needs none of them. Sharing it would have put three gigabytes behind two
# containers whose whole job is to POST /parsing/start and poll a status
# route, and would have coupled the orchestrator's rebuilds to the
# pipeline's.

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt/dagster/app:/shared

WORKDIR /opt/dagster/app

COPY orchestration/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY orchestration/ ./orchestration/
COPY telemetry/ /shared/telemetry/

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /opt/dagster/dagster_home /var/log/qa \
    && chown -R app:app /opt/dagster /var/log/qa

USER app

EXPOSE 3000

# The command is in compose.yaml: one image serves the webserver and the
# daemon, and what differs is which of them it runs.
