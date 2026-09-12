# Build context is the REPO ROOT, for the shared telemetry package only. The
# frontend still knows nothing about the backend beyond one URL: telemetry is
# cross-cutting infrastructure, not a service.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/shared

WORKDIR /app

COPY frontend/requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY frontend/ .
COPY telemetry/ /shared/telemetry/

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8501

# Healthcheck is defined in compose.yaml so it stays next to the port mapping.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
