# Build context is the REPO ROOT: the service imports blob_store, database and
# the backend service packages from outside its own directory. See compose.yaml.

# ── Resolver ───────────────────────────────────────────────────────────────
#
# Not part of the runtime image, and never built by `compose build`. This is
# the stage that turns requirements.txt into requirements.lock: it installs
# the ranges and prints what they resolved to. `make lock` runs it.
#
# It exists because the runtime stage installs the lock and never reads
# requirements.txt, so a build could not be what regenerates the lock: a
# dependency added to requirements.txt would never reach pip.
FROM python:3.13-slim AS resolve

ENV PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY backend/api/requirements.txt .

# CPU-only torch first, so resolving the ranges below finds torch already
# satisfied and never reaches PyPI for a CUDA build. Unpinned on purpose:
# what this stage resolves is what the lock is about to say.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -r requirements.txt \
    && ! pip list --format=freeze | grep -q '^nvidia-'

CMD ["pip", "freeze"]

# ── Runtime ────────────────────────────────────────────────────────────────
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/shared

WORKDIR /app

# Docling reads page images through OpenCV, which links against these at
# import time and finds none of them in a slim image. Without them the
# converter fails on its first document with a bare ImportError.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libxcb1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# The lock, not the ranges. requirements.txt says what we depend on;
# requirements.lock says which versions, transitive ones included, so this
# image is the same image next month. See the header of the lock file for
# how to regenerate it.
COPY backend/api/requirements.lock backend/api/requirements.txt ./

# Fails the build if requirements.txt declares something the lock does not
# pin. Only the lock is installed, so without this a dependency added to
# requirements.txt alone builds a green image that crashes on first import -
# which is how the topic worker shipped without gensim. Names are folded to
# PEP 503 form so SQLAlchemy and sqlalchemy are one name.
RUN set -eu; \
    fold() { tr 'A-Z' 'a-z' | tr '_.' '--' | sort -u; }; \
    sed 's/#.*//' requirements.txt \
      | sed -n 's/^\([A-Za-z0-9][A-Za-z0-9._-]*\).*/\1/p' | fold > /tmp/declared; \
    sed -n 's/^\([A-Za-z0-9][A-Za-z0-9._-]*\)==.*/\1/p' requirements.lock \
      | fold > /tmp/locked; \
    missing="$(comm -23 /tmp/declared /tmp/locked | tr '\n' ' ')"; \
    rm -f /tmp/declared /tmp/locked; \
    [ -z "$missing" ] || { \
      echo "requirements.lock does not pin: $missing" >&2; \
      echo "requirements.txt declares them. Run \`make lock\`." >&2; \
      exit 1; }

# CPU-only torch, from PyTorch's index. Versions read from the lock.
RUN grep -E '^(torch|torchvision)==' requirements.lock > /tmp/torch.txt \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        -r /tmp/torch.txt \
    && rm /tmp/torch.txt

RUN pip install --no-cache-dir -r requirements.lock

# The spaCy pipelines NLP_MODELS names. Baked in because the runtime has no
# network: a pipeline missing at run time fails every passage of every
# document, and nothing downloads one on demand.
ARG SPACY_MODELS="de_core_news_md en_core_web_md"
RUN for model in ${SPACY_MODELS}; do python -m spacy download "$model"; done

# Fails the build if any CUDA wheel is present.
RUN ! pip list --format=freeze | grep -q '^nvidia-'

COPY backend/api/ ./api/
COPY backend/database/ ./database/
COPY backend/blob_store/ ./blob_store/
COPY backend/settings/ ./settings/
COPY backend/stages/ ./stages/
COPY backend/ingestion/ ./ingestion/
COPY backend/preprocessing/ ./preprocessing/
COPY backend/extraction/ ./extraction/
COPY backend/topic_modelling/ ./topic_modelling/
COPY backend/question_generation/ ./question_generation/
COPY backend/assessment/ ./assessment/
COPY backend/nlp/ ./nlp/
COPY backend/llm/ ./llm/
# The one first-party module that is a file rather than a package, and so
# the one that a line per directory above does not carry. Four packages
# import it - extraction, question_generation and the two that read a
# score back - and without it the api and every worker died on
# `ModuleNotFoundError: No module named 'confidence'` at import.
COPY backend/confidence.py ./confidence.py
# The version stamped on a stored object is read from it at run time.
COPY pyproject.toml ./
COPY telemetry/ /shared/telemetry/

RUN useradd --create-home --uid 1000 app

# Where the JSON log lines go, when LOG_DIR points here. Created before the
# volume is mounted over it and owned by the user that writes: an empty
# named volume takes the ownership of the image directory it covers, and a
# directory that only exists at mount time arrives owned by root, which this
# process is not.
RUN mkdir -p /var/log/qa && chown app:app /var/log/qa

USER app

EXPOSE 8000

# Healthcheck is defined in compose.yaml so it stays next to the port mapping.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
