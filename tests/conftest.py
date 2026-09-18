"""The environment every test runs in, and the containers some need.

The tuning values are read from configs/env/backend.env, the file the
services read, rather than copied, and they are set here because pytest
loads this file before it collects a test module.

The container fixtures are session-scoped and lazy: nothing is started
until a test asks for one, so the unit layer pays nothing for them. The
schema is applied by alembic, not by `create_all`, so what the integration
layer runs against is what a deployment gets.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TUNING = ROOT / "configs/env/backend.env"

for line in TUNING.read_text().splitlines():
    name, sign, value = line.partition("=")
    if sign and not name.lstrip().startswith("#"):
        os.environ.setdefault(name.strip(), value.strip())

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://unused:unused@localhost/x")

#: The same images compose runs.
POSTGRES_IMAGE = "pgvector/pgvector:pg18"
SEAWEEDFS_IMAGE = "chrislusf/seaweedfs:4.44"

#: What configs/postgres/init.sh installs before the schema is built. The
#: migrations create these too; tests/static asserts the two agree.
EXTENSIONS = ("vector", "pg_trgm")


def _runtime_ready() -> str | None:
    """Says why containers cannot be started here, or nothing."""
    try:
        import docker
    except ImportError:  # pragma: no cover - the dev group installs it
        return "the docker client library is not installed"
    try:
        docker.from_env().ping()
    except Exception as exc:  # noqa: BLE001 - any failure is the same answer
        return f"no container runtime: {type(exc).__name__}: {exc}"
    return None


@pytest.fixture(scope="session")
def runtime() -> None:
    """Skips the whole layer when there is nowhere to run a container."""
    refused = _runtime_ready()
    if refused:
        pytest.skip(refused, allow_module_level=True)


def _started(build, attempts: int = 3):
    """Starts a container, retrying a start that fails outright.

    Not a readiness wait, which testcontainers already does: an engine under
    load occasionally refuses the run itself, and one retry is the difference
    between a gate and a coin toss.
    """
    for remaining in range(attempts - 1, -1, -1):
        container = build()
        try:
            return container.start()
        except Exception:
            if not remaining:
                raise
            with contextlib.suppress(Exception):
                container.stop()


@pytest.fixture(scope="session")
def postgres(runtime) -> Iterator[str]:
    """A Postgres with the extensions installed and the schema migrated."""
    from testcontainers.community.postgres import PostgresContainer

    container = _started(lambda: PostgresContainer(POSTGRES_IMAGE, driver="psycopg"))
    try:
        url = container.get_connection_url()
        _install_extensions(url)
        _migrate(url)
        yield url
    finally:
        container.stop()


def _install_extensions(url: str) -> None:
    """Runs what init.sh runs before the application schema is built."""
    from sqlalchemy import create_engine, text

    engine = create_engine(url)
    with engine.begin() as connection:
        for extension in EXTENSIONS:
            connection.execute(text(f"CREATE EXTENSION IF NOT EXISTS {extension}"))
    engine.dispose()


def _migrate(url: str) -> None:
    """Brings the database up to the newest revision."""
    from alembic import command
    from alembic.config import Config

    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "backend/database/migrations"))
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = url
    try:
        command.upgrade(config, "head")
    finally:
        if previous is None:
            del os.environ["DATABASE_URL"]
        else:
            os.environ["DATABASE_URL"] = previous


@pytest.fixture(scope="session")
def engine(postgres: str):
    """An engine on the migrated database, for the tests that read directly."""
    from sqlalchemy import create_engine

    built = create_engine(postgres)
    yield built
    built.dispose()


@pytest.fixture
def database(postgres: str, engine, monkeypatch) -> Iterator[None]:
    """Points the repositories at the container, and empties it afterwards.

    The repositories reach the database through the cached engine in
    `database.qa_generator.engine`, which reads DATABASE_URL once per
    process. Both caches are cleared either side so a test never inherits
    the connection of whatever ran before it.
    """
    from sqlalchemy import text

    from database.qa_generator.engine import engine as connect
    from database.qa_generator.engine import sessions

    monkeypatch.setenv("DATABASE_URL", postgres)
    connect.cache_clear()
    sessions.cache_clear()
    try:
        yield
    finally:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "TRUNCATE documents, ingest_events, passages, facts, "
                    "questions, question_facts, passage_topics, topics, "
                    "service_settings "
                    "RESTART IDENTITY CASCADE"
                )
            )
        connect.cache_clear()
        sessions.cache_clear()


#: The credentials the gateway is given, and the ones the buckets use.
S3_KEY, S3_SECRET = "qa-test-key", "qa-test-secret"

#: The identity configs/seaweedfs/entrypoint.sh writes from .env. Without
#: one the gateway serves anonymously and lists no bucket at all, which is
#: what the health check reads.
S3_IDENTITY = json.dumps(
    {
        "identities": [
            {
                "name": "app",
                "credentials": [{"accessKey": S3_KEY, "secretKey": S3_SECRET}],
                "actions": ["Admin", "Read", "Write", "List", "Tagging"],
            }
        ]
    }
)


@pytest.fixture(scope="session")
def s3(runtime) -> Iterator[dict[str, str]]:
    """A SeaweedFS S3 gateway, every role in one container."""
    from testcontainers.core.container import DockerContainer
    from testcontainers.core.wait_strategies import LogMessageWaitStrategy

    start = (
        "mkdir -p /etc/seaweedfs && "
        f"printf '%s' '{S3_IDENTITY}' > /etc/seaweedfs/s3.json && "
        # -volume.max: a bucket is a SeaweedFS collection, and a collection
        # does not claim one volume on its first write - it claims seven, in
        # one growth batch, and `weed server` exposes no flag to make that
        # smaller. Three buckets is therefore 21 volumes, plus the six the
        # default collection takes for the filer's own metadata: 27.
        #
        # At 20 the third bucket got none. `export` answered head_bucket and
        # then refused every PutObject with an InternalError, which took the
        # three tests that store a topic figure whenever the other two
        # buckets were written to first - so it passed alone and failed in a
        # full run, which is the worst way for a gate to be wrong. Measured
        # on the container this builds: documents 7, parsed 7, default 6,
        # free 0.
        #
        # 60 is headroom rather than a second guess. A volume is created on
        # demand, so a slot nothing uses costs nothing.
        "exec weed server -dir=/data -s3 -ip=0.0.0.0 -volume.max=60 "
        "-s3.config=/etc/seaweedfs/s3.json"
    )
    # The image's entrypoint is `weed`, so the shell that writes the config
    # has to replace it rather than be passed to it.
    container = _started(
        lambda: (
            DockerContainer(SEAWEEDFS_IMAGE)
            .with_kwargs(entrypoint="/bin/sh")
            .with_command(["-c", start])
            .with_exposed_ports(8333)
            .waiting_for(
                LogMessageWaitStrategy(
                    "Start Seaweed S3 API Server"
                ).with_startup_timeout(120)
            )
        )
    )
    try:
        yield {
            "S3_ENDPOINT": (
                f"http://{container.get_container_host_ip()}"
                f":{container.get_exposed_port(8333)}"
            ),
            "S3_ACCESS_KEY": S3_KEY,
            "S3_SECRET_KEY": S3_SECRET,
        }
    finally:
        container.stop()


#: What S3_BUCKETS names, which configs/seaweedfs/bucket-init.sh creates at
#: start-up. Nothing in the application creates a bucket.
BUCKETS = ("documents", "parsed", "export")


@pytest.fixture
def buckets(s3: dict[str, str], monkeypatch) -> Iterator[None]:
    """Points the buckets at the gateway and empties them afterwards.

    `Bucket.count` answers from a process-wide cache with a 30 second life,
    so that is emptied too: two tests in the same second would otherwise
    read each other's total.
    """
    from blob_store.seaweedfs import client
    from blob_store.seaweedfs.bucket import _COUNTS

    for name, value in s3.items():
        monkeypatch.setenv(name, value)
    client.s3_client.cache_clear()
    _COUNTS.clear()

    made = client.s3_client()
    _ensure(made)
    _empty(made)
    try:
        yield
    finally:
        _empty(made)
        _COUNTS.clear()
        client.s3_client.cache_clear()


def _ensure(client, attempts: int = 10) -> None:
    """Makes sure every bucket is there before a test writes to one.

    Creating one used to be wrapped in a bare suppress, which is fine for the
    usual case - the bucket exists and creation is refused - and wrong for the
    one that matters. `test_a_missing_bucket_is_named` deletes `export` to see
    what the health check says, and SeaweedFS does not free the name at once:
    the next test's creation was refused, suppressed, and every later write to
    that bucket failed with an InternalError after three retries. In a full
    run that took the three tests that put a topic figure.

    Raises:
        AssertionError: If a bucket cannot be created, which is the object
            store being broken rather than a test having deleted one.
    """
    for name in BUCKETS:
        for attempt in range(attempts):
            with contextlib.suppress(Exception):
                client.create_bucket(Bucket=name)
            with contextlib.suppress(Exception):
                client.head_bucket(Bucket=name)
                break
            time.sleep(0.2 * (attempt + 1))
        else:
            raise AssertionError(f"the {name} bucket could not be created")


def _empty(client) -> None:
    """Removes every object from every bucket that is there.

    Tolerant of a bucket that is not: a test may have deleted one to see
    what the health check says about it.
    """
    for name in BUCKETS:
        with contextlib.suppress(client.exceptions.NoSuchBucket):
            for page in client.get_paginator("list_objects_v2").paginate(Bucket=name):
                for held in page.get("Contents", []):
                    client.delete_object(Bucket=name, Key=held["Key"])


@pytest.fixture(scope="session")
def application(postgres: str, s3: dict[str, str]):
    """The FastAPI application, built against the containers.

    api.dependencies is the composition root and builds every service as it
    is imported, binding each to the addresses in the environment at that
    moment. So it is imported here, once, after the containers are up - and
    nothing is dropped from sys.modules afterwards, because a second copy of
    blob_store would carry a second object count cache and the fixtures
    would be clearing the wrong one.

    The routes hold those services as module-level names rather than through
    `Depends`, so there is nothing to override: what these tests exercise is
    the real wiring.

    The caches are cleared BEFORE that import, and that is the whole fixture.
    Each repository captures `sessions()` as it is constructed, so a cache
    populated by anything that ran earlier binds every route for the rest of
    the process - and the `database` fixture clearing it afterwards cannot
    reach the factories already captured. A unit test that runs a stage's
    command line populates it against the ambient DATABASE_URL, which is the
    developer's own database: the API tests then read and write there while
    the fixtures set up a container nothing touches.
    """
    from database.qa_generator.engine import engine as connect
    from database.qa_generator.engine import sessions

    os.environ["DATABASE_URL"] = postgres
    os.environ.update(s3)
    # Model identity, which compose gives the api and configs/env/backend.env
    # does not carry: the settings routes serve these and refuse a bad value
    # for one, and both need the value the workers are reading.
    os.environ.setdefault("LLM_MODEL", "ollama_chat/test-model")
    connect.cache_clear()
    sessions.cache_clear()

    from api.main import app

    return app


@pytest.fixture
def client(application, database, buckets):
    """A client on an empty database and empty buckets."""
    from fastapi.testclient import TestClient

    with TestClient(application) as opened:
        yield opened


@pytest.fixture
def pdf() -> bytes:
    """A one-page PDF, small enough to keep here and real enough to read."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
        b"trailer<</Root 1 0 R>>\n"
    )
