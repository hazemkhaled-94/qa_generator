"""Every revision, applied and taken back out.

Run against a database of its own, created and dropped here, because
downgrading to base empties the one every other test shares.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def spare(postgres: str) -> Iterator[str]:
    """A database of its own, dropped when the test is done."""
    name = f"rollback_{uuid.uuid4().hex[:12]}"
    admin = create_engine(postgres, isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{name}"'))
    url = postgres.rsplit("/", 1)[0] + "/" + name

    built = create_engine(url)
    with built.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
    built.dispose()

    try:
        yield url
    finally:
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


def alembic(url: str):
    """An alembic config pointed at one database."""
    from alembic.config import Config

    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "backend/database/migrations"))
    os.environ["DATABASE_URL"] = url
    return config


def tables(url: str) -> set[str]:
    """The tables one database holds."""
    engine = create_engine(url)
    try:
        return set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


def test_the_revisions_apply_to_an_empty_database(spare: str) -> None:
    """What a new deployment runs."""
    from alembic import command

    command.upgrade(alembic(spare), "head")

    assert "documents" in tables(spare)


def test_every_revision_can_be_taken_back_out(spare: str) -> None:
    """Down to base leaves nothing but alembic's own bookkeeping."""
    from alembic import command

    config = alembic(spare)
    command.upgrade(config, "head")
    command.downgrade(config, "base")

    assert tables(spare) <= {"alembic_version"}


def test_the_schema_survives_a_round_trip(spare: str) -> None:
    """Up, down and up again builds the same tables it did the first time.

    A downgrade that drops more than its upgrade created, or an upgrade that
    cannot run twice, shows up here.
    """
    from alembic import command

    config = alembic(spare)
    command.upgrade(config, "head")
    first = tables(spare)

    command.downgrade(config, "base")
    command.upgrade(config, "head")

    assert tables(spare) == first


def test_the_models_still_match_after_a_round_trip(spare: str) -> None:
    """The second upgrade builds the schema the models describe."""
    from alembic import command
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from database.qa_generator import Base

    config = alembic(spare)
    command.upgrade(config, "head")
    command.downgrade(config, "base")
    command.upgrade(config, "head")

    engine = create_engine(spare)
    try:
        with engine.connect() as connection:
            context = MigrationContext.configure(
                connection,
                opts={"compare_type": True, "compare_server_default": True},
            )
            difference = compare_metadata(context, Base.metadata)
    finally:
        engine.dispose()

    assert not difference, "\n".join(str(item) for item in difference)


def test_stepping_down_one_revision_at_a_time_reaches_base(spare: str) -> None:
    """Each downgrade in turn, rather than one jump, so a broken one names itself."""
    from alembic import command
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    config = alembic(spare)
    command.upgrade(config, "head")

    scripts = ScriptDirectory.from_config(Config(str(ROOT / "alembic.ini")))
    for _ in scripts.walk_revisions():
        command.downgrade(config, "-1")

    assert tables(spare) <= {"alembic_version"}
