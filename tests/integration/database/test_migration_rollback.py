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


def test_a_populated_table_survives_the_scopes_migration(spare: str) -> None:
    """A data backfill has to run in an order the constraints allow.

    This is the only layer that can catch it. Autogenerate never writes a
    backfill, an empty database runs every UPDATE against no rows, and the
    models compare equal either way - so an UPDATE that writes a value the
    constraint it has not dropped yet forbids passes everything except a
    real table with real rows in it. That is what happened here: the band
    was written while the old `difficulty` CHECK was still in force.
    """
    from alembic import command
    from sqlalchemy import create_engine, text

    scopes = "c41f8b7d2e06"
    config = alembic(spare)
    command.upgrade(config, "head")
    command.downgrade(config, f"{scopes}-1")

    engine = create_engine(spare)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO questions "
                "  (question_text, target_answer, answerable, difficulty, "
                "   language, status) "
                "VALUES "
                "  ('Q1?', '48 hours', true, 'single_passage', 'en', 'accepted'), "
                "  ('Q2?', 'a much longer answer than sixty characters would be, "
                "written out at length', true, 'cross_document', 'en', 'accepted'), "
                "  ('Q3?', NULL, false, 'cross_passage', 'en', 'accepted')"
            )
        )

    command.upgrade(config, scopes)

    with engine.connect() as connection:
        rows = {
            row.question_text: row
            for row in connection.execute(
                text(
                    "SELECT question_text, difficulty, passage_scope, "
                    "document_scope, topic_scope, answer_chars, thread_position "
                    "FROM questions"
                )
            ).all()
        }
    engine.dispose()

    # The mechanical translation, one row per old value.
    assert rows["Q1?"].passage_scope == "single_passage"
    assert rows["Q1?"].document_scope == "single_document"
    assert rows["Q2?"].document_scope == "cross_document"
    assert rows["Q3?"].passage_scope == "multi_passage"
    assert rows["Q3?"].document_scope == "single_document"

    # Every row lands in a band, and the band agrees with what it was built
    # from: Q1 has none of the five, Q2 has a wide spread and a long answer.
    assert rows["Q1?"].difficulty == "easy"
    assert rows["Q2?"].difficulty == "hard"
    assert rows["Q1?"].answer_chars == len("48 hours")
    assert all(row.topic_scope == "single_topic" for row in rows.values())
    assert all(row.thread_position == 1 for row in rows.values())


def test_a_thread_is_deleted_rather_than_orphaned_on_the_way_back(spare: str) -> None:
    """A follow-up without its parent is not a question anybody can score.

    It may lean on the thread for context, so the downgrade takes it rather
    than leaving a row the older schema has no way to read.
    """
    from alembic import command
    from sqlalchemy import create_engine, text

    scopes = "c41f8b7d2e06"
    config = alembic(spare)
    command.upgrade(config, "head")

    engine = create_engine(spare)
    with engine.begin() as connection:
        root = connection.execute(
            text(
                "INSERT INTO questions (question_text, target_answer, answerable, "
                "language, status, difficulty) VALUES "
                "('Root?', '48 hours', true, 'en', 'accepted', 'easy') RETURNING id"
            )
        ).scalar_one()
        connection.execute(
            text(
                "INSERT INTO questions (question_text, target_answer, answerable, "
                "language, status, difficulty, follows_id, thread_position) VALUES "
                "('And urgent?', '4 hours', true, 'en', 'accepted', 'easy', "
                ":root, 2)"
            ),
            {"root": root},
        )

    command.downgrade(config, f"{scopes}-1")

    with engine.connect() as connection:
        left = (
            connection.execute(text("SELECT question_text FROM questions"))
            .scalars()
            .all()
        )
    engine.dispose()

    assert left == ["Root?"], "the follow-up outlived the column linking it"


def test_the_type_columns_arrive_empty_and_the_topics_are_queued_again(
    spare: str,
) -> None:
    """Nothing written before the plan knew what it was asked for.

    The columns cannot be backfilled honestly - a question written under one
    prompt for every kind is not a factoid, it is a question nobody chose the
    kind of - so the rows go and the topics return to `new`. Regenerating is
    what makes the columns true.
    """
    from alembic import command
    from sqlalchemy import create_engine, text

    types = "3f7b21d9e4a5"
    config = alembic(spare)
    command.upgrade(config, f"{types}-1")

    engine = create_engine(spare)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO questions (question_text, target_answer, answerable, "
                "language, status, difficulty) VALUES "
                "('Written before the plan?', '48 hours', true, 'en', 'accepted', "
                "'easy')"
            )
        )

    command.upgrade(config, types)

    with engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM questions")).scalar() == 0
        columns = (
            connection.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'questions'"
                )
            )
            .scalars()
            .all()
        )
    engine.dispose()

    assert {"question_type", "answer_form", "planned_difficulty"} <= set(columns)
