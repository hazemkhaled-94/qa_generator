"""The schema alembic builds on an empty database."""

from __future__ import annotations

import pytest
from sqlalchemy import inspect, text

pytestmark = pytest.mark.integration


def test_the_migrations_build_every_table_the_models_declare(engine) -> None:
    """`alembic upgrade head` alone, on a database holding nothing."""
    from database.qa_generator import Base

    built = set(inspect(engine).get_table_names())
    declared = set(Base.metadata.tables)

    assert declared <= built, f"never created: {sorted(declared - built)}"


def test_the_extensions_the_schema_needs_are_installed(engine) -> None:
    """A vector column and a trigram index need these present first."""
    with engine.connect() as connection:
        installed = set(
            connection.scalars(text("SELECT extname FROM pg_extension")).all()
        )

    assert {"vector", "pg_trgm"} <= installed, installed


def test_the_delete_orphan_trigger_is_installed(engine) -> None:
    """Autogenerate never sees a trigger, so a migration has to name it."""
    with engine.connect() as connection:
        triggers = set(
            connection.scalars(
                text("SELECT tgname FROM pg_trigger WHERE NOT tgisinternal")
            ).all()
        )

    assert "question_facts_delete_orphan_question" in triggers, triggers


def test_every_check_constraint_the_models_declare_exists(engine) -> None:
    """Autogenerate does not compare CHECK constraints, so nothing else does.

    Two of these were declared on the model and never created by a revision:
    a fact could be stored `validated` with a rejection code, and a rejection
    code no check raises would have been accepted.
    """
    from sqlalchemy import CheckConstraint

    from database.qa_generator import Base

    declared = {
        constraint.name
        for table in Base.metadata.tables.values()
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint) and constraint.name
    }
    with engine.connect() as connection:
        built = set(
            connection.scalars(
                text(
                    "SELECT conname FROM pg_constraint "
                    "WHERE contype = 'c' AND connamespace = 'public'::regnamespace"
                )
            ).all()
        )

    assert declared <= built, f"declared but never created: {sorted(declared - built)}"


def test_the_models_and_the_migrations_agree(engine) -> None:
    """Autogenerate against the built schema must find nothing to write.

    A model changed without a revision, or a revision that does not build
    what the model describes, shows up here and nowhere else until a
    deployment.
    """
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from database.qa_generator import Base

    with engine.connect() as connection:
        context = MigrationContext.configure(
            connection,
            opts={"compare_type": True, "compare_server_default": True},
        )
        difference = compare_metadata(context, Base.metadata)

    assert not difference, "\n".join(str(item) for item in difference)


def test_every_enum_check_matches_its_model(engine) -> None:
    """A CHECK that has fallen behind its enum, which autogenerate cannot see.

    `test_the_models_and_the_migrations_agree` above runs
    `compare_metadata`, and that compares tables, columns, indexes and
    foreign keys. **It does not compare CHECK constraints.** So a member
    added to a StrEnum widens the constraint the model declares, no
    revision is generated for it, and nothing fails - until a long-lived
    database meets a gate that was added late and Postgres refuses the row.

    That is not hypothetical. `restates_question` and `answer_incomplete`
    reached `QuestionRejection` and the checker without reaching
    `questions_rejected_reason_valid`, and the first run to fire either one
    failed the topic with an IntegrityError instead of storing a verdict.

    Read off the MIGRATED database rather than off the models, because the
    models are the side that is right by construction.
    """
    import re

    from sqlalchemy import text

    from database.qa_generator import Base

    def members(sql: str) -> set[str]:
        """The literals one CHECK allows."""
        return set(re.findall(r"'([a-z_0-9]+)'", sql))

    declared = {
        constraint.name: members(str(constraint.sqltext))
        for table in Base.metadata.tables.values()
        for constraint in table.constraints
        if type(constraint).__name__ == "CheckConstraint"
        and len(members(str(constraint.sqltext))) >= 2
    }
    assert declared, "no enum-backed CHECK constraints found on the models"

    with engine.connect() as connection:
        built = {
            name: members(ddl)
            for name, ddl in connection.execute(
                text(
                    "SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint "
                    "WHERE contype = 'c' AND connamespace = 'public'::regnamespace"
                )
            ).all()
        }

    drifted = {
        name: sorted(values - built[name])
        for name, values in declared.items()
        if name in built and values - built[name]
    }

    assert not drifted, (
        "the database refuses values the models allow, and autogenerate will "
        "not write the revision for you:\n"
        + "\n".join(
            f"  {name} is missing {missing}" for name, missing in drifted.items()
        )
    )
