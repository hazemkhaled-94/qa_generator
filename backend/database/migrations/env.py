"""How Alembic reaches the database and what it compares against.

The target is `Base.metadata`, which holds every model the package imports,
so `alembic revision --autogenerate` diffs the models against the live
schema. The URL comes from the environment, like every other setting.
"""

from __future__ import annotations

import os

from alembic import context
from pgvector.sqlalchemy import Vector
from sqlalchemy import engine_from_config, pool

from database.qa_generator import Base

config = context.config

# Not in alembic.ini: the URL is a credential, and .env is where those live.
config.set_main_option("sqlalchemy.url", os.environ["DATABASE_URL"])

target_metadata = Base.metadata


def _render_item(type_, obj, autogen_context) -> bool:
    """Adds the import a rendered type needs."""
    if type_ == "type" and isinstance(obj, Vector):
        # Rendered as pgvector.sqlalchemy.vector.VECTOR, which is a
        # NameError in the revision without this.
        autogen_context.imports.add("import pgvector.sqlalchemy")
    return False


#: Autogenerate reads these off the models, so a changed CHECK constraint or
#: a changed column comment produces a revision instead of being skipped.
_COMPARE = {
    "compare_type": True,
    "compare_server_default": True,
    "render_item": _render_item,
}


def run_migrations_offline() -> None:
    """Emits the migrations as SQL, without connecting.

    For a deployment where the person applying the change is not the person
    who wrote it.
    """
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        **_COMPARE,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Applies the migrations against the live database."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata, **_COMPARE
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
