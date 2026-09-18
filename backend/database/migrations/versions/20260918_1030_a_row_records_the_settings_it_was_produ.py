"""A row records the settings it was produced under.

Settings used to be read once, out of a file, when a service started. A
corpus was therefore produced under one configuration, whatever the files
said, and a reader asking what a fact was extracted under could read the
file.

Now a setting can be changed while the pipeline runs, and the next row a
worker claims is worked under the new one. That makes the question real:
these twenty thousand facts were not all read the same way, and nothing
recorded which of them were read which way.

So one column on each of the three tables that hold produced work. It holds
the version `settings.store` computes - a digest of the overrides, or
`environment` where there are none - which names a configuration by its
content. Two rows carrying the same version were produced under the same
settings, and a reader comparing two runs can select on it.

A digest rather than a description. What the settings were is in
service_settings, and what a row needs is a name for the set of them; the
existing columns already carry the values worth reading off a single fact,
which is the model, the prompt version and the temperature.

NULL on every row written before this, and on every row a worker writes
until it is restarted. That is honest rather than unfortunate: those rows
were produced under settings this column cannot name, because nothing was
recording one.

Additive and nullable, so the five workers running against this database
when it is applied keep inserting exactly as they do now.

Revision: d3c81f4a2e57
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d3c81f4a2e57"
down_revision: str | None = "b8f24e07d3a1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: The tables that hold work a stage produced, and the comment each column
#: carries. Written out rather than built from a template: these are the
#: comments the models declare, and tests/integration asserts the two agree.
_TABLES = {
    "facts": "The configuration this fact was extracted under, as the digest "
    "settings.store computes, or 'environment' when nothing was overridden. "
    "The columns above name the model and the prompt; this names the rest of "
    "it, including the shares the checks held this fact to. NULL for a fact "
    "written before a setting could be changed without a restart.",
    "topics": "The configuration this topic was fitted under, as the digest "
    "settings.store computes, or 'environment' when nothing was overridden. "
    "A fit is not reproducible without the parameters it used, and the seed "
    "is only one of them. NULL for a topic fitted before a setting could be "
    "changed without a restart.",
    "questions": "The configuration this question was written under, as the "
    "digest settings.store computes, or 'environment' when nothing was "
    "overridden. The only provenance a question carries: which model wrote "
    "it, what mix the plan aimed for and what bounds the gates held it to "
    "are all in the settings that version names. NULL for a question written "
    "before a setting could be changed without a restart.",
}


def upgrade() -> None:
    """Adds the settings version to every table that holds produced work."""
    for table, comment in _TABLES.items():
        op.add_column(
            table,
            sa.Column("settings_version", sa.Text(), nullable=True, comment=comment),
        )
        # Partial: what anyone asks for is the rows produced under one
        # configuration, or the ones produced under none, and the column is
        # low-cardinality either way.
        op.create_index(
            f"ix_{table}_settings_version",
            table,
            ["settings_version"],
            postgresql_where=sa.text("settings_version IS NOT NULL"),
        )


def downgrade() -> None:
    """Takes the column back off, and every version recorded with it."""
    for table in _TABLES:
        op.drop_index(f"ix_{table}_settings_version", table_name=table)
        op.drop_column(table, "settings_version")
