"""A row records the settings it was produced under.

A setting can now be changed while the pipeline runs, and the next row a
worker claims is worked under the new one. So a corpus is no longer produced
under one configuration, and nothing recorded which rows were produced under
which.

One column on each of the three tables that hold produced work, holding the
version settings.store computes: a digest of the overrides, or `environment`
where there are none. Two rows carrying the same version were produced the
same way.

A digest rather than a description. What the settings were is in
service_settings; the existing columns carry the model, the prompt version
and the temperature.

NULL on every row written before this, and by any worker not yet restarted.
Those rows were produced under settings nothing was recording.

Additive and nullable, so the five workers running against this database when
it is applied keep inserting as they do now.

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
#: carries. Written out because these are the comments the models declare,
#: which tests/integration asserts against.
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
        # Partial: the column is low-cardinality and mostly NULL.
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
