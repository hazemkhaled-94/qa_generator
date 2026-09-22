"""Two runs under one configuration are two runs.

`settings_version` names a configuration, as a digest of the stored
overrides. That answers "what was this produced under" and cannot answer
"which run produced it": run the same stage twice over the same corpus
without touching a setting and both sets of rows carry one version.

Which is the position anybody comparing runs is in. The A/B recorded in
`evaluation/README.md` compared two phrasing models over one topic, and the
only reason the two could be separated afterwards is that the first run's
questions were deleted before the second started. That is not a method, it
is a thing that happened to work once, and it destroys the baseline it is
measuring against.

So one more column on each of the three tables a stage writes, filled from
`settings.runs.run_id`: RUN_ID where a caller set one and a uuid otherwise.
Indexed, because every query this exists for groups on it.

Nothing is backfilled HERE, and the claim this file used to make - that
nothing could be - was wrong. The rows already stored do carry which run
wrote them, in `created_at`: a run is hours of continuous writing with
hours of nothing either side. The revision after this one recovers them,
and says how it was checked.

Revision ID: 7a3f2c9e4b18
Revises: d511a6ee5767
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "7a3f2c9e4b18"
down_revision: str | None = "d511a6ee5767"
branch_labels: str | None = None
depends_on: str | None = None

#: The three tables a stage writes a row of its own into. `passages` and
#: `documents` are not here: a passage is what the chunker made of a
#: document and carries no verdict, so nothing about it is worth comparing
#: between two runs of anything.
TABLES = ("facts", "questions", "topics")

COMMENT = (
    "Which RUN produced this row, as `settings.runs.run_id` named it: "
    "RUN_ID where a caller set one, and a uuid otherwise. settings_version "
    "beside it names the CONFIGURATION, and two runs under one "
    "configuration carry the same version - so comparing a change against "
    "the run before it needs this column and cannot be done with that one. "
    "NULL for a row written before runs were named."
)


def upgrade() -> None:
    """Adds the run column to each table a stage writes, and indexes it."""
    for table in TABLES:
        op.add_column(
            table,
            sa.Column("run_id", sa.Text(), nullable=True, comment=COMMENT),
        )
        op.create_index(f"ix_{table}_run_id", table, ["run_id"])


def downgrade() -> None:
    """Takes it back out. The runs already named are lost with it."""
    for table in TABLES:
        op.drop_index(f"ix_{table}_run_id", table_name=table)
        op.drop_column(table, "run_id")
