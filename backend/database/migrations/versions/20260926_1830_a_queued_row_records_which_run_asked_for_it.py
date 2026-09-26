"""A queued row records which run asked for it.

Every produced row already carried `run_id`, and it named the wrong thing.
`settings.runs.run_id` was one uuid per PROCESS, and a stage worker is
long-lived: a Dagster run, the Start button and `make extract` all produced
rows stamped with the same worker, and nothing in the data said which of
them had asked. Following a corpus back to what set it going was therefore
impossible from either end.

So each of the six queues gets a column the queueing verb writes and the
claim reads back. `StageQueue._claim` adopts it for as long as the worker
holds the row, and `run_id` answers with it - so the id on a produced row
is now the id of the thing that asked for it. RUN_ID still overrides both.

Parsing and chunking both queue over `documents` and topic modelling and
question generation both queue over `topics`, so those two tables carry two
of these each: a queue's columns are prefixed with its own stage.

No index. These are read one row at a time by primary key on the way out of
a claim, never selected on.

NULL for every row already queued when this runs, which reads as "nobody
named a run" and falls back to the worker's own id exactly as before.

Revision ID: c5a71e83b94f
Revises: a92e0f4c7d61
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "c5a71e83b94f"
down_revision: str | None = "a92e0f4c7d61"
branch_labels: str | None = None
depends_on: str | None = None

#: What each stage's queue calls the run that asked, by the table it queues
#: over. The first carries the whole explanation and the rest point at it,
#: the way the model declarations do.
_ASKED = (
    "Which run asked for this document to be parsed, written by whatever "
    "queued it and adopted by the worker that claims it. NULL where nobody "
    "named a run, and the worker's own id is used instead."
)

#: Table, column, and what the column holds.
_COLUMNS = (
    ("documents", "parse_trigger", _ASKED),
    (
        "documents",
        "chunk_trigger",
        "Which run asked for this document to be chunked. See `parse_trigger`.",
    ),
    (
        "passages",
        "extract_trigger",
        ("Which run asked for this passage to be read. See `documents.parse_trigger`."),
    ),
    (
        "topics",
        "trigger",
        "Which run asked for this fit. See `documents.parse_trigger`.",
    ),
    (
        "topics",
        "question_trigger",
        ("Which run asked for questions on this topic. See `documents.parse_trigger`."),
    ),
    (
        "assessments",
        "assess_trigger",
        (
            "Which run asked for this artefact to be judged. See "
            "`documents.parse_trigger`."
        ),
    ),
)


def upgrade() -> None:
    """Applies the change."""
    for table, column, comment in _COLUMNS:
        op.add_column(
            table, sa.Column(column, sa.Text(), nullable=True, comment=comment)
        )


def downgrade() -> None:
    """Takes it back out."""
    for table, column, _ in reversed(_COLUMNS):
        op.drop_column(table, column)
