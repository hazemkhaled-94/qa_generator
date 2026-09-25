"""A row records how close it came to the verdict that would have refused it.

Every gate that is a measurement read a number and compared it to a
threshold, and only the comparison survived. A question 0.92 from an
accepted twin and one 0.01 from it are both `accepted` against a 0.93
ceiling, and nothing said which of the two a reviewer should open first.

`gate_scores` holds one entry per measuring gate that read the row - the
value, the threshold it was read against, and the margin between them.
`confidence` is the smallest of those margins, which is the only thing one
number can honestly say: what is the weakest thing about this row.

NULL on both for every row written before this, and NULL on `confidence`
for a row no measuring gate read. That is not zero: a question whose gates
were all judgements or all free rules has no confidence, where zero means
it survived one by nothing at all.

Revision ID: b4f1c7d20e85
Revises: c1a7e93b04d6
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "b4f1c7d20e85"
down_revision: str | None = "c1a7e93b04d6"
branch_labels: str | None = None
depends_on: str | None = None

_SCORES = (
    "One entry per measuring gate that read this row: the gate, the value it "
    "measured, the threshold it was read against and the margin between "
    "them. A gate that is a judgement rather than a measurement records "
    "nothing here, because an opinion has no number behind it. NULL for a "
    "row written before this column."
)

_CONFIDENCE = (
    "The smallest margin in gate_scores, in [0, 1]. How close this row came "
    "to the verdict that would have refused it, NOT a probability: the "
    "readings are on different scales and only the margins are comparable. "
    "NULL when no measuring gate read the row, which is different from 0."
)


def upgrade() -> None:
    """Applies the change."""
    for table in ("questions", "facts"):
        op.add_column(
            table,
            sa.Column("gate_scores", JSONB(), nullable=True, comment=_SCORES),
        )
        op.add_column(
            table,
            sa.Column("confidence", sa.Float(), nullable=True, comment=_CONFIDENCE),
        )
        op.create_check_constraint(
            f"{table}_confidence_is_a_share",
            table,
            "confidence IS NULL OR confidence BETWEEN 0 AND 1",
        )
        # What a review queue orders by, and it is read newest-worst-first
        # over a filtered set rather than scanned.
        op.create_index(
            f"ix_{table}_confidence",
            table,
            ["confidence"],
            postgresql_where=sa.text("confidence IS NOT NULL"),
        )


def downgrade() -> None:
    """Takes it back out."""
    for table in ("questions", "facts"):
        op.drop_index(f"ix_{table}_confidence", table_name=table)
        op.drop_constraint(f"{table}_confidence_is_a_share", table, type_="check")
        op.drop_column(table, "confidence")
        op.drop_column(table, "gate_scores")
