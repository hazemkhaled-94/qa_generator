"""A question records every gate that read it, not only the one that stopped it.

`gates_ran` went on the span alone. `rejected_reason` says what stopped a
question and can never say what it got past, so a stored question could not
answer "which gates did this pass" without Phoenix holding the trace still.

Two of the six gates are conditional, so the sequence cannot be derived
from the verdict and a fixed order. It has to be stored.

NULL for every question written before this, which is distinct from the
empty array: nothing read it is a different answer from nobody recorded
what read it.

Revision ID: 6e2f8c0d3a91
Revises: 4d7e2b9a15c3
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY

revision: str = "6e2f8c0d3a91"
down_revision: str | None = "4d7e2b9a15c3"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Applies the change."""
    op.add_column(
        "questions",
        sa.Column(
            "gates_ran",
            ARRAY(sa.Text()),
            nullable=True,
            comment=(
                "Every gate that read this question, in the order they read it, "
                "the one that refused it last. question_generation.checker.GATES "
                "gives each its fixed position. Two gates are conditional, so "
                "this cannot be derived from rejected_reason and a fixed order. "
                "NULL for a question written before this column; empty means no "
                "gate ran."
            ),
        ),
    )


def downgrade() -> None:
    """Takes it back out."""
    op.drop_column("questions", "gates_ran")
