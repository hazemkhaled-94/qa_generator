"""A question records which attempt at its slot wrote it.

`QUESTIONS_RETRIES` asks the writer again when a gate said something the
writer could have acted on, and every draft is stored - so a run's rows
hold both attempts and nothing said which was which. What a retry BOUGHT
is the acceptance rate of the second draft against the first, and that was
not answerable from the table: the best draft is moved to the end of the
list before it is written, so even row order did not say.

1 on every question written before this, which is the truth about them:
the column did not exist, so none of them is known to be a retry, and the
overwhelming majority of any run's rows are first attempts anyway.

Revision ID: a92e0f4c7d61
Revises: b4f1c7d20e85
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "a92e0f4c7d61"
down_revision: str | None = "b4f1c7d20e85"
branch_labels: str | None = None
depends_on: str | None = None

_COMMENT = (
    "Which attempt at this question's slot wrote it: 1 is the first draft "
    "and anything above it is a retry QUESTIONS_RETRIES paid for. Every "
    "draft is stored, refused ones included, so this is what makes the "
    "yield of a retry a query rather than a guess."
)


def upgrade() -> None:
    """Applies the change."""
    op.add_column(
        "questions",
        sa.Column(
            "attempt",
            sa.Integer(),
            nullable=False,
            server_default="1",
            comment=_COMMENT,
        ),
    )
    op.create_check_constraint(
        "questions_attempt_is_positive", "questions", "attempt >= 1"
    )


def downgrade() -> None:
    """Takes it back out."""
    op.drop_constraint("questions_attempt_is_positive", "questions", type_="check")
    op.drop_column("questions", "attempt")
