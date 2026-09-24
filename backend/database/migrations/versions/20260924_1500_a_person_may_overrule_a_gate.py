"""A person may overrule a gate, and the hold-out column goes.

Two changes to `questions`, both about what a review can be read back as.

**The reviewer's verdict.** `questions.status` was described as the human
column and `rejected_reason` as the gates', but `decide()` cleared the
reason on every human decision - so a person accepting what a gate refused
erased the only record that the two had disagreed. The agreement rate
between a reviewer and the checker was therefore not a query, which is the
one number a review sample exists to produce.

So the split facts got in `7a3e15c9b204` is copied back to questions, which
is where that revision said it came from. `reviewed_verdict` is the
person's and `rejected_reason` stays the gate's, and because the two are now
separate columns the reason is no longer stale once somebody overrules it -
it is the thing being compared against. `decide()` stops clearing it.

NULL means nobody has looked, which is every question written before this.

**`holdout_set_id` goes.** It was declared in the initial schema and no code
ever wrote it: nothing in `backend/`, `frontend/`, `evaluation/` or
`review/` referenced it, so the column carried NULL on every row it ever
had. Its comment promised more than a column can keep on its own - that the
questions it names are "withheld from the open export and kept encrypted" -
and there was neither an export to withhold them from nor anything doing the
encrypting. A promise nothing implements is worse than a missing feature,
because a reader checking the schema finds it and believes it.

Nothing is backfilled, and nothing could be: there are no human verdicts to
backfill and no hold-out draws to keep.

Revision: 3c81a9e4d7b2
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "3c81a9e4d7b2"
down_revision: str | None = "175d0f5da371"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Adds the reviewer's verdict, and drops the hold-out column."""
    op.add_column(
        "questions",
        sa.Column(
            "reviewed_verdict",
            sa.Text(),
            nullable=True,
            comment="What a person decided about this question: accepted or "
            "rejected. NULL means nobody has looked, which is most questions - "
            "a review is a sample. Separate from `status`, which is what the "
            "question IS, and from `rejected_reason`, which is the gate's; "
            "holding all three is what makes the agreement between a reviewer "
            "and the checker a query.",
        ),
    )
    op.add_column(
        "questions",
        sa.Column(
            "reviewed_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="When the verdict above was recorded. NULL exactly when "
            "reviewed_verdict is.",
        ),
    )
    op.create_check_constraint(
        "questions_reviewed_verdict_valid",
        "questions",
        "reviewed_verdict IS NULL OR reviewed_verdict IN ('accepted', 'rejected')",
    )
    # One fact about a row, so half of it written is a write that failed.
    op.create_check_constraint(
        "questions_reviewed_together",
        "questions",
        "(reviewed_verdict IS NULL) = (reviewed_at IS NULL)",
    )
    # Partial: NULL for almost every row, and what anyone asks for is the
    # reviewed ones, or the count of them beside the gates'.
    op.create_index(
        "ix_questions_reviewed_verdict",
        "questions",
        ["reviewed_verdict"],
        postgresql_where=sa.text("reviewed_verdict IS NOT NULL"),
    )
    op.drop_column("questions", "holdout_set_id")


def downgrade() -> None:
    """Puts the hold-out column back, and takes the verdicts off."""
    op.add_column(
        "questions",
        sa.Column(
            "holdout_set_id",
            sa.Uuid(),
            nullable=True,
            comment="The hold-out draw this question belongs to; one UUID per "
            "release. Nothing writes this column.",
        ),
    )
    op.drop_index("ix_questions_reviewed_verdict", table_name="questions")
    op.drop_constraint("questions_reviewed_together", "questions", type_="check")
    op.drop_constraint("questions_reviewed_verdict_valid", "questions", type_="check")
    op.drop_column("questions", "reviewed_at")
    op.drop_column("questions", "reviewed_verdict")
