"""A person may overrule a fact check.

Facts had nowhere to record a human verdict. `validated` and
`rejection_code` are the checker's, and `extract-revalidate` rewrites both
from scratch every time it runs - so a reviewer's decision written there
would survive exactly until the next re-judgement, and would then be gone
with nothing saying it had been.

Questions already have this separation and it is the model copied here.
`questions.status` is the human column and `questions.rejected_reason` is
the gates', which is why `questions-reverify` "only ever rejects": a
re-check that un-rejected would overturn somebody's decision on its next
run.

So two columns of their own. `reviewed_verdict` NULL means nobody has
looked, which is every fact written before this and most facts after it -
a review is a sample, not a pass over twenty thousand rows. Nothing reads
these to decide whether a fact may seed a question yet; that is a separate
decision from being able to record the review at all, and this revision
only makes the recording possible.

Nothing is backfilled, and nothing could be: there are no human verdicts
to backfill.

Revision: 7a3e15c9b204
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "7a3e15c9b204"
down_revision: str | None = "f1b9d6c30a47"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Adds the reviewer's verdict and when it was given."""
    op.add_column(
        "facts",
        sa.Column(
            "reviewed_verdict",
            sa.Text(),
            nullable=True,
            comment="What a person decided about this fact: accepted or "
            "rejected. NULL means nobody has looked, which is most facts - a "
            "review is a sample. Separate from `validated`, which is the "
            "checker's and is rewritten in full by extract-revalidate.",
        ),
    )
    op.add_column(
        "facts",
        sa.Column(
            "reviewed_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="When the verdict above was recorded. NULL exactly when "
            "reviewed_verdict is.",
        ),
    )
    op.create_check_constraint(
        "facts_reviewed_verdict_valid",
        "facts",
        "reviewed_verdict IS NULL OR reviewed_verdict IN ('accepted', 'rejected')",
    )
    # The two are one fact about a row, so a verdict with no timestamp - or a
    # timestamp with no verdict - is a write that went half in.
    op.create_check_constraint(
        "facts_reviewed_together",
        "facts",
        "(reviewed_verdict IS NULL) = (reviewed_at IS NULL)",
    )
    # Partial: the column is NULL for almost every row, and what anyone asks
    # for is the reviewed ones, or the count of them beside the checker's.
    op.create_index(
        "ix_facts_reviewed_verdict",
        "facts",
        ["reviewed_verdict"],
        postgresql_where=sa.text("reviewed_verdict IS NOT NULL"),
    )


def downgrade() -> None:
    """Takes the reviewer's verdict back off, and every verdict with it."""
    op.drop_index("ix_facts_reviewed_verdict", table_name="facts")
    op.drop_constraint("facts_reviewed_together", "facts", type_="check")
    op.drop_constraint("facts_reviewed_verdict_valid", "facts", type_="check")
    op.drop_column("facts", "reviewed_at")
    op.drop_column("facts", "reviewed_verdict")
