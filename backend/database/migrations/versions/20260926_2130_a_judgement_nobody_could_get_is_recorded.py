"""A judgement nobody could get is recorded.

Three stages ask a model for an opinion about an artefact that already
exists, and all three ABSTAIN when the model cannot be reached: losing one
opinion must not fail a row the rest of the pipeline judged fine. That part
is right and is unchanged. What was missing is any record that the opinion
was asked for and not had, which made an abstention unfindable and
unrepeatable - and for one of the three, invisible in a way that changed
what shipped.

**`questions.gates_abstained`.** The phrasing gates are the reason this is
a defect and not a wart. `checker` reads a judgement as a rejection only
when it comes back False, and an abstention comes back None - so a gate
that could not run read exactly like a gate that passed, and the question
was ACCEPTED. Recorded here, `questions-reverify` can find them and a
released set can be told what was never judged.

**`assessments.metrics_due`.** A row reads `assessed` whether the judge
answered six metrics or two. The count that was due is in the source, under
the template version, so SQL could not see it: a thin verdict and a whole
one were the same row. Written on the row rather than derived, for the
reason the `prompts` table exists at all - the row outlives the source.

NULL on every row already written, which reads as "not known" rather than
as zero. Neither column is indexed: both are read with the row.

Revision ID: f47b2c9e1a08
Revises: d3f8a1c47b52
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY

revision: str = "f47b2c9e1a08"
down_revision: str | None = "d3f8a1c47b52"
branch_labels: str | None = None
depends_on: str | None = None

#: Word for word what the models declare, because
#: `tests/integration/database/test_schema.py` compares the two.
_ABSTAINED = (
    "Every judgement this question was meant to get and could not, because the "
    "model could not be reached. A gate that abstained is NOT a gate that "
    "passed: the checker rejects on False and an abstention is None, so without "
    "this the question reads as having cleared a gate nobody ran. NULL for a "
    "question written before this column; empty means every gate that was asked "
    "answered."
)

_DUE = (
    "How many metrics this artefact's kind was due under its prompt_version. "
    "The judge abstains per metric when the model cannot be reached, so a row "
    "with fewer assessment_metrics than this was judged on less than it should "
    "have been. Stored rather than derived because the count lives in the "
    "source under a version the row outlives. NULL for a row assessed before "
    "this column."
)


def upgrade() -> None:
    """Applies the change."""
    op.add_column(
        "questions",
        sa.Column(
            "gates_abstained", ARRAY(sa.Text()), nullable=True, comment=_ABSTAINED
        ),
    )
    op.add_column(
        "assessments",
        sa.Column("metrics_due", sa.Integer(), nullable=True, comment=_DUE),
    )


def downgrade() -> None:
    """Takes it back out."""
    op.drop_column("assessments", "metrics_due")
    op.drop_column("questions", "gates_abstained")
