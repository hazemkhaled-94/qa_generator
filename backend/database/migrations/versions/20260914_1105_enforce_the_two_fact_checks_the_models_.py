"""Enforce the two fact checks the models already declare.

`facts_verdict_agrees` and `facts_rejection_code_valid` are on the model and
were never created: autogenerate does not compare CHECK constraints, so no
revision ever carried them and nothing reported them missing.

Both are true of every row the pipeline writes - FactChecker sets `validated`
from whether a code was raised, and raises only the declared codes - so this
adds them without a backfill. A row that breaks either was not written by
this pipeline, and the migration stopping is the right answer to that.

Revision: 8c1e4f2a7b03
Parent:   29b6c8dff20e
Created:  2026-09-14
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "8c1e4f2a7b03"
down_revision: str | None = "29b6c8dff20e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: The verdict and the code are one fact stated twice; they cannot disagree.
_VERDICT = "validated = (rejection_code IS NULL)"

#: Every refusal carries a code a quality report can group on.
_CODE = (
    "rejection_code IS NULL OR rejection_code IN ('evidence_absent', 'copied', "
    "'not_atomic', 'unsupported_addition', 'unresolved_reference')"
)


def upgrade() -> None:
    """Applies the change."""
    op.create_check_constraint("facts_verdict_agrees", "facts", _VERDICT)
    op.create_check_constraint("facts_rejection_code_valid", "facts", _CODE)


def downgrade() -> None:
    """Takes it back out."""
    op.drop_constraint("facts_rejection_code_valid", "facts", type_="check")
    op.drop_constraint("facts_verdict_agrees", "facts", type_="check")
