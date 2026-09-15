"""A question may not name its own source.

`leaks_source` refuses a question that says WHERE the answer is - naming or
quoting a document, a report, a circular, a named regulation, a section or a
heading.

This corrects an overshoot in the two revisions before it. Told to name what
it was asking about, the writer named the document: seven of ten questions in
one topic opened `Laut den 'Risiken im Fokus 2026', ...` or `Gemäß der MaRisk,
...`, echoing back titles that are rows in this database's own documents
table. The `unanchored` gate had asked for exactly that - its criterion read
"names its subject: the institution, the document, the rule" - which conflated
two different things.

Naming the SUBJECT is required: without it a question is vague, or the fact
with one part replaced by a question word. Naming the SOURCE is refused:
nobody asks a service desk a question while telling it which file to open, and
a question carrying its own source has already done the retrieving it was
written to measure.

Both halves of the gate are new. The free one refuses a question containing
its own document's title verbatim, for a title long enough to be a name. The
other is the verifier's, on the call it was already making, which catches the
paraphrases a title match cannot - `Laut dem Jahresbericht 2025` names a
source without quoting `Druckversion - Jahresbericht 2025` exactly.

Nothing is backfilled. Every row written before this was written under the
prompt that asked for the leak.

Revision: 5e8c07a3b6f1
Parent:   c41f8b7d2e06
Created:  2026-09-15
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "5e8c07a3b6f1"
down_revision: str | None = "c41f8b7d2e06"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NAME = "questions_rejected_reason_valid"

_WAS = (
    "rejected_reason IS NULL OR rejected_reason IN ('malformed', "
    "'answer_too_short', 'restates_fact', 'unanchored', 'not_recoverable', "
    "'duplicate', 'answerable_after_all', 'source_changed')"
)

_NOW = (
    "rejected_reason IS NULL OR rejected_reason IN ('malformed', "
    "'answer_too_short', 'leaks_source', 'restates_fact', 'unanchored', "
    "'not_recoverable', 'duplicate', 'answerable_after_all', 'source_changed')"
)


def upgrade() -> None:
    """Applies the change."""
    op.drop_constraint(_NAME, "questions", type_="check")
    op.create_check_constraint(_NAME, "questions", _NOW)


def downgrade() -> None:
    """Takes it back out.

    A question the new gate refused goes back to `draft`: the reason it
    carries is not a value the older constraint allows, and the order is the
    order every revision here uses - clear the values first, then narrow the
    constraint, because the reverse writes a value the constraint in force
    forbids.
    """
    op.execute(
        "UPDATE questions SET status = 'draft', rejected_reason = NULL "
        "WHERE rejected_reason = 'leaks_source'"
    )
    op.drop_constraint(_NAME, "questions", type_="check")
    op.create_check_constraint(_NAME, "questions", _WAS)
