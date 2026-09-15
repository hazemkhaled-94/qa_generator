"""Two more gates, for a question a person would actually ask.

The first run of question generation produced cloze deletions: the fact with
one part replaced by a question word. `Geopolitische Konflikte schüren
Unsicherheit` became `Was schüren geopolitische Konflikte?`. Both new codes
name a way of failing that the existing five could not see, and both are
counted separately because each says something different about what to
change.

`restates_fact` is structural and free. The question's content lemmas are a
subset of its facts', so it is about exactly what the fact is about and adds
nothing a person searching a corpus would have to know in order to ask it.
That is the writer ignoring its prompt, and the number is how you tell
whether a prompt change worked.

`unanchored` is the verifier's judgement, taken from the call it was already
making. A question can be perfectly answerable by the passages it cites and
still be one nobody could have typed - naming no institution, no document and
no period, or referring to `the requirements` without saying which. No
structural check makes that call; a model reading the question and the
material can.

Nothing is backfilled. The rows already stored were written by the prompt
these gates exist to correct, and re-judging them is `questions-reverify`'s
to do rather than a migration's.

Revision: 7f3a1c95e2d8
Parent:   b4d29e0af715
Created:  2026-09-15
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "7f3a1c95e2d8"
down_revision: str | None = "b4d29e0af715"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NAME = "questions_rejected_reason_valid"

#: Every gate that can refuse a question. NULL is a person rejecting one from
#: the page, which names no gate.
_WAS = (
    "rejected_reason IS NULL OR rejected_reason IN ('malformed', "
    "'not_recoverable', 'duplicate', 'answerable_after_all', 'source_changed')"
)

_NOW = (
    "rejected_reason IS NULL OR rejected_reason IN ('malformed', "
    "'restates_fact', 'unanchored', 'not_recoverable', 'duplicate', "
    "'answerable_after_all', 'source_changed')"
)


def upgrade() -> None:
    """Applies the change."""
    op.drop_constraint(_NAME, "questions", type_="check")
    op.create_check_constraint(_NAME, "questions", _NOW)


def downgrade() -> None:
    """Takes it back out.

    Any question either new gate rejected is returned to `draft`: the reason
    it carries is not a value the older constraint allows, and a row the
    constraint would refuse is one this migration cannot leave behind.
    """
    op.execute(
        "UPDATE questions SET status = 'draft', rejected_reason = NULL "
        "WHERE rejected_reason IN ('restates_fact', 'unanchored')"
    )
    op.drop_constraint(_NAME, "questions", type_="check")
    op.create_check_constraint(_NAME, "questions", _WAS)
