"""A question that asks two things, and the gate that could not tell a kind.

`compound` replaces nothing and catches what made every hard question fail.
Told to use both passages of a wide sample, the writer welded two unrelated
questions together: `Welche Vorschriften gelten für das Kreditgeschäft und wie
hoch ist die Gebühr für das inländische Investmentwesen?`. 17 of the first 19
multi-passage questions had that shape, and every one was rejected downstream
for an answer the verifier could only half recover - which is the right
verdict arrived at for the wrong reason, three model calls too late.

It is read off the tagger: a question using more than one interrogative word
is two questions. A finite-verb count does not separate them - `Welche Arten
von Kryptowerten gelten als reguliert?` carries two verbs and is one question
- and measured over 17 real questions the interrogatives separated 16.

`wrong_type` goes, and the reason is worth recording. The verifier was asked
whether a question was the kind it had been planned as, and answered `true`
for a bare `Wie viele Anlassprüfungen wurden 2025 durchgeführt?` written into
a `reason` slot. It fired zero times over 71 questions while the kind was
plainly wrong on six of the 27 accepted. A yes-or-no judgement from a model
this size is worth nothing, which is what the phrasing judgement had already
shown. What survives is structural: a kind declares an answer form, and
`wrong_form` reads the answer against it.

Any question rejected under the old code returns to `draft`.

Revision: 6b4e91a0c37d
Parent:   6b1e93f7c204
Created:  2026-09-16
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "6b4e91a0c37d"
down_revision: str | None = "6b1e93f7c204"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NAME = "questions_rejected_reason_valid"

_WAS = (
    "rejected_reason IS NULL OR rejected_reason IN ('malformed', "
    "'answer_too_short', 'answer_too_long', 'wrong_form', 'wrong_type', "
    "'leaks_source', 'unanchored', 'not_recoverable', 'duplicate', "
    "'answerable_after_all', 'source_changed')"
)

_NOW = (
    "rejected_reason IS NULL OR rejected_reason IN ('malformed', 'compound', "
    "'answer_too_short', 'answer_too_long', 'wrong_form', "
    "'leaks_source', 'unanchored', 'not_recoverable', 'duplicate', "
    "'answerable_after_all', 'source_changed')"
)


def upgrade() -> None:
    """Applies the change."""
    op.execute(
        "UPDATE questions SET status = 'draft', rejected_reason = NULL "
        "WHERE rejected_reason = 'wrong_type'"
    )
    op.drop_constraint(_NAME, "questions", type_="check")
    op.create_check_constraint(_NAME, "questions", _NOW)


def downgrade() -> None:
    """Takes it back out."""
    op.execute(
        "UPDATE questions SET status = 'draft', rejected_reason = NULL "
        "WHERE rejected_reason = 'compound'"
    )
    op.drop_constraint(_NAME, "questions", type_="check")
    op.create_check_constraint(_NAME, "questions", _WAS)
