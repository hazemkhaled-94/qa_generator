"""No gate can tell a good question from its own fact.

`restates_fact` is removed. It refused a question whose content lemmas were a
subset of its facts' - the idea being that a question adding nothing a
searcher would have to know is the fact rearranged with a question word.

Measured against 61 real rows it refused 19, and the reason it cannot work is
that the premise is false: for a single atomic fact, a good question IS the
fact minus its answer, put as a question. That is what asking about a fact
means. These two have the same shape and the same overlap with their facts:

  Wie hoch war die Arbeitslosenquote im August 2025?  -> 6,4 Prozent
  Was schüren geopolitische Konflikte?                -> Unsicherheit

The first is as good as a benchmark question gets and the gate refused it.
A second formulation - the question reproducing one whole fact but for the
answer - refused 25 including that one and four more like it. What separates
the two is whether the answer is determinate, and that is not lexical either:
requiring a number, name or date in the answer refused 7 of the 15 accepted
answers, `knapp ein Fünftel` and `Fest- und Stufenzinszertifikate` among
them.

Whether a question is worth asking is a judgement, so the verifier makes it.
`not_recoverable` already carries it - it cannot get a determinate answer back
out of the passages for a vague question - and it correlates: 10 of its 14
rejections had an answer with no number, name or date in it, against 8 of 19
for the gate being removed.

The exact-match check stays. A fact handed back with a question mark on the
end is not a judgement call.

Rows carrying the code go back to `draft` rather than being deleted: the
question was written by a model and only the verdict was wrong, which is the
same bargain `extract-revalidate` makes with a fact.

Revision: 9a2d6f04c1b8
Parent:   5e8c07a3b6f1
Created:  2026-09-15
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "9a2d6f04c1b8"
down_revision: str | None = "5e8c07a3b6f1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NAME = "questions_rejected_reason_valid"

_WAS = (
    "rejected_reason IS NULL OR rejected_reason IN ('malformed', "
    "'answer_too_short', 'leaks_source', 'restates_fact', 'unanchored', "
    "'not_recoverable', 'duplicate', 'answerable_after_all', 'source_changed')"
)

_NOW = (
    "rejected_reason IS NULL OR rejected_reason IN ('malformed', "
    "'answer_too_short', 'leaks_source', 'unanchored', 'not_recoverable', "
    "'duplicate', 'answerable_after_all', 'source_changed')"
)


def upgrade() -> None:
    """Applies the change."""
    op.execute(
        "UPDATE questions SET status = 'draft', rejected_reason = NULL "
        "WHERE rejected_reason = 'restates_fact'"
    )
    op.drop_constraint(_NAME, "questions", type_="check")
    op.create_check_constraint(_NAME, "questions", _NOW)


def downgrade() -> None:
    """Puts the value back in the vocabulary, and nothing else.

    The gate that wrote it is gone, so no row gets it back: which questions
    it would have refused is not something a migration can work out.
    """
    op.drop_constraint(_NAME, "questions", type_="check")
    op.create_check_constraint(_NAME, "questions", _WAS)
