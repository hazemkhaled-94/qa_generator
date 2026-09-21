"""An answer that is matched and an answer that is read are two answers.

`target_answer` was doing two jobs that pull against each other. It is the
key a chatbot is scored against, so it has to be short and exact enough to
be matched; and it was the only place a question said what its answer
MEANS, so it had to be long enough to be worth reading. It cannot be both.

The gate is where the two meet. Recoverability compares a `list` or an
`explanation` by how much of it a second model, shown only the cited
passages, independently wrote back - and the denominator is the target's
own content lemmas. Every lemma added to an answer is another one the
verifier has to reproduce, so an answer written to teach somebody is an
answer the gate refuses. Measured over the run this column was added for:
explanations sit at a median of 157 characters against a ceiling of 600,
and `not_recoverable` was already a third of every rejection. Lengthening
the target would have bought worse prose and a lower acceptance rate.

So the target stays as it is and `answer_explanation` carries the reading.
It faces a different gate, because it is a different claim: recoverability
asks whether the passages STATE the answer, and an explanation is refused
instead for asserting a number or a name the passages do not carry. That
is extraction's `unsupported_addition` check pointed the other way, which
is the same bargain `aggregation` already makes.

NULL on every row written before this, and deliberately not backfilled: a
question written without an explanation has no explanation, and copying
the target into the column would say a reader was given something nobody
wrote for them. NULL on every unanswerable question too, by CHECK - one
with no answer has nothing to explain, the same reason it carries no
target.

Three rejection codes arrive with it. `explanation_unusable` is the gate
above. `wrong_type` returns, structural this time: the judgement a model
was asked for fired zero times over 71 questions and was deleted, but 85%
of accepted `entity` questions named no agent, which a rule settles.
`asks_nothing_new` and `off_thread` are the two ways a follow-up fails to
be one.

Revision: 5b2e9c74a1f6
Parent:   3c91a4e17b02
Created:  2026-09-21
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from database.qa_generator import QuestionRejection, one_of

revision: str = "5b2e9c74a1f6"
down_revision: str | None = "3c91a4e17b02"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_REASONS = "questions_rejected_reason_valid"
_NO_EXPLANATION = "questions_unanswerable_has_no_explanation"

#: The codes the column arrives with. A CHECK naming only these would
#: refuse every question the new gates reject, so the upgrade widens it and
#: the downgrade clears what the narrower one cannot hold.
_ARRIVING = (
    "wrong_type",
    "explanation_unusable",
    "asks_nothing_new",
    "off_thread",
)


def upgrade() -> None:
    """Adds the column, its CHECK, and the four new rejection codes."""
    op.add_column(
        "questions",
        sa.Column(
            "answer_explanation",
            sa.Text(),
            nullable=True,
            comment="The same answer said at length, for a reader who has not "
            "seen the material. Separate from target_answer because the two "
            "are scored differently: recoverability compares the target whole, "
            "and every lemma added to it is another one the verifier has to "
            "reproduce, so an answer that teaches cannot also be the answer "
            "that is matched. NULL on every question written before this "
            "column, and on every unanswerable one.",
        ),
    )
    op.create_check_constraint(
        _NO_EXPLANATION,
        "questions",
        "answerable OR answer_explanation IS NULL",
    )

    op.drop_constraint(_REASONS, "questions", type_="check")
    op.create_check_constraint(
        _REASONS,
        "questions",
        f"rejected_reason IS NULL OR {one_of('rejected_reason', QuestionRejection)}",
    )


def downgrade() -> None:
    """Drops the column, after clearing the codes the old CHECK refuses."""
    op.execute(
        "UPDATE questions SET rejected_reason = 'source_changed' "
        "WHERE rejected_reason IN (" + ", ".join(f"'{one}'" for one in _ARRIVING) + ")"
    )
    op.drop_constraint(_REASONS, "questions", type_="check")
    op.create_check_constraint(
        _REASONS,
        "questions",
        "rejected_reason IS NULL OR rejected_reason IN ("
        + ", ".join(
            f"'{one}'" for one in QuestionRejection if str(one) not in _ARRIVING
        )
        + ")",
    )
    op.drop_constraint(_NO_EXPLANATION, "questions", type_="check")
    op.drop_column("questions", "answer_explanation")
