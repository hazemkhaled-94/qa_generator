"""Question types, answer forms, and the band the plan asked for.

Three columns and four rejection codes.

`question_type` is what the question asks for - factoid, definition, entity,
enumeration, condition, reason, procedure, consequence, comparison,
aggregation or temporal. It is requested before the question is written
rather than read off it afterwards, because the prompt, the answer bounds
and the gates all differ by type.

`answer_form` is the shape of the target answer, declared by the type: a
value carries no verb, a list carries several items, an explanation is
prose. The verb gate now applies to `value` alone; applied to every answer
it made every why, how and procedure question unwritable.

`planned_difficulty` is the band the plan asked for. `difficulty` beside it
stays derived from what the question turned out to cite, so the two together
say how often the plan was met.

Every existing row is deleted. The columns cannot be backfilled honestly -
nothing written before this knew what it was asked for - and regenerating is
what makes them true.

Revision: 3f7b21d9e4a5
Parent:   9a2d6f04c1b8
Created:  2026-09-15
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "3f7b21d9e4a5"
down_revision: str | None = "9a2d6f04c1b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_REASON = "questions_rejected_reason_valid"

_REASON_WAS = (
    "rejected_reason IS NULL OR rejected_reason IN ('malformed', "
    "'answer_too_short', 'leaks_source', 'unanchored', 'not_recoverable', "
    "'duplicate', 'answerable_after_all', 'source_changed')"
)

_REASON_NOW = (
    "rejected_reason IS NULL OR rejected_reason IN ('malformed', "
    "'answer_too_short', 'answer_too_long', 'wrong_form', 'wrong_type', "
    "'leaks_source', 'unanchored', 'not_recoverable', 'duplicate', "
    "'answerable_after_all', 'source_changed')"
)

_TYPES = (
    "factoid",
    "definition",
    "entity",
    "enumeration",
    "condition",
    "reason",
    "procedure",
    "consequence",
    "comparison",
    "aggregation",
    "temporal",
)

_FORMS = ("value", "list", "explanation")

_BANDS = ("easy", "medium", "hard")

_COLUMNS = {
    "question_type": (
        _TYPES,
        (
            "What the question asks for, enforced by a CHECK constraint: factoid, "
            "definition, entity, enumeration, condition, reason, procedure, "
            "consequence, comparison, aggregation or temporal. Requested by the plan "
            "before the question is written, not read off it afterwards. "
            "QUESTIONS_TYPE_MIX sets which are written and in what proportion."
        ),
    ),
    "answer_form": (
        _FORMS,
        (
            "value | list | explanation, enforced by a CHECK constraint. The shape "
            "the target answer takes, declared by the question type. The gates read "
            "it: a value carries no verb, an explanation does, and each form has its "
            "own length bounds in QUESTIONS_ANSWER_CHARS."
        ),
    ),
    "planned_difficulty": (
        _BANDS,
        (
            "The band QUESTIONS_DIFFICULTY_MIX asked for, enforced by a CHECK "
            "constraint. `difficulty` beside it is what the question turned out to "
            "be. The two disagree when the writer cited fewer facts than it was "
            "offered, which is a measurement of the plan rather than a fault in the "
            "row."
        ),
    ),
}


def upgrade() -> None:
    """Applies the change."""
    # Follow-ups first: the cascade would take them anyway, and deleting the
    # parents first leaves nothing for it to run over.
    op.execute("DELETE FROM questions WHERE follows_id IS NOT NULL")
    op.execute("DELETE FROM questions")
    op.execute("UPDATE topics SET question_status = 'new', question_error = NULL")

    for column, (_, comment) in _COLUMNS.items():
        op.add_column(
            "questions", sa.Column(column, sa.Text(), nullable=True, comment=comment)
        )
    for column, (values, _) in _COLUMNS.items():
        allowed = ", ".join(f"'{one}'" for one in values)
        op.create_check_constraint(
            f"questions_{column}_valid",
            "questions",
            f"{column} IS NULL OR {column} IN ({allowed})",
        )
    op.create_index("ix_questions_question_type", "questions", ["question_type"])

    op.drop_constraint(_REASON, "questions", type_="check")
    op.create_check_constraint(_REASON, "questions", _REASON_NOW)


def downgrade() -> None:
    """Takes the three columns back out, and leaves the rows.

    A question written under a type is still a question; what it loses is
    the record of which gates read it. A re-check under the older rules
    rejects whatever no longer holds, which is visible, where deleting the
    rows would not be.
    """
    op.execute(
        "UPDATE questions SET status = 'draft', rejected_reason = NULL "
        "WHERE rejected_reason IN "
        "('answer_too_long', 'wrong_form', 'wrong_type')"
    )
    op.drop_constraint(_REASON, "questions", type_="check")
    op.create_check_constraint(_REASON, "questions", _REASON_WAS)

    op.drop_index("ix_questions_question_type", table_name="questions")
    for column in _COLUMNS:
        op.drop_constraint(f"questions_{column}_valid", "questions", type_="check")
        op.drop_column("questions", column)
