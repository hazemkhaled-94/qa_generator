"""A question says what it asks of a reader, not only where the answer is.

`difficulty` measures how far an answer is spread - over two passages, two
documents, two subjects - which is how hard it is to FIND. It says nothing
about what has to be done once it is found, and a question spanning two
documents can still be a bare lookup with both of them in hand.

`cognitive_level` is the other axis. Declared by the question's type rather
than judged per row, the way difficulty is read off the scopes: a type says
what it asks for, and what it asks for decides this. Nobody holds an
opinion about an individual question.

  recall      factoid, entity
  understand  definition, enumeration
  apply       condition, procedure, application
  analyse     reason, consequence, comparison, aggregation, temporal,
              implication

The two types at the end of that list are new, and they are what makes the
column worth having. Every other type's answer is STATED in the passages -
the recoverability gate refuses one whose answer is not - so the level of a
retrieval question is a label on a lookup. `implication` and `application`
are derived: the premises are in the material and the conclusion is not, so
answering means reasoning rather than finding. They face a gate of their
own, as `aggregation` already did, because recoverability asks the wrong
question of them.

NULL on every row written before this. Those questions were written by
types that now declare a level, so it could be backfilled from
`question_type` - and deliberately is not. The level is a property of the
type at the time a question was written, and a type's level can change; a
column filled in afterwards would say the row was judged when nothing had
judged it.

Revision: a17f4c93e2b8
Parent:   d3c81f4a2e57
Created:  2026-09-19
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from database.qa_generator import CognitiveLevel, one_of

revision: str = "a17f4c93e2b8"
down_revision: str | None = "d3c81f4a2e57"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_LEVEL = "questions_cognitive_level_valid"
_TYPES = "questions_question_type_valid"

#: The types the column arrives with. Two are new, and a CHECK naming the
#: old eleven would refuse every question written under either.
_TYPES_BEFORE = (
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


def upgrade() -> None:
    """Adds the column and widens the type vocabulary to reach it."""
    op.add_column(
        "questions",
        sa.Column(
            "cognitive_level",
            sa.Text(),
            nullable=True,
            comment="recall | understand | apply | analyse, enforced by a "
            "CHECK constraint. How much the question asks of whoever answers "
            "it, declared by its type rather than judged per row - the same "
            "bargain difficulty makes with the scopes. A different axis from "
            "difficulty, and deliberately not merged with it: difficulty says "
            "how far the answer is spread and so how hard it is to FIND, and "
            "a question spanning two documents can still be a bare lookup "
            "once both are in hand. This says how much has to be done with "
            "what was found.",
        ),
    )
    op.create_check_constraint(
        _LEVEL,
        "questions",
        f"cognitive_level IS NULL OR {one_of('cognitive_level', CognitiveLevel)}",
    )

    op.drop_constraint(_TYPES, "questions", type_="check")
    op.create_check_constraint(
        _TYPES,
        "questions",
        "question_type IS NULL OR question_type IN ("
        + ", ".join(f"'{one}'" for one in (*_TYPES_BEFORE, "implication", "application"))
        + ")",
    )


def downgrade() -> None:
    """Drops the column, after clearing the types the old CHECK refuses."""
    op.execute(
        "UPDATE questions SET question_type = NULL, status = 'rejected', "
        "rejected_reason = 'source_changed' "
        "WHERE question_type IN ('implication', 'application')"
    )
    op.drop_constraint(_TYPES, "questions", type_="check")
    op.create_check_constraint(
        _TYPES,
        "questions",
        "question_type IS NULL OR question_type IN ("
        + ", ".join(f"'{one}'" for one in _TYPES_BEFORE)
        + ")",
    )
    op.drop_constraint(_LEVEL, "questions", type_="check")
    op.drop_column("questions", "cognitive_level")
