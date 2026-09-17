"""A cap on atomic, two gates for the unanswerable, and a balanced release.

Four changes, all of them vocabulary or a column; nothing existing moves.

`facts.rejection_code` gains `over_cap`. EXTRACTION_MIN_OTHER_SHARE is a
floor on the share of a passage's facts that are not atomic, which works out
to a cap on the atomic ones, and the facts above it are refused rather than
dropped - the same bargain every other check here makes. Read without a cap
this corpus gave 18.4 atomic facts a passage against two digests, so atomic
was 91.9% of everything extracted and an author list became fifty facts.

`questions.rejected_reason` gains `off_topic` and `answerable_elsewhere`,
the two gates an unanswerable question now faces. It had none of its own:
the only thing that could refuse one was the verifier finding an answer in
its own passages after all, and they passed at 87% where answerable
questions passed at 30%. They were not better, they were less tested.

`questions.release_id` says which questions make up the balanced set.
Accepting a question says it is sound; what the SET looks like is a
different problem, and one no gate can answer - every gate did its job and
the accepted set still came out 45% unanswerable and 96.5% easy.

And a GIN index on `passages.lemmas`, which is what the corpus-wide probe
behind `answerable_elsewhere` searches: a question claiming the material is
silent about something is a claim about every passage, not about the two it
cites.

Revision: f1b9d6c30a47
Parent:   c4a7f2e918bd
Created:  2026-09-17
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from database.qa_generator import QuestionRejection, Rejection

revision: str = "f1b9d6c30a47"
down_revision: str | None = "c4a7f2e918bd"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_FACT_CODES = "facts_rejection_code_valid"
_QUESTION_CODES = "questions_rejected_reason_valid"
_LEMMAS = "ix_passages_lemmas_gin"

#: The codes each column held before this revision, for the way back down.
_FACT_CODES_BEFORE = (
    "evidence_absent",
    "copied",
    "asserts_nothing",
    "not_atomic",
    "unsupported_addition",
    "unresolved_reference",
    "not_condensed",
    "not_listed",
    "not_bridging",
)
_QUESTION_CODES_BEFORE = (
    "malformed",
    "compound",
    "answer_too_short",
    "answer_too_long",
    "wrong_form",
    "leaks_source",
    "unanchored",
    "not_recoverable",
    "duplicate",
    "answerable_after_all",
    "source_changed",
)


def _listed(column: str, values: Sequence[str]) -> str:
    """The CHECK expression allowing NULL or one of these values."""
    joined = ", ".join(f"'{value}'" for value in values)
    return f"{column} IS NULL OR {column} IN ({joined})"


def upgrade() -> None:
    """Widens the two vocabularies, adds the column and the index."""
    op.drop_constraint(_FACT_CODES, "facts", type_="check")
    op.create_check_constraint(
        _FACT_CODES, "facts", _listed("rejection_code", tuple(Rejection))
    )

    op.drop_constraint(_QUESTION_CODES, "questions", type_="check")
    op.create_check_constraint(
        _QUESTION_CODES,
        "questions",
        _listed("rejected_reason", tuple(QuestionRejection)),
    )

    op.add_column(
        "questions",
        sa.Column(
            "release_id",
            sa.Uuid(),
            nullable=True,
            comment="The balanced release this question was chosen into; one "
            "UUID per draw. Accepting a question says it is sound, which is a "
            "different question from what the SET should look like: `make "
            "questions-balance` fills an even quota of kinds and difficulty "
            "bands out of everything accepted, and writes this. NULL means "
            "accepted but not drawn - kept, queryable, and available to the "
            "next draw.",
        ),
    )
    op.create_index("ix_questions_release_id", "questions", ["release_id"])

    # postgresql_using, because the default btree cannot answer the `&&` the
    # probe filters with. It is the array of content lemmas chunking already
    # wrote, so this indexes a column that was there and costs no backfill.
    op.create_index(_LEMMAS, "passages", ["lemmas"], postgresql_using="gin")


def downgrade() -> None:
    """Narrows both vocabularies again, after clearing what they would refuse."""
    op.drop_index(_LEMMAS, table_name="passages")
    op.drop_index("ix_questions_release_id", table_name="questions")
    op.drop_column("questions", "release_id")

    # A row carrying a code the older constraint does not allow would make
    # it unaddable. The verdict is kept and the code is cleared to the one
    # that still exists for "a check refused this".
    op.execute(
        "UPDATE facts SET rejection_code = 'asserts_nothing' "
        f"WHERE rejection_code = '{Rejection.OVER_CAP}'"
    )
    op.execute(
        "UPDATE questions SET rejected_reason = 'not_recoverable' "
        f"WHERE rejected_reason IN ('{QuestionRejection.OFF_TOPIC}', "
        f"'{QuestionRejection.ANSWERABLE_ELSEWHERE}')"
    )

    op.drop_constraint(_QUESTION_CODES, "questions", type_="check")
    op.create_check_constraint(
        _QUESTION_CODES, "questions", _listed("rejected_reason", _QUESTION_CODES_BEFORE)
    )
    op.drop_constraint(_FACT_CODES, "facts", type_="check")
    op.create_check_constraint(
        _FACT_CODES, "facts", _listed("rejection_code", _FACT_CODES_BEFORE)
    )
