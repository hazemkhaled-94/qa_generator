"""Three scopes, a difficulty band, and follow-up threads.

`difficulty` held how far a question's evidence was spread - single_passage,
cross_passage, cross_document - which conflated three separate things and
measured none of them as difficulty. It is now a band, and the three things
it conflated are columns of their own:

  passage_scope    single_passage | multi_passage
  document_scope   single_document | cross_document
  topic_scope      single_topic | multi_topic

`topic_scope` is the one that was impossible before. A sample was drawn from
the facts of one claimed topic, so every question was single-topic by
construction; selection now also offers bridge passages, which are passages
whose dominant topic differs from the claimed one but which carry it above
the weight floor. 703 of this corpus's 831 placed passages sit in more than
one topic, so the material was there and unreachable.

`difficulty` is derived from those three, from whether the answer is long,
and from whether the question follows another. Each is worth a point and the
band is the total, so it is reproducible and a reviewer can stratify on it
without holding an opinion.

`answer_chars` is stored rather than measured on read, so the band it fed can
be recomputed and checked against the row.

`follows_id` and `thread_position` carry a multi-turn thread. A follow-up may
rely on its parent for context - `And for urgent requests?` - which is what
it is for, so the phrasing gate is not applied to one and the verifier is
shown the thread when it judges recoverability. The foreign key cascades:
a follow-up whose parent is gone has no thread to be read in.

The scopes are backfilled from the old `difficulty` where it said anything,
because the translation is mechanical. The band is then computed from them.
Both are approximations for rows written before the writer reported which
facts its question used: the old value was wrong on 91 of the first 140 rows,
which is why the writer now reports. Regenerating is what makes them true.

Revision: c41f8b7d2e06
Parent:   7f3a1c95e2d8
Created:  2026-09-15
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c41f8b7d2e06"
down_revision: str | None = "7f3a1c95e2d8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_REASON = "questions_rejected_reason_valid"

_REASON_WAS = (
    "rejected_reason IS NULL OR rejected_reason IN ('malformed', "
    "'restates_fact', 'unanchored', 'not_recoverable', 'duplicate', "
    "'answerable_after_all', 'source_changed')"
)

_REASON_NOW = (
    "rejected_reason IS NULL OR rejected_reason IN ('malformed', "
    "'answer_too_short', 'restates_fact', 'unanchored', 'not_recoverable', "
    "'duplicate', 'answerable_after_all', 'source_changed')"
)

_DIFFICULTY_WAS = (
    "difficulty IS NULL OR difficulty IN ('single_passage', 'cross_passage', "
    "'cross_document')"
)

_DIFFICULTY_NOW = "difficulty IS NULL OR difficulty IN ('easy', 'medium', 'hard')"

#: The three new vocabularies, by the column that holds each.
_SCOPES = {
    "passage_scope": ("single_passage", "multi_passage"),
    "document_scope": ("single_document", "cross_document"),
    "topic_scope": ("single_topic", "multi_topic"),
}

_COMMENTS = {
    "passage_scope": (
        "single_passage | multi_passage, enforced by a CHECK constraint. How "
        "many distinct passages the facts this question cites come from."
    ),
    "document_scope": (
        "single_document | cross_document, enforced by a CHECK constraint. The "
        "scope a retriever cannot fake: a cross-document question has no one "
        "chunk holding its answer."
    ),
    "topic_scope": (
        "single_topic | multi_topic, enforced by a CHECK constraint. Read off "
        "the dominant topic of each cited fact's passage. A passage usually "
        "sits in several topics above the weight floor, so a question bridging "
        "two subjects is one about the material rather than about a section of "
        "it."
    ),
}


def upgrade() -> None:
    """Applies the change."""
    for column, comment in _COMMENTS.items():
        op.add_column(
            "questions", sa.Column(column, sa.Text(), nullable=True, comment=comment)
        )
    op.add_column(
        "questions",
        sa.Column(
            "answer_chars",
            sa.Integer(),
            nullable=True,
            comment=(
                "Length of target_answer, stored rather than measured on read so "
                "the difficulty band it fed can be recomputed and checked. NULL "
                "on an unanswerable question, which has no answer to measure."
            ),
        ),
    )
    op.add_column(
        "questions",
        sa.Column(
            "follows_id",
            sa.BigInteger(),
            nullable=True,
            comment=(
                "The question this one follows, for a multi-turn thread. NULL on "
                "a root question. A follow-up may rely on its parent for context "
                "- that is what it is for - so it is deliberately not "
                "self-contained and the phrasing gate is not applied to one. "
                "Cascades: a follow-up whose parent is gone has no thread to be "
                "read in."
            ),
        ),
    )
    op.add_column(
        "questions",
        sa.Column(
            "thread_position",
            sa.Integer(),
            nullable=False,
            server_default="1",
            comment=(
                "Where this question sits in its thread: 1 is a root, 2 is the "
                "first follow-up. QUESTIONS_MAX_FOLLOWUPS caps how far it goes."
            ),
        ),
    )
    op.create_foreign_key(
        "questions_follows_id_fkey",
        "questions",
        "questions",
        ["follows_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index("ix_questions_follows_id", "questions", ["follows_id"])

    # Before either backfill, and this order is load-bearing: the second
    # UPDATE writes `easy`, which the constraint still in force forbids. A
    # populated table is the only place that shows, which is why there is a
    # test that migrates one.
    op.drop_constraint("questions_difficulty_valid", "questions", type_="check")

    # The mechanical half of the translation. `answer_chars` is measured from
    # the column that is still there, so it is exact.
    op.execute(
        "UPDATE questions SET "
        "  passage_scope = CASE WHEN difficulty = 'single_passage' "
        "                       THEN 'single_passage' ELSE 'multi_passage' END, "
        "  document_scope = CASE WHEN difficulty = 'cross_document' "
        "                        THEN 'cross_document' ELSE 'single_document' END, "
        "  topic_scope = 'single_topic', "
        "  answer_chars = length(target_answer) "
        "WHERE difficulty IS NOT NULL"
    )
    # Then the band, from the scopes and the answer. Three is hard and two
    # is medium, which is the same arithmetic the service applies to a new
    # row: `cross_document` implies `multi_passage`, so three is as high as
    # evidence spread alone can reach.
    op.execute(
        "UPDATE questions SET difficulty = CASE "
        "  WHEN (CASE WHEN passage_scope = 'multi_passage' THEN 1 ELSE 0 END "
        "      + CASE WHEN document_scope = 'cross_document' THEN 1 ELSE 0 END "
        "      + CASE WHEN coalesce(answer_chars, 0) >= 60 THEN 1 ELSE 0 END) >= 3 "
        "    THEN 'hard' "
        "  WHEN (CASE WHEN passage_scope = 'multi_passage' THEN 1 ELSE 0 END "
        "      + CASE WHEN document_scope = 'cross_document' THEN 1 ELSE 0 END "
        "      + CASE WHEN coalesce(answer_chars, 0) >= 60 THEN 1 ELSE 0 END) >= 2 "
        "    THEN 'medium' "
        "  ELSE 'easy' END "
        "WHERE difficulty IS NOT NULL"
    )

    op.create_check_constraint(
        "questions_difficulty_valid", "questions", _DIFFICULTY_NOW
    )
    for column, values in _SCOPES.items():
        allowed = ", ".join(f"'{one}'" for one in values)
        op.create_check_constraint(
            f"questions_{column}_valid",
            "questions",
            f"{column} IS NULL OR {column} IN ({allowed})",
        )
    op.create_check_constraint(
        "questions_thread_position_agrees",
        "questions",
        "(follows_id IS NULL) = (thread_position = 1)",
    )
    op.create_check_constraint(
        "questions_thread_position_valid", "questions", "thread_position >= 1"
    )

    op.drop_constraint(_REASON, "questions", type_="check")
    op.create_check_constraint(_REASON, "questions", _REASON_NOW)

    op.alter_column(
        "questions",
        "difficulty",
        existing_type=sa.Text(),
        comment=(
            "easy | medium | hard, enforced by a CHECK constraint. Derived "
            "rather than judged: the three scope columns, a long answer and "
            "being a follow-up are each worth a point, and the band is the "
            "total. Two readers cannot disagree about it, and a review sample "
            "can stratify on it."
        ),
    )


def downgrade() -> None:
    """Takes it back out.

    A follow-up is deleted rather than orphaned: it may rely on its parent
    for context, so without a thread it is not a question anybody can score.
    Any question the new gate rejected is returned to `draft`, as in
    7f3a1c95e2d8.
    """
    op.execute("DELETE FROM questions WHERE follows_id IS NOT NULL")
    op.execute(
        "UPDATE questions SET status = 'draft', rejected_reason = NULL "
        "WHERE rejected_reason = 'answer_too_short'"
    )
    op.drop_constraint(_REASON, "questions", type_="check")
    op.create_check_constraint(_REASON, "questions", _REASON_WAS)

    op.drop_constraint("questions_thread_position_valid", "questions", type_="check")
    op.drop_constraint("questions_thread_position_agrees", "questions", type_="check")
    for column in _SCOPES:
        op.drop_constraint(f"questions_{column}_valid", "questions", type_="check")

    # Before the UPDATE, and for the same reason it comes first on the way
    # up: the value being written is one the constraint in force forbids.
    op.drop_constraint("questions_difficulty_valid", "questions", type_="check")
    op.execute(
        "UPDATE questions SET difficulty = CASE "
        "  WHEN document_scope = 'cross_document' THEN 'cross_document' "
        "  WHEN passage_scope = 'multi_passage' THEN 'cross_passage' "
        "  ELSE 'single_passage' END "
        "WHERE difficulty IS NOT NULL"
    )
    op.create_check_constraint(
        "questions_difficulty_valid", "questions", _DIFFICULTY_WAS
    )
    op.alter_column(
        "questions",
        "difficulty",
        existing_type=sa.Text(),
        comment=(
            "single_passage | cross_passage | cross_document, enforced by a "
            "CHECK constraint. Read off how far the facts the question was "
            "written from are spread rather than judged, so it is measurable "
            "and two readers cannot disagree about it. Used to stratify the "
            "review sample."
        ),
    )

    op.drop_index("ix_questions_follows_id", table_name="questions")
    op.drop_constraint("questions_follows_id_fkey", "questions", type_="foreignkey")
    for column in ("thread_position", "follows_id", "answer_chars", *_SCOPES):
        op.drop_column("questions", column)
