"""Queue question generation over topics, and constrain what a question says.

Question generation's unit of work is one topic: the facts it writes from are
the ones whose passage has that topic as its strongest, and a group spanning
two documents is what makes a cross-document question. So the queue goes on
`topics`, in the same three columns every other stage owns.

`topics` already held one queue - a row with a NULL `topic_index` is a request
to refit rather than a topic - so this second one is meaningful on only half
the table. The partial index below covers the queue states, and the stage's
own repository carries `topic_index IS NOT NULL`; without that, `start` would
queue the fit request as though it were a subject.

A fit replaces every row in `topics`, so a refit returns every topic to `new`
and the questions are generated again. That is deliberate: the questions
themselves belong to their facts and survive it, and selection skips the facts
that already carry an accepted question, so the second run writes only what
the first did not.

The three constraints on `questions` are the vocabulary the models declare and
autogenerate does not compare, as in 8c1e4f2a7b03. The table is empty until
this stage runs, so none of them needs a backfill.

Revision: b4d29e0af715
Parent:   8c1e4f2a7b03
Created:  2026-09-14
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b4d29e0af715"
down_revision: str | None = "8c1e4f2a7b03"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: The states a topic moves through for question generation: the four every
#: stage shares, plus this one's own finished value.
_QUESTION_STATUS = (
    "question_status IN ('new', 'pending', 'in_progress', 'failed', 'generated')"
)

#: What the partial queue index covers: the rows a worker still has to act on.
_QUEUED = "question_status IN ('pending', 'in_progress')"

#: Read off how far a question's facts are spread, so it is measurable.
_DIFFICULTY = (
    "difficulty IS NULL OR difficulty IN ('single_passage', 'cross_passage', "
    "'cross_document')"
)

#: Every gate that can refuse a question. NULL is a person rejecting one from
#: the page, which names no gate.
_REJECTED = (
    "rejected_reason IS NULL OR rejected_reason IN ('malformed', "
    "'not_recoverable', 'duplicate', 'answerable_after_all', 'source_changed')"
)

#: The table now carries two queues rather than one, and says so.
_TOPICS_WAS = (
    "Clusters over one language's own vocabulary, factorised out of a tf-idf "
    "weighted term matrix. passage_topics carries the membership; facts and "
    "questions reach their topics by joining through their passage. Also the "
    "queue: a row with a NULL topic_index is a request to refit."
)

_TOPICS_NOW = (
    "Clusters over one language's own vocabulary, factorised out of a tf-idf "
    "weighted term matrix. passage_topics carries the membership; facts and "
    "questions reach their topics by joining through their passage. Also two "
    "queues: a row with a NULL topic_index is a request to refit, and a row "
    "that is a topic is a unit of question generation."
)


def upgrade() -> None:
    """Applies the change."""
    op.add_column(
        "topics",
        sa.Column(
            "question_status",
            sa.Text(),
            nullable=False,
            server_default="new",
            comment=(
                "new | pending | in_progress | failed | generated, enforced by a "
                "CHECK constraint. Question generation's own queue, over topics "
                "rather than over fits: the unit of work is one topic's facts. "
                "Meaningless on a fit request, which is why that queue carries "
                "topic_index IS NOT NULL. A fit replaces every row here, so a "
                "refit returns every topic to `new`; the questions themselves are "
                "not a topic's and survive it."
            ),
        ),
    )
    op.add_column(
        "topics",
        sa.Column(
            "question_error",
            sa.Text(),
            nullable=True,
            comment="Why question generation failed on this topic.",
        ),
    )
    op.add_column(
        "topics",
        sa.Column(
            "question_claimed_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment=(
                "When a worker claimed this topic for question generation, NULL "
                "when none holds it."
            ),
        ),
    )
    op.create_index(
        "ix_topics_question_queue",
        "topics",
        ["question_status"],
        postgresql_where=sa.text(_QUEUED),
    )
    op.create_check_constraint(
        "topics_question_status_valid", "topics", _QUESTION_STATUS
    )
    op.create_table_comment("topics", _TOPICS_NOW, existing_comment=_TOPICS_WAS)

    op.create_index("ix_questions_status", "questions", ["status"])
    op.create_check_constraint("questions_difficulty_valid", "questions", _DIFFICULTY)
    op.create_check_constraint(
        "questions_rejected_reason_valid", "questions", _REJECTED
    )
    op.alter_column(
        "questions",
        "difficulty",
        existing_type=sa.Text(),
        comment=(
            "single_passage | cross_passage | cross_document, enforced by a CHECK "
            "constraint. Read off how far the facts the question was written from "
            "are spread rather than judged, so it is measurable and two readers "
            "cannot disagree about it. Used to stratify the review sample."
        ),
        existing_comment="Difficulty band, used to stratify the review sample.",
    )
    op.alter_column(
        "questions",
        "rejected_reason",
        existing_type=sa.Text(),
        comment=(
            "Which gate rejected the question, enforced by a CHECK constraint. "
            "NULL on a question a person rejected from the page, which names no "
            "gate."
        ),
        existing_comment="Which gate rejected the question.",
    )


def downgrade() -> None:
    """Takes it back out."""
    op.alter_column(
        "questions",
        "rejected_reason",
        existing_type=sa.Text(),
        comment="Which gate rejected the question.",
    )
    op.alter_column(
        "questions",
        "difficulty",
        existing_type=sa.Text(),
        comment="Difficulty band, used to stratify the review sample.",
    )
    op.drop_constraint("questions_rejected_reason_valid", "questions", type_="check")
    op.drop_constraint("questions_difficulty_valid", "questions", type_="check")
    op.drop_index("ix_questions_status", table_name="questions")

    op.create_table_comment("topics", _TOPICS_WAS, existing_comment=_TOPICS_NOW)
    op.drop_constraint("topics_question_status_valid", "topics", type_="check")
    op.drop_index("ix_topics_question_queue", table_name="topics")
    op.drop_column("topics", "question_claimed_at")
    op.drop_column("topics", "question_error")
    op.drop_column("topics", "question_status")
