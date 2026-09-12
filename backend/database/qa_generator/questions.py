"""The questions table."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CHAR,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Index,
    Text,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.qa_generator.base import Base
from database.qa_generator.outcomes import QuestionStatus, one_of

if TYPE_CHECKING:
    from database.qa_generator.question_facts import QuestionFact


class Question(Base):
    """One test question, with its expected answer or expected behaviour.

    Reaches its facts through :class:`QuestionFact` - one for an ordinary
    question, two for a cross-document one - and through them its passages,
    documents and topics. Rejected questions keep their row; only losing
    every source fact removes a question.
    """

    __tablename__ = "questions"
    __table_args__ = (
        CheckConstraint(
            one_of("status", QuestionStatus), name="questions_status_valid"
        ),
        # The dedup and unanswerability gates cosine-search every accepted
        # question. Without this the search is a sequential scan of the
        # table, which is the one query that grows with the deliverable.
        Index(
            "ix_questions_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # Shape, not policy, as on documents.language.
        CheckConstraint(
            "language ~ '^[a-z]{2}$'", name="questions_language_is_iso_639_1"
        ),
        # An unanswerable question is scored on behaviour, not on content.
        CheckConstraint(
            "answerable OR target_answer IS NULL",
            name="questions_unanswerable_has_no_target",
        ),
        {
            "comment": "The test questions themselves. Append-only and never "
            "hard-deleted: rejected questions are the drop-rate evidence in the "
            "coverage report. Deleting every fact a question came from also "
            "deletes the question: a foreign key cannot cascade that direction, "
            "so an AFTER DELETE trigger on question_facts does it."
        },
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    question_text: Mapped[str] = mapped_column(
        Text, comment="The question as it would be put to the chatbot."
    )
    target_answer: Mapped[str | None] = mapped_column(
        Text,
        comment="The expected answer. Must be NULL when answerable is false, "
        "enforced by a CHECK constraint: an unanswerable question is scored on "
        "behaviour, not on content.",
    )
    answerable: Mapped[bool] = mapped_column(
        Boolean,
        comment="Whether the document supports an answer at all. False for the "
        "deliberately unanswerable questions that test whether the chatbot "
        "recognises the limits of its knowledge.",
    )
    difficulty: Mapped[str | None] = mapped_column(
        Text, comment="Difficulty band, used to stratify the review sample."
    )
    language: Mapped[str] = mapped_column(
        CHAR(2),
        comment="ISO 639-1 language the question is written in, constrained to "
        "the shape of a code rather than to a list of them. Deliberately "
        "independent of the source document's language: a German document can "
        "carry English questions.",
    )
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(1024),
        comment="Question embedding, used by the semantic dedup and unanswerability "
        "gates. Those do cosine search over every accepted question, which the HNSW "
        "index above serves. 1024 dimensions because that is the width of "
        "EMBEDDING_MODEL, which is also the tokenizer chunking sizes a passage by - "
        "one name in .env for both, so a passage cannot be sized against one model "
        "and embedded by another. Changing that model means a migration here and "
        "re-embedding every row; pgvector indexes HNSW up to 2000 dimensions, so a "
        "wider model than that must be truncated, which Matryoshka training "
        "supports.",
    )
    status: Mapped[str] = mapped_column(
        Text,
        server_default=QuestionStatus.DRAFT,
        comment="draft | accepted | rejected, enforced by a CHECK constraint. "
        "Rejected questions are never deleted: they are the drop-rate evidence in "
        "the coverage report.",
    )
    rejected_reason: Mapped[str | None] = mapped_column(
        Text, comment="Which gate rejected the question."
    )
    holdout_set_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid,
        comment="The hold-out draw this question belongs to; one UUID per release. "
        "Non-NULL means the question is withheld from the open export and kept "
        "encrypted, so there is a test the development team has not tuned against. "
        "NULL means it is in the open set.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="When the question was generated.",
    )
    status_changed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), comment="When status last changed."
    )

    fact_links: Mapped[list[QuestionFact]] = relationship(
        back_populates="question", cascade="all, delete-orphan", passive_deletes=True
    )
