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
    ForeignKey,
    Index,
    Integer,
    Text,
    Uuid,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.qa_generator.base import Base
from database.qa_generator.outcomes import (
    Difficulty,
    DocumentScope,
    PassageScope,
    QuestionRejection,
    QuestionStatus,
    TopicScope,
    one_of,
)

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
        # The dedup probe filters to the accepted questions before it orders
        # by distance, and the quality report groups on the same column.
        Index("ix_questions_status", "status"),
        # Shape, not policy, as on documents.language.
        CheckConstraint(
            "language ~ '^[a-z]{2}$'", name="questions_language_is_iso_639_1"
        ),
        # An unanswerable question is scored on behaviour, not on content.
        CheckConstraint(
            "answerable OR target_answer IS NULL",
            name="questions_unanswerable_has_no_target",
        ),
        CheckConstraint(
            f"difficulty IS NULL OR {one_of('difficulty', Difficulty)}",
            name="questions_difficulty_valid",
        ),
        *(
            CheckConstraint(
                f"{column} IS NULL OR {one_of(column, values)}",
                name=f"questions_{column}_valid",
            )
            for column, values in (
                ("passage_scope", PassageScope),
                ("document_scope", DocumentScope),
                ("topic_scope", TopicScope),
            )
        ),
        # A thread runs 1, 2, 3: the root is 1 and a follow-up is one past
        # whatever it follows. The two halves cannot disagree.
        CheckConstraint(
            "(follows_id IS NULL) = (thread_position = 1)",
            name="questions_thread_position_agrees",
        ),
        CheckConstraint("thread_position >= 1", name="questions_thread_position_valid"),
        # Reached on every read of a thread, and by the cascade a deleted
        # parent runs.
        Index("ix_questions_follows_id", "follows_id"),
        # A person rejecting a question from the page names no gate, so a
        # rejected question may carry no reason; a reason that is not a gate
        # is what this refuses.
        CheckConstraint(
            f"rejected_reason IS NULL OR {one_of('rejected_reason', QuestionRejection)}",
            name="questions_rejected_reason_valid",
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
        Text,
        comment="easy | medium | hard, enforced by a CHECK constraint. Derived "
        "rather than judged: the three scope columns, a long answer and being a "
        "follow-up are each worth a point, and the band is the total. Two readers "
        "cannot disagree about it, and a review sample can stratify on it.",
    )
    passage_scope: Mapped[str | None] = mapped_column(
        Text,
        comment="single_passage | multi_passage, enforced by a CHECK constraint. "
        "How many distinct passages the facts this question cites come from.",
    )
    document_scope: Mapped[str | None] = mapped_column(
        Text,
        comment="single_document | cross_document, enforced by a CHECK "
        "constraint. The scope a retriever cannot fake: a cross-document "
        "question has no one chunk holding its answer.",
    )
    topic_scope: Mapped[str | None] = mapped_column(
        Text,
        comment="single_topic | multi_topic, enforced by a CHECK constraint. "
        "Read off the dominant topic of each cited fact's passage. A passage "
        "usually sits in several topics above the weight floor, so a question "
        "bridging two subjects is one about the material rather than about a "
        "section of it.",
    )
    answer_chars: Mapped[int | None] = mapped_column(
        Integer,
        comment="Length of target_answer, stored rather than measured on read so "
        "the difficulty band it fed can be recomputed and checked. NULL on an "
        "unanswerable question, which has no answer to measure.",
    )
    follows_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("questions.id", ondelete="CASCADE"),
        comment="The question this one follows, for a multi-turn thread. NULL on "
        "a root question. A follow-up may rely on its parent for context - that "
        "is what it is for - so it is deliberately not self-contained and the "
        "phrasing gate is not applied to one. Cascades: a follow-up whose parent "
        "is gone has no thread to be read in.",
    )
    thread_position: Mapped[int] = mapped_column(
        Integer,
        server_default=text("1"),
        comment="Where this question sits in its thread: 1 is a root, 2 is the "
        "first follow-up. QUESTIONS_MAX_FOLLOWUPS caps how far it goes.",
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
        Text,
        comment="Which gate rejected the question, enforced by a CHECK "
        "constraint. NULL on a question a person rejected from the page, which "
        "names no gate.",
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
