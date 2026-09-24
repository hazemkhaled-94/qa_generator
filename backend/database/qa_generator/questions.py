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
    AnswerForm,
    CognitiveLevel,
    Difficulty,
    DocumentScope,
    PassageScope,
    QuestionRejection,
    QuestionStatus,
    QuestionType,
    ReviewVerdict,
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
        # And so it has nothing to explain either.
        CheckConstraint(
            "answerable OR answer_explanation IS NULL",
            name="questions_unanswerable_has_no_explanation",
        ),
        CheckConstraint(
            f"difficulty IS NULL OR {one_of('difficulty', Difficulty)}",
            name="questions_difficulty_valid",
        ),
        CheckConstraint(
            f"planned_difficulty IS NULL OR {one_of('planned_difficulty', Difficulty)}",
            name="questions_planned_difficulty_valid",
        ),
        CheckConstraint(
            f"question_type IS NULL OR {one_of('question_type', QuestionType)}",
            name="questions_question_type_valid",
        ),
        CheckConstraint(
            f"answer_form IS NULL OR {one_of('answer_form', AnswerForm)}",
            name="questions_answer_form_valid",
        ),
        CheckConstraint(
            f"cognitive_level IS NULL OR {one_of('cognitive_level', CognitiveLevel)}",
            name="questions_cognitive_level_valid",
        ),
        # The listing filters on the type and the quality report groups on it.
        Index("ix_questions_question_type", "question_type"),
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
        CheckConstraint(
            f"reviewed_verdict IS NULL OR {one_of('reviewed_verdict', ReviewVerdict)}",
            name="questions_reviewed_verdict_valid",
        ),
        # One fact about a row, so half of it written is a write that failed.
        CheckConstraint(
            "(reviewed_verdict IS NULL) = (reviewed_at IS NULL)",
            name="questions_reviewed_together",
        ),
        # Partial: NULL for almost every row, and what anyone asks for is
        # the reviewed ones.
        Index(
            "ix_questions_reviewed_verdict",
            "reviewed_verdict",
            postgresql_where=text("reviewed_verdict IS NOT NULL"),
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
    answer_explanation: Mapped[str | None] = mapped_column(
        Text,
        comment="The same answer said at length, for a reader who has not seen "
        "the material. Separate from target_answer because the two are scored "
        "differently: recoverability compares the target whole, and every lemma "
        "added to it is another one the verifier has to reproduce, so an answer "
        "that teaches cannot also be the answer that is matched. NULL on every "
        "question written before this column, and on every unanswerable one.",
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
    question_type: Mapped[str | None] = mapped_column(
        Text,
        comment="What the question asks for, enforced by a CHECK constraint: "
        "factoid, definition, entity, enumeration, condition, reason, procedure, "
        "consequence, comparison, aggregation or temporal. Requested by the plan "
        "before the question is written, not read off it afterwards. "
        "QUESTIONS_TYPE_MIX sets which are written and in what proportion.",
    )
    answer_form: Mapped[str | None] = mapped_column(
        Text,
        comment="value | list | explanation, enforced by a CHECK constraint. The "
        "shape the target answer takes, declared by the question type. The gates "
        "read it: a value carries no verb, an explanation does, and each form has "
        "its own length bounds in QUESTIONS_ANSWER_CHARS.",
    )
    planned_difficulty: Mapped[str | None] = mapped_column(
        Text,
        comment="The band QUESTIONS_DIFFICULTY_MIX asked for, enforced by a CHECK "
        "constraint. `difficulty` beside it is what the question turned out to be. "
        "The two disagree when the writer cited fewer facts than it was offered, "
        "which is a measurement of the plan rather than a fault in the row.",
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
    reviewed_verdict: Mapped[str | None] = mapped_column(
        Text,
        comment="What a person decided about this question: accepted or rejected. "
        "NULL means nobody has looked, which is most questions - a review is a "
        "sample. Separate from `status`, which is what the question IS, and from "
        "`rejected_reason`, which is the gate's; holding all three is what makes "
        "the agreement between a reviewer and the checker a query.",
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        comment="When the verdict above was recorded. NULL exactly when "
        "reviewed_verdict is.",
    )
    cognitive_level: Mapped[str | None] = mapped_column(
        Text,
        comment="recall | understand | apply | analyse, enforced by a CHECK "
        "constraint. How much the question asks of whoever answers it, "
        "declared by its type rather than judged per row - the same bargain "
        "difficulty makes with the scopes. A different axis from difficulty, "
        "and deliberately not merged with it: difficulty says how far the "
        "answer is spread and so how hard it is to FIND, and a question "
        "spanning two documents can still be a bare lookup once both are in "
        "hand. This says how much has to be done with what was found.",
    )
    release_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid,
        index=True,
        comment="The balanced release this question was chosen into; one UUID "
        "per draw. Accepting a question says it is sound, which is a different "
        "question from what the SET should look like: `make questions-balance` "
        "fills an even quota of kinds and difficulty bands out of everything "
        "accepted, and writes this. NULL means accepted but not drawn - kept, "
        "queryable, and available to the next draw.",
    )
    settings_version: Mapped[str | None] = mapped_column(
        Text,
        index=True,
        comment="The configuration this question was written under, as the digest "
        "settings.store computes, or 'environment' when nothing was overridden. "
        "Which model wrote it, what mix the plan aimed for and what bounds the "
        "gates held it to are all in the settings that version names. NULL for a "
        "question written before a setting could be changed without a restart.",
    )
    trace_id: Mapped[str | None] = mapped_column(
        Text,
        index=True,
        comment="The OpenTelemetry trace this question was written and judged "
        "in, as 32 hex characters. What gets from a row to the calls that "
        "produced it: Phoenix resolves a trace from this alone, so "
        "<phoenix>/redirects/traces/<trace_id> opens it without anything "
        "knowing Phoenix's internal ids. NULL for a question written before "
        "this column, and for one written by a process exporting no spans.",
    )
    span_id: Mapped[str | None] = mapped_column(
        Text,
        index=True,
        comment="The span the gates ran in, as 16 hex characters, which is the "
        "span this question's verdict annotations hang off. Stored beside the "
        "trace because it is the more useful of the two: /redirects/spans/ "
        "opens the gate decision itself rather than the whole topic. The span "
        "cannot carry the question id instead - the question has none yet when "
        "the gates run.",
    )
    prompt_version: Mapped[str | None] = mapped_column(
        Text,
        index=True,
        comment="Version of the prompt that wrote this question, as "
        "question_generation.types.PROMPT_VERSION declares it. The prompt decides "
        "what a question IS, so two prompts are two datasets - the same reason "
        "facts.prompt_version exists. Not covered by settings_version, which is a "
        "digest of the stored overrides and does not move when a prompt is "
        "edited. NULL for a question written before the version was recorded.",
    )
    run_id: Mapped[str | None] = mapped_column(
        Text,
        index=True,
        comment="Which RUN produced this row, as `settings.runs.run_id` "
        "named it: RUN_ID where a caller set one, and a uuid otherwise. "
        "settings_version beside it names the CONFIGURATION, and two runs "
        "under one configuration carry the same version - so comparing a "
        "change against the run before it needs this column and cannot be "
        "done with that one. NULL for a row written before runs were named.",
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
