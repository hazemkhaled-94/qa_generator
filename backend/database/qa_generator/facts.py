"""The facts table."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
    false,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.qa_generator.base import Base
from database.qa_generator.outcomes import Rejection, one_of

if TYPE_CHECKING:
    from database.qa_generator.passages import Passage
    from database.qa_generator.question_facts import QuestionFact


class Fact(Base):
    """One atomic statement drawn from one passage, with its source span."""

    __tablename__ = "facts"
    __table_args__ = (
        Index(
            "ix_facts_statement_trgm",
            "statement",
            postgresql_using="gin",
            postgresql_ops={"statement": "gin_trgm_ops"},
        ),
        Index(
            "ix_facts_evidence_trgm",
            "evidence_text",
            postgresql_using="gin",
            postgresql_ops={"evidence_text": "gin_trgm_ops"},
        ),
        # Groups the quality report; without it every scan reads the table.
        Index("ix_facts_rejection_code", "rejection_code"),
        CheckConstraint(
            "extraction_method IN ('llm', 'deterministic')",
            name="facts_extraction_method_valid",
        ),
        CheckConstraint(
            "evidence_end >= evidence_start", name="facts_evidence_span_ordered"
        ),
        CheckConstraint(
            f"rejection_code IS NULL OR {one_of('rejection_code', Rejection)}",
            name="facts_rejection_code_valid",
        ),
        CheckConstraint(
            "validated = (rejection_code IS NULL)", name="facts_verdict_agrees"
        ),
        {
            "comment": "One atomic statement drawn from one passage, with the "
            "source span supporting it. The unit a question is generated from "
            "and scored against."
        },
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    passage_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("passages.id", ondelete="CASCADE"),
        index=True,
        comment="The passage this fact was drawn from, and the route to both its "
        "document and its topics.",
    )
    statement: Mapped[str] = mapped_column(
        Text, comment="The fact as a single self-contained sentence."
    )
    evidence_text: Mapped[str] = mapped_column(
        Text,
        comment="The cited sentences of the passage, resolved from "
        "evidence_sentence_ids.",
    )
    evidence_sentence_ids: Mapped[list[int]] = mapped_column(
        ARRAY(Integer),
        comment="Which of passages.sentences the claim was drawn from. What the "
        "extractor chooses; the span below is resolved from it, so a citation is "
        "exact by construction rather than by searching for a quote.",
    )
    evidence_start: Mapped[int] = mapped_column(
        Integer,
        comment="Offset of the evidence's first character within passages.text.",
    )
    evidence_end: Mapped[int] = mapped_column(
        Integer, comment="Offset one past the evidence's last character."
    )
    extraction_method: Mapped[str] = mapped_column(Text, comment="llm | deterministic.")
    extraction_model: Mapped[str | None] = mapped_column(
        Text, comment="The model that wrote the statement, NULL if none did."
    )
    prompt_version: Mapped[str | None] = mapped_column(
        Text,
        comment="Version of the extraction prompt. The prompt decides what counts "
        "as a fact, so two prompts are two datasets.",
    )
    extraction_temperature: Mapped[float | None] = mapped_column(
        Float, comment="Sampling temperature used."
    )
    spacy_model: Mapped[str | None] = mapped_column(
        Text, comment="The spaCy pipeline that judged this fact."
    )
    spacy_version: Mapped[str | None] = mapped_column(
        Text,
        comment="spaCy version. The parser decides the verdict, so it is "
        "provenance in the same way the prompt version is.",
    )
    statement_predicates: Mapped[int] = mapped_column(
        Integer,
        server_default="0",
        comment="Finite verbs in the statement. One is atomic; more means several "
        "claims in one sentence, none means no claim at all.",
    )
    evidence_predicates: Mapped[int] = mapped_column(
        Integer,
        server_default="0",
        comment="Finite verbs in the cited sentences. Against "
        "statement_predicates, how far the passage was decomposed.",
    )
    units_statement: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        comment="The numbers, proper nouns and predicates the statement asserts, "
        "which are the parts of it a paraphrase cannot change.",
    )
    units_added: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        comment="Units of the statement that occur nowhere in the cited sentences, "
        "in any form. Empty on every accepted fact, which is what says nothing was "
        "invented.",
    )
    unresolved_references: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        comment="Pronouns leaving the statement dependent on its context. Empty on "
        "every accepted fact, which is what makes a question from it answerable "
        "alone.",
    )
    validated: Mapped[bool] = mapped_column(
        Boolean, server_default=false(), comment="Whether every check passed."
    )
    rejection_code: Mapped[str | None] = mapped_column(
        Text, comment="Which check failed, NULL when none did."
    )
    validation_error: Mapped[str | None] = mapped_column(
        Text, comment="That failure in words."
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="When the fact was extracted.",
    )

    passage: Mapped[Passage] = relationship(back_populates="facts")
    question_links: Mapped[list[QuestionFact]] = relationship(
        back_populates="fact", cascade="all, delete-orphan", passive_deletes=True
    )
