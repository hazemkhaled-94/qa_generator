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
    Index,
    Integer,
    Text,
    false,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.qa_generator.base import Base
from database.qa_generator.outcomes import FactKind, Rejection, one_of

if TYPE_CHECKING:
    from database.qa_generator.fact_passages import FactPassage
    from database.qa_generator.question_facts import QuestionFact


class Fact(Base):
    """One statement drawn from the passages listed in fact_passages."""

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
        # Filters the listing and groups the quality report by kind.
        Index("ix_facts_kind", "kind"),
        CheckConstraint(
            "extraction_method IN ('llm', 'deterministic')",
            name="facts_extraction_method_valid",
        ),
        CheckConstraint(one_of("kind", FactKind), name="facts_kind_valid"),
        CheckConstraint(
            f"rejection_code IS NULL OR {one_of('rejection_code', Rejection)}",
            name="facts_rejection_code_valid",
        ),
        CheckConstraint(
            "validated = (rejection_code IS NULL)", name="facts_verdict_agrees"
        ),
        {
            "comment": "One statement and the verdicts the checks reached on it. "
            "Which passages it rests on, and where in each, is fact_passages. The "
            "unit a question is generated from and scored against."
        },
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    kind: Mapped[str] = mapped_column(
        Text,
        server_default=FactKind.ATOMIC,
        comment="atomic | summary | outline | bridge. Decides which checks the "
        "statement is held to and how many passages it rests on: atomic, summary "
        "and outline rest on one, a bridge on two or more.",
    )
    statement: Mapped[str] = mapped_column(
        Text,
        comment="The fact as written. One self-contained sentence for an atomic "
        "fact or a bridge, a short paragraph for a summary, and newline-separated "
        "`- ` bullets for an outline.",
    )
    evidence_text: Mapped[str] = mapped_column(
        Text,
        comment="Every span in fact_passages, resolved and joined in position "
        "order. The denormalised copy the search index reads; the spans "
        "themselves are the record.",
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

    passage_links: Mapped[list[FactPassage]] = relationship(
        back_populates="fact", cascade="all, delete-orphan", passive_deletes=True
    )
    question_links: Mapped[list[QuestionFact]] = relationship(
        back_populates="fact", cascade="all, delete-orphan", passive_deletes=True
    )
