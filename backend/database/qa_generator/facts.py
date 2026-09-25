"""The facts table."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
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
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.qa_generator.base import Base
from database.qa_generator.outcomes import FactKind, Rejection, ReviewVerdict, one_of

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
        # The dedup gate cosine-searches every validated fact once per
        # candidate. Without this the search is a sequential scan, and
        # extraction makes one per fact a passage yields.
        Index(
            "ix_facts_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # Groups the quality report; without it every scan reads the table.
        Index("ix_facts_rejection_code", "rejection_code"),
        # Filters the listing and groups the quality report by kind.
        Index("ix_facts_kind", "kind"),
        # A margin, so it is a share like every other one in this schema.
        CheckConstraint(
            "confidence IS NULL OR confidence BETWEEN 0 AND 1",
            name="facts_confidence_is_a_share",
        ),
        # What a review queue orders by: the rows that survived a check by
        # the least are the ones worth a person's time first.
        Index(
            "ix_facts_confidence",
            "confidence",
            postgresql_where=text("confidence IS NOT NULL"),
        ),
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
        # What an accepted fact means, in the table rather than only in the
        # checker's control flow: a bug there could otherwise store the
        # opposite and question generation would offer it.
        #
        # `llm` only, for both. A statement composed from a grid faces the
        # copy check and nothing else, because it is neither written nor a
        # sentence, so it is accepted carrying whatever the reading found.
        CheckConstraint(
            "extraction_method <> 'llm' OR validated IS FALSE OR "
            "cardinality(units_added) = 0",
            name="facts_accepted_invented_nothing",
        ),
        # Narrower than the one above, because `_self_contained` is applied
        # to a claim and not to a digest: a summary standing in for a whole
        # passage may open with a pronoun and 291 accepted ones do. So this
        # holds for the kinds a question is actually written from, which is
        # what QUESTIONS_FACT_KINDS names, and not for the two that stand
        # in for a passage rather than assert something about it.
        CheckConstraint(
            "extraction_method <> 'llm' OR validated IS FALSE OR "
            "kind NOT IN ('atomic', 'bridge') OR "
            "cardinality(unresolved_references) = 0",
            name="facts_accepted_claim_stands_alone",
        ),
        CheckConstraint(
            f"reviewed_verdict IS NULL OR {one_of('reviewed_verdict', ReviewVerdict)}",
            name="facts_reviewed_verdict_valid",
        ),
        # One fact about a row, so half of it written is a write that failed.
        CheckConstraint(
            "(reviewed_verdict IS NULL) = (reviewed_at IS NULL)",
            name="facts_reviewed_together",
        ),
        # Partial: NULL for almost every row, and what anyone asks for is
        # the reviewed ones.
        Index(
            "ix_facts_reviewed_verdict",
            "reviewed_verdict",
            postgresql_where=text("reviewed_verdict IS NOT NULL"),
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
    settings_version: Mapped[str | None] = mapped_column(
        Text,
        index=True,
        comment="The configuration this fact was extracted under, as the digest "
        "settings.store computes, or 'environment' when nothing was overridden. "
        "The columns above name the model and the prompt; this names the rest of "
        "it, including the shares the checks held this fact to. NULL for a fact "
        "written before a setting could be changed without a restart.",
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
    trace_id: Mapped[str | None] = mapped_column(
        Text,
        index=True,
        comment="The OpenTelemetry trace this fact was extracted and checked in, "
        "as 32 hex characters.",
    )
    span_id: Mapped[str | None] = mapped_column(
        Text,
        index=True,
        comment="The span the extraction ran in, as 16 hex characters. One span "
        "per passage, so every fact from one passage shares it.",
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
        "every accepted atomic fact and bridge, which is what makes a question from "
        "one answerable alone. A summary and an outline are not held to it: they "
        "stand in for a whole passage rather than assert something about it, so "
        "they are checked on length and on inventing nothing and may open with a "
        "pronoun.",
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
    gate_scores: Mapped[list[dict] | None] = mapped_column(
        JSONB,
        comment="One entry per measuring check that read this fact: the check, "
        "the value it measured, the threshold it was read against and the margin "
        "between them. A check that is structural rather than a measurement "
        "records nothing here - a citation resolves or it does not. NULL for a "
        "fact written before this column.",
    )
    confidence: Mapped[float | None] = mapped_column(
        Float,
        comment="The smallest margin in gate_scores, in [0, 1]. How close this "
        "fact came to the verdict that would have refused it, NOT a probability. "
        "NULL when no measuring check read it, which is most facts: nearly every "
        "check here is structural.",
    )
    reviewed_verdict: Mapped[str | None] = mapped_column(
        Text,
        comment="What a person decided about this fact: accepted or rejected. "
        "NULL means nobody has looked, which is most facts - a review is a "
        "sample. Separate from `validated`, which is the checker's and is "
        "rewritten in full by extract-revalidate.",
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        comment="When the verdict above was recorded. NULL exactly when "
        "reviewed_verdict is.",
    )
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(1024),
        comment="Statement embedding, used by the dedup gate. That gate cosine-"
        "searches every validated fact, which the HNSW index above serves. The "
        "same width and the same model as questions.embedding and "
        "passages.embedding: one corpus is measured in one space, so a fact and "
        "the question resting on it are comparable. NULL on a fact extracted "
        "before the column existed, and on one no deployment asked to embed - "
        "EXTRACTION_DUPLICATE_COSINE at 0 turns the gate and the embedding off "
        "together.",
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
