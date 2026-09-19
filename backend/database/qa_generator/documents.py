"""The documents table."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    CHAR,
    CheckConstraint,
    DateTime,
    Float,
    Index,
    Integer,
    Text,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.qa_generator.base import Base
from database.qa_generator.status import Status, check, queued

if TYPE_CHECKING:
    from database.qa_generator.ingest_events import IngestEvent
    from database.qa_generator.passages import Passage


class Document(Base):
    """One stored source document.

    Ingestion writes sha256, mime_type, page_count and char_count; parsing
    writes the content columns. Each stage owns its own status, error and
    claim columns. Filename, size and upload time are in :class:`IngestEvent`.
    """

    __tablename__ = "documents"
    __table_args__ = (
        # Partial: the queue is a shrinking fraction of the table, and this is
        # the only predicate a worker selects on.
        Index(
            "ix_documents_parse_queue",
            "sha256",
            postgresql_where=text(queued("parse_status")),
        ),
        Index(
            "ix_documents_chunk_queue",
            "sha256",
            postgresql_where=text(queued("chunk_status")),
        ),
        CheckConstraint(
            check("parse_status", Status.PARSED), name="documents_parse_status_valid"
        ),
        CheckConstraint(
            check("chunk_status", Status.CHUNKED), name="documents_chunk_status_valid"
        ),
        # Shape, not policy: a constraint listing PARSING_LANGUAGES would need a
        # migration every time that setting changed.
        CheckConstraint(
            "language ~ '^[a-z]{2}$'", name="documents_language_is_iso_639_1"
        ),
        CheckConstraint(
            "parse_confidence BETWEEN 0 AND 1 AND parse_confidence_low BETWEEN 0 AND 1",
            name="documents_parse_confidence_is_a_probability",
        ),
        {
            "comment": "One stored source document. Everything about the act of "
            "uploading it lives in ingest_events."
        },
    )

    sha256: Mapped[str] = mapped_column(
        CHAR(64),
        primary_key=True,
        comment="SHA-256 of the raw file bytes, and the object key in both "
        "buckets: documents/{sha[0:2]}/{sha[2:4]}/{sha}.{ext} and "
        "parsed/.../{sha}.json. Both keys are derived in code, never stored.",
    )
    mime_type: Mapped[str] = mapped_column(
        Text, comment="Detected from the leading bytes at upload."
    )
    page_count: Mapped[int | None] = mapped_column(
        Integer, comment="Pages, counted by PyMuPDF at upload."
    )
    char_count: Mapped[int | None] = mapped_column(
        Integer, comment="Extractable characters, counted by PyMuPDF at upload."
    )
    language: Mapped[str | None] = mapped_column(
        CHAR(2),
        comment="ISO 639-1, NULL until parsing detects it. The fallback "
        "pipeline for a passage of this document too short to tell its own "
        "language; `passages.language` is what selects one where it has it.",
    )
    title: Mapped[str | None] = mapped_column(
        Text, comment="Document title, written by parsing."
    )
    content_sha256: Mapped[str | None] = mapped_column(
        CHAR(64),
        comment="SHA-256 of the body text with whitespace collapsed and case "
        'folded. Answers "do we already have this content?" on request; nothing '
        "acts on it at ingest.",
    )
    parse_confidence: Mapped[float | None] = mapped_column(
        Float, comment="Mean confidence the parser reported, in [0, 1]."
    )
    parse_confidence_low: Mapped[float | None] = mapped_column(
        Float,
        comment="Confidence of the worst page, which is the one worth reviewing.",
    )
    parse_status: Mapped[str] = mapped_column(
        Text,
        server_default=Status.NEW,
        comment="new | pending | in_progress | parsed | failed.",
    )
    parse_error: Mapped[str | None] = mapped_column(
        Text, comment="Why a failed document failed to parse."
    )
    parse_claimed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        comment="When a worker claimed this document for parsing, NULL when none "
        "holds it. What tells a live claim from one a dead worker left behind.",
    )
    chunk_status: Mapped[str] = mapped_column(
        Text,
        server_default=Status.NEW,
        comment="new | pending | in_progress | chunked | failed. The only column "
        "chunking reads to find work: it does not consult parse_status, because a "
        "stage knows of no other stage.",
    )
    chunk_error: Mapped[str | None] = mapped_column(
        Text, comment="Why a failed document failed to chunk."
    )
    chunk_claimed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        comment="When a worker claimed this document for chunking, NULL when none "
        "holds it.",
    )
    oversized: Mapped[int | None] = mapped_column(
        Integer,
        comment="Passages the last chunking run stored above EMBEDDING_MAX_TOKENS. "
        "Expected to be 0: the chunker splits on that budget, so anything above it "
        "means the split did not happen. A tripwire; chunking discards nothing.",
    )

    # passive_deletes: without it SQLAlchemy NULLs the child foreign key, and
    # passages.doc_sha256 is NOT NULL.
    passages: Mapped[list[Passage]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )
    # No cascade: the foreign key is ON DELETE SET NULL, so the audit trail
    # outlives the document it describes.
    ingest_events: Mapped[list[IngestEvent]] = relationship(
        back_populates="document", passive_deletes=True
    )
