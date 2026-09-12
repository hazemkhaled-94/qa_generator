"""The ingest_events table."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    CHAR,
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.qa_generator.base import Base
from database.qa_generator.outcomes import Outcome, one_of

if TYPE_CHECKING:
    from database.qa_generator.documents import Document


class IngestEvent(Base):
    """One upload attempt, refused ones included.

    The same bytes arriving three times under three names is one
    :class:`Document` row and three rows here; a refused upload is a row
    here and no document at all.
    """

    __tablename__ = "ingest_events"
    __table_args__ = (
        CheckConstraint(one_of("outcome", Outcome), name="ingest_events_outcome_valid"),
        {
            "comment": "Every upload attempt, including refused ones. A refused upload "
            "writes nothing to documents, so this is the only record that it happened. "
            "Also holds ingest provenance: the same bytes arriving three times under "
            "three names is one documents row and three rows here."
        },
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    submitted_filename: Mapped[str] = mapped_column(
        Text,
        comment="Filename as submitted. The earliest stored event for a hash is "
        "that document's original filename.",
    )
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="When the upload was received.",
    )
    size_bytes: Mapped[int] = mapped_column(
        BigInteger, comment="Size of the submitted file."
    )
    sha256: Mapped[str | None] = mapped_column(
        CHAR(64),
        ForeignKey("documents.sha256", ondelete="SET NULL"),
        comment="SHA-256 of the submitted bytes, and the link to the document they "
        "became. NULL for uploads refused before hashing, which is why the foreign "
        "key tolerates NULL.",
    )
    outcome: Mapped[str] = mapped_column(
        Text,
        comment="stored | duplicate_bytes | too_large | unsupported_type. "
        "Enforced by a CHECK constraint.",
    )
    detail: Mapped[str | None] = mapped_column(
        Text,
        comment="Human-readable reason, shown to the uploader when a file is refused.",
    )

    document: Mapped[Document | None] = relationship(back_populates="ingest_events")
