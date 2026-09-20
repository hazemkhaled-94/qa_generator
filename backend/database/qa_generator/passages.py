"""The passages table."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CHAR,
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.qa_generator.base import Base
from database.qa_generator.status import Status, check, queued

if TYPE_CHECKING:
    from database.qa_generator.documents import Document
    from database.qa_generator.fact_passages import FactPassage
    from database.qa_generator.passage_topics import PassageTopic


class Passage(Base):
    """A document split into retrievable chunks.

    Immutable: re-chunking deletes and re-inserts every passage of a document,
    which cascades to its facts, topics and questions.
    """

    __tablename__ = "passages"
    __table_args__ = (
        UniqueConstraint("doc_sha256", "ordinal"),
        # GIN trigram, not btree: every search here is ILIKE '%term%'.
        Index(
            "ix_passages_text_trgm",
            "text",
            postgresql_using="gin",
            postgresql_ops={"text": "gin_trgm_ops"},
        ),
        # GIN over the array, which is what the `&&` in question generation's
        # corpus-wide probe needs: an unanswerable question claims the whole
        # corpus is silent about something, so that gate reads every passage
        # rather than the two the question cites.
        Index("ix_passages_lemmas_gin", "lemmas", postgresql_using="gin"),
        # Serves the nearest-passage search: what a topic's bridge candidates
        # are grouped by, and what a reviewer reads to find the passage a
        # question should have cited.
        Index(
            "ix_passages_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # Partial: the queue is a shrinking fraction of the table, and this is
        # the only predicate a worker selects on.
        Index(
            "ix_passages_extract_queue",
            "id",
            postgresql_where=text(queued("extract_status")),
        ),
        CheckConstraint("ordinal > 0", name="passages_ordinal_positive"),
        CheckConstraint(
            "language IS NULL OR language ~ '^[a-z]{2}$'",
            name="passages_language_is_iso_639_1",
        ),
        CheckConstraint(
            check("extract_status", Status.EXTRACTED),
            name="passages_extract_status_valid",
        ),
        {
            "comment": "A document split into retrievable chunks. Immutable: "
            "re-chunking deletes and re-inserts every passage of a document."
        },
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    doc_sha256: Mapped[str] = mapped_column(
        CHAR(64),
        ForeignKey("documents.sha256", ondelete="CASCADE"),
        comment="The document this passage was cut from.",
    )
    ordinal: Mapped[int] = mapped_column(
        Integer,
        comment="Position in reading order within the document, from 1 and "
        "contiguous, so ordinal +/- 1 are the neighbouring passages. Reassigned on "
        "every re-chunk.",
    )
    extract_status: Mapped[str] = mapped_column(
        Text,
        server_default=Status.NEW,
        comment="new | pending | in_progress | extracted | failed. Extraction "
        "queues over passages rather than documents, which is the unit that "
        "divides evenly between workers.",
    )
    extract_error: Mapped[str | None] = mapped_column(
        Text, comment="Why this passage could not be read."
    )
    extract_claimed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        comment="When a worker claimed this passage, NULL when none holds it.",
    )
    text: Mapped[str] = mapped_column(
        Text,
        comment="The passage content. The evidence offsets in fact_passages are "
        "relative to this string.",
    )
    language: Mapped[str | None] = mapped_column(
        CHAR(2),
        index=True,
        comment="ISO 639-1, detected on this passage rather than inherited from "
        "the document: one file carries a German report and its English summary, "
        "so a document-wide label reads half the passages with the wrong "
        "pipeline. Selects the spaCy pipeline, and partitions the topic model. "
        "NULL when the passage was too short to tell.",
    )
    sentences: Mapped[list[dict] | None] = mapped_column(
        JSONB(none_as_null=True),
        comment="The sentences spaCy found, [{i, start, end, predicates}], with "
        "offsets into text. Extraction cites one by index, so a citation needs no "
        "quote to search for. NULL until chunking has run under a version that "
        "writes it.",
    )
    lemmas: Mapped[list[str] | None] = mapped_column(
        ARRAY(Text),
        comment="Content lemmas, the vocabulary topic modelling is fitted over. "
        "Written here so one segmentation serves both stages and a fit reads a "
        "column instead of re-tokenising the corpus.",
    )
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(1024),
        comment="Passage embedding, in the same space as facts.embedding and "
        "questions.embedding. What a nearest-passage search reads. Written by "
        "extraction rather than by chunking, which owns every other column "
        "here: the extraction worker loads the model anyway for the fact dedup "
        "gate, and a second worker loading two gigabytes to write one column is "
        "the cost this avoids. NULL until a passage has been through extraction "
        "under a deployment that embeds.",
    )
    page_from: Mapped[int | None] = mapped_column(
        Integer, comment="Page the passage starts on."
    )
    page_to: Mapped[int | None] = mapped_column(
        Integer, comment="Page the passage ends on."
    )
    section_path: Mapped[str | None] = mapped_column(
        Text, comment='Heading trail, for example "3 > 3.2 > Delivery".'
    )
    block_type: Mapped[str | None] = mapped_column(
        Text,
        comment="Docling's DocItemLabel for the source item. Routes extraction: a "
        "table goes to the cell reader, everything else to the model. A passage "
        "holding items of several kinds is labelled table if any of them is one.",
    )
    doc_item_refs: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        comment="The converter's identifiers for the items this passage was cut "
        "from, and the only non-fuzzy way back to the converted document.",
    )
    bbox: Mapped[list[dict] | None] = mapped_column(
        JSONB(none_as_null=True),
        comment="Where the passage sits, one box per page it spans: "
        "[{page, l, t, r, b}], on a top-left origin. NULL when the converter "
        "recorded no position.",
    )
    table_cells: Mapped[list[dict] | None] = mapped_column(
        # none_as_null: otherwise None is stored as the JSON value null, and
        # every passage answers IS NOT NULL whether it holds a table or not.
        JSONB(none_as_null=True),
        comment="The cell grids behind a table passage: [{caption, num_rows, "
        "num_cols, cells:[{row, col, row_span, col_span, column_header, "
        "row_header, text, line}]}]. `line` is the rendered Markdown row the cell "
        "sits in and the evidence a fact drawn from it cites; NULL on a header "
        "cell. Only the rows this passage renders are kept. NULL unless "
        "block_type is table.",
    )

    document: Mapped[Document] = relationship(back_populates="passages")
    # Topics attach to the passage, not the fact: inference needs enough text
    # for term statistics.
    topic_links: Mapped[list[PassageTopic]] = relationship(
        back_populates="passage", cascade="all, delete-orphan", passive_deletes=True
    )
    #: Every fact resting on this passage, of any kind. Deleting the passage
    #: deletes these rows, and a trigger on them deletes the facts.
    fact_links: Mapped[list[FactPassage]] = relationship(
        back_populates="passage", cascade="all, delete-orphan", passive_deletes=True
    )
