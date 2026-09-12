"""Database access for the chunking service."""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy import delete, func, insert, or_, select

from database.qa_generator import Document, Passage, Status
from database.qa_generator.repository import Repository
from preprocessing.chunking.models import (
    Chunk,
    Chunking,
    ClaimedDocument,
    PassageDetail,
    StoredPassage,
)
from stages import Columns, RowQueue

#: The next document to chunk. This stage's own column and nothing else: it
#: does not read parse_status, because it knows of no parsing stage. A
#: document queued here without a parsed form fails on the missing object.
_NEXT_PENDING = (
    select(Document.sha256)
    .where(Document.chunk_status == Status.PENDING)
    .order_by(Document.sha256)
    .with_for_update(skip_locked=True)
    .limit(1)
    .scalar_subquery()
)

#: Which columns the search box looks in, by the name the API accepts.
SEARCH_FIELDS = {
    "text": (Passage.text,),
    "section": (Passage.section_path,),
    "both": (Passage.text, Passage.section_path),
}

DEFAULT_FIELD = "text"


def _filtered(query, document, search, block_type, field):
    """Applies the document, text and type filters to a passage query.

    One place, so a listing and its count cannot disagree about what they are
    looking at.
    """
    if document:
        query = query.where(Passage.doc_sha256 == document)
    if search:
        columns = SEARCH_FIELDS.get(field or "", SEARCH_FIELDS[DEFAULT_FIELD])
        query = query.where(
            or_(*(c.icontains(search, autoescape=True) for c in columns))
        )
    if block_type:
        query = query.where(Passage.block_type == block_type)
    return query


class ChunkQueue(RowQueue):
    """Reads the chunking queue and records what became of each document."""

    columns = Columns(
        entity=Document,
        key=Document.sha256,
        status=Document.chunk_status,
        error=Document.chunk_error,
        claimed_at=Document.chunk_claimed_at,
    )
    done = Status.CHUNKED
    lease = timedelta(minutes=30)
    next_pending = _NEXT_PENDING

    def claim(self) -> ClaimedDocument | None:
        """Takes the next unchunked document off the queue."""
        claimed = self._claim(Document.sha256, Document.language)
        if claimed is None:
            return None
        return ClaimedDocument(sha256=claimed.sha256, language=claimed.language)

    def replace(self, sha256: str, chunking: Chunking) -> int:
        """Replaces a document's passages with a new set, then finishes it.

        One transaction: the passages, the oversized count and the status land
        together, so a document can never read as chunked while holding
        another run's passages. The delete cascades to the facts drawn from
        them and to the topic memberships they held.
        """
        chunks: list[Chunk] = chunking.passages
        with self._session.begin() as session:
            session.execute(delete(Passage).where(Passage.doc_sha256 == sha256))
            if chunks:
                session.execute(
                    insert(Passage),
                    [
                        {
                            "doc_sha256": sha256,
                            "ordinal": chunk.ordinal,
                            "text": chunk.text,
                            "language": chunk.language,
                            "sentences": chunk.sentences or None,
                            "lemmas": chunk.lemmas,
                            "page_from": chunk.page_from,
                            "page_to": chunk.page_to,
                            "section_path": chunk.section_path,
                            "block_type": chunk.block_type,
                            "doc_item_refs": chunk.doc_item_refs,
                            "bbox": chunk.bbox or None,
                            "table_cells": chunk.table_cells or None,
                        }
                        for chunk in chunks
                    ],
                )
            self._finish(sha256, session=session, oversized=chunking.oversized)
        return len(chunks)

    def counts(self) -> dict[str, int]:
        """Reports what this service owns, for the status panel."""
        with self._session() as session:
            passages = session.scalar(select(func.count()).select_from(Passage))
        return {**self.counts_by_status(), "passages": passages}


class PassageCatalog(Repository):
    """Reads back the passages chunking produced.

    Separate from the queue: the API serves these and never claims a row.
    """

    def block_types(self) -> list[str]:
        """Lists the block types actually present, for a filter to offer."""
        with self._session() as session:
            return list(
                session.scalars(
                    select(Passage.block_type)
                    .where(Passage.block_type.is_not(None))
                    .distinct()
                    .order_by(Passage.block_type)
                ).all()
            )

    def page(
        self,
        document: str | None = None,
        limit: int = 100,
        offset: int = 0,
        search: str | None = None,
        block_type: str | None = None,
        field: str | None = None,
    ) -> tuple[int, list[StoredPassage]]:
        """Reads one page of passages and the total behind it.

        Both in one call: a listing and its pager asked separately, which cost
        two round trips and let the count describe a corpus the rows no longer
        matched.
        """
        narrowed = _filtered(
            select(Passage).order_by(Passage.doc_sha256, Passage.ordinal),
            document,
            search,
            block_type,
            field,
        )
        counting = _filtered(
            select(func.count()).select_from(Passage),
            document,
            search,
            block_type,
            field,
        )
        with self._session() as session:
            total = session.scalar(counting) or 0
            rows = session.scalars(narrowed.limit(limit).offset(offset)).all()
        return total, [_stored(row) for row in rows]

    def passage(self, passage_id: int) -> PassageDetail | None:
        """Reads one passage in full, its cell grids and sentences included."""
        with self._session() as session:
            row = session.get(Passage, passage_id)
            if row is None:
                return None
            return PassageDetail(
                passage=_stored(row),
                table_cells=row.table_cells or [],
                sentences=row.sentences or [],
                extract_status=row.extract_status,
                extract_error=row.extract_error,
            )


def _stored(row: Passage) -> StoredPassage:
    """Reads one passage row into the shape the API publishes."""
    return StoredPassage(
        id=row.id,
        doc_sha256=row.doc_sha256,
        ordinal=row.ordinal,
        text=row.text,
        page_from=row.page_from,
        page_to=row.page_to,
        section_path=row.section_path,
        block_type=row.block_type,
        language=row.language,
        doc_item_refs=row.doc_item_refs or [],
        bbox=row.bbox or [],
        table_count=len(row.table_cells or []),
        sentence_count=len(row.sentences or []),
    )
