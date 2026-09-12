"""Database access for the ingestion service."""

from __future__ import annotations

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert

from database.qa_generator import (
    Document,
    IngestEvent,
    Outcome,
    Passage,
    Status,
)
from database.qa_generator.repository import Repository
from ingestion.models import DocumentName, PdfFacts, StoredDocument, UploadedFile

#: The name and date a digest first arrived under. DISTINCT ON rather than two
#: independent MIN aggregates: those take the earliest timestamp and the
#: alphabetically smallest name, which come from different rows whenever the
#: same bytes were uploaded twice.
_FIRST_EVENT = (
    select(
        IngestEvent.sha256.label("sha256"),
        IngestEvent.submitted_filename.label("filename"),
        IngestEvent.submitted_at.label("first_seen"),
    )
    .where(IngestEvent.sha256.is_not(None))
    .distinct(IngestEvent.sha256)
    .order_by(IngestEvent.sha256, IngestEvent.submitted_at)
    .subquery()
)


def _passages(*where) -> object:
    """Counts one document's passages, as a scalar subquery.

    Correlated rather than a grouped join: the join grouped every passage in
    the corpus to return one page of documents.
    """
    return (
        select(func.count())
        .select_from(Passage)
        .where(Passage.doc_sha256 == Document.sha256, *where)
        .scalar_subquery()
    )


def _filtered(query, search: str | None):
    """Applies the search filter to a document query.

    One place, so a listing and its count cannot disagree about what they are
    looking at. Wildcards are escaped.
    """
    if not search:
        return query
    return query.where(
        or_(
            _FIRST_EVENT.c.filename.icontains(search, autoescape=True),
            Document.title.icontains(search, autoescape=True),
            Document.sha256.icontains(search, autoescape=True),
        )
    )


class DocumentRepository(Repository):
    """Reads and writes `documents` and `ingest_events`.

    Both tables are behind one repository because a stored upload writes both
    in a single transaction.
    """

    def exists(self, sha256: str) -> bool:
        """Reports whether these exact bytes are already stored."""
        with self._session() as session:
            return (
                session.scalar(select(Document.sha256).where(Document.sha256 == sha256))
                is not None
            )

    def media_type(self, sha256: str) -> str | None:
        """Reads the type detected for a stored document."""
        with self._session() as session:
            return session.scalar(
                select(Document.mime_type).where(Document.sha256 == sha256)
            )

    def delete(self, sha256: str) -> int:
        """Removes a document row and everything derived from it.

        The cascades do the work. Ingest events survive, on ON DELETE SET NULL.
        """
        with self._session.begin() as session:
            passages = session.scalar(
                select(func.count())
                .select_from(Passage)
                .where(Passage.doc_sha256 == sha256)
            )
            session.execute(delete(Document).where(Document.sha256 == sha256))
        return passages or 0

    def delete_derived(self, sha256: str) -> int:
        """Drops a document's passages and marks it unchunked again.

        Left `new` rather than `pending`: this restores the state a freshly
        parsed document is in, and starting chunking stays a decision.
        """
        with self._session.begin() as session:
            passages = session.scalar(
                select(func.count())
                .select_from(Passage)
                .where(Passage.doc_sha256 == sha256)
            )
            session.execute(delete(Passage).where(Passage.doc_sha256 == sha256))
            session.execute(
                update(Document)
                .where(Document.sha256 == sha256)
                .values(chunk_status=Status.NEW)
            )
        return passages or 0

    def counts(self) -> dict[str, int]:
        """Reports what this service owns, for the status panel."""
        with self._session() as session:
            return {
                "documents": session.scalar(select(func.count()).select_from(Document)),
                "upload_attempts": session.scalar(
                    select(func.count()).select_from(IngestEvent)
                ),
            }

    def names(self) -> list[DocumentName]:
        """Lists just enough of each document to put it in a picker.

        Separate from `page` because three pages draw that picker and none of
        them needs the passage counts.
        """
        with self._session() as session:
            return [
                DocumentName(sha256=row.sha256, filename=row.filename)
                for row in session.execute(
                    select(Document.sha256, _FIRST_EVENT.c.filename)
                    .join(
                        _FIRST_EVENT,
                        _FIRST_EVENT.c.sha256 == Document.sha256,
                        isouter=True,
                    )
                    .order_by(_FIRST_EVENT.c.first_seen.desc().nullslast())
                ).all()
            ]

    def page(
        self, search: str | None = None, limit: int = 100, offset: int = 0
    ) -> tuple[int, list[StoredDocument]]:
        """Reads one page of documents and the total behind it."""
        listing = _filtered(
            select(
                Document.sha256,
                _FIRST_EVENT.c.filename,
                _FIRST_EVENT.c.first_seen,
                Document.page_count,
                Document.title,
                Document.language,
                Document.parse_status,
                Document.parse_error,
                Document.chunk_status,
                Document.chunk_error,
                Document.oversized,
                _passages().label("total_passages"),
                _passages(Passage.extract_status == Status.EXTRACTED).label(
                    "extracted_passages"
                ),
            )
            .join(_FIRST_EVENT, _FIRST_EVENT.c.sha256 == Document.sha256, isouter=True)
            # nullslast: DESC sorts NULLs first in PostgreSQL, which put a
            # document whose upload events were removed above every real one.
            .order_by(_FIRST_EVENT.c.first_seen.desc().nullslast()),
            search,
        )
        counting = _filtered(
            select(func.count())
            .select_from(Document)
            .join(_FIRST_EVENT, _FIRST_EVENT.c.sha256 == Document.sha256, isouter=True),
            search,
        )
        with self._session() as session:
            total = session.scalar(counting) or 0
            rows = session.execute(listing.limit(limit).offset(offset)).all()
        return total, [StoredDocument(**row._asdict()) for row in rows]

    def store(self, upload: UploadedFile, facts: PdfFacts) -> str:
        """Inserts a document and its event in one transaction.

        Answers DUPLICATE_BYTES when another upload of the same bytes won the
        race between the existence check and this insert.
        """
        with self._session.begin() as session:
            inserted = session.scalar(
                insert(Document)
                .values(
                    sha256=upload.sha256,
                    mime_type=upload.media_type,
                    page_count=facts.page_count,
                    char_count=facts.char_count,
                )
                .on_conflict_do_nothing(index_elements=[Document.sha256])
                .returning(Document.sha256)
            )
            outcome = Outcome.STORED if inserted else Outcome.DUPLICATE_BYTES
            session.add(
                self._event(
                    upload.filename, upload.size_bytes, outcome, None, upload.sha256
                )
            )
        return outcome

    def record_attempt(
        self,
        filename: str,
        size_bytes: int,
        outcome: str,
        detail: str | None = None,
        sha256: str | None = None,
    ) -> None:
        """Records an upload that did not produce a new document.

        Pass a digest only when a document row exists for it:
        ingest_events.sha256 is a foreign key into documents.
        """
        with self._session.begin() as session:
            session.add(self._event(filename, size_bytes, outcome, detail, sha256))

    @staticmethod
    def _event(
        filename: str,
        size_bytes: int,
        outcome: str,
        detail: str | None,
        sha256: str | None,
    ) -> IngestEvent:
        """Builds one ingest_events row."""
        return IngestEvent(
            submitted_filename=filename,
            size_bytes=size_bytes,
            sha256=sha256,
            outcome=outcome,
            detail=detail,
        )
