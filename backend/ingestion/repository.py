"""Database access for the ingestion service."""

from __future__ import annotations

from sqlalchemy import ScalarSelect, delete, func, select, update
from sqlalchemy.dialects.postgresql import insert

from database.qa_generator import (
    Document,
    IngestEvent,
    Outcome,
    Passage,
    Status,
)
from database.qa_generator.repository import Repository, matching
from ingestion.models import DocumentName, PdfFacts, StoredDocument, UploadedFile

#: The name and date a digest first arrived under. DISTINCT ON, so the name
#: and the timestamp come from the same row.
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


def _passages(*where) -> ScalarSelect[int]:
    """Counts one document's passages, as a correlated scalar subquery."""
    return (
        select(func.count())
        .select_from(Passage)
        .where(Passage.doc_sha256 == Document.sha256, *where)
        .scalar_subquery()
    )


def _filtered(query, search: str | None):
    """Applies the search filter to a document query. Wildcards are escaped."""
    if not search:
        return query
    return query.where(
        matching(search, _FIRST_EVENT.c.filename, Document.title, Document.sha256)
    )


class DocumentRepository(Repository):
    """Reads and writes `documents` and `ingest_events`."""

    def exists(self, sha256: str) -> bool:
        """Reports whether these exact bytes are already stored."""
        with self._session() as session:
            return (
                session.scalar(select(Document.sha256).where(Document.sha256 == sha256))
                is not None
            )

    def media_type(self, sha256: str) -> str | None:
        """Reads the media type detected for a stored document."""
        with self._session() as session:
            return session.scalar(
                select(Document.mime_type).where(Document.sha256 == sha256)
            )

    def digests(self) -> list[str]:
        """Every stored document's digest, oldest first."""
        with self._session() as session:
            return list(
                session.scalars(select(Document.sha256).order_by(Document.sha256))
            )

    def forget_uploads(self) -> int:
        """Drops the upload history, which a document's deletion keeps.

        `ingest_events` survives a deletion on ON DELETE SET NULL, because it
        records what was attempted rather than what is held. Emptying the
        corpus is the one time that is wrong: the status panel would report
        upload attempts for documents nothing has.
        """
        with self._session.begin() as session:
            return session.execute(delete(IngestEvent)).rowcount

    def delete(self, sha256: str) -> int:
        """Removes a document row and everything derived from it.

        Ingest events survive, on ON DELETE SET NULL.

        Args:
            sha256: The document's digest.

        Returns:
            Passages the cascade took with it.
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
        """Drops a document's passages and returns chunk_status to `new`.

        Args:
            sha256: The document's digest.

        Returns:
            Passages deleted.
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
        """Lists every document by digest and first filename, newest first."""
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
        """Reads one page of documents and the total behind it.

        Args:
            search: Matched against filename, title and digest.
            limit: Rows to return.
            offset: Rows to skip.

        Returns:
            The total matching the search, and the requested page.
        """
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
            # nullslast: DESC sorts NULLs first in PostgreSQL.
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

        Args:
            upload: The file as it arrived.
            facts: What reading the PDF measured.

        Returns:
            STORED, or DUPLICATE_BYTES when another upload of the same bytes
            won the race with the existence check.
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

        Args:
            filename: Name the file was submitted under.
            size_bytes: Size of the submitted file.
            outcome: One of the `Outcome` values.
            detail: The refusal in words.
            sha256: A digest a document row already exists for;
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
