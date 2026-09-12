"""Database access for the parsing service."""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy import select

from database.qa_generator import Document, Status
from preprocessing.parsing.models import ClaimedDocument, ParsedDocument
from stages import Columns, RowQueue

#: The next document to parse. FOR UPDATE SKIP LOCKED lets a second worker
#: take the following row instead of blocking on this one.
_NEXT_PENDING = (
    select(Document.sha256)
    .where(Document.parse_status == Status.PENDING)
    .order_by(Document.sha256)
    .with_for_update(skip_locked=True)
    .limit(1)
    .scalar_subquery()
)


class ParseQueue(RowQueue):
    """Reads the parsing queue and records what became of each document.

    Uses the shared models and session factory in database.qa_generator.
    """

    #: This stage's columns on `documents`, which is what StageQueue writes
    #: every claim, failure and requeue against.
    columns = Columns(
        entity=Document,
        key=Document.sha256,
        status=Document.parse_status,
        error=Document.parse_error,
        claimed_at=Document.parse_claimed_at,
    )
    done = Status.PARSED
    #: Four times PARSING_TIMEOUT_SECONDS, which is thirty minutes.
    lease = timedelta(hours=2)
    next_pending = _NEXT_PENDING

    def claim(self) -> ClaimedDocument | None:
        """Takes the next pending document off the queue."""
        claimed = self._claim(
            Document.sha256,
            Document.mime_type,
            Document.page_count,
            Document.char_count,
        )
        if claimed is None:
            return None
        return ClaimedDocument(
            sha256=claimed.sha256,
            media_type=claimed.mime_type,
            page_count=claimed.page_count,
            char_count=claimed.char_count,
        )

    def complete(self, sha256: str, parsed: ParsedDocument) -> None:
        """Records a successful parse, releasing the claim.

        Writes the title, language, content digest and both confidences.
        page_count is left alone: ingestion owns that column.

        chunk_status is not touched. This stage does not know that a
        chunking stage exists, so it cannot set one going, and a re-parse
        therefore leaves the old passages standing until someone asks for
        them to be rebuilt. That is the orchestrator's job: after re-parsing,
        redo chunking.
        """
        self._finish(
            sha256,
            title=parsed.title,
            language=parsed.language,
            content_sha256=parsed.content_sha256,
            parse_confidence=parsed.confidence,
            parse_confidence_low=parsed.confidence_low,
        )
