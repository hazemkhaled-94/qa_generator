"""The ingest flow, top to bottom."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from database.qa_generator import Outcome
from ingestion.models import (
    DETECTABLE,
    DocumentName,
    IngestResult,
    StoredDocument,
    UploadedFile,
)
from ingestion.pdf import UnreadablePdf, read_pdf
from ingestion.repository import DocumentRepository
from ingestion.stores import DocumentStore
from telemetry import tracer

log = logging.getLogger(__name__)
span = tracer(__name__)


class IngestService:
    """Accepts one uploaded file and decides what becomes of it.

    Checks run cheapest first and nothing is written until all of them pass.
    Duplicate detection is byte-exact. Deleting a document is
    :class:`RemovalService`.
    """

    def __init__(
        self,
        *,
        repository: DocumentRepository,
        store: DocumentStore,
        max_file_size_bytes: int,
        allowed_media_types: tuple[str, ...],
        pipeline_version: str,
    ) -> None:
        """Initialises the service with its collaborators.

        Args:
            repository: Where documents and ingest events are written.
            store: Where the accepted bytes are put.
            max_file_size_bytes: Largest upload accepted.
            allowed_media_types: Media types to store, each of DETECTABLE.
            pipeline_version: Recorded in the stored object's metadata.

        Raises:
            ValueError: If the allowlist names a type this build cannot
                detect.
        """
        unknown = set(allowed_media_types) - DETECTABLE
        if unknown:
            raise ValueError(
                f"ALLOWED_MIME_TYPES names {', '.join(sorted(unknown))}, which "
                f"this build cannot detect. Detectable: "
                f"{', '.join(sorted(DETECTABLE))}."
            )
        self._repository = repository
        self._store = store
        self._max_file_size_bytes = max_file_size_bytes
        self._allowed_media_types = allowed_media_types
        self._pipeline_version = pipeline_version

    @property
    def max_file_size_bytes(self) -> int:
        """Largest upload accepted."""
        return self._max_file_size_bytes

    def documents(
        self,
        search: str | None = None,
        limit: int = 100,
        offset: int = 0,
        parse_status: str | None = None,
    ) -> tuple[int, list[StoredDocument]]:
        """Lists one page of what has been ingested, and the total behind it.

        Args:
            search: Matched against filename, title and digest.
            limit: Rows to return.
            offset: Rows to skip.
            parse_status: Narrows to one parse state.

        Returns:
            The total matching the filters, and the requested page.
        """
        return self._repository.page(search, limit, offset, parse_status)

    def document_names(self) -> list[DocumentName]:
        """Lists every document by the name it arrived under."""
        return self._repository.names()

    def stored_file(self, sha256: str) -> tuple[bytes, str] | None:
        """Reads one stored document back, exactly as it was uploaded.

        The digest is looked up before it becomes an object key.

        Args:
            sha256: The document's digest.

        Returns:
            The bytes and their media type, or None if no document has that
            digest.
        """
        media_type = self._repository.media_type(sha256)
        if media_type is None:
            return None
        return self._store.get(self._store.key_for(sha256, media_type)), media_type

    def ingest(self, upload: UploadedFile) -> IngestResult:
        """Runs one upload through every check and stores it if they pass.

        Every path records an ingest_events row, refusals included.

        Args:
            upload: The file as it arrived.

        Returns:
            What became of the upload.
        """
        with span.start_as_current_span("ingest") as current:
            current.set_attribute("upload.filename", upload.filename)
            current.set_attribute("upload.size_bytes", upload.size_bytes)
            result = self._ingest(upload)
            current.set_attribute("ingest.outcome", result.outcome)
            if result.sha256:
                current.set_attribute("document.sha256", result.sha256)
            return result

    def refuse_oversized(self, filename: str, size_bytes: int) -> IngestResult:
        """Refuses an upload on its declared size, without reading the body.

        Writes the ingest_events row only: with no bytes there is no digest
        and no document.

        Args:
            filename: Name the file was submitted under.
            size_bytes: Size the request declared.

        Returns:
            A TOO_LARGE result carrying the measurement.
        """
        detail = self._over_limit(size_bytes)
        self._repository.record_attempt(filename, size_bytes, Outcome.TOO_LARGE, detail)
        log.info("too_large %s (%d bytes declared): %s", filename, size_bytes, detail)
        return IngestResult(Outcome.TOO_LARGE, None, detail)

    def _ingest(self, upload: UploadedFile) -> IngestResult:
        """Runs the checks and the writes. See `ingest`."""
        if upload.size_bytes > self._max_file_size_bytes:
            return self._attempted(
                upload, Outcome.TOO_LARGE, self._over_limit(upload.size_bytes)
            )

        if upload.media_type not in self._allowed_media_types:
            return self._attempted(
                upload,
                Outcome.UNSUPPORTED_TYPE,
                f"not a supported document (detected: {upload.media_type or 'unknown'})",
            )

        # Ahead of reading the document: hashing bytes already in memory is
        # cheaper than walking every page.
        if self._repository.exists(upload.sha256):
            return self._attempted(
                upload, Outcome.DUPLICATE_BYTES, sha256=upload.sha256
            )

        try:
            facts = read_pdf(upload.data)
        except UnreadablePdf as exc:
            return self._attempted(upload, Outcome.UNSUPPORTED_TYPE, str(exc))

        # Object before row: the key is content-addressed, so an object with
        # no row is harmless and a row with no object is not.
        self._store.put(
            self._store.key_for(upload.sha256, upload.media_type),
            upload.data,
            content_type=upload.media_type,
            metadata=self._metadata(upload),
        )

        # An unwritable row takes its object back out.
        try:
            outcome = self._repository.store(upload, facts)
        except Exception:
            self._store.remove(self._store.key_for(upload.sha256, upload.media_type))
            log.exception(
                "removed the stored object for %s: its row could not be written",
                upload.sha256,
            )
            raise
        log.info(
            "%s %s (%d bytes) -> %s",
            outcome,
            upload.filename,
            upload.size_bytes,
            upload.sha256,
        )
        return IngestResult(outcome, upload.sha256)

    def _attempted(
        self,
        upload: UploadedFile,
        outcome: str,
        detail: str | None = None,
        *,
        sha256: str | None = None,
    ) -> IngestResult:
        """Records an upload that did not produce a new document."""
        self._repository.record_attempt(
            upload.filename, upload.size_bytes, outcome, detail, sha256
        )
        log.info(
            "%s %s (%d bytes)%s",
            outcome,
            upload.filename,
            upload.size_bytes,
            f": {detail}" if detail else "",
        )
        return IngestResult(outcome, sha256, detail)

    def _over_limit(self, size_bytes: int) -> str:
        """Describes an upload that exceeds the size limit."""
        return (
            f"{size_bytes / 1048576:.1f} MB exceeds the "
            f"{self._max_file_size_bytes / 1048576:.0f} MB limit"
        )

    def _metadata(self, upload: UploadedFile) -> dict[str, str]:
        """Builds the S3 user metadata for a document.

        Enough for the bucket alone to rebuild the row. Nothing mutable is
        included.
        """
        return {
            "sha256": upload.sha256,
            "original-filename": upload.filename,
            "ingested-at": datetime.now(UTC).isoformat(),
            "pipeline-version": self._pipeline_version,
        }
