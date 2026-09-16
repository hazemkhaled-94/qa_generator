"""Removing a document, or only what the pipeline built from it."""

from __future__ import annotations

import logging

from ingestion.models import Removal
from ingestion.repository import DocumentRepository
from ingestion.stores import DocumentStore, ParsedStore

log = logging.getLogger(__name__)


class RemovalService:
    """Deletes a document, its file, its converted form and its derived rows.

    Nothing is atomic across two stores, so the order is fixed: the objects
    go first and the row last.
    """

    def __init__(
        self,
        *,
        repository: DocumentRepository,
        store: DocumentStore,
        parsed: ParsedStore,
    ) -> None:
        """Initialises the service with its collaborators.

        Args:
            repository: Where the document and passage rows are.
            store: Where the uploaded file is.
            parsed: Where the converted document is.
        """
        self._repository = repository
        self._store = store
        self._parsed = parsed

    def delete(self, sha256: str) -> Removal | None:
        """Removes a document completely: its rows, its file, its parsed form.

        Args:
            sha256: The document's digest.

        Returns:
            What was removed, or None if no document has that digest.
        """
        media_type = self._repository.media_type(sha256)
        if media_type is None:
            return None

        file_gone = self._store.remove(self._store.key_for(sha256, media_type))
        parsed_gone = self._parsed.remove(self._parsed.key_for(sha256))
        passages = self._repository.delete(sha256)

        log.info(
            "deleted %s: %d passage(s), file=%s parsed=%s",
            sha256,
            passages,
            file_gone,
            parsed_gone,
        )
        return Removal(
            sha256=sha256,
            document=True,
            passages=passages,
            file=file_gone,
            parsed=parsed_gone,
        )

    def delete_derived(self, sha256: str) -> Removal | None:
        """Drops what the pipeline built from a document, keeping the document.

        The file and the converted form stay, and chunking returns to `new`.

        Args:
            sha256: The document's digest.

        Returns:
            What was removed, or None if no document has that digest.
        """
        if self._repository.media_type(sha256) is None:
            return None
        passages = self._repository.delete_derived(sha256)
        log.info("dropped derived data for %s: %d passage(s)", sha256, passages)
        return Removal(
            sha256=sha256, document=False, passages=passages, file=False, parsed=False
        )
