"""Removing a document, or only what the pipeline built from it.

Separate from the ingest flow: same repository, different trigger, different
lifecycle, and a different reason to change. Accepting an upload is a
validation problem; deleting one is an ordering problem across three stores.
"""

from __future__ import annotations

import logging

from ingestion.models import Removal
from ingestion.repository import DocumentRepository
from ingestion.stores import DocumentStore, ParsedStore

log = logging.getLogger(__name__)


class RemovalService:
    """Deletes a document, its file, its converted form and its derived rows.

    Nothing is atomic across two stores, so the order is fixed: the objects
    go first and the row last. An orphaned object is wasted space; a row
    pointing at a missing object fails every later stage.
    """

    def __init__(
        self,
        *,
        repository: DocumentRepository,
        store: DocumentStore,
        parsed: ParsedStore,
    ) -> None:
        """Initialises the service with its collaborators."""
        self._repository = repository
        self._store = store
        self._parsed = parsed

    def delete(self, sha256: str) -> Removal | None:
        """Removes a document completely: its rows, its file, its parsed form."""
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

        The file and the converted form both stay, and chunking returns to
        `new`, so the next run rebuilds the passages and facts without
        re-parsing once somebody asks for it.
        """
        if self._repository.media_type(sha256) is None:
            return None
        passages = self._repository.delete_derived(sha256)
        log.info("dropped derived data for %s: %d passage(s)", sha256, passages)
        return Removal(
            sha256=sha256, document=False, passages=passages, file=False, parsed=False
        )
