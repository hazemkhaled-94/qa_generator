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

    def delete_every(self) -> list[Removal]:
        """Removes every document, its file, its converted form and its rows.

        What a document's deletion cascades to is everything drawn from it:
        its passages, their topic memberships, the facts on them, and by the
        orphan trigger the questions resting on those facts. What it does not
        reach is `topics`, which has no foreign key to a document because a
        fit is over the corpus rather than over a file. Emptying the corpus
        is therefore two commands, and `make wipe` is the one that runs both
        in the order that leaves nothing: documents, then topics.

        The upload history goes too. It survives a single deletion on purpose
        - it records what was attempted rather than what is held - but an
        empty corpus reporting nine upload attempts is a status panel
        describing documents nothing has.

        Returns:
            What was removed, one entry per document.
        """
        removed = [
            gone
            for digest in self._repository.digests()
            if (gone := self.delete(digest)) is not None
        ]
        events = self._repository.forget_uploads()
        log.info(
            "deleted %d document(s) and %d upload record(s); topics are not a "
            "document's and are deleted separately",
            len(removed),
            events,
        )
        return removed

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
