"""What this package needs from object storage.

Declared here rather than imported from blob_store, so the dependency points
inwards: the services name the operations they use and nothing supplies them
but the composition root.
"""

from __future__ import annotations

from typing import Protocol


class DocumentStore(Protocol):
    """The store holding source documents, exactly as uploaded."""

    def key_for(self, sha256: str, media_type: str) -> str:
        """Returns the key a document of this type is stored under."""
        ...

    def get(self, key: str) -> bytes:
        """Reads an object."""
        ...

    def remove(self, key: str) -> bool:
        """Deletes an object, reporting whether one was there."""
        ...

    def put(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str,
        metadata: dict[str, str],
    ) -> None:
        """Stores an object under `key`."""
        ...


class ParsedStore(Protocol):
    """The store holding converted documents.

    Read-and-delete only from here: parsing is what writes them. Deleting a
    document has to take its converted form, or the object is orphaned.
    """

    def key_for(self, sha256: str) -> str:
        """Returns the key a converted document is stored under."""
        ...

    def remove(self, key: str) -> bool:
        """Deletes an object, reporting whether one was there."""
        ...
