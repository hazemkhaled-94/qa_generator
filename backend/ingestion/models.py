"""The things ingestion passes around."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property

#: Leading bytes that identify a media type.
_MAGIC = {b"%PDF-": "application/pdf"}

#: Every media type this build can detect.
DETECTABLE = frozenset(_MAGIC.values())


@dataclass(frozen=True)
class UploadedFile:
    """One file as it arrived, and everything derived from its bytes.

    Attributes:
        filename: Name the file was submitted under.
        data: The raw bytes.
    """

    filename: str
    data: bytes

    @property
    def size_bytes(self) -> int:
        """Size of the upload in bytes."""
        return len(self.data)

    @cached_property
    def sha256(self) -> str:
        """SHA-256 of the raw bytes, which is the document's identity."""
        return hashlib.sha256(self.data).hexdigest()

    @cached_property
    def media_type(self) -> str | None:
        """Media type detected from the leading bytes, or None if unrecognised."""
        return next(
            (
                mime
                for signature, mime in _MAGIC.items()
                if self.data.startswith(signature)
            ),
            None,
        )


@dataclass(frozen=True)
class PdfFacts:
    """Measurements taken from a PDF by reading it.

    Attributes:
        page_count: Pages in the document.
        char_count: Characters extractable from its text layer.
    """

    page_count: int
    char_count: int


@dataclass(frozen=True)
class DocumentName:
    """One document as a picker shows it.

    Attributes:
        sha256: The document's digest.
        filename: Name it first arrived under, if any is recorded.
    """

    sha256: str
    filename: str | None


@dataclass(frozen=True)
class StoredDocument:
    """One document as the catalogue reports it.

    Attributes:
        sha256: The document's digest.
        filename: Name it first arrived under.
        first_seen: When it first arrived.
        page_count: Pages, counted at upload.
        title: Title, written by parsing.
        language: ISO 639-1 code, written by parsing.
        parse_status: Where parsing has got to.
        parse_error: Why parsing failed.
        chunk_status: Where chunking has got to.
        chunk_error: Why chunking failed.
        extracted_passages: Passages extraction has read.
        total_passages: Passages chunking produced.
        oversized: Passages stored above the token budget.
    """

    sha256: str
    filename: str | None
    first_seen: datetime | None
    page_count: int | None
    title: str | None
    language: str | None
    parse_status: str
    parse_error: str | None
    chunk_status: str
    chunk_error: str | None
    extracted_passages: int
    total_passages: int
    oversized: int | None = None


@dataclass(frozen=True)
class Removal:
    """What deleting a document took with it, reported per store.

    Attributes:
        sha256: The document's digest.
        document: Whether the document row was deleted.
        passages: Passages deleted.
        file: Whether the uploaded file was deleted.
        parsed: Whether the converted form was deleted.
    """

    sha256: str
    document: bool
    passages: int
    file: bool
    parsed: bool


@dataclass(frozen=True)
class IngestResult:
    """What happened to one upload.

    Attributes:
        outcome: One of the `Outcome` values.
        sha256: The document's digest, when one was hashed.
        detail: The refusal in words, when it was refused.
    """

    outcome: str
    sha256: str | None = None
    detail: str | None = None
