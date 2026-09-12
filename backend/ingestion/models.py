"""The things ingestion passes around."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property

#: Leading bytes that identify a type, checked instead of trusting the
#: upload's declared content type. ALLOWED_MIME_TYPES may name only types
#: listed here, which the ingest service checks at start-up.
_MAGIC = {b"%PDF-": "application/pdf"}

#: Every type this build can detect, for a caller to check its allowlist
#: against.
DETECTABLE = frozenset(_MAGIC.values())


@dataclass(frozen=True)
class UploadedFile:
    """One file as it arrived, and everything derivable from the bytes.

    Size, digest and media type are properties rather than fields, so no
    instance can hold a digest that disagrees with its content.
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
        """Type detected from the leading bytes, or None if unrecognised.

        The trust boundary: neither the declared content type nor the
        filename extension is evidence of anything.
        """
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

    Measurements, not judgements: whether a document counts as scanned is
    parsing's decision.
    """

    page_count: int
    char_count: int


@dataclass(frozen=True)
class DocumentName:
    """One document as a picker shows it."""

    sha256: str
    filename: str | None


@dataclass(frozen=True)
class StoredDocument:
    """One document as the catalogue reports it.

    A declared shape rather than a mapping, so the API publishes its fields.
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
    """What deleting a document took with it.

    Reported per store rather than assumed, since a document's rows, its
    file and its converted form are deleted separately.
    """

    sha256: str
    document: bool
    passages: int
    file: bool
    parsed: bool


@dataclass(frozen=True)
class IngestResult:
    """What happened to one upload."""

    outcome: str
    sha256: str | None = None
    detail: str | None = None
