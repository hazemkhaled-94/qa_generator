"""The things parsing passes around."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Behind TYPE_CHECKING: the repository imports this module, and the API
    # imports the repository.
    from docling_core.types.doc.document import DoclingDocument


@dataclass(frozen=True)
class ClaimedDocument:
    """One document taken off the parsing queue.

    Attributes:
        sha256: The document's digest.
        media_type: Media type detected at ingest.
        page_count: Pages, counted at ingest.
        char_count: Characters in the text layer, counted at ingest.
    """

    sha256: str
    media_type: str
    page_count: int | None
    char_count: int | None


@dataclass(frozen=True)
class SourceDocument:
    """One stored file, as a pipeline receives it.

    Attributes:
        sha256: The document's digest.
        data: The raw file bytes.
        scanned: Whether the document carries too little text to read.
    """

    sha256: str
    data: bytes
    scanned: bool


@dataclass(frozen=True)
class Conversion:
    """What a pipeline produced from one file.

    Attributes:
        document: The converted document.
        confidence: Mean confidence the converter reported, or None.
        confidence_low: Confidence of its worst page, or None.
    """

    document: DoclingDocument
    confidence: float | None
    confidence_low: float | None


@dataclass(frozen=True)
class ParsedDocument:
    """One document as the pipeline understood it. Only what a column holds.

    Attributes:
        title: The document's own title, or None.
        language: ISO 639-1 code, or None when too short to tell.
        content_sha256: Digest of the normalised body text.
        page_count: Pages the converted document holds.
        confidence: Mean confidence the converter reported, or None.
        confidence_low: Confidence of its worst page, or None.
    """

    title: str | None
    language: str | None
    content_sha256: str
    page_count: int
    confidence: float | None
    confidence_low: float | None
