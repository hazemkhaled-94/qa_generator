"""The things parsing passes around."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Behind TYPE_CHECKING: the repository imports this module, so an
    # unconditional import would put Docling and torch in the API process.
    from docling_core.types.doc.document import DoclingDocument


@dataclass(frozen=True)
class ClaimedDocument:
    """One document taken off the parsing queue.

    A plain value rather than the row it was read from, which belongs to the
    repository's session.
    """

    sha256: str
    media_type: str
    page_count: int | None
    char_count: int | None


@dataclass(frozen=True)
class SourceDocument:
    """One stored file, as a pipeline receives it.

    The same shape for every format, so the service hands a pipeline its
    work without knowing which format it holds.
    """

    sha256: str
    data: bytes
    scanned: bool


@dataclass(frozen=True)
class Conversion:
    """What a pipeline produced from one file.

    Confidence travels beside the document because the converter reports it
    on the result, so it is absent from the exported model.
    """

    document: DoclingDocument
    confidence: float | None
    confidence_low: float | None


@dataclass(frozen=True)
class ParsedDocument:
    """One document as the pipeline understood it.

    Only what a column holds. Bounding boxes, cell grids, figures and
    captions stay in the parsed bucket.
    """

    title: str | None
    language: str | None
    content_sha256: str
    page_count: int
    confidence: float | None
    confidence_low: float | None
