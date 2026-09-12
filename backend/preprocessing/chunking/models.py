"""The things chunking passes around."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ClaimedDocument:
    """One document taken off the chunking queue.

    Carries the language so the passages are split by the right pipeline.
    """

    sha256: str
    language: str | None


@dataclass(frozen=True)
class Chunk:
    """One passage, as the chunker cut it and before it is stored."""

    ordinal: int
    text: str
    page_from: int | None
    page_to: int | None
    section_path: str | None
    block_type: str | None
    doc_item_refs: list[str] = field(default_factory=list)
    bbox: list[dict] = field(default_factory=list)
    table_cells: list[dict] = field(default_factory=list)
    #: Detected on this passage, not inherited from the document: these
    #: documents mix languages within one file.
    language: str | None = None
    #: The units a fact may cite, [{i, start, end, predicates}], with offsets
    #: into text. Sentences for prose, rendered rows for a table.
    sentences: list[dict] = field(default_factory=list)
    #: Content lemmas, the vocabulary topic modelling is fitted over.
    lemmas: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Chunking:
    """What the chunker made of one document.

    Nothing is discarded on size: `oversized` is a tripwire, because the
    chunker splits on the same budget it counts against.
    """

    passages: list[Chunk]
    oversized: int


@dataclass(frozen=True)
class StoredPassage:
    """One passage as it is read back out, for display.

    Carries counts rather than the grids and sentences themselves, so a
    listing can say which passages hold a table without shipping every cell.
    """

    id: int
    doc_sha256: str
    ordinal: int
    text: str
    page_from: int | None
    page_to: int | None
    section_path: str | None
    block_type: str | None
    language: str | None
    doc_item_refs: list[str]
    bbox: list[dict]
    table_count: int
    sentence_count: int


@dataclass(frozen=True)
class PassageDetail:
    """One passage in full, cell grids and sentences included."""

    passage: StoredPassage
    table_cells: list[dict]
    sentences: list[dict]
    extract_status: str
    extract_error: str | None
