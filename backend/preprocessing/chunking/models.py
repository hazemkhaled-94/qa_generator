"""The things chunking passes around."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ClaimedDocument:
    """One document taken off the chunking queue.

    Attributes:
        sha256: The document's digest.
        language: The document's language, the fallback for a passage too
            short to detect.
    """

    sha256: str
    language: str | None


@dataclass(frozen=True)
class Chunk:
    """One passage, as the chunker cut it and before it is stored.

    Attributes:
        ordinal: Position in reading order, from 1.
        text: The passage content. Every evidence offset indexes this.
        page_from: Page the passage starts on.
        page_to: Page the passage ends on.
        section_path: Heading trail, joined with " > ".
        block_type: The converter's label for the source item.
        doc_item_refs: The converter's identifiers for those items.
        bbox: One box per page spanned, [{page, l, t, r, b}], top-left
            origin.
        table_cells: The cell grids behind a table passage.
        language: Detected on this passage, not inherited from the document.
        lemmas: Content lemmas, the vocabulary topic modelling is fitted
            over.
        sentences: The units a fact may cite, [{i, start, end, predicates}],
            with offsets into text. Sentences for prose, rendered rows for a
            table.
    """

    ordinal: int
    text: str
    page_from: int | None
    page_to: int | None
    section_path: str | None
    block_type: str | None
    doc_item_refs: list[str] = field(default_factory=list)
    bbox: list[dict] = field(default_factory=list)
    table_cells: list[dict] = field(default_factory=list)
    language: str | None = None
    sentences: list[dict] = field(default_factory=list)
    lemmas: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Chunking:
    """What the chunker made of one document.

    Attributes:
        passages: Every passage, numbered.
        oversized: Passages above the token budget. A tripwire; nothing is
            discarded on size.
    """

    passages: list[Chunk]
    oversized: int


@dataclass(frozen=True)
class StoredPassage:
    """One passage as it is read back out, for display.

    Carries counts rather than the grids and sentences themselves.

    Attributes:
        id: The passage's row id.
        doc_sha256: The document it was cut from.
        ordinal: Position in reading order, from 1.
        text: The passage content.
        page_from: Page the passage starts on.
        page_to: Page the passage ends on.
        section_path: Heading trail.
        block_type: The converter's label for the source item.
        language: ISO 639-1 code, or None.
        doc_item_refs: The converter's identifiers for those items.
        bbox: One box per page spanned.
        table_count: Cell grids this passage holds.
        sentence_count: Numbered units a fact may cite.
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
    """One passage in full, cell grids and sentences included.

    Attributes:
        passage: The passage itself.
        table_cells: Its cell grids.
        sentences: Its numbered units.
        extract_status: Where extraction has got to on it.
        extract_error: Why extraction failed on it.
    """

    passage: StoredPassage
    table_cells: list[dict]
    sentences: list[dict]
    extract_status: str
    extract_error: str | None
