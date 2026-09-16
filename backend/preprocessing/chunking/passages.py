"""Cutting a converted document into passages."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace

from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.serializer.markdown import (
    MarkdownPictureSerializer,
    MarkdownTableSerializer,
)
from docling_core.types.doc.common.reference import RefItem
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.node import DocItem
from docling_core.types.doc.labels import DocItemLabel

from nlp.analysis import read
from nlp.language import detect
from preprocessing.chunking.models import Chunk, Chunking


class NoPassages(Exception):
    """Raised when a document yielded no passages to store."""


class _Markdown(ChunkingSerializerProvider):
    """Renders every kind of block into the passage text, in Markdown."""

    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        """Builds the serializer the chunker uses for one document."""
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            picture_serializer=MarkdownPictureSerializer(),
        )


def chunking_of(
    chunks: Iterable,
    *,
    max_tokens: int,
    count_tokens: Callable[[str], int],
    to_passage: Callable[[int, object], Chunk],
) -> Chunking:
    """Numbers every chunk the chunker produced and counts the oversized ones.

    All of the builder's decisions and none of its machinery.

    Args:
        chunks: What the chunker returned.
        max_tokens: The token budget.
        count_tokens: Counts the tokens in a string.
        to_passage: Turns an ordinal and a chunk into a passage.

    Returns:
        The numbered passages and how many exceed the budget.

    Raises:
        NoPassages: If no chunk holds anything but whitespace.
    """
    kept = [chunk for chunk in chunks if chunk.text.strip()]
    if not kept:
        raise NoPassages("the document yielded no passages")

    passages = [to_passage(ordinal, chunk) for ordinal, chunk in enumerate(kept, 1)]
    oversized = sum(
        1 for passage in passages if count_tokens(passage.text) > max_tokens
    )
    return Chunking(passages=passages, oversized=oversized)


def lines_of(text: str) -> list[dict]:
    """Locates every non-empty line of a rendered table passage.

    A table has no sentences, so its rows are what a fact cites.

    Args:
        text: The rendered passage text.

    Returns:
        One {i, start, end, predicates} per non-empty line, numbered from
        zero, with offsets into `text`.
    """
    found: list[dict] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped:
            start = offset + line.index(stripped[0])
            found.append(
                {
                    "i": len(found),
                    "start": start,
                    "end": start + len(stripped),
                    "predicates": 0,
                }
            )
        offset += len(line)
    return found


class PassageBuilder:
    """Splits a converted document into retrievable passages.

    Uses the converter's own hybrid chunker, which cuts on the document's
    structure before merging or dividing by token count, so a passage keeps
    its heading trail.
    """

    def __init__(
        self, *, embedding_model: str, max_tokens: int, merge_peers: bool
    ) -> None:
        """Initialises the builder and its chunker.

        Args:
            embedding_model: The model whose tokenizer sizes a passage.
            max_tokens: The token budget a passage is cut to.
            merge_peers: Whether to combine undersized neighbours that share
                a heading.
        """
        self._max_tokens = max_tokens
        self._tokenizer = HuggingFaceTokenizer.from_pretrained(
            model_name=embedding_model, max_tokens=max_tokens
        )
        self._chunker = HybridChunker(
            tokenizer=self._tokenizer,
            merge_peers=merge_peers,
            serializer_provider=_Markdown(),
        )

    def build(self, document: DoclingDocument, language: str | None) -> Chunking:
        """Cuts one document into passages and reads their linguistic surface.

        Args:
            document: The converted document.
            language: The document's language, used where a passage is too
                short to detect its own.

        Returns:
            The numbered passages and how many exceed the budget.

        Raises:
            NoPassages: If the document yielded nothing but whitespace.
        """
        chunking = chunking_of(
            self._chunker.chunk(document),
            max_tokens=self._max_tokens,
            count_tokens=self._tokenizer.count_tokens,
            to_passage=lambda ordinal, chunk: self._passage(document, ordinal, chunk),
        )
        return replace(chunking, passages=self._read(chunking.passages, language))

    @staticmethod
    def _read(passages: list[Chunk], language: str | None) -> list[Chunk]:
        """Adds each passage's language, numbered units and lemmas.

        The language is detected per passage; one too short to judge keeps
        the document's. Every passage is read for vocabulary, a table
        included. Only the numbering differs: a table's rows are numbered
        rather than its sentences split. Passages are parsed in one batch per
        language.

        Args:
            passages: The numbered passages.
            language: The document's language.

        Returns:
            The same passages, each carrying its language, units and lemmas.
        """
        spoken = [
            (index, detect(p.text) or language) for index, p in enumerate(passages)
        ]
        parsed: dict[int, tuple] = {}
        for code in {found for _, found in spoken}:
            batch = [index for index, found in spoken if found == code]
            parsed |= dict(
                zip(
                    batch,
                    read((passages[index].text for index in batch), code),
                    strict=True,
                )
            )

        by_index = dict(spoken)
        surfaced = []
        for index, passage in enumerate(passages):
            sentences, lemmas = parsed[index]
            numbered = (
                lines_of(passage.text)
                if passage.table_cells
                else [
                    {
                        "i": s.index,
                        "start": s.start,
                        "end": s.end,
                        "predicates": s.predicates,
                    }
                    for s in sentences
                ]
            )
            surfaced.append(
                replace(
                    passage,
                    language=by_index[index],
                    sentences=numbered,
                    lemmas=lemmas,
                )
            )
        return surfaced

    @staticmethod
    def _passage(document: DoclingDocument, ordinal: int, chunk) -> Chunk:
        """Reads one chunk into the values the passages table holds."""
        # Every evidence offset is an index into this string.
        text = chunk.text.strip()
        # Resolved once and handed to both readers below, so the cells and the
        # box drawn over them agree on which rows this passage rendered.
        tables = _tables_rendered(document, chunk.meta.doc_items, text)
        boxes = _boxes(document, chunk.meta.doc_items, tables)
        return Chunk(
            ordinal=ordinal,
            text=text,
            page_from=boxes[0]["page"] if boxes else None,
            page_to=boxes[-1]["page"] if boxes else None,
            section_path=" > ".join(chunk.meta.headings)
            if chunk.meta.headings
            else None,
            block_type=_block_type(chunk.meta.doc_items),
            doc_item_refs=[item.self_ref for item in chunk.meta.doc_items],
            bbox=boxes,
            table_cells=_grids(document, chunk.meta.doc_items, tables),
        )


def _block_type(items: list[DocItem]) -> str | None:
    """Labels a passage for the extractor that will read it.

    A table anywhere in a merged passage wins; otherwise the first item's
    label.
    """
    if not items:
        return None
    if any(item.label == DocItemLabel.TABLE for item in items):
        return DocItemLabel.TABLE.value
    return items[0].label.value


def _tables_rendered(
    document: DoclingDocument, items: list[DocItem], rendered: str
) -> dict[str, tuple]:
    """Resolves every table in a passage and the rows this passage renders.

    Args:
        document: The converted document.
        items: The items this passage was cut from.
        rendered: The rendered passage text.

    Returns:
        The table and its paired rows, by the item's self reference.
    """
    numbered = lines_of(rendered)
    tables = {}
    for reference in items:
        if reference.label != DocItemLabel.TABLE:
            continue
        # Through the alias rather than the field name: it is the serialised
        # form and the only one a checker sees on the constructor.
        table = RefItem.model_validate({"$ref": reference.self_ref}).resolve(document)
        tables[reference.self_ref] = (
            table,
            _rendered_rows(table, rendered, numbered),
        )
    return tables


def _cell_box(table, rows: list[tuple[list, dict | None]]) -> tuple[int, list] | None:
    """Locates the rows of a table that one passage renders.

    Header cells are left out although the piece repeats them.

    Args:
        table: The resolved table item.
        rows: Its rows, each paired with the line rendering it or None.

    Returns:
        The page and the cell boxes on it, or None when either is missing.
    """
    boxes = [
        cell.bbox
        for cells, line in rows
        if line is not None
        for cell in cells
        if cell.bbox is not None
    ]
    page = next((prov.page_no for prov in table.prov or []), None)
    # ponytail: one page per table, which holds for every table in this
    # corpus. A table broken across a page boundary would need its cells
    # grouped by which page each one falls on.
    return (page, boxes) if page is not None and boxes else None


def _boxes(
    document: DoclingDocument, items: list[DocItem], tables: dict[str, tuple]
) -> list[dict]:
    """Locates a passage on the pages it spans, one box per page.

    Every box is normalised to a top-left origin.

    Args:
        document: The converted document.
        items: The items this passage was cut from.
        tables: What `_tables_rendered` resolved.

    Returns:
        One {page, l, t, r, b} per page spanned, in page order.
    """
    by_page: dict[int, list] = {}
    for reference in items:
        if reference.self_ref in tables:
            placed = _cell_box(*tables[reference.self_ref])
            if placed is not None:
                page_no, boxes = placed
                by_page.setdefault(page_no, []).extend(boxes)
                continue
        for provenance in reference.prov or []:
            page = document.pages.get(provenance.page_no)
            if page is None:
                continue
            box = provenance.bbox.to_top_left_origin(page_height=page.size.height)
            by_page.setdefault(provenance.page_no, []).append(box)

    return [
        {
            "page": page_no,
            "l": round(min(box.l for box in boxes), 1),
            "t": round(min(box.t for box in boxes), 1),
            "r": round(max(box.r for box in boxes), 1),
            "b": round(max(box.b for box in boxes), 1),
        }
        for page_no, boxes in sorted(by_page.items())
    ]


def _fields(line: str) -> list[str]:
    """Splits one rendered Markdown row into its cell values."""
    return [field.strip() for field in line.strip("|").split("|")]


def _rendered_rows(
    table, rendered: str, numbered: list[dict]
) -> list[tuple[list, dict | None]]:
    """Pairs each row of a table with the numbered line that renders it.

    A row matches by position: every one of its values must sit at its own
    column index. Lines are consumed in order, so two identical rows take two
    different lines. A data row this passage does not render is dropped.
    Header rows are kept whether matched or not, and so carry no line; a
    `row_header` cell does not make its row one.

    Args:
        table: The resolved table item.
        rendered: The rendered passage text.
        numbered: What `lines_of` found in it.

    Returns:
        One (cells, line) per kept row, in row order.
    """
    lines = [
        line
        for line in numbered
        if (text := rendered[line["start"] : line["end"]]).startswith("|")
        and set(text) - set("|-: ")
    ]
    taken = [False] * len(lines)

    by_row: dict[int, list] = {}
    for cell in table.data.table_cells:
        by_row.setdefault(cell.start_row_offset_idx, []).append(cell)

    kept: list[tuple[list, dict | None]] = []
    for _, cells in sorted(by_row.items()):
        if any(cell.column_header for cell in cells):
            kept.append((cells, None))
            continue
        placed = [
            (cell.start_col_offset_idx, cell.text.strip())
            for cell in cells
            if cell.text.strip()
        ]
        if not placed:
            continue
        for index, line in enumerate(lines):
            if taken[index]:
                continue
            fields = _fields(rendered[line["start"] : line["end"]])
            if all(col < len(fields) and fields[col] == value for col, value in placed):
                taken[index] = True
                kept.append((cells, line))
                break
    return kept


def _grids(
    document: DoclingDocument, items: list[DocItem], tables: dict[str, tuple]
) -> list[dict]:
    """Reads the cell grid out of every table in a passage.

    Header flags, position and spans are all lost in the rendered string.
    Each data cell carries the numbered line it sits in, which is what a fact
    drawn from it cites.

    Args:
        document: The converted document.
        items: The items this passage was cut from.
        tables: What `_tables_rendered` resolved.

    Returns:
        One grid per table, each {caption, num_rows, num_cols, cells}.
    """
    grids = []
    for reference in items:
        if reference.self_ref not in tables:
            continue
        table, rows = tables[reference.self_ref]
        grids.append(
            {
                "caption": table.caption_text(document).strip() or None,
                "num_rows": table.data.num_rows,
                "num_cols": table.data.num_cols,
                "cells": [
                    {
                        "row": cell.start_row_offset_idx,
                        "col": cell.start_col_offset_idx,
                        "row_span": cell.row_span,
                        "col_span": cell.col_span,
                        # Two flags, not one: a corner cell heads its column
                        # and not its row.
                        "column_header": bool(cell.column_header),
                        "row_header": bool(cell.row_header),
                        "text": cell.text,
                        "line": None if line is None else line["i"],
                    }
                    for cells, line in rows
                    for cell in cells
                ],
            }
        )
    return grids
