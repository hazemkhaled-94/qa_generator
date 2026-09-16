"""The deterministic table extractor."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import ClassVar

from extraction.extractors.base import Extractor
from extraction.models import CandidateFact, PassageToExtract

log = logging.getLogger(__name__)


class TableExtractor(Extractor):
    """Reads facts straight out of a table's cells, with no model involved.

    Walks the stored cell grid, labelling each value cell with the headers
    above and to the left of it. Every fact cites the numbered line the
    chunker recorded on the cell, so a table citation is an index like any
    other and the checker needs no rule of its own.
    """

    block_types: ClassVar[tuple[str, ...]] = ("table",)
    method: ClassVar[str] = "deterministic"

    def extract(self, passage: PassageToExtract) -> list[CandidateFact]:
        """Reads every data cell of every table in the passage.

        Args:
            passage: The table passage to read.

        Returns:
            One candidate per labelled value cell the passage renders.
        """
        facts: list[CandidateFact] = []
        for grid in passage.table_cells:
            # The caption in preference to the heading trail: it is the one
            # that says what the table holds and from when.
            facts.extend(self._grid(grid, grid.get("caption") or _nearest(passage)))
        log.debug("passage %d: %d cell fact(s)", passage.id, len(facts))
        return facts

    def _grid(self, grid: dict, scope: str | None) -> list[CandidateFact]:
        """Reads one table into one candidate per labelled data cell.

        Both a row label and a column label are required, and every header
        above a cell is used rather than only the nearest: a grouped table
        would otherwise give several columns the same label, and so several
        facts identical in wording and different in value.
        """
        cells = grid.get("cells", [])
        columns = self._column_labels(cells)
        rows = self._row_labels(cells)

        facts: list[CandidateFact] = []
        for cell in cells:
            text = (cell.get("text") or "").strip()
            line = cell.get("line")
            if not text or _is_header(cell) or line is None:
                continue
            row_label = " ".join(rows.get(cell.get("row"), ()))
            column_label = " ".join(columns.get(cell.get("col"), ()))
            if not row_label or not column_label or text in (row_label, column_label):
                continue
            labels = [part for part in (scope, row_label, column_label) if part]
            facts.append(
                CandidateFact(
                    statement=f"{' - '.join(labels)}: {text}", sentences=(line,)
                )
            )
        return facts

    @staticmethod
    def _column_labels(cells: list[dict]) -> dict[int, tuple[str, ...]]:
        """Reads the full header stack above each column, spans included."""
        labels: dict[int, list[tuple[int, str]]] = {}
        for cell in cells:
            text = (cell.get("text") or "").strip()
            if not cell.get("column_header") or not text:
                continue
            first = cell["col"]
            for col in range(first, first + max(1, cell.get("col_span", 1))):
                labels.setdefault(col, []).append((cell["row"], text))
        return {
            col: tuple(text for _, text in sorted(found))
            for col, found in labels.items()
        }

    @staticmethod
    def _row_labels(cells: list[dict]) -> dict[int, tuple[str, ...]]:
        """Reads the full label stack to the left of each row.

        ponytail: falls back to the first column, which labels the row for a
        column-oriented table and not for a transposed one. The stored grid
        carries enough to settle it, once a document needs it.
        """
        marked = _stack(cells, lambda cell: bool(cell.get("row_header")))
        if marked:
            return marked
        return _stack(cells, lambda cell: cell.get("col") == 0 and not _is_header(cell))


def _stack(
    cells: list[dict], keep: Callable[[dict], bool]
) -> dict[int, tuple[str, ...]]:
    """Collects the labels covering each row, ordered by column."""
    labels: dict[int, list[tuple[int, str]]] = {}
    for cell in cells:
        text = (cell.get("text") or "").strip()
        if not keep(cell) or not text:
            continue
        first = cell["row"]
        for row in range(first, first + max(1, cell.get("row_span", 1))):
            labels.setdefault(row, []).append((cell["col"], text))
    return {
        row: tuple(text for _, text in sorted(found)) for row, found in labels.items()
    }


def _nearest(passage: PassageToExtract) -> str | None:
    """Names a table by the heading directly above it.

    The last segment only: the whole trail would repeat the document title in
    every one of a table's sixty facts.
    """
    if not passage.section_path:
        return None
    return passage.section_path.rsplit(" > ", 1)[-1].strip() or None


def _is_header(cell: dict) -> bool:
    """Reports whether a cell heads its column or its row."""
    return bool(cell.get("column_header") or cell.get("row_header"))
