"""Reading facts off a table's grid.

A cell is labelled from its coordinates and cites the numbered row its own
value sits in, rather than a row found by searching the rendered text.
"""

from __future__ import annotations

from typing import Any

import pytest

from extraction.extractors.table import TableExtractor
from extraction.models import PassageToExtract
from extraction.validation import FactChecker
from nlp.models import Sentence
from preprocessing.chunking.passages import lines_of

RENDERED = "Table 1: Models\n\n| Model | Mass |\n| --- | --- |\n| Compact | 4 |\n"


def cell(row: int, col: int, text: str, line: int | None, **flags: bool) -> dict:
    """Builds one stored cell."""
    return {
        "row": row,
        "col": col,
        "row_span": 1,
        "col_span": 1,
        "column_header": flags.get("column_header", False),
        "row_header": flags.get("row_header", False),
        "text": text,
        "line": line,
    }


#: The Standard row is in the table but not in this passage, so the chunker
#: kept no line for it.
GRID: dict[str, Any] = {
    "caption": "Table 1: Models",
    "num_rows": 3,
    "num_cols": 2,
    "cells": [
        cell(0, 0, "Model", None, column_header=True),
        cell(0, 1, "Mass", None, column_header=True),
        cell(2, 0, "Compact", 3, row_header=True),
        cell(2, 1, "4", 3),
    ],
}


def table(grid: dict) -> PassageToExtract:
    """Builds the table passage, numbered as chunking stores it."""
    return PassageToExtract(
        id=1,
        text=RENDERED,
        section_path="Models",
        block_type="table",
        language="en",
        sentences=[
            Sentence(
                index=line["i"],
                start=line["start"],
                end=line["end"],
                text=RENDERED[line["start"] : line["end"]],
                predicates=0,
            )
            for line in lines_of(RENDERED)
        ],
        table_cells=[grid],
    )


def test_a_value_cell_becomes_one_labelled_fact() -> None:
    """The header cells label the value; they produce no fact of their own."""
    facts = TableExtractor().extract(table(GRID))

    assert len(facts) == 1, [f.statement for f in facts]
    assert facts[0].statement == "Table 1: Models - Compact - Mass: 4"
    assert facts[0].sentences == (3,), facts[0].sentences


@pytest.mark.nlp
def test_the_citation_resolves_to_the_row_holding_the_value() -> None:
    """The cited row is the row the value sits in."""
    passage = table(GRID)
    fact = TableExtractor().extract(passage)[0]

    checked = FactChecker().check(passage, fact, "deterministic")
    assert checked.validated, checked.validation_error
    assert checked.evidence_text == "| Compact | 4 |", checked.evidence_text


def test_a_row_the_passage_does_not_render_produces_no_fact() -> None:
    """A cell with no line has no evidence, so it yields nothing."""
    orphan = dict(
        GRID,
        cells=[
            *GRID["cells"],
            cell(1, 0, "Standard", None, row_header=True),
            cell(1, 1, "9", None),
        ],
    )
    assert len(TableExtractor().extract(table(orphan))) == 1
