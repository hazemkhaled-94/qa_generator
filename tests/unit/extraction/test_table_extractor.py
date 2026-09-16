"""Reading facts off a table's grid.

A cell is labelled from its coordinates and cites the numbered row its own
value sits in, rather than a row found by searching the rendered text.
"""

from __future__ import annotations

import pytest
from drivers import GRID, Checker, cell, table_passage

from extraction.extractors.table import TableExtractor
from extraction.models import CandidateFact


def test_a_value_cell_becomes_one_labelled_fact() -> None:
    """The header cells label the value; they produce no fact of their own."""
    facts = TableExtractor().extract(table_passage())

    assert len(facts) == 1, [f.statement for f in facts]
    assert facts[0].statement == "Table 1: Models - Compact - Mass: 4"
    assert facts[0].sentences == (3,), facts[0].sentences


@pytest.mark.nlp
def test_the_citation_resolves_to_the_row_holding_the_value() -> None:
    """The cited row is the row the value sits in."""
    under = table_passage()
    fact = TableExtractor().extract(under)[0]

    checked = Checker(under).checker.check(under, fact, "deterministic")
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
    assert len(TableExtractor().extract(table_passage(orphan))) == 1


def test_a_cell_repeating_one_of_its_own_labels_yields_nothing() -> None:
    """`Compact - Model: Compact` states nothing the label did not."""
    echoed = dict(
        GRID,
        cells=[
            cell(0, 0, "Model", None, column_header=True),
            cell(0, 1, "Mass", None, column_header=True),
            cell(2, 0, "Compact", 3, row_header=True),
            cell(2, 1, "Compact", 3),
        ],
    )
    assert TableExtractor().extract(table_passage(echoed)) == []


def test_a_grid_with_no_row_header_falls_back_to_the_first_column() -> None:
    """A column-oriented table labels its rows there."""
    unmarked = dict(
        GRID,
        cells=[
            cell(0, 0, "Model", None, column_header=True),
            cell(0, 1, "Mass", None, column_header=True),
            cell(2, 0, "Compact", 3),
            cell(2, 1, "4", 3),
        ],
    )
    facts = TableExtractor().extract(table_passage(unmarked))

    assert [fact.statement for fact in facts] == ["Table 1: Models - Compact - Mass: 4"]


def test_a_spanning_header_labels_every_column_it_covers() -> None:
    """A grouped table would otherwise give two columns one label."""
    spanned = dict(
        GRID,
        cells=[
            {**cell(0, 0, "Measurements", None, column_header=True), "col_span": 2},
            cell(1, 0, "Model", None, column_header=True),
            cell(1, 1, "Mass", None, column_header=True),
            cell(2, 0, "Compact", 3, row_header=True),
            cell(2, 1, "4", 3),
        ],
    )
    facts = TableExtractor().extract(table_passage(spanned))

    assert facts[0].statement == ("Table 1: Models - Compact - Measurements Mass: 4"), (
        facts[0].statement
    )


def test_a_table_with_no_caption_is_named_by_the_heading_above_it() -> None:
    """The last segment only, so sixty facts do not repeat the whole trail."""
    from dataclasses import replace

    uncaptioned = {k: v for k, v in GRID.items() if k != "caption"}
    under = replace(table_passage(uncaptioned), section_path="Report > 3 > Models")
    facts = TableExtractor().extract(under)

    assert facts[0].statement == "Models - Compact - Mass: 4", facts[0].statement


def test_a_table_with_neither_caption_nor_heading_is_labelled_by_its_cells() -> None:
    """Row and column are required; a scope is not."""
    from dataclasses import replace

    uncaptioned = {k: v for k, v in GRID.items() if k != "caption"}
    under = replace(table_passage(uncaptioned), section_path=None)
    facts = TableExtractor().extract(under)

    assert facts[0].statement == "Compact - Mass: 4", facts[0].statement


def test_a_fact_from_a_grid_is_atomic_and_names_no_group() -> None:
    """The kind is what routes it to the checks a composed statement faces."""
    from database.qa_generator import FactKind

    fact = TableExtractor().extract(table_passage())[0]

    assert isinstance(fact, CandidateFact)
    assert fact.kind == FactKind.ATOMIC
    assert fact.passages == ()
