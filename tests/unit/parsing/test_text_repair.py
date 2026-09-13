"""Repairing the text a PDF's content stream carried.

These fonts map their discretionary hyphen to U+0002, which is invisible: a
word stored with it in place of each line break matches no search and no
vocabulary term.
"""

from __future__ import annotations

import pytest

from preprocessing.parsing.pipelines.pdf import _mend, _repaired


@pytest.mark.parametrize(
    ("broken", "whole"),
    [
        ("Lebens\x02 versiche\x02 rer", "Lebensversicherer"),
        ("Risikoklassifizie\x02rung", "Risikoklassifizierung"),
        ("Lebens­ versiche­rer", "Lebensversicherer"),
    ],
)
def test_a_discretionary_hyphen_joins_the_halves(broken: str, whole: str) -> None:
    """U+0002 and U+00AD are the same mark and join the same way."""
    assert _mend(broken) == whole


@pytest.mark.parametrize(
    ("broken", "whole"),
    [
        ("Registrierungsformu- lare", "Registrierungsformulare"),
        ("Ge- schäftsleiter", "Geschäftsleiter"),
        ("ergän- zende", "ergänzende"),
        ("Institu- te", "Institute"),
        ("Nied- rig", "Niedrig"),
    ],
)
def test_a_real_hyphen_at_a_line_break_joins(broken: str, whole: str) -> None:
    """The converter turned the break into a space; the halves are one word."""
    assert _mend(broken) == whole


@pytest.mark.parametrize(
    "kept",
    [
        "Zoll- und Steuerrecht",
        "Über- oder Unterdeckung",
        "Qualitäts- bzw. Quantitätsprüfung",
        "Geschäftsfortführungs- sowie Notfallpläne",
        "pre- and post-trade",
        "Schaden- / Unfallversicherer",
        "Anlage- Verwalter",
        "Krypto-Transaktionen",
    ],
)
def test_a_hyphen_the_author_typed_survives(kept: str) -> None:
    """A shared suffix completed by a conjunction is not a broken word."""
    assert _mend(kept) == kept


def test_control_characters_go_and_newlines_and_tabs_stay() -> None:
    """Only the two the passage text is allowed to keep survive."""
    assert _mend("a\nb\tc\x07d") == "a\nb\tcd"


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("Unternehmenskrediten →", "Unternehmenskrediten →"),
        ("12,50 EUR netto", "12,50 EUR netto"),
        ("Bafin\u200b﻿ meldet", "Bafin meldet"),
    ],
)
def test_every_space_becomes_the_one_a_model_types_back(given, expected) -> None:
    """Non-breaking and zero-width marks do not survive into passage text."""
    assert _mend(given) == expected


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("Bußgelder für Häuser", "Bußgelder für Häuser"),
        ("ü", "ü"),
    ],
)
def test_a_decomposed_umlaut_is_composed(given: str, expected: str) -> None:
    """Composed and decomposed umlauts look identical and compare unequal."""
    assert _mend(given) == expected


def test_the_repair_reaches_the_text_and_the_grid() -> None:
    """A passage's table_cells are read from the grid, not from the rendering."""
    from docling_core.types.doc.document import DoclingDocument, TableCell, TableData
    from docling_core.types.doc.labels import DocItemLabel

    document = DoclingDocument(name="t")
    document.add_text(label=DocItemLabel.TEXT, text="Ände\x02 rung zum Vor\x02 jahr")
    document.add_table(
        data=TableData(
            num_rows=1,
            num_cols=1,
            table_cells=[
                TableCell(
                    text="Pensi\x02 onskas\x02 sen",
                    start_row_offset_idx=0,
                    end_row_offset_idx=1,
                    start_col_offset_idx=0,
                    end_col_offset_idx=1,
                )
            ],
        )
    )

    mended = _repaired(document)
    assert mended.texts[0].text == "Änderung zum Vorjahr", mended.texts[0].text
    cell = mended.tables[0].data.table_cells[0].text
    assert cell == "Pensionskassen", cell
