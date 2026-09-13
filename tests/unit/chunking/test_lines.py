"""Numbering a rendered table, which a cell fact cites the same way prose is."""

from __future__ import annotations

from preprocessing.chunking.passages import lines_of

RENDERED = "Table 1: Models\n\n| Model | Mass |\n| --- | --- |\n| Compact | 4 |\n"


def test_every_rendered_line_is_numbered_in_order() -> None:
    """A table's lines carry contiguous indices from zero."""
    assert [line["i"] for line in lines_of(RENDERED)] == [0, 1, 2, 3]


def test_a_line_slices_to_itself_without_surrounding_space() -> None:
    """Each span holds the line's own text and no whitespace around it."""
    for line in lines_of(RENDERED):
        sliced = RENDERED[line["start"] : line["end"]]
        assert sliced.strip() == sliced and sliced, repr(sliced)


def test_a_row_resolves_to_the_row() -> None:
    """The numbered span of a row is that row."""
    lines = lines_of(RENDERED)
    assert RENDERED[lines[3]["start"] : lines[3]["end"]] == "| Compact | 4 |"


def test_leading_whitespace_is_not_counted_into_the_span() -> None:
    """An indented row starts where its text does."""
    indented = lines_of("  | a | b |\n")
    assert indented[0]["start"] == 2, indented
