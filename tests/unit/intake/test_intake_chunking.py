"""Reading a chunk into a passage, and the flow that stores them.

The hybrid chunker is not run here: constructing it downloads the embedding
model's tokenizer. Everything it hands over - the rendered text, the items it
was cut from, the heading trail - is what these read, off real Docling
documents.
"""

from __future__ import annotations

import pytest
from docling_core.types.doc.base import CoordOrigin
from docling_core.types.doc.labels import DocItemLabel
from intake_drivers import (
    ChunkDriver,
    RevocabularyDriver,
    ScriptedBuilder,
    cell,
    chunk_of,
    docling_document,
    placed,
    table_of,
)

from preprocessing.chunking.models import Chunk, Chunking, ClaimedDocument
from preprocessing.chunking.passages import (
    NoPassages,
    PassageBuilder,
    _block_type,
    _boxes,
    _grids,
    _rendered_rows,
    _tables_rendered,
    chunking_of,
    lines_of,
)

SHA = "a" * 64
OTHER = "b" * 64

#: A rendered table as the Markdown serializer produces one.
RENDERED = (
    "Table 1: Models\n\n"
    "| Model | Mass |\n"
    "| --- | --- |\n"
    "| Compact | 4 |\n"
    "| Large | 9 |\n"
)


def passage_of(document, chunk) -> Chunk:
    """Reads one chunk the way the builder reads it."""
    return PassageBuilder._passage(document, 1, chunk)


# ── Numbering and the token budget ─────────────────────────────────────────


class _Chunk:
    """A stand-in carrying the attribute the chunk reader reads off a chunk."""

    def __init__(self, text: str) -> None:
        """Initialises the chunk with its text."""
        self.text = text


def counted(texts: tuple[str, ...], max_tokens: int = 10) -> Chunking:
    """Runs the chunk reader over some texts, counting a word as a token."""
    return chunking_of(
        [_Chunk(text) for text in texts],
        max_tokens=max_tokens,
        count_tokens=lambda text: len(text.split()),
        to_passage=lambda ordinal, chunk: Chunk(
            ordinal=ordinal,
            text=chunk.text.strip(),
            page_from=None,
            page_to=None,
            section_path=None,
            block_type=None,
        ),
    )


def test_a_chunk_exactly_at_the_budget_is_not_oversized() -> None:
    """The budget is what fits, not what does not."""
    assert counted(("x " * 10,), max_tokens=10).oversized == 0
    assert counted(("x " * 11,), max_tokens=10).oversized == 1


def test_the_budget_is_counted_on_the_text_that_is_stored() -> None:
    """Not on the raw chunk, which carries whitespace the passage will not."""
    assert counted(("  " + "x " * 10 + "  ",), max_tokens=10).oversized == 0


def test_the_ordinals_are_assigned_after_the_blanks_are_dropped() -> None:
    """So a reader counts on there being no hole."""
    stored = counted(("a", "   ", "b", "\n", "c")).passages

    assert [one.ordinal for one in stored] == [1, 2, 3]
    assert [one.text for one in stored] == ["a", "b", "c"]


# ── Numbering a rendered table ─────────────────────────────────────────────


def test_a_table_with_windows_line_endings_is_numbered_the_same() -> None:
    """The converter's output is not guaranteed to use one of them."""
    assert [one["i"] for one in lines_of("| a |\r\n| b |\r\n")] == [0, 1]
    assert lines_of("| a |\r\n")[0]["end"] == 5


def test_a_line_of_only_whitespace_is_not_numbered() -> None:
    """Nothing could be cited from it."""
    assert [one["i"] for one in lines_of("| a |\n   \n\t\n| b |\n")] == [0, 1]


def test_text_with_no_lines_is_numbered_as_nothing() -> None:
    """An empty passage has nothing a fact could cite."""
    assert lines_of("") == []
    assert lines_of("\n\n") == []


def test_a_numbered_table_row_carries_no_predicates() -> None:
    """A table makes no claims; its rows are cited, not parsed."""
    assert all(one["predicates"] == 0 for one in lines_of(RENDERED))


# ── Labelling a passage ────────────────────────────────────────────────────


def test_a_passage_cut_from_nothing_has_no_block_type() -> None:
    """Which is what the column holds for a passage with no items."""
    assert _block_type([]) is None


def test_a_table_anywhere_in_a_merged_passage_wins() -> None:
    """Sending a table to the model is the more expensive mistake."""
    document = docling_document(paragraphs=("Before.", "After."))
    table = table_of(
        document, [cell("Model", 0, 0, column_header=True)], num_rows=1, num_cols=1
    )
    items = [document.texts[0], table, document.texts[1]]

    assert _block_type(items) == DocItemLabel.TABLE.value


def test_otherwise_a_passage_is_labelled_by_its_first_item() -> None:
    """Which is the item the extractor's router reads."""
    document = docling_document(headings=("1 Scope",), paragraphs=("Text.",))

    assert _block_type(list(document.texts)) == DocItemLabel.SECTION_HEADER.value


# ── Pairing a table's rows with the lines that render them ─────────────────


def rows_of(cells, *, num_rows, num_cols, rendered=RENDERED):
    """Pairs one table's rows with the lines of a rendered passage."""
    document = docling_document()
    table = table_of(document, cells, num_rows=num_rows, num_cols=num_cols)
    return _rendered_rows(table, rendered, lines_of(rendered))


HEADERS = [
    cell("Model", 0, 0, column_header=True),
    cell("Mass", 0, 1, column_header=True),
]


def test_a_header_row_is_kept_and_carries_no_line() -> None:
    """The piece repeats the headers; they are not what a fact cites."""
    rows = rows_of(
        [*HEADERS, cell("Compact", 1, 0), cell("4", 1, 1)], num_rows=2, num_cols=2
    )

    assert [line is None for _, line in rows] == [True, False]


def test_a_data_row_cites_the_line_that_renders_it() -> None:
    """By position, so a table of repeated counts cannot match every row."""
    rows = rows_of(
        [
            *HEADERS,
            cell("Compact", 1, 0),
            cell("4", 1, 1),
            cell("Large", 2, 0),
            cell("9", 2, 1),
        ],
        num_rows=3,
        num_cols=2,
    )

    assert [None if line is None else line["i"] for _, line in rows] == [None, 3, 4]


def test_two_identical_rows_take_two_different_lines() -> None:
    """Consumed in order, so both are citable rather than both citing the first."""
    rendered = "| Model | Mass |\n| --- | --- |\n| Compact | 4 |\n| Compact | 4 |\n"
    rows = rows_of(
        [
            *HEADERS,
            cell("Compact", 1, 0),
            cell("4", 1, 1),
            cell("Compact", 2, 0),
            cell("4", 2, 1),
        ],
        num_rows=3,
        num_cols=2,
        rendered=rendered,
    )

    cited = [line["i"] for _, line in rows if line is not None]
    assert cited == [2, 3], cited


def test_a_row_this_passage_does_not_render_is_dropped() -> None:
    """It belongs to another piece of a table the chunker split."""
    rows = rows_of(
        [
            *HEADERS,
            cell("Compact", 1, 0),
            cell("4", 1, 1),
            cell("Elsewhere", 5, 0),
            cell("99", 5, 1),
        ],
        num_rows=6,
        num_cols=2,
    )

    assert [values[0].text for values, _ in rows] == ["Model", "Compact"]


def test_the_separator_line_is_never_matched() -> None:
    """It carries no values, so nothing may be cited from it."""
    rows = rows_of(
        [*HEADERS, cell("---", 1, 0), cell("---", 1, 1)], num_rows=2, num_cols=2
    )

    assert all(line is None or line["i"] != 2 for _, line in rows)


def test_a_row_whose_cells_are_all_empty_is_skipped() -> None:
    """There is nothing in it to label or to cite."""
    rows = rows_of([*HEADERS, cell("  ", 1, 0), cell("", 1, 1)], num_rows=2, num_cols=2)

    assert len(rows) == 1, rows


def test_a_row_label_does_not_make_its_row_a_header_row() -> None:
    """row_header marks the stub of an ordinary data row."""
    rows = rows_of(
        [*HEADERS, cell("Compact", 1, 0, row_header=True), cell("4", 1, 1)],
        num_rows=2,
        num_cols=2,
    )

    cited = [line["i"] for _, line in rows if line is not None]
    assert cited == [3], "a labelled data row was read as a header and left uncitable"


def test_a_row_wider_than_the_line_that_renders_it_is_not_matched() -> None:
    """A value at a column the rendering does not have is not that row."""
    rows = rows_of(
        [
            *HEADERS,
            cell("Compact", 1, 0),
            cell("4", 1, 1),
            cell("extra", 1, 7),
        ],
        num_rows=2,
        num_cols=8,
    )

    assert all(line is None for _, line in rows)


# ── The cell grid ──────────────────────────────────────────────────────────


def grid_of(cells, *, num_rows, num_cols, caption=None, rendered=RENDERED):
    """Reads one table's grid the way the chunk reader reads it."""
    document = docling_document()
    table = table_of(
        document, cells, num_rows=num_rows, num_cols=num_cols, caption=caption
    )
    tables = _tables_rendered(document, [table], rendered)
    return _grids(document, [table], tables)[0]


def test_the_grid_carries_the_shape_the_rendering_loses() -> None:
    """Header flags, position and spans are all gone in the string."""
    grid = grid_of(
        [
            cell("Mass", 0, 1, column_header=True, col_span=2),
            cell("Compact", 1, 0, row_header=True),
            cell("4", 1, 1),
        ],
        num_rows=2,
        num_cols=3,
        caption="Table 1: Models",
    )

    assert (grid["num_rows"], grid["num_cols"]) == (2, 3)
    assert grid["caption"] == "Table 1: Models"
    header = next(one for one in grid["cells"] if one["text"] == "Mass")
    assert (header["col_span"], header["column_header"], header["row_header"]) == (
        2,
        True,
        False,
    )
    stub = next(one for one in grid["cells"] if one["text"] == "Compact")
    assert (stub["row_header"], stub["column_header"]) == (True, False)


def test_a_caption_of_only_whitespace_is_no_caption() -> None:
    """Otherwise every fact from the table is labelled with a blank."""
    grid = grid_of(
        [*HEADERS, cell("Compact", 1, 0)], num_rows=2, num_cols=2, caption="  "
    )

    assert grid["caption"] is None


def test_a_header_cell_cites_nothing_and_a_data_cell_cites_its_row() -> None:
    """Which is what lets the checker treat a table citation like any other."""
    grid = grid_of(
        [*HEADERS, cell("Compact", 1, 0), cell("4", 1, 1)], num_rows=2, num_cols=2
    )

    lines = {one["text"]: one["line"] for one in grid["cells"]}
    assert lines == {"Model": None, "Mass": None, "Compact": 3, "4": 3}


def test_a_passage_holding_no_table_has_no_grid() -> None:
    """NULL rather than an empty list, which is what the column holds."""
    document = docling_document(paragraphs=("Text.",))

    assert _grids(document, list(document.texts), {}) == []


# ── Where a passage sits on the page ───────────────────────────────────────


def test_a_passage_is_located_once_per_page_it_spans() -> None:
    """One box per page, in page order."""
    document = docling_document(pages=((1, 200.0, 300.0), (2, 200.0, 300.0)))
    first = placed(document, "One.", page_no=1, box=(10, 250, 100, 200))
    second = placed(document, "Two.", page_no=2, box=(20, 150, 120, 100))

    boxes = _boxes(document, [first, second], {})

    assert [one["page"] for one in boxes] == [1, 2]
    assert boxes[0] == {"page": 1, "l": 10.0, "t": 50.0, "r": 100.0, "b": 100.0}


def test_a_box_is_normalised_to_a_top_left_origin() -> None:
    """The converter reports text from the bottom left."""
    document = docling_document(pages=((1, 200.0, 300.0),))
    item = placed(document, "One.", box=(10, 250, 100, 200))

    bottom_left = _boxes(document, [item], {})[0]

    top_left_document = docling_document(pages=((1, 200.0, 300.0),))
    already = placed(
        top_left_document,
        "One.",
        box=(10, 50, 100, 100),
        origin=CoordOrigin.TOPLEFT,
    )
    assert _boxes(top_left_document, [already], {})[0] == bottom_left


def test_two_items_on_one_page_are_one_box_covering_both() -> None:
    """A passage is a region, not a list of lines."""
    document = docling_document(pages=((1, 200.0, 300.0),))
    top = placed(document, "One.", box=(10, 280, 100, 260), origin=CoordOrigin.TOPLEFT)
    below = placed(
        document, "Two.", box=(20, 250, 150, 200), origin=CoordOrigin.TOPLEFT
    )

    boxes = _boxes(document, [top, below], {})

    assert boxes == [{"page": 1, "l": 10.0, "t": 250.0, "r": 150.0, "b": 260.0}]


def test_an_item_on_a_page_the_document_does_not_hold_is_not_placed() -> None:
    """There is no page height to flip its coordinates against."""
    document = docling_document()
    item = placed(document, "One.", page_no=4)

    assert _boxes(document, [item], {}) == []


def test_an_item_the_converter_never_placed_has_no_box() -> None:
    """Which is the NULL the column holds."""
    document = docling_document(paragraphs=("Text.",), pages=((1, 200.0, 300.0),))

    assert _boxes(document, list(document.texts), {}) == []


def test_a_table_is_boxed_over_the_rows_this_passage_renders() -> None:
    """A split table points every piece at the whole item."""
    document = docling_document(pages=((1, 200.0, 300.0),))
    table = table_of(
        document,
        [
            *HEADERS,
            cell("Compact", 1, 0, box=(10, 100, 60, 120)),
            cell("4", 1, 1, box=(70, 100, 90, 120)),
            cell("Elsewhere", 5, 0, box=(10, 900, 60, 920)),
            cell("99", 5, 1, box=(70, 900, 90, 920)),
        ],
        num_rows=6,
        num_cols=2,
        page_no=1,
    )
    tables = _tables_rendered(document, [table], RENDERED)

    boxes = _boxes(document, [table], tables)

    assert boxes == [{"page": 1, "l": 10.0, "t": 100.0, "r": 90.0, "b": 120.0}]


def test_a_table_whose_cells_carry_no_position_falls_back_to_the_item() -> None:
    """The item's own provenance is all there is to place it with."""
    document = docling_document(pages=((1, 200.0, 300.0),))
    table = table_of(
        document,
        [*HEADERS, cell("Compact", 1, 0), cell("4", 1, 1)],
        num_rows=2,
        num_cols=2,
        page_no=1,
    )
    tables = _tables_rendered(document, [table], RENDERED)

    assert _boxes(document, [table], tables) == [
        {"page": 1, "l": 0.0, "t": 0.0, "r": 1.0, "b": 1.0}
    ]


# ── One chunk into one passage ─────────────────────────────────────────────


def test_a_passage_is_stored_as_the_text_every_offset_indexes() -> None:
    """Stripped, because that is what a sentence offset counts from."""
    document = docling_document(paragraphs=("Text.",))

    read = passage_of(document, chunk_of("  The device weighs 4 kg.\n", ()))

    assert read.text == "The device weighs 4 kg."


def test_a_passage_carries_its_heading_trail() -> None:
    """Which is the only context the extractor is given."""
    document = docling_document(paragraphs=("Text.",))

    read = passage_of(
        document, chunk_of("Text.", (), headings=("3", "3.2", "Delivery"))
    )

    assert read.section_path == "3 > 3.2 > Delivery"


def test_a_passage_the_chunker_found_no_heading_for_has_none() -> None:
    """NULL rather than an empty string."""
    assert passage_of(docling_document(), chunk_of("Text.", ())).section_path is None


def test_a_passage_names_the_items_it_was_cut_from() -> None:
    """The only non-fuzzy way back to the converted document."""
    document = docling_document(paragraphs=("One.", "Two."))

    read = passage_of(document, chunk_of("One. Two.", tuple(document.texts)))

    assert read.doc_item_refs == ["#/texts/0", "#/texts/1"]


def test_a_passage_spanning_two_pages_records_the_first_and_the_last() -> None:
    """Which is what the Passages page shows as a range."""
    document = docling_document(pages=((1, 200.0, 300.0), (2, 200.0, 300.0)))
    first = placed(document, "One.", page_no=1)
    second = placed(document, "Two.", page_no=2)

    read = passage_of(document, chunk_of("One. Two.", (first, second)))

    assert (read.page_from, read.page_to) == (1, 2)


def test_a_passage_the_converter_never_placed_has_no_pages() -> None:
    """A document converted without a layout still yields passages."""
    document = docling_document(paragraphs=("Text.",))

    read = passage_of(document, chunk_of("Text.", tuple(document.texts)))

    assert (read.page_from, read.page_to) == (None, None)


def test_the_cells_and_the_box_agree_on_which_rows_were_rendered() -> None:
    """Both read one resolution, so they cannot describe different rows."""
    document = docling_document(pages=((1, 200.0, 300.0),))
    table = table_of(
        document,
        [
            *HEADERS,
            cell("Compact", 1, 0, box=(10, 100, 60, 120)),
            cell("4", 1, 1, box=(70, 100, 90, 120)),
            cell("Elsewhere", 5, 0, box=(10, 900, 60, 920)),
        ],
        num_rows=6,
        num_cols=2,
        page_no=1,
    )

    read = passage_of(document, chunk_of(RENDERED, (table,)))

    assert [one["text"] for one in read.table_cells[0]["cells"]] == [
        "Model",
        "Mass",
        "Compact",
        "4",
    ]
    assert read.bbox == [{"page": 1, "l": 10.0, "t": 100.0, "r": 90.0, "b": 120.0}]
    assert read.block_type == DocItemLabel.TABLE.value


# ── The flow ───────────────────────────────────────────────────────────────


def test_an_empty_queue_cuts_nothing() -> None:
    """Draining an empty queue is not an error."""
    driver = ChunkDriver()

    assert driver.run() is None
    assert driver.builder.calls == 0


def test_the_document_is_read_back_out_of_the_parsed_bucket() -> None:
    """Never the original file, so re-chunking never re-parses."""
    driver = ChunkDriver(
        queued=(ClaimedDocument(sha256=SHA, language="de"),),
        document=docling_document(name="held", paragraphs=("Text.",)),
    )

    driver.run()

    assert driver.builder.built[0][0].name == "held"


def test_the_builder_is_handed_the_documents_language_as_a_fallback() -> None:
    """A passage too short to judge is read under it."""
    driver = ChunkDriver(queued=(ClaimedDocument(sha256=SHA, language="de"),))

    driver.run()

    assert driver.language_given() == "de"


def test_a_document_with_no_language_is_still_chunked() -> None:
    """Detection then decides every passage on its own."""
    driver = ChunkDriver(queued=(ClaimedDocument(sha256=SHA, language=None),))

    assert driver.run() == SHA
    assert driver.language_given() is None
    assert driver.failure is None


def test_what_the_builder_produced_is_what_is_stored() -> None:
    """The passages and the oversized count land together."""
    made = Chunking(
        passages=[
            Chunk(
                ordinal=1,
                text="Text.",
                page_from=1,
                page_to=1,
                section_path=None,
                block_type="text",
            )
        ],
        oversized=0,
    )
    driver = ChunkDriver(
        queued=(ClaimedDocument(sha256=SHA, language="en"),),
        builder=ScriptedBuilder(made),
    )

    driver.run()

    assert driver.stored == {SHA: made}


def test_a_document_that_yielded_no_passages_is_failed() -> None:
    """Refused rather than stored as a document with nothing in it."""
    driver = ChunkDriver(
        queued=(ClaimedDocument(sha256=SHA, language="en"),),
        builder=ScriptedBuilder(raises=NoPassages("the document yielded no passages")),
    )

    assert driver.run() == SHA
    assert driver.failure == "the document yielded no passages"
    assert driver.stored == {}


def test_a_document_with_no_converted_form_is_failed() -> None:
    """Chunking does not read parse_status, so this is how it finds out."""
    driver = ChunkDriver(queued=(ClaimedDocument(sha256=SHA, language="en"),))
    driver.parsed.objects.clear()

    driver.run()

    assert "KeyError" in (driver.failure or "")


def test_a_converted_form_that_is_not_a_document_is_failed() -> None:
    """A truncated or rewritten object fails this document and no other."""
    driver = ChunkDriver(queued=(ClaimedDocument(sha256=SHA, language="en"),))
    driver.parsed.objects[driver.parsed.key_for(SHA)] = b"{not json"

    driver.run()

    assert driver.failure is not None
    assert driver.stored == {}


def test_an_unexpected_failure_is_recorded_as_its_own_type() -> None:
    """Not raised out of the drain loop."""
    driver = ChunkDriver(
        queued=(ClaimedDocument(sha256=SHA, language="en"),),
        builder=ScriptedBuilder(raises=MemoryError("out of memory")),
    )

    assert driver.run() == SHA
    assert driver.failure == "MemoryError: out of memory"


def test_a_passage_over_the_budget_is_stored_and_reported_loudly(caplog) -> None:
    """The chunker splits on that budget, so one above it is a missed split."""
    made = Chunking(
        passages=[
            Chunk(
                ordinal=1,
                text="Text.",
                page_from=None,
                page_to=None,
                section_path=None,
                block_type="text",
            )
        ],
        oversized=1,
    )
    driver = ChunkDriver(
        queued=(ClaimedDocument(sha256=SHA, language="en"),),
        builder=ScriptedBuilder(made),
    )

    with caplog.at_level("WARNING"):
        driver.run()

    assert driver.stored[SHA].oversized == 1
    assert "over the token budget" in caplog.text


def test_every_queued_document_is_worked_and_one_failure_stops_nothing() -> None:
    """Which is what makes a batch a batch."""
    driver = ChunkDriver(
        queued=(
            ClaimedDocument(sha256=SHA, language="en"),
            ClaimedDocument(sha256=OTHER, language="en"),
        )
    )
    driver.parsed.objects.pop(driver.parsed.key_for(OTHER))

    assert driver.drain() == 2
    assert list(driver.stored) == [SHA]
    assert list(driver.rows.failures) == [OTHER]


def test_a_drain_sweeps_the_claims_an_earlier_run_left_behind(caplog) -> None:
    """Before it claims anything of its own."""
    driver = ChunkDriver(abandoned=2)

    with caplog.at_level("WARNING"):
        assert driver.drain() == 0

    assert "2 document(s) left claimed" in caplog.text


# ── Re-reading stored passages ─────────────────────────────────────────────

GERMAN = (
    "Die Bundesanstalt für Finanzdienstleistungsaufsicht beaufsichtigt die "
    "Institute und veröffentlicht ihre Feststellungen jedes Jahr."
)
ENGLISH = (
    "The supervisory authority oversees the institutions and publishes its "
    "findings every year."
)


def test_a_corpus_with_no_passages_writes_nothing(caplog) -> None:
    """And says so rather than reporting a successful run over nothing."""
    driver = RevocabularyDriver()

    with caplog.at_level("WARNING"):
        assert driver.run() == 0

    assert "no passages to read" in caplog.text


@pytest.mark.nlp
def test_each_passage_is_written_with_the_language_of_its_own_text() -> None:
    """One file carries a German report and its English summary."""
    driver = RevocabularyDriver((1, GERMAN, "de"), (2, ENGLISH, "de"))

    assert driver.run() == 2
    assert driver.languages == {1: "de", 2: "en"}


@pytest.mark.nlp
def test_a_passage_too_short_to_judge_keeps_its_documents_language() -> None:
    """Below the detector's floor it is guessing from a few words."""
    driver = RevocabularyDriver((1, "Anhang", "de"))

    driver.run()

    assert driver.languages == {1: "de"}


@pytest.mark.nlp
def test_a_passage_whose_document_has_no_language_either_is_written_as_none() -> None:
    """Which is what puts it outside every topic model."""
    driver = RevocabularyDriver((1, "Anhang", None))

    assert driver.run() == 1
    assert driver.languages == {1: None}


@pytest.mark.nlp
def test_one_batch_is_written_per_language() -> None:
    """Which is where the time goes."""
    driver = RevocabularyDriver(
        (1, GERMAN, "de"), (2, ENGLISH, "de"), (3, GERMAN, "de")
    )

    assert driver.run() == 3
    assert sorted(driver.batches) == [1, 2]


@pytest.mark.nlp
def test_the_vocabulary_written_is_the_content_words_of_the_passage() -> None:
    """Nouns, proper nouns and adjectives, lemmatised, and nothing else."""
    driver = RevocabularyDriver((1, ENGLISH, "en"))

    driver.run()

    terms = driver.lemmas[1]
    assert "authority" in terms
    assert "institution" in terms
    assert "publish" not in terms, terms
    assert "the" not in terms
