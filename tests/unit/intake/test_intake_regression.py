"""The defects this service has already had, one test each.

Everything here is a fix that nothing at run time would report if it were
undone: a character that survives into passage text, a converter option that
silently changes what a document becomes. The behaviour tests beside this
file say what the code does; these say what it must not go back to doing.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from docling_core.types.doc.document import DoclingDocument, TableCell, TableData
from docling_core.types.doc.labels import DocItemLabel

from preprocessing.parsing.pipelines import pdf as pdf_module
from preprocessing.parsing.pipelines.pdf import PdfPipeline, _mend, _repaired

#: Every codepoint _SPACES folds to U+0020. Written out one at a time: the
#: character class used to name only some of them, and three survived into
#: passage text, where a model asked to quote a span types U+0020 for all of
#: them and the evidence gate reported the difference as an invented quote.
FOLDED_SPACES = (
    "\u0085",
    "\u00a0",
    "\u1680",
    "\u2000",
    "\u2001",
    "\u2002",
    "\u2003",
    "\u2004",
    "\u2005",
    "\u2006",
    "\u2007",
    "\u2008",
    "\u2009",
    "\u200a",
    "\u2028",
    "\u2029",
    "\u202f",
    "\u205f",
    "\u3000",
)

#: Every codepoint _INVISIBLE removes. They render as nothing, so no quote of
#: a span holding one can be reproduced.
REMOVED_MARKS = (
    "\u200b",
    "\u200c",
    "\u200d",
    "\u2060",
    "\ufeff",
)


@pytest.mark.parametrize("space", FOLDED_SPACES)
def test_every_kind_of_space_becomes_the_one_a_model_types_back(space: str) -> None:
    """One codepoint left out of the class is one rejected evidence quote."""
    mended = _mend(f"Bafin{space}meldet")

    assert mended == "Bafin meldet", repr(mended)


@pytest.mark.parametrize("mark", REMOVED_MARKS)
def test_every_zero_width_mark_is_removed(mark: str) -> None:
    """A character that renders as nothing cannot be quoted back."""
    assert _mend(f"Bafin{mark}meldet") == "Bafinmeldet"


@pytest.mark.parametrize("code", [*range(0x09), *range(0x0B, 0x20), 0x7F])
def test_every_control_character_but_newline_and_tab_is_dropped(code: int) -> None:
    """U+0002 is a discretionary hyphen in some fonts; the rest are noise."""
    mended = _mend(f"a{chr(code)}b")

    assert mended in ("ab", "a b"), repr(mended)


@pytest.mark.parametrize("kept", ["\n", "\t"])
def test_the_two_characters_passage_text_may_keep_survive(kept: str) -> None:
    """A rendered table is lines, and a line break is what separates them."""
    assert _mend(f"a{kept}b") == f"a{kept}b"


def test_the_repair_reaches_the_original_spelling_as_well_as_the_text() -> None:
    """`orig` is what a serialiser falls back to, so it carries the same text."""
    document = DoclingDocument(name="t")
    document.add_text(
        label=DocItemLabel.TEXT,
        text="Lebens\x02 versicherer",
        orig="Lebens\x02 versicherer",
    )

    mended = _repaired(document)

    assert mended.texts[0].text == "Lebensversicherer"
    assert mended.texts[0].orig == "Lebensversicherer"


def test_the_repair_reaches_a_cell_the_serialiser_never_touches() -> None:
    """A passage's table_cells are read from the grid, not the rendering."""
    document = DoclingDocument(name="t")
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

    assert _repaired(document).tables[0].data.table_cells[0].text == "Pensionskassen"


# ── The converter's options ────────────────────────────────────────────────


class _Converter:
    """Stands in for Docling's converter, recording how it was configured."""

    built: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, **kwargs: Any) -> None:
        """Records the format options rather than loading any model."""
        type(self).built.append(kwargs)


@pytest.fixture
def options(monkeypatch):
    """Builds a pipeline's converter and hands back the PDF format option."""

    def build(**overrides: Any):
        """Configures a pipeline and returns what it asked the converter for."""
        from docling.datamodel.base_models import InputFormat

        _Converter.built = []
        monkeypatch.setattr(pdf_module, "DocumentConverter", _Converter)
        settings = {
            "table_mode": "accurate",
            "heading_hierarchy": True,
            "document_timeout_seconds": 1800.0,
            "artifacts_path": None,
            **overrides,
        }
        pipeline = PdfPipeline(**settings)
        pipeline._converter()
        return _Converter.built[0]["format_options"][InputFormat.PDF], pipeline

    return build


def test_the_font_change_that_split_a_word_is_turned_off(options) -> None:
    """Left on, an f-ligature arrives as a bare `f` between two halves.

    Measured on a 73-page document: 397 splits with the flag on, none with it
    off. Every later stage reads the broken spelling.
    """
    chosen, _ = options()

    assert chosen.backend_options is not None
    assert chosen.backend_options.enforce_same_font is False


def test_the_converter_reads_the_page_itself(options) -> None:
    """No backend override: Docling's own reports the font style and cells.

    A backend that does not - pypdfium2 - leaves every section header at
    level one and no word cells for the table matcher.
    """
    chosen, _ = options()

    assert "DoclingParse" in chosen.backend.__name__, chosen.backend


def test_table_structure_is_on_and_the_mode_is_the_configured_one(options) -> None:
    """Without it no cell grid is recovered and no cell is quotable."""
    chosen, _ = options(table_mode="fast")

    assert chosen.pipeline_options.do_table_structure is True
    assert chosen.pipeline_options.table_structure_options.mode.value == "fast"


def test_a_table_mode_that_is_not_one_is_refused_at_construction() -> None:
    """Rather than at the first conversion, an hour into a batch."""
    with pytest.raises(ValueError, match="thorough"):
        PdfPipeline(
            table_mode="thorough",
            heading_hierarchy=True,
            document_timeout_seconds=None,
            artifacts_path=None,
        )


@pytest.mark.parametrize("wanted", [True, False])
def test_the_parsed_pages_the_heading_tree_needs_follow_it(options, wanted) -> None:
    """The hierarchy reads font style off the parsed cells."""
    chosen, _ = options(heading_hierarchy=wanted)

    assert chosen.pipeline_options.heading_hierarchy_options.enabled is wanted
    assert chosen.pipeline_options.generate_parsed_pages is wanted


def test_ocr_is_off_in_the_converter_as_well_as_in_the_guard(options) -> None:
    """No engine is installed, so asking for OCR would fail every conversion."""
    chosen, _ = options()

    assert chosen.pipeline_options.do_ocr is False


def test_the_timeout_and_the_weights_path_are_passed_through(options) -> None:
    """Docling has no timeout of its own, and an empty path is not a path."""
    chosen, _ = options(
        document_timeout_seconds=42.0, artifacts_path="/opt/docling-models"
    )

    assert chosen.pipeline_options.document_timeout == 42.0
    assert str(chosen.pipeline_options.artifacts_path) == "/opt/docling-models"


def test_the_converter_is_built_once_and_reused(options) -> None:
    """Constructing it loads the layout and table models."""
    _, pipeline = options()

    pipeline._converter()
    pipeline._converter()

    assert len(_Converter.built) == 1
