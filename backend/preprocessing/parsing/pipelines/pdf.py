"""The PDF pipeline."""

from __future__ import annotations

import io
import logging
import math
import re
import unicodedata
from typing import ClassVar

from docling.datamodel.backend_options import ThreadedDoclingParseBackendOptions
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    HeadingHierarchyOptions,
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument

from preprocessing.parsing.models import Conversion, SourceDocument
from preprocessing.parsing.pipelines.base import ConversionFailed, Pipeline

log = logging.getLogger(__name__)

#: A word the page broke across two lines. Some fonts map their discretionary
#: hyphen to U+0002 rather than to a hyphen and pypdfium2 passes the mapping
#: through, so the two halves arrive joined by a control character and
#: whatever space the assembler put between the cells; others use U+00AD,
#: which is the same mark spelled properly. Both join the halves.
_SOFT_HYPHEN = re.compile(r"[\x02\u00ad]\s*")

#: Anything else the content stream carried that is not text. Newline and
#: tab are the two the passage text is allowed to keep.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

#: Zero-width marks, which render as nothing and so cannot be quoted back.
_INVISIBLE = re.compile(r"[\u200b-\u200d\u2060\ufeff]")

#: The conjunctions a hanging hyphen is followed by. German writes a shared
#: suffix as "Zoll- und Steuerrecht", and the Ergaenzungsstrich is only ever
#: completed by a coordinating conjunction, so a hyphen followed by anything
#: else is a broken word rather than an abbreviated compound. Not a guess at
#: the language: it is the rule the mark is defined by.
_JOINING = "und|oder|bzw|sowie|beziehungsweise|and|or"

#: A word the page broke across two lines with a real hyphen.
#:
#: Docling applies the same test - a hyphen before a lower-case continuation
#: is a split word - in ReadingOrderModel._merge_elements, but only where it
#: joins two layout elements, and without the conjunction guard. Within one
#: element the backend has already turned the line break into a space before
#: anything here runs, so no item text holds a hyphen before a newline and
#: Docling's rule never sees these. This is that same rule for that case.
#:
#: The guard is what the space costs: with the newline gone, "Zoll- und" and
#: "ergaen- zende" are the same shape, and joining blindly gives "Zollund".
#: pd3f's `dehyphen` scores the alternatives with a character language model
#: instead, which is the thorough answer; it was last released in 2020 against
#: a fork of Flair and declares no Python above 3.8.
#:
#: ponytail: measured over this corpus, 127 of 131 joins are right and all 222
#: hanging hyphens are left alone. The four misses are one table cell, "Mittel-
#: niedrig", where two ratings sit side by side and neither is a broken word.
#: Nothing in the punctuation separates that from "Nied- rig"; telling them
#: apart needs the language model above, or the cell geometry either side.
_BROKEN_WORD = re.compile(
    rf"(?<=[A-Za-z\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df])-\s+(?!(?:{_JOINING})\b)(?=[a-z\u00e4\u00f6\u00fc\u00df])"
)

#: Every space that is not U+0020. A model asked to copy a span verbatim
#: returns an ordinary space for all of them, so leaving one in the passage
#: text rejects an evidence quote that is correct in every visible respect -
#: eight of this corpus's seventeen rejected facts differed from their
#: passage by one U+00A0 and nothing else.
_SPACES = re.compile(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")


class PdfPipeline(Pipeline):
    """Converts PDFs, tuned for born-digital documents with tables.

    Heading hierarchy and table structure are both on, which Docling leaves
    off: without them every section header comes back at level one and no
    cell grid is recovered. OCR is not implemented, so a scanned document is
    refused rather than converted into an empty one.

    Text comes off the page through Docling's own backend with
    `enforce_same_font` turned off. Left on, docling-parse cuts a text cell
    wherever the font changes, and an f-ligature is set in a font of its own,
    so a word carrying one arrives split around a bare `f` - `identify` as
    `identi f y`. Every later stage then reads the broken spelling: the
    chunker cuts on it, the vocabulary keeps `f` as a term, and the evidence
    gate matches against it. Measured on a 73-page test document: 397 splits
    with the flag on, none with it off, and the same section headers at the
    same levels.

    The converter is built once and reused, since constructing it loads the
    layout and table models.
    """

    media_types: ClassVar[tuple[str, ...]] = ("application/pdf",)

    def __init__(
        self,
        *,
        table_mode: str,
        heading_hierarchy: bool,
        document_timeout_seconds: float | None,
        artifacts_path: str | None,
    ) -> None:
        """Initialises the pipeline.

        No defaults: every value comes from the environment, and one written
        here would silently disagree with the one in `.env.example`.
        """
        self._table_mode = TableFormerMode(table_mode)
        self._heading_hierarchy = heading_hierarchy
        self._timeout = document_timeout_seconds
        self._artifacts_path = artifacts_path
        self._built: DocumentConverter | None = None

    def convert(self, source: SourceDocument) -> Conversion:
        """Converts one PDF, refusing a scanned one.

        Raises:
            ConversionFailed: If the document needs OCR, or if Docling
                cannot convert it.
        """
        # ── OCR placeholder ────────────────────────────────────────────────
        # Refused here, where the reason is still known, rather than as "no
        # extractable body text" three steps later.
        #
        # To enable OCR: install an engine (docling supports EasyOCR,
        # RapidOCR and Tesseract; none is installed and Tesseract also needs
        # its system binary in the image), delete this guard, and pass
        # `do_ocr=True` with an explicit `ocr_options` in `_converter`. Note
        # that OCR wants a language hint before conversion while
        # documents.language is only detected after it, and that the hint
        # uses ISO 639-2 codes where that column uses 639-1.
        if source.scanned:
            raise ConversionFailed(
                "the document carries too little text to read without OCR, "
                "which is not enabled"
            )

        try:
            result = self._converter().convert(
                # Docling picks its input format from the extension, so
                # naming the stream is where this pipeline states its format.
                DocumentStream(
                    name=f"{source.sha256}.pdf", stream=io.BytesIO(source.data)
                )
            )
        except Exception as exc:
            raise ConversionFailed(f"{type(exc).__name__}: {exc}") from exc
        return Conversion(
            document=_repaired(result.document),
            confidence=_score(result, "mean_score"),
            confidence_low=_score(result, "low_score"),
        )

    def _converter(self) -> DocumentConverter:
        """Returns the converter, building it once."""
        if self._built is None:
            options = PdfPipelineOptions(
                do_ocr=False,
                do_table_structure=True,
                table_structure_options=TableStructureOptions(mode=self._table_mode),
                heading_hierarchy_options=HeadingHierarchyOptions(
                    enabled=self._heading_hierarchy
                ),
                # Required by the line above, not independent of it: the
                # hierarchy reads font style off the parsed cells.
                generate_parsed_pages=self._heading_hierarchy,
                document_timeout=self._timeout,
                artifacts_path=self._artifacts_path,
            )
            self._built = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=options,
                        # No backend=: Docling's own is the one that reports
                        # the font style heading levels are ranked by and the
                        # word cells table structure is matched against.
                        backend_options=ThreadedDoclingParseBackendOptions(
                            enforce_same_font=False
                        ),
                    )
                }
            )
            log.info("built the PDF converter")
        return self._built


def _mend(text: str) -> str:
    """Rejoins a line-broken word and spells the rest so it can be copied.

    Runs before anything reads the document, so passages.text, the cell grid
    and every evidence offset are all indices into the mended spelling. A
    character a model cannot reproduce is one no quote of it can match, and
    the evidence gate reports that as the model inventing a quote.

    NFC first: the same umlaut arrives composed from one document and
    decomposed from another, and the two compare unequal character for
    character while looking identical on the page.
    """
    composed = unicodedata.normalize("NFC", text)
    cleaned = _SPACES.sub(
        " ", _INVISIBLE.sub("", _CONTROL.sub("", _SOFT_HYPHEN.sub("", composed)))
    )
    # After the space normalisation, so the broken-word rule sees one kind of
    # space rather than having to spell every one of them again.
    return _BROKEN_WORD.sub("", cleaned)


def _repaired(document: DoclingDocument) -> DoclingDocument:
    """Mends every string a later stage reads out of a converted document.

    Runs before the document is stored, so the parsed object, the passages
    and the cell grid all carry one spelling. Both places are needed: a
    passage's text comes from the text items, and its table_cells come from
    the table grid, which the serialiser does not touch.
    """
    for item in document.texts:
        item.text = _mend(item.text)
        item.orig = _mend(item.orig)
    for table in document.tables:
        for cell in table.data.table_cells:
            cell.text = _mend(cell.text)
    return document


def _score(result: ConversionResult, name: str) -> float | None:
    """Reads one score off a conversion, as a value the database accepts.

    Docling reports an unmeasured score as NaN, which reaches PostgreSQL as
    a value comparing equal to nothing, so a later `WHERE confidence < 0.8`
    would skip those rows silently. NULL behaves.
    """
    value = getattr(getattr(result, "confidence", None), name, None)
    return None if value is None or math.isnan(value) else float(value)
