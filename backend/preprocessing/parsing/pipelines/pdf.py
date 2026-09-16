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

#: A discretionary hyphen, spelled either as U+00AD or as the U+0002 some
#: fonts map it to, with whatever space follows it.
_SOFT_HYPHEN = re.compile(r"[\x02\u00ad]\s*")

#: Control characters other than newline and tab.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

#: Zero-width marks.
_INVISIBLE = re.compile(r"[\u200b-\u200d\u2060\ufeff]")

#: The coordinating conjunctions that complete a hanging hyphen, as in
#: "Zoll- und Steuerrecht".
_JOINING = "und|oder|bzw|sowie|beziehungsweise|and|or"

#: A word the page broke across two lines with a real hyphen: a hyphen after
#: a letter and before a lower-case continuation, unless a conjunction
#: follows it.
#:
#: ponytail: punctuation alone cannot separate a broken word from two ratings
#: written side by side in one cell ("Mittel- niedrig"). Measured over this
#: corpus, 127 of 131 joins are right and all 222 hanging hyphens are left
#: alone. Telling the remaining four apart needs a character language model
#: or the geometry of the cells either side.
_BROKEN_WORD = re.compile(
    rf"(?<=[A-Za-z\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df])-\s+(?!(?:{_JOINING})\b)(?=[a-z\u00e4\u00f6\u00fc\u00df])"
)

#: Every space that is not U+0020.
_SPACES = re.compile(r"[\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]")


class PdfPipeline(Pipeline):
    """Converts PDFs, tuned for born-digital documents with tables.

    Heading hierarchy and table structure are both on. OCR is not
    implemented, so a scanned document is refused rather than converted into
    an empty one.

    Text comes off the page through Docling's own backend with
    `enforce_same_font` turned off, so a word carrying an f-ligature arrives
    whole rather than split around a bare `f`.

    The converter is built once and reused.
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
        """Initialises the pipeline. Every value comes from the environment.

        Args:
            table_mode: TableFormer mode, `fast` or `accurate`.
            heading_hierarchy: Whether to rebuild the heading tree.
            document_timeout_seconds: Longest one conversion may run.
            artifacts_path: Where the model weights are, or None to let
                Docling download them.
        """
        self._table_mode = TableFormerMode(table_mode)
        self._heading_hierarchy = heading_hierarchy
        self._timeout = document_timeout_seconds
        self._artifacts_path = artifacts_path
        self._built: DocumentConverter | None = None

    def convert(self, source: SourceDocument) -> Conversion:
        """Converts one PDF, refusing a scanned one.

        Args:
            source: The stored file and what is known about it.

        Returns:
            The mended document and its confidences.

        Raises:
            ConversionFailed: If the document needs OCR, or if Docling cannot
                convert it.
        """
        # OCR extension point. See this package's README for what enabling
        # it takes.
        if source.scanned:
            raise ConversionFailed(
                "the document carries too little text to read without OCR, "
                "which is not enabled"
            )

        try:
            result = self._converter().convert(
                # Docling picks its input format from the extension, so the
                # stream's name is where this pipeline states its format.
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
                # Required by the line above: the hierarchy reads font style
                # off the parsed cells.
                generate_parsed_pages=self._heading_hierarchy,
                document_timeout=self._timeout,
                artifacts_path=self._artifacts_path,
            )
            self._built = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=options,
                        # No backend=: Docling's own reports the font style
                        # heading levels are ranked by and the word cells
                        # table structure is matched against.
                        backend_options=ThreadedDoclingParseBackendOptions(  # pyright: ignore[reportCallIssue]
                            enforce_same_font=False
                        ),
                    )
                }
            )
            log.info("built the PDF converter")
        return self._built


def _mend(text: str) -> str:
    """Rejoins a line-broken word and spells the rest so it can be copied.

    Composes to NFC, joins soft and line-break hyphens, drops control and
    zero-width characters, and turns every other kind of space into U+0020.

    Args:
        text: One string as the converter read it.

    Returns:
        The mended string.
    """
    composed = unicodedata.normalize("NFC", text)
    cleaned = _SPACES.sub(
        " ", _INVISIBLE.sub("", _CONTROL.sub("", _SOFT_HYPHEN.sub("", composed)))
    )
    # After the space normalisation, so the broken-word rule sees one kind of
    # space.
    return _BROKEN_WORD.sub("", cleaned)


def _repaired(document: DoclingDocument) -> DoclingDocument:
    """Mends every string a later stage reads out of a converted document.

    Both the text items and the table grid: a passage's text comes from the
    first and its table_cells from the second.

    Args:
        document: The converted document, modified in place.

    Returns:
        The same document.
    """
    for item in document.texts:
        item.text = _mend(item.text)
        item.orig = _mend(item.orig)
    for table in document.tables:
        for cell in table.data.table_cells:
            cell.text = _mend(cell.text)
    return document


def _score(result: ConversionResult, name: str) -> float | None:
    """Reads one confidence score off a conversion.

    Args:
        result: What Docling returned.
        name: The score's attribute name.

    Returns:
        The score, or None when it is absent or NaN.
    """
    value = getattr(getattr(result, "confidence", None), name, None)
    return None if value is None or math.isnan(value) else float(value)
