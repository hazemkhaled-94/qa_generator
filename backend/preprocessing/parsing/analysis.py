"""Reading a converted document: body text, language and content hash."""

from __future__ import annotations

import hashlib

from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from nlp.analysis import normalised
from nlp.language import detect
from preprocessing.parsing.models import Conversion, ParsedDocument

#: Labels that carry a document's own title.
_TITLE_LABELS = (DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER)


class EmptyDocument(Exception):
    """Raised when a conversion produced no body text to work with."""


class DocumentAnalyser:
    """Turns a converted document into the values the database holds.

    Format-independent, and so separate from the pipelines: once a document
    is a DoclingDocument, reading it is the same work whatever it arrived
    as.
    """

    def analyse(self, conversion: Conversion) -> ParsedDocument:
        """Reads everything the database needs from a converted document.

        Raises:
            EmptyDocument: If the conversion succeeded but produced no body
                text.
        """
        document = conversion.document
        body = self.body(document)
        if not body:
            raise EmptyDocument("the document has no extractable body text")

        return ParsedDocument(
            title=self.title(document),
            language=self.language(body),
            content_sha256=self.content_hash(body),
            page_count=len(document.pages),
            confidence=conversion.confidence,
            confidence_low=conversion.confidence_low,
        )

    @staticmethod
    def body(document: DoclingDocument) -> str:
        """Renders the document's content as text, table values included.

        Delegates to the converter's own serialiser, which walks the body
        content layer: that is what carries table cells, and what leaves out
        running headers and footers.
        """
        return document.export_to_text().strip()

    @staticmethod
    def title(document: DoclingDocument) -> str | None:
        """Finds the document's title."""
        for label in _TITLE_LABELS:
            for item in document.texts:
                if item.label == label and (item.text or "").strip():
                    return item.text.strip()
        return None

    @staticmethod
    def language(body: str) -> str | None:
        """Detects the language of the body text."""
        return detect(body)

    @staticmethod
    def content_hash(body: str) -> str:
        """Hashes the body with whitespace collapsed and case folded.

        Punctuation and digits are kept: in financial text they are the
        content.

        The same fold as every other text comparison in the project. It used
        to be `lower`, which leaves ß alone, so a document written STRASSE
        and one written Straße hashed differently.
        """
        return hashlib.sha256(normalised(body).encode()).hexdigest()
