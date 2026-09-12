"""The parsing flow: claim, dispatch, convert, store, analyse, record."""

from __future__ import annotations

import json
import logging
from typing import ClassVar, Protocol

from opentelemetry.trace import Span

from preprocessing.parsing.analysis import DocumentAnalyser, EmptyDocument
from preprocessing.parsing.models import ClaimedDocument, SourceDocument
from preprocessing.parsing.pipelines import (
    ConversionFailed,
    PipelineRegistry,
    UnsupportedFormat,
)
from preprocessing.parsing.repository import ParseQueue
from stages import StageService
from telemetry import tracer

log = logging.getLogger(__name__)
span = tracer(__name__)


class SourceStore(Protocol):
    """What this service needs from the store holding source documents.

    Declared here rather than imported from blob_store, so the dependency
    points inwards.
    """

    def key_for(self, sha256: str, media_type: str) -> str:
        """Returns the key a document of this type is stored under."""
        ...

    def get(self, key: str) -> bytes:
        """Reads an object."""
        ...


class ParsedStore(Protocol):
    """What this service needs from the store holding converted documents."""

    def key_for(self, sha256: str) -> str:
        """Returns the key a converted document is stored under."""
        ...

    def put(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str,
        metadata: dict[str, str],
    ) -> None:
        """Stores an object under `key`."""
        ...


class ParsingService(StageService):
    """Converts stored documents into structured ones, one at a time.

    The order is fixed: dispatch on the media type recorded at ingest,
    convert, store the converted document, and only then read out of it, so
    a bug in the analysis does not mean paying for the conversion twice.

    Work arrives through documents.parse_status.
    """

    name: ClassVar[str] = "parsing"
    unit: ClassVar[str] = "document"

    def __init__(
        self,
        *,
        repository: ParseQueue,
        documents: SourceStore,
        parsed: ParsedStore,
        pipelines: PipelineRegistry,
        analyser: DocumentAnalyser,
        ocr_char_threshold: int,
    ) -> None:
        """Initialises the service with its collaborators."""
        super().__init__(repository)
        self._repository: ParseQueue = repository
        self._documents = documents
        self._parsed = parsed
        self._pipelines = pipelines
        self._analyser = analyser
        self._ocr_char_threshold = ocr_char_threshold

    def process_next(self) -> str | None:
        """Parses one queued document.

        A failure is recorded against the document rather than raised, so one
        unreadable file does not stop a batch.
        """
        document = self._repository.claim()
        if document is None:
            return None

        with span.start_as_current_span("parse") as current:
            current.set_attribute("document.sha256", document.sha256)
            current.set_attribute("document.media_type", document.media_type)
            try:
                self._parse(document, current)
            except (UnsupportedFormat, ConversionFailed, EmptyDocument) as exc:
                self._fail(document.sha256, str(exc), current)
            except Exception as exc:
                self._fail(document.sha256, f"{type(exc).__name__}: {exc}", current)
                log.exception("failed %s", document.sha256)
        return document.sha256

    def _parse(self, document: ClaimedDocument, current: Span) -> None:
        """Runs one claimed document through its pipeline.

        Raises:
            UnsupportedFormat: If no pipeline handles the media type.
            ConversionFailed: If the pipeline could not convert the file.
            EmptyDocument: If the conversion yielded no body text.
        """
        pipeline = self._pipelines.for_media_type(document.media_type)
        scanned = self._is_scanned(document)
        current.set_attribute("parse.pipeline", type(pipeline).__name__)
        current.set_attribute("parse.scanned", scanned)

        data = self._documents.get(
            self._documents.key_for(document.sha256, document.media_type)
        )
        converted = pipeline.convert(
            SourceDocument(sha256=document.sha256, data=data, scanned=scanned)
        )

        self._parsed.put(
            self._parsed.key_for(document.sha256),
            json.dumps(
                converted.document.export_to_dict(), ensure_ascii=False
            ).encode(),
            content_type="application/json",
            metadata={"sha256": document.sha256},
        )

        parsed = self._analyser.analyse(converted)
        self._repository.complete(document.sha256, parsed)

        self._done(current)
        current.set_attribute("parse.pages", parsed.page_count)
        if parsed.confidence is not None:
            current.set_attribute("parse.confidence", parsed.confidence)
        log.info(
            "parsed %s: %d pages, language=%s, confidence=%s, title=%r",
            document.sha256,
            parsed.page_count,
            parsed.language,
            "n/a" if parsed.confidence is None else f"{parsed.confidence:.3f}",
            parsed.title,
        )

    def _is_scanned(self, document: ClaimedDocument) -> bool:
        """Decides whether a document carries a text layer worth reading.

        Read from the counts ingestion already took, so it costs nothing.
        """
        if document.char_count is None or not document.page_count:
            return True
        return document.char_count / document.page_count < self._ocr_char_threshold
