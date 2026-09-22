"""Drivers the ingestion, parsing and chunking tests act through.

A driver is the page object of a service test: one per thing under test,
exposing the operations a reader cares about and holding the wiring out of
the test body. The doubles below stand in for the database and the two
buckets; everything else is the real code.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import (
    DoclingDocument,
    ProvenanceItem,
    TableCell,
    TableData,
)
from docling_core.types.doc.labels import DocItemLabel

from database.qa_generator import Outcome, Status
from ingestion.models import PdfFacts, UploadedFile
from ingestion.removal import RemovalService
from ingestion.service import IngestService
from preprocessing.chunking.models import Chunking
from preprocessing.chunking.models import ClaimedDocument as ChunkClaim
from preprocessing.chunking.passages import NoPassages
from preprocessing.chunking.service import ChunkingService, revocabulary
from preprocessing.parsing.analysis import DocumentAnalyser
from preprocessing.parsing.models import (
    ClaimedDocument as ParseClaim,
)
from preprocessing.parsing.models import (
    Conversion,
    SourceDocument,
)
from preprocessing.parsing.pipelines import Pipeline, PipelineRegistry
from preprocessing.parsing.service import ParsingService

#: A one-page PDF, small enough to keep here and real enough for PyMuPDF.
PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
    b"trailer<</Root 1 0 R>>\n"
)

#: Bytes no magic number claims.
NOT_A_PDF = b"GIF89a and nothing else"


# ── Documents ──────────────────────────────────────────────────────────────


def docling_document(
    *,
    name: str = "t",
    title: str | None = None,
    headings: tuple[str, ...] = (),
    paragraphs: tuple[str, ...] = (),
    pages: tuple[tuple[int, float, float], ...] = (),
) -> DoclingDocument:
    """Builds a converted document out of the parts a test names.

    Args:
        name: The document's name.
        title: A title item, if one is wanted.
        headings: Section headers, in order.
        paragraphs: Body paragraphs, in order.
        pages: One (page number, width, height) per page.

    Returns:
        The document.
    """
    document = DoclingDocument(name=name)
    for page_no, width, height in pages:
        document.add_page(page_no=page_no, size=Size(width=width, height=height))
    if title is not None:
        document.add_title(text=title)
    for heading in headings:
        document.add_heading(text=heading)
    for paragraph in paragraphs:
        document.add_text(label=DocItemLabel.TEXT, text=paragraph)
    return document


def placed(
    document: DoclingDocument,
    text: str,
    *,
    page_no: int = 1,
    box: tuple[float, float, float, float] = (10, 250, 100, 200),
    origin: CoordOrigin = CoordOrigin.BOTTOMLEFT,
):
    """Adds one body paragraph with a position on a page.

    Args:
        document: The document to add to.
        text: The paragraph text.
        page_no: The page it sits on.
        box: Its (l, t, r, b) in that page's own origin.
        origin: Which corner those coordinates count from.

    Returns:
        The added item.
    """
    left, top, right, bottom = box
    return document.add_text(
        label=DocItemLabel.TEXT,
        text=text,
        prov=ProvenanceItem(
            page_no=page_no,
            bbox=BoundingBox(l=left, t=top, r=right, b=bottom, coord_origin=origin),
            charspan=(0, len(text)),
        ),
    )


def cell(
    text: str,
    row: int,
    col: int,
    *,
    column_header: bool = False,
    row_header: bool = False,
    row_span: int = 1,
    col_span: int = 1,
    box: tuple[float, float, float, float] | None = None,
) -> TableCell:
    """Builds one table cell.

    Args:
        text: The cell's value.
        row: Its first row index.
        col: Its first column index.
        column_header: Whether it heads its column.
        row_header: Whether it labels its row.
        row_span: Rows it covers.
        col_span: Columns it covers.
        box: Its (l, t, r, b) on the page, if a test needs one.

    Returns:
        The cell.
    """
    return TableCell(
        text=text,
        start_row_offset_idx=row,
        end_row_offset_idx=row + row_span,
        start_col_offset_idx=col,
        end_col_offset_idx=col + col_span,
        row_span=row_span,
        col_span=col_span,
        column_header=column_header,
        row_header=row_header,
        bbox=None
        if box is None
        else BoundingBox(
            l=box[0], t=box[1], r=box[2], b=box[3], coord_origin=CoordOrigin.TOPLEFT
        ),
    )


def table_of(
    document: DoclingDocument,
    cells: list[TableCell],
    *,
    num_rows: int,
    num_cols: int,
    caption: str | None = None,
    page_no: int | None = None,
):
    """Adds one table to a document.

    Args:
        document: The document to add to.
        cells: The table's cells.
        num_rows: Rows the whole table has.
        num_cols: Columns the whole table has.
        caption: A caption, if one is wanted.
        page_no: The page the table sits on, if a test needs one.

    Returns:
        The added table item.
    """
    return document.add_table(
        data=TableData(num_rows=num_rows, num_cols=num_cols, table_cells=cells),
        caption=None
        if caption is None
        else document.add_text(label=DocItemLabel.CAPTION, text=caption),
        prov=None
        if page_no is None
        else ProvenanceItem(
            page_no=page_no,
            bbox=BoundingBox(l=0, t=0, r=1, b=1, coord_origin=CoordOrigin.TOPLEFT),
            charspan=(0, 0),
        ),
    )


def chunk_of(text: str, items: tuple = (), headings: tuple[str, ...] = ()) -> Any:
    """Stands in for one chunk the hybrid chunker produced.

    Args:
        text: The rendered chunk text.
        items: The document items it was cut from.
        headings: Its heading trail.

    Returns:
        An object carrying what the chunk reader reads.
    """
    return SimpleNamespace(
        text=text, meta=SimpleNamespace(doc_items=list(items), headings=list(headings))
    )


# ── Ingestion ──────────────────────────────────────────────────────────────


class MemoryDocuments:
    """The documents bucket, held in memory."""

    name = "documents"

    def __init__(self) -> None:
        """Starts empty, refusing nothing."""
        self.objects: dict[str, bytes] = {}
        self.content_types: dict[str, str] = {}
        self.metadata: dict[str, dict[str, str]] = {}
        self.removed: list[str] = []
        self.refuses_put: Exception | None = None
        self.refuses_get: Exception | None = None

    @staticmethod
    def key_for(sha256: str, media_type: str) -> str:
        """The key a document of this type is stored under."""
        return f"{media_type.replace('/', '-')}/{sha256}"

    def get(self, key: str) -> bytes:
        """Reads an object."""
        if self.refuses_get is not None:
            raise self.refuses_get
        return self.objects[key]

    def put(
        self, key: str, data: bytes, *, content_type: str, metadata: dict[str, str]
    ) -> None:
        """Stores an object."""
        if self.refuses_put is not None:
            raise self.refuses_put
        self.objects[key] = data
        self.content_types[key] = content_type
        self.metadata[key] = dict(metadata)

    def remove(self, key: str) -> bool:
        """Deletes an object, reporting whether one was there."""
        self.removed.append(key)
        return self.objects.pop(key, None) is not None


class MemoryParsed:
    """The parsed bucket, held in memory."""

    name = "parsed"

    def __init__(self) -> None:
        """Starts empty, refusing nothing."""
        self.objects: dict[str, bytes] = {}
        self.content_types: dict[str, str] = {}
        self.metadata: dict[str, dict[str, str]] = {}
        self.removed: list[str] = []
        self.refuses_put: Exception | None = None
        self.refuses_get: Exception | None = None

    @staticmethod
    def key_for(sha256: str) -> str:
        """The key a converted document is stored under."""
        return f"parsed/{sha256}.json"

    def get(self, key: str) -> bytes:
        """Reads an object."""
        if self.refuses_get is not None:
            raise self.refuses_get
        return self.objects[key]

    def put(
        self, key: str, data: bytes, *, content_type: str, metadata: dict[str, str]
    ) -> None:
        """Stores an object."""
        if self.refuses_put is not None:
            raise self.refuses_put
        self.objects[key] = data
        self.content_types[key] = content_type
        self.metadata[key] = dict(metadata)

    def remove(self, key: str) -> bool:
        """Deletes an object, reporting whether one was there."""
        self.removed.append(key)
        return self.objects.pop(key, None) is not None


class MemoryArchive:
    """The archive bucket, held in memory, over the buckets it takes from."""

    def __init__(self, **origins: dict[str, bytes]) -> None:
        """Wires the archive over the object dict of each named bucket."""
        self.origins = origins
        self.objects: dict[str, bytes] = {}

    def take(self, origin: str, key: str) -> bool:
        """Moves an object out of `origin`, reporting whether one was there."""
        data = self.origins[origin].pop(key, None)
        if data is None:
            return False
        self.objects[f"{origin}/{key}"] = data
        return True


class DocumentRows:
    """The documents and ingest_events tables, held in memory."""

    def __init__(self) -> None:
        """Starts empty, refusing nothing."""
        self.documents: dict[str, dict] = {}
        self.passages: dict[str, int] = {}
        self.events: list[dict] = []
        self.chunk_status: dict[str, str] = {}
        self.refuses_store: Exception | None = None
        #: Simulates another upload of the same bytes landing between the
        #: existence check and the insert.
        self.raced = False

    def exists(self, sha256: str) -> bool:
        """Whether these exact bytes are already stored."""
        return sha256 in self.documents

    def media_type(self, sha256: str) -> str | None:
        """The media type detected for a stored document."""
        held = self.documents.get(sha256)
        return None if held is None else held["media_type"]

    def store(self, upload: UploadedFile, facts: PdfFacts) -> str:
        """Inserts a document and its event."""
        if self.refuses_store is not None:
            raise self.refuses_store
        taken = self.raced or upload.sha256 in self.documents
        if not taken:
            self.documents[upload.sha256] = {
                "media_type": upload.media_type,
                "page_count": facts.page_count,
                "char_count": facts.char_count,
            }
            self.chunk_status[upload.sha256] = Status.NEW
        outcome = Outcome.DUPLICATE_BYTES if taken else Outcome.STORED
        self.record_attempt(
            upload.filename, upload.size_bytes, outcome, None, upload.sha256
        )
        return outcome

    def record_attempt(
        self,
        filename: str,
        size_bytes: int,
        outcome: str,
        detail: str | None = None,
        sha256: str | None = None,
    ) -> None:
        """Records one upload attempt."""
        self.events.append(
            {
                "filename": filename,
                "size_bytes": size_bytes,
                "outcome": outcome,
                "detail": detail,
                "sha256": sha256,
            }
        )

    def delete(self, sha256: str) -> int:
        """Removes a document row and everything derived from it."""
        self.documents.pop(sha256, None)
        self.chunk_status.pop(sha256, None)
        return self.passages.pop(sha256, 0)

    def delete_derived(self, sha256: str) -> int:
        """Drops a document's passages and returns chunk_status to `new`."""
        self.chunk_status[sha256] = Status.NEW
        return self.passages.pop(sha256, 0)

    def page(
        self, search: str | None = None, limit: int = 100, offset: int = 0
    ) -> tuple[int, list]:
        """One page of documents, and the total behind it."""
        held = [
            SimpleNamespace(sha256=sha, **row) for sha, row in self.documents.items()
        ]
        if search:
            held = [one for one in held if search in one.sha256]
        return len(held), held[offset : offset + limit]

    def names(self) -> list:
        """Every document by digest."""
        return [SimpleNamespace(sha256=sha, filename=None) for sha in self.documents]


class ReaderSpy:
    """Stands in for `read_pdf`, counting the calls and answering as told."""

    def __init__(
        self, facts: PdfFacts | None = None, raises: Exception | None = None
    ) -> None:
        """Takes the answer every call gets.

        Args:
            facts: What to answer with.
            raises: What to raise instead.
        """
        self.facts = facts or PdfFacts(page_count=1, char_count=500)
        self.raises = raises
        self.reads: list[bytes] = []

    @property
    def calls(self) -> int:
        """How many times the PDF was read."""
        return len(self.reads)

    def __call__(self, data: bytes) -> PdfFacts:
        """Answers, recording what it was handed."""
        self.reads.append(data)
        if self.raises is not None:
            raise self.raises
        return self.facts


class IngestDriver:
    """The ingest flow, over the database and the bucket in memory."""

    def __init__(
        self,
        *,
        max_file_size_bytes: int = 1024 * 1024,
        allowed: tuple[str, ...] = ("application/pdf",),
        pipeline_version: str = "0.0.1",
        reader: ReaderSpy | None = None,
        monkeypatch: Any = None,
    ) -> None:
        """Wires the service over the doubles.

        Args:
            max_file_size_bytes: The upload limit.
            allowed: The media types to accept.
            pipeline_version: Recorded in the object's metadata.
            reader: A stand-in for `read_pdf`, if a test wants one.
            monkeypatch: Required when `reader` is given.
        """
        self.rows = DocumentRows()
        self.store = MemoryDocuments()
        self.reader = reader
        if reader is not None:
            monkeypatch.setattr("ingestion.service.read_pdf", reader)
        self.service = IngestService(
            repository=self.rows,  # pyright: ignore[reportArgumentType]
            store=self.store,  # pyright: ignore[reportArgumentType]
            max_file_size_bytes=max_file_size_bytes,
            allowed_media_types=allowed,
            pipeline_version=pipeline_version,
        )

    def upload(self, filename: str = "report.pdf", data: bytes = PDF):
        """Ingests one file.

        Args:
            filename: Name to submit it under.
            data: The bytes.

        Returns:
            What became of the upload.
        """
        return self.service.ingest(UploadedFile(filename, data))

    def refuse(self, filename: str, size_bytes: int):
        """Refuses an upload on its declared size alone."""
        return self.service.refuse_oversized(filename, size_bytes)

    @property
    def outcomes(self) -> list[str]:
        """Every recorded outcome, in order."""
        return [event["outcome"] for event in self.rows.events]

    @property
    def stored(self) -> list[str]:
        """The digests the database holds."""
        return list(self.rows.documents)

    @property
    def objects(self) -> list[str]:
        """The keys the bucket holds."""
        return list(self.store.objects)

    def metadata_of(self, sha256: str) -> dict[str, str]:
        """The metadata written beside one stored document."""
        return self.store.metadata[self.store.key_for(sha256, "application/pdf")]


class RemovalDriver:
    """Deleting a document, over the database and both buckets in memory."""

    def __init__(self, *, held: tuple[str, ...] = (), passages: int = 0) -> None:
        """Wires the service over the doubles.

        Args:
            held: Digests the corpus already holds.
            passages: Passages each of them has.
        """
        self.rows = DocumentRows()
        self.store = MemoryDocuments()
        self.parsed = MemoryParsed()
        self.archive = MemoryArchive(
            documents=self.store.objects, parsed=self.parsed.objects
        )
        for sha in held:
            self.rows.documents[sha] = {
                "media_type": "application/pdf",
                "page_count": 1,
                "char_count": 500,
            }
            self.rows.chunk_status[sha] = Status.CHUNKED
            self.rows.passages[sha] = passages
            self.store.objects[self.store.key_for(sha, "application/pdf")] = PDF
            self.parsed.objects[self.parsed.key_for(sha)] = b"{}"
        self.service = RemovalService(
            repository=self.rows,  # pyright: ignore[reportArgumentType]
            store=self.store,  # pyright: ignore[reportArgumentType]
            parsed=self.parsed,  # pyright: ignore[reportArgumentType]
            archive=self.archive,
        )

    def delete(self, sha256: str):
        """Removes a document completely."""
        return self.service.delete(sha256)

    def delete_derived(self, sha256: str):
        """Drops only what the pipeline built from it."""
        return self.service.delete_derived(sha256)

    def holds_file(self, sha256: str) -> bool:
        """Whether the uploaded file is still there."""
        return self.store.key_for(sha256, "application/pdf") in self.store.objects

    def holds_parsed(self, sha256: str) -> bool:
        """Whether the converted form is still there."""
        return self.parsed.key_for(sha256) in self.parsed.objects


# ── Parsing ────────────────────────────────────────────────────────────────


class ScriptedPipeline(Pipeline):
    """A converter that answers with a document instead of reading one."""

    media_types = ("application/pdf",)

    def __init__(
        self,
        *,
        document: DoclingDocument | None = None,
        confidence: float | None = 0.9,
        confidence_low: float | None = 0.9,
        raises: Exception | None = None,
    ) -> None:
        """Takes the answer every conversion gets.

        Args:
            document: The document to answer with.
            confidence: The mean confidence to report.
            confidence_low: The lower-bound confidence to report.
            raises: What to raise instead of answering.
        """
        self.document = document or docling_document(
            title="The Device", paragraphs=("The device weighs 4 kg.",)
        )
        self.confidence = confidence
        self.confidence_low = confidence_low
        self.raises = raises
        self.converted: list[SourceDocument] = []

    @property
    def calls(self) -> int:
        """How many documents were handed over."""
        return len(self.converted)

    def convert(self, source: SourceDocument) -> Conversion:
        """Answers, recording what it was handed."""
        self.converted.append(source)
        if self.raises is not None:
            raise self.raises
        return Conversion(
            document=self.document,
            confidence=self.confidence,
            confidence_low=self.confidence_low,
        )


class ParseRows:
    """The parsing queue, held in memory."""

    done = Status.PARSED

    def __init__(
        self,
        *queued: ParseClaim,
        holders: dict[str, str] | None = None,
        abandoned: int = 0,
    ) -> None:
        """Takes the documents to hand out and the text already held.

        Args:
            queued: The documents to claim, in order.
            holders: Which document holds each content digest already.
            abandoned: What a sweep reports.
        """
        self.pending = list(queued)
        self.holders = holders or {}
        self.abandoned = abandoned
        self.claims: list[str] = []
        self.completed: dict[str, Any] = {}
        self.failures: dict[str, str] = {}

    def claim(self) -> ParseClaim | None:
        """Takes the next pending document."""
        if not self.pending:
            return None
        taken = self.pending.pop(0)
        self.claims.append(taken.sha256)
        return taken

    def holder_of(self, content_sha256: str, besides: str) -> str | None:
        """Names another document holding this exact text."""
        held = self.holders.get(content_sha256)
        return None if held is None or held == besides else held

    def complete(self, sha256: str, parsed) -> None:
        """Records a successful parse.

        The content digest becomes this document's, which is what the real
        repository writes here and what `holder_of` reads back.
        """
        self.completed[sha256] = parsed
        self.holders.setdefault(parsed.content_sha256, sha256)

    def fail(self, key: str, error: str) -> None:
        """Records a document this stage could not process."""
        self.failures[key] = error

    def abandon(self) -> int:
        """Sweeps claims an earlier run left behind."""
        return self.abandoned


class ParseDriver:
    """The parsing flow, with the queue and both buckets in memory."""

    def __init__(
        self,
        *,
        queued: tuple[ParseClaim, ...] = (),
        pipeline: ScriptedPipeline | None = None,
        pipelines: PipelineRegistry | None = None,
        holders: dict[str, str] | None = None,
        ocr_char_threshold: int = 100,
        min_confidence: float = 0.5,
        abandoned: int = 0,
    ) -> None:
        """Wires the service over the doubles.

        Args:
            queued: The documents to hand out.
            pipeline: The converter to use.
            pipelines: A registry to use instead of one built from `pipeline`.
            holders: Which document holds each content digest already.
            ocr_char_threshold: Characters per page below which a document
                counts as scanned.
            min_confidence: Lowest confidence a conversion may carry.
            abandoned: What a sweep reports.
        """
        self.rows = ParseRows(*queued, holders=holders, abandoned=abandoned)
        self.documents = MemoryDocuments()
        self.parsed = MemoryParsed()
        self.pipeline = pipeline or ScriptedPipeline()
        for one in queued:
            self.documents.objects[
                self.documents.key_for(one.sha256, one.media_type)
            ] = PDF
        self.service = ParsingService(
            repository=self.rows,  # pyright: ignore[reportArgumentType]
            documents=self.documents,  # pyright: ignore[reportArgumentType]
            parsed=self.parsed,  # pyright: ignore[reportArgumentType]
            pipelines=pipelines or PipelineRegistry((self.pipeline,)),
            analyser=DocumentAnalyser(),
            ocr_char_threshold=ocr_char_threshold,
            min_confidence=min_confidence,
        )

    def run(self) -> str | None:
        """Parses one queued document."""
        return self.service.process_next()

    def drain(self) -> int:
        """Parses every queued document."""
        return self.service.drain()

    @property
    def completed(self) -> dict[str, Any]:
        """What each finished document was recorded as."""
        return self.rows.completed

    @property
    def failure(self) -> str | None:
        """Why the last document failed, if it did."""
        return next(iter(self.rows.failures.values()), None)

    @property
    def failures(self) -> dict[str, str]:
        """Every failure, by digest."""
        return self.rows.failures

    @property
    def parsed_keys(self) -> list[str]:
        """The keys the parsed bucket holds."""
        return list(self.parsed.objects)

    def scanned_as(self, index: int = 0) -> bool:
        """How the service judged the document it handed over."""
        return self.pipeline.converted[index].scanned


# ── Chunking ───────────────────────────────────────────────────────────────


class ScriptedBuilder:
    """A passage builder that answers with a chunking instead of cutting one."""

    def __init__(
        self, chunking: Chunking | None = None, raises: Exception | None = None
    ) -> None:
        """Takes the answer every build gets.

        Args:
            chunking: What to answer with.
            raises: What to raise instead.
        """
        self.chunking = chunking or Chunking(passages=[], oversized=0)
        self.raises = raises
        self.built: list[tuple[Any, str | None]] = []

    @property
    def calls(self) -> int:
        """How many documents were cut."""
        return len(self.built)

    def build(self, document, language: str | None) -> Chunking:
        """Answers, recording what it was handed."""
        self.built.append((document, language))
        if self.raises is not None:
            raise self.raises
        return self.chunking


class ChunkRows:
    """The chunking queue, held in memory."""

    done = Status.CHUNKED

    def __init__(self, *queued: ChunkClaim, abandoned: int = 0) -> None:
        """Takes the documents to hand out.

        Args:
            queued: The documents to claim, in order.
            abandoned: What a sweep reports.
        """
        self.pending = list(queued)
        self.abandoned = abandoned
        self.claims: list[str] = []
        self.stored: dict[str, Chunking] = {}
        self.failures: dict[str, str] = {}
        self.refuses: Exception | None = None

    def claim(self) -> ChunkClaim | None:
        """Takes the next pending document."""
        if not self.pending:
            return None
        taken = self.pending.pop(0)
        self.claims.append(taken.sha256)
        return taken

    def replace(self, sha256: str, chunking: Chunking) -> int:
        """Replaces a document's passages with a new set."""
        if self.refuses is not None:
            raise self.refuses
        self.stored[sha256] = chunking
        return len(chunking.passages)

    def fail(self, key: str, error: str) -> None:
        """Records a document this stage could not process."""
        self.failures[key] = error

    def abandon(self) -> int:
        """Sweeps claims an earlier run left behind."""
        return self.abandoned


class ChunkDriver:
    """The chunking flow, with the queue and the parsed bucket in memory."""

    def __init__(
        self,
        *,
        queued: tuple[ChunkClaim, ...] = (),
        builder: ScriptedBuilder | None = None,
        document: DoclingDocument | None = None,
        abandoned: int = 0,
    ) -> None:
        """Wires the service over the doubles.

        Args:
            queued: The documents to hand out.
            builder: The passage builder to use.
            document: The converted document the bucket holds for each.
            abandoned: What a sweep reports.
        """
        self.rows = ChunkRows(*queued, abandoned=abandoned)
        self.parsed = MemoryParsed()
        self.builder = builder or ScriptedBuilder()
        held = document or docling_document(paragraphs=("The device weighs 4 kg.",))
        for one in queued:
            self.parsed.objects[self.parsed.key_for(one.sha256)] = (
                held.model_dump_json().encode()
            )
        self.service = ChunkingService(
            repository=self.rows,  # pyright: ignore[reportArgumentType]
            parsed=self.parsed,  # pyright: ignore[reportArgumentType]
            builder=self.builder,  # pyright: ignore[reportArgumentType]
        )

    def run(self) -> str | None:
        """Chunks one queued document."""
        return self.service.process_next()

    def drain(self) -> int:
        """Chunks every queued document."""
        return self.service.drain()

    @property
    def stored(self) -> dict[str, Chunking]:
        """What each finished document was cut into."""
        return self.rows.stored

    @property
    def failure(self) -> str | None:
        """Why the last document failed, if it did."""
        return next(iter(self.rows.failures.values()), None)

    def language_given(self, index: int = 0) -> str | None:
        """Which language the builder was handed."""
        return self.builder.built[index][1]


class PassageRows:
    """The stored passages a re-read walks, held in memory."""

    def __init__(self, *passages: tuple[int, str, str | None]) -> None:
        """Takes one (id, text, document language) per passage."""
        self.passages = list(passages)
        self.written: list[tuple[int, str | None, list[str]]] = []
        self.batches: list[int] = []

    def texts(self, within=None) -> list[tuple[int, str, str | None]]:
        """Every passage's id, text and its document's language."""
        return list(self.passages)

    def revocabulary(self, read: list[tuple[int, str | None, list[str]]]) -> int:
        """Replaces the stored language and lemmas."""
        self.written.extend(read)
        self.batches.append(len(read))
        return len(read)


class RevocabularyDriver:
    """Re-reading the language and vocabulary of stored passages."""

    def __init__(self, *passages: tuple[int, str, str | None]) -> None:
        """Takes the passages the corpus holds."""
        self.rows = PassageRows(*passages)

    def run(self) -> int:
        """Reads every passage again."""
        return revocabulary(self.rows)  # pyright: ignore[reportArgumentType]

    @property
    def languages(self) -> dict[int, str | None]:
        """The language each passage was written as."""
        return {one: language for one, language, _ in self.rows.written}

    @property
    def lemmas(self) -> dict[int, list[str]]:
        """The lemmas each passage was written with."""
        return {one: terms for one, _, terms in self.rows.written}

    @property
    def batches(self) -> list[int]:
        """How many passages each write carried."""
        return self.rows.batches


__all__ = [
    "NOT_A_PDF",
    "PDF",
    "ChunkDriver",
    "ChunkRows",
    "DocumentRows",
    "IngestDriver",
    "MemoryDocuments",
    "MemoryParsed",
    "NoPassages",
    "ParseDriver",
    "ParseRows",
    "PassageRows",
    "ReaderSpy",
    "RemovalDriver",
    "RevocabularyDriver",
    "ScriptedBuilder",
    "ScriptedPipeline",
    "cell",
    "chunk_of",
    "docling_document",
    "placed",
    "table_of",
]
