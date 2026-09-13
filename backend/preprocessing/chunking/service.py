"""The chunking flow: claim, read, cut, store."""

from __future__ import annotations

import logging
from itertools import groupby
from typing import ClassVar, Protocol

from docling_core.types.doc.document import DoclingDocument
from opentelemetry.trace import Span

from nlp.analysis import read
from nlp.language import detect
from preprocessing.chunking.passages import NoPassages, PassageBuilder
from preprocessing.chunking.repository import ChunkQueue, PassageCatalog
from stages import StageService
from telemetry import tracer

log = logging.getLogger(__name__)
span = tracer(__name__)


def revocabulary(catalog: PassageCatalog, within=None) -> int:
    """Reads every stored passage's language and vocabulary again, in place.

    The same two readings chunking makes, over passages already stored: the
    language is detected per passage and falls back to the document's, and
    the vocabulary is read with the pipeline that language names. Applies a
    change to either without re-chunking, which would delete every passage
    and take its facts with it.
    """
    passages = catalog.texts(within)
    if not passages:
        log.warning("no passages to read")
        return 0

    # Detected first, so each passage is read by the pipeline for the language
    # it is actually in, and one batch is parsed per language.
    spoken = [
        (passage_id, text, detect(text) or fallback)
        for passage_id, text, fallback in passages
    ]
    written = 0
    for language, group in groupby(sorted(spoken, key=_spoken), key=_spoken):
        batch = list(group)
        found = [
            (passage_id, language, terms)
            for (passage_id, _, _), (_, terms) in zip(
                batch, read([text for _, text, _ in batch], language), strict=True
            )
        ]
        written += catalog.revocabulary(found)
        log.info(
            "%s: read %d passage(s), %d term(s)",
            language or "no language",
            len(found),
            sum(len(terms) for _, _, terms in found),
        )
    return written


def _spoken(row: tuple[int, str, str | None]) -> str:
    """Names the language a passage was detected as, for grouping."""
    return row[2] or ""


class ParsedStore(Protocol):
    """What this service needs from the store holding converted documents.

    Declared here rather than imported from blob_store, so the dependency
    points inwards.
    """

    def key_for(self, sha256: str) -> str:
        """Returns the key a converted document is stored under."""
        ...

    def get(self, key: str) -> bytes:
        """Reads an object."""
        ...


class ChunkingService(StageService):
    """Splits converted documents into passages, one document at a time.

    Reads the parsed bucket rather than the original file, so re-chunking
    never re-parses. Work arrives through documents.chunk_status.
    """

    name: ClassVar[str] = "chunking"
    unit: ClassVar[str] = "document"

    def __init__(
        self,
        *,
        repository: ChunkQueue,
        parsed: ParsedStore,
        builder: PassageBuilder,
    ) -> None:
        """Initialises the service with its collaborators."""
        super().__init__(repository)
        self._repository: ChunkQueue = repository
        self._parsed = parsed
        self._builder = builder

    def process_next(self) -> str | None:
        """Chunks one queued document, recording a failure against the row."""
        claimed = self._repository.claim()
        if claimed is None:
            return None

        with span.start_as_current_span("chunk") as current:
            current.set_attribute("document.sha256", claimed.sha256)
            try:
                self._chunk(claimed.sha256, claimed.language, current)
            except NoPassages as exc:
                self._fail(claimed.sha256, str(exc), current)
            except Exception as exc:
                self._fail(claimed.sha256, f"{type(exc).__name__}: {exc}", current)
                log.exception("failed %s", claimed.sha256)
        return claimed.sha256

    def _chunk(self, sha256: str, language: str | None, current: Span) -> None:
        """Cuts one claimed document into passages and stores them."""
        document = DoclingDocument.model_validate_json(
            self._parsed.get(self._parsed.key_for(sha256))
        )
        chunking = self._builder.build(document, language)
        stored = self._repository.replace(sha256, chunking)

        self._done(current)
        current.set_attribute("chunk.passages", stored)
        current.set_attribute("chunk.oversized", chunking.oversized)
        log.info(
            "chunked %s: %d passage(s), %d sentence(s)",
            sha256,
            stored,
            sum(len(passage.sentences) for passage in chunking.passages),
        )
        if chunking.oversized:
            # Loud on purpose: the chunker splits on this budget, so a passage
            # above it means the split did not happen.
            log.warning(
                "%d passage(s) of %s came back over the token budget and were "
                "stored anyway; they will be truncated when embedded.",
                chunking.oversized,
                sha256,
            )
