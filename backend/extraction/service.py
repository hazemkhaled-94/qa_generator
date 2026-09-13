"""The extraction flow: claim, skip what asserts nothing, read, check, store."""

from __future__ import annotations

import logging
from typing import ClassVar

from extraction.extractors import ExtractionFailed, ExtractorRegistry
from extraction.models import CheckedFact, PassageToExtract
from extraction.repository import FactCatalog, PassageQueue
from extraction.validation import FactChecker
from nlp.analysis import normalised
from stages import StageService
from telemetry import tracer

log = logging.getLogger(__name__)
span = tracer(__name__)

#: How many re-judged facts to write at once.
_REJUDGE_BATCH = 500


def revalidate(catalog: FactCatalog, within=None) -> int:
    """Judges every stored fact again, without calling the model.

    What the model wrote is the record of one extraction and is kept; what
    the checks read off it is derived, and is replaced with what today's
    checks read. This is what applies a change to the checks to facts that
    were extracted before it.
    """
    checker = FactChecker()
    verdicts: list[tuple[int, CheckedFact]] = []
    written = 0

    def flush() -> None:
        """Writes what has piled up."""
        nonlocal written
        written += catalog.rejudge(verdicts)
        verdicts.clear()

    for fact_id, passage, candidate, method in catalog.judged(within):
        verdicts.append((fact_id, checker.check(passage, candidate, method)))
        if len(verdicts) >= _REJUDGE_BATCH:
            flush()
    flush()
    log.info("re-judged %d fact(s)", written)
    return written


def _is_heading(passage: PassageToExtract) -> bool:
    """Reports whether a passage is its own heading and nothing more."""
    if not passage.section_path:
        return False
    text = normalised(passage.text)
    return any(
        text == normalised(segment) for segment in passage.section_path.split(" > ")
    )


def skipped(passage: PassageToExtract) -> str | None:
    """Says why a passage is not worth a model call, or None if it is.

    A passage whose sentences carry no finite verb asserts nothing - a
    heading, a caption, a navigation line, a bare list fragment. Asking a
    model to find a claim in one costs a full call and returns the text back.
    """
    # Ahead of the table exemption: nothing can be cited from a passage with
    # nothing numbered, whatever kind it is. A table read without them
    # produced a fact per cell, every one citing a row that did not exist.
    if not passage.sentences:
        return "no sentences"
    if passage.block_type in ("table",):
        return None
    if _is_heading(passage):
        return "heading"
    if not passage.claims:
        return "no finite verb"
    return None


class ExtractionService(StageService):
    """Draws atomic facts out of passages, one passage at a time.

    The unit of work is a passage rather than a document, so the queue
    divides evenly between workers and one unreadable passage fails only
    itself. A passage that yields no facts is not a failure, and neither is a
    fact that fails a check: that one is stored with the reason.
    """

    name: ClassVar[str] = "extraction"
    unit: ClassVar[str] = "passage"

    def __init__(
        self,
        *,
        repository: PassageQueue,
        extractors: ExtractorRegistry,
        checker: FactChecker,
    ) -> None:
        """Initialises the service with its collaborators."""
        super().__init__(repository)
        self._repository: PassageQueue = repository
        self._extractors = extractors
        self._checker = checker

    def process_next(self) -> int | None:
        """Reads one queued passage and stores what it yielded."""
        passage = self._repository.claim()
        if passage is None:
            return None

        with span.start_as_current_span("extract") as current:
            current.set_attribute("passage.id", passage.id)
            current.set_attribute("passage.block_type", passage.block_type or "")
            try:
                facts = self._passage(passage, current)
                stored = self._repository.store(passage.id, facts)
                validated = sum(1 for fact in facts if fact.validated)
                self._done(current)
                current.set_attribute("extract.facts", stored)
                current.set_attribute("extract.validated", validated)
                log.info(
                    "passage %d: %d fact(s), %d validated (%d rejected)",
                    passage.id,
                    stored,
                    validated,
                    stored - validated,
                )
            except ExtractionFailed as exc:
                self._fail(passage.id, str(exc), current)
            except Exception as exc:
                self._fail(passage.id, f"{type(exc).__name__}: {exc}", current)
                log.exception("failed passage %d", passage.id)
        return passage.id

    def _passage(self, passage: PassageToExtract, current) -> list[CheckedFact]:
        """Reads one passage and checks everything it produced."""
        reason = skipped(passage)
        if reason is not None:
            # Marked read with nothing found, not failed: there was nothing
            # here to read, which is an answer and not an error.
            current.set_attribute("extract.skipped", reason)
            return []

        extractor = self._extractors.for_block_type(passage.block_type)
        return [
            self._checker.check(
                passage, candidate, extractor.method, extractor.provenance
            )
            for candidate in extractor.extract(passage)
        ]
