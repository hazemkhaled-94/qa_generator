"""The extraction flow: claim, skip what asserts nothing, read, check, store."""

from __future__ import annotations

import logging
from itertools import zip_longest
from typing import ClassVar

from database.qa_generator import FactKind
from extraction.extractors import (
    BridgeExtractor,
    DigestExtractor,
    ExtractionFailed,
    ExtractorRegistry,
)
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

#: How many claims a passage must carry before a digest is asked for.
_DIGESTIBLE = 2


def revalidate(catalog: FactCatalog, within=None) -> int:
    """Judges every stored fact again, without calling the model.

    Args:
        catalog: Where the facts are read from and written back to.
        within: A condition narrowing which facts are re-judged, or None for
            all of them.

    Returns:
        How many facts were written back.
    """
    checker = FactChecker()
    verdicts: list[tuple[int, CheckedFact]] = []
    written = 0

    def flush() -> None:
        """Writes what has piled up."""
        nonlocal written
        written += catalog.rejudge(verdicts)
        verdicts.clear()

    skipped = 0
    for fact_id, passages, candidate, method in catalog.judged(within):
        if _uncitable(candidate):
            skipped += 1
            continue
        verdicts.append((fact_id, _rejudge(checker, passages, candidate, method)))
        if len(verdicts) >= _REJUDGE_BATCH:
            flush()
    flush()
    log.info("re-judged %d fact(s)", written)
    if skipped:
        log.warning(
            "left %d bridge(s) alone: they were drawn before the prompt recorded "
            "which sentences they rest on, so there is nothing to judge them "
            "against. Run --bridge to replace them.",
            skipped,
        )
    return written


def _uncitable(candidate) -> bool:
    """Whether a stored bridge recorded no sentence to judge it against.

    A bridge drawn before prompt version 2 named its passages and not the
    sentences in them. Judging one would resolve nothing and refuse it as
    evidence_absent, which would throw away a fact that was correct under the
    prompt that wrote it.
    """
    return candidate.kind == FactKind.BRIDGE and not any(
        one.sentences for one in candidate.passages
    )


def _rejudge(
    checker: FactChecker,
    passages: list[PassageToExtract],
    candidate,
    method: str,
) -> CheckedFact:
    """Judges one stored fact against the passages it was drawn from."""
    if candidate.kind == FactKind.BRIDGE:
        return checker.check_bridge(passages, candidate)
    return checker.check(passages[0], candidate, method)


def grouped(
    passages: list[PassageToExtract], wanted: int, size: int
) -> list[list[PassageToExtract]]:
    """Splits one topic's passages into the groups a bridge is read from.

    Documents are taken in turn, and the groups are strided over the whole
    topic rather than taken from its start.

    Args:
        passages: The topic's passages, in document and reading order.
        wanted: The most groups to return.
        size: How many passages one group holds.

    Returns:
        Up to `wanted` groups of exactly `size` passages. Empty when the
        topic holds fewer than `size`.
    """
    if size < 2 or wanted < 1 or len(passages) < size:
        return []

    by_document: dict[str | None, list[PassageToExtract]] = {}
    for passage in passages:
        by_document.setdefault(passage.doc_sha256, []).append(passage)

    interleaved = [
        passage
        for taken in zip_longest(*by_document.values())
        for passage in taken
        if passage is not None
    ]
    whole = [
        interleaved[at : at + size]
        for at in range(0, len(interleaved) - size + 1, size)
    ]
    return whole[:: max(1, len(whole) // wanted)][:wanted]


def bridge(
    catalog: FactCatalog,
    extractor: BridgeExtractor,
    checker: FactChecker,
    groups_per_topic: int,
    size: int,
    within=None,
) -> int:
    """Writes the bridge facts a corpus's topics offer.

    One model call per group. Every bridge in scope is deleted first, so a
    second run replaces what the first wrote instead of adding to it.

    Args:
        catalog: Where the groups are read from and the facts written to.
        extractor: The model-backed reader of a group.
        checker: What judges what it proposed.
        groups_per_topic: How many groups one topic is worth.
        size: How many passages one group holds.
        within: A condition narrowing which passages are read, or None.

    Returns:
        How many bridge facts were stored, refused ones included.
    """
    log.info("replacing %d bridge fact(s)", catalog.clear_bridges(within))
    written = 0
    for topic_id, passages in catalog.by_topic(within):
        for offered in grouped(passages, groups_per_topic, size):
            try:
                proposed = extractor.extract(offered)
            except ExtractionFailed as exc:
                log.warning("topic %d: a group failed: %s", topic_id, exc)
                continue
            written += catalog.add_bridges(
                [
                    checker.check_bridge(offered, candidate, extractor.provenance)
                    for candidate in proposed
                ]
            )
    log.info("wrote %d bridge fact(s)", written)
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
    """Says why a passage is not worth a model call.

    Args:
        passage: The passage about to be read.

    Returns:
        The reason it was skipped, or None when it is worth reading.
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
    """Draws facts out of passages, one passage at a time.

    A passage that yields no facts is not a failure, and neither is a fact
    that fails a check: that one is stored with the reason.
    """

    name: ClassVar[str] = "extraction"
    unit: ClassVar[str] = "passage"

    def __init__(
        self,
        *,
        repository: PassageQueue,
        extractors: ExtractorRegistry,
        checker: FactChecker,
        digest: DigestExtractor | None = None,
    ) -> None:
        """Initialises the service with its collaborators.

        Args:
            repository: The queue it claims passages from.
            extractors: One reader per block type, with a default.
            checker: What judges everything they propose.
            digest: The reader of a passage's summary and outline, or None
                when this deployment writes neither.
        """
        super().__init__(repository)
        self._repository: PassageQueue = repository
        self._extractors = extractors
        self._checker = checker
        self._digest = digest

    def process_next(self) -> int | None:
        """Reads one queued passage and stores what it yielded.

        Returns:
            The passage's id, or None when the queue is empty.
        """
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
        facts = [
            self._checker.check(
                passage, candidate, extractor.method, extractor.provenance
            )
            for candidate in extractor.extract(passage)
        ]
        return facts + self._digested(passage)

    def _digested(self, passage: PassageToExtract) -> list[CheckedFact]:
        """Reads what the passage is about, when there is enough to condense."""
        if self._digest is None or passage.claims < _DIGESTIBLE:
            return []
        return [
            self._checker.check(
                passage, candidate, self._digest.method, self._digest.provenance
            )
            for candidate in self._digest.extract(passage)
        ]
