"""The extraction flow: claim, skip what asserts nothing, read, check, store."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import replace
from itertools import zip_longest
from typing import ClassVar

from database.qa_generator import FactKind, Rejection
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
from telemetry import tracer, working

log = logging.getLogger(__name__)
span = tracer(__name__)

#: How many re-judged facts to write at once.
_REJUDGE_BATCH = 500

#: How many claims a passage must carry before a digest is asked for.
_DIGESTIBLE = 2


def revalidate(catalog: FactCatalog, checker: FactChecker, within=None) -> int:
    """Judges every stored fact again, without calling the model.

    Args:
        catalog: Where the facts are read from and written back to.
        checker: What judges them, holding the shares this configuration
            sets. Passed rather than built here, as the bridge pass already
            does: built here it would read the process environment, and a
            deployment that changed EXTRACTION_DIGEST_MAX_SHARE through
            /settings stores the new value rather than exporting it - so a
            digest would be re-judged against the file while the fact was
            stamped with the stored configuration's version.
        within: A condition narrowing which facts are re-judged, or None for
            all of them.

    Returns:
        How many facts were written back.
    """
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
            "left %d fact(s) alone: their citation resolved to no sentence, so "
            "there is nothing to judge them against and re-judging would "
            "overwrite the only record of what they cited. A bridge among them "
            "was drawn before the prompt recorded its sentences; run --bridge "
            "to replace those.",
            skipped,
        )
    return written


def recap(catalog: FactCatalog, cap: int | None, within=None) -> int:
    """Applies the atomic cap to facts already stored.

    Args:
        catalog: Where the facts are read from and written back to.
        cap: The most validated atomic facts one passage may keep, or None
            when EXTRACTION_MIN_OTHER_SHARE sets no cap.
        within: A condition narrowing which passages are read, or None.

    Returns:
        How many facts were refused.
    """
    if cap is None:
        log.info("EXTRACTION_MIN_OTHER_SHARE caps nothing; no fact was touched")
        return 0
    written = catalog.recap(
        cap,
        f"the passage yielded more than the {cap} atomic fact(s) "
        f"EXTRACTION_MIN_OTHER_SHARE leaves room for, and the ones asserting "
        f"a number, a date or a name were kept ahead of it",
        within,
    )
    log.info("refused %d atomic fact(s) over the cap of %d a passage", written, cap)
    return written


def _uncitable(candidate) -> bool:
    """Whether a stored fact recorded no sentence to judge it against.

    Two kinds of row look like this and neither is worth re-reading.

    A bridge drawn before prompt version 2 named its passages and not the
    sentences in them. Judging one would resolve nothing and refuse it as
    evidence_absent, which would throw away a fact that was correct under the
    prompt that wrote it.

    A fact already refused as evidence_absent is the other. The numbers its
    citation named survive nowhere but its own `validation_error`: the link
    rows carry NULL, which is what says they resolved to nothing. Re-judging
    one rebuilds the candidate from those NULLs, reaches the same verdict,
    and rewrites the message as `names (none)` - so the only record of what
    the model actually cited is lost to a pass that learned nothing.

    Read off the field `_rejudge` will read, which is not the same one for
    both kinds: a bridge is judged against the sentences named per passage
    and everything else against the candidate's own.
    """
    if candidate.kind == FactKind.BRIDGE:
        return not any(one.sentences for one in candidate.passages)
    return not candidate.sentences


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


#: Block types that say where something is rather than what it says. A table
#: of contents yields "Das Team managen beginnt auf Seite 69"; the index this
#: corpus renders as a code block yielded 95 facts of the form "Der Lehrplan
#: umfasst das Thema X" from four passages. Both are navigation, and a
#: question asked of either measures a reader's page-turning.
_NAVIGATION = frozenset({"document_index", "code"})

#: How many characters a passage may carry per claim before it stops
#: reading as prose. The parser labels what it recognises, and an index it
#: does not recognise arrives as `text`: one did, at 1,568 characters
#: carrying a single finite verb, and became questions asking which pages a
#: term appears on. Prose in this corpus runs 166 characters per claim at
#: the median and 357 at the 95th percentile.
#:
#: Why not `claims` alone: that check already exists and refuses only a
#: passage with none. A thousand characters of index with one stray verb in
#: it passes, which is what happened.
_CHARS_PER_CLAIM = 500

#: The share of a passage's words that may carry a digit before it reads as
#: a list of references. Sparseness alone is not enough to judge on: a
#: German bullet list is sparse because its points are infinitives rather
#: than sentences, and those are content. Measured over the passages this
#: pair of rules selects:
#:
#:     content bullets      0.0%      bibliography    9-16%
#:     content prose        1.0%      contents page    25%
#:                                    index          41-56%
#:
#: So both must hold. Digits rather than any word, so this says nothing
#: about a language or a subject; a passage genuinely about numbers is a
#: `table` and exempt above.
_NUMERIC_WORDS = 0.05

#: The glyph a copyright notice opens with, in any language. Front matter
#: asserts things a model will turn into facts - who holds a right, which
#: edition a thing belongs to - and they are about the document rather than
#: about what it says.
_COPYRIGHT = "\u00a9"

_HAS_DIGIT = re.compile(r"\d")


def _reference_list(passage: PassageToExtract) -> bool:
    """Whether the passage is a list of references rather than prose.

    Both halves, because either alone is wrong. Sparse-in-claims catches a
    content bullet list; numeric-in-words catches a passage that is simply
    about measurements. An index, a bibliography and a contents page are
    the only things that are both.
    """
    text = normalised(passage.text)
    words = text.split()
    if not words or len(text) <= _CHARS_PER_CLAIM * passage.claims:
        return False
    numeric = sum(1 for one in words if _HAS_DIGIT.search(one)) / len(words)
    return numeric > _NUMERIC_WORDS


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
    if passage.block_type in _NAVIGATION:
        return f"{passage.block_type}: navigation rather than content"
    if passage.block_type in ("table",):
        return None
    if _is_heading(passage):
        return "heading"
    if not passage.claims:
        return "no finite verb"
    # What the checks above are for, applied to what the parser did not
    # label. Between them these produced questions asking which pages a
    # term appears on and how the document is licensed - answerable,
    # checkable, and about the document rather than about its subject.
    if _COPYRIGHT in passage.text:
        return "a copyright notice, which is about the document"
    if _reference_list(passage):
        return "an index, a bibliography or a contents page rather than prose"
    return None


def over_cap(facts: Sequence[CheckedFact], cap: int | None) -> set[int]:
    """Which of one passage's facts the atomic cap refuses, by position.

    Only the validated atomic ones compete for the budget and only they are
    refused. A fact a check already threw out is not taking a place from
    anything, and overwriting its code would lose why it really went.

    The ones asserting a value are kept, which is the rule question
    generation already offers facts by: a fact carrying a number, a date or
    a name is what a checkable question is written from. Ties go to the
    order the model wrote them in, so one passage read twice keeps the same
    facts.
    """
    if cap is None:
        return set()
    competing = [
        (position, fact)
        for position, fact in enumerate(facts)
        if fact.kind == FactKind.ATOMIC and fact.validated
    ]
    if len(competing) <= cap:
        return set()
    ranked = sorted(competing, key=lambda one: (not one[1].units_statement, one[0]))
    return {position for position, _ in ranked[cap:]}


def _refuse(fact: CheckedFact, cap: int) -> CheckedFact:
    """Marks one atomic fact as over the cap, leaving the rest of it alone."""
    return replace(
        fact,
        validated=False,
        rejection_code=Rejection.OVER_CAP,
        validation_error=(
            f"the passage yielded more than the {cap} atomic fact(s) "
            f"EXTRACTION_MIN_OTHER_SHARE leaves room for, and the ones "
            f"asserting a number, a date or a name were kept ahead of it"
        ),
    )


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
        atomic_cap: int | None = None,
        digest_min_chars: int = 0,
    ) -> None:
        """Initialises the service with its collaborators.

        Args:
            repository: The queue it claims passages from.
            extractors: One reader per block type, with a default.
            checker: What judges everything they propose.
            digest: The reader of a passage's summary and outline, or None
                when this deployment writes neither.
            atomic_cap: The most atomic facts one passage keeps, or None for
                no cap. EXTRACTION_MIN_OTHER_SHARE is where it comes from.
            digest_min_chars: The shortest passage worth digesting. 0
                digests every passage carrying enough claims.
        """
        super().__init__(repository)
        self._repository: PassageQueue = repository
        self._extractors = extractors
        self._checker = checker
        self._digest = digest
        self._atomic_cap = atomic_cap
        self._digest_min_chars = digest_min_chars

    def process_next(self) -> int | None:
        """Reads one queued passage and stores what it yielded.

        Returns:
            The passage's id, or None when the queue is empty.
        """
        passage = self._repository.claim()
        if passage is None:
            return None

        with working(
            span,
            "extract",
            {
                "stage": self.name,
                "passage.id": passage.id,
                "passage.block_type": passage.block_type or "",
            },
        ) as current:
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
                    extra={
                        "facts.stored": stored,
                        "facts.validated": validated,
                        "facts.rejected": stored - validated,
                    },
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
        cap = self._atomic_cap
        refused = over_cap(facts, cap)
        if cap is not None and refused:
            current.set_attribute("extract.over_cap", len(refused))
            facts = [
                _refuse(fact, cap) if position in refused else fact
                for position, fact in enumerate(facts)
            ]
        return facts + self._digested(passage)

    def _digested(self, passage: PassageToExtract) -> list[CheckedFact]:
        """Reads what the passage is about, when there is enough to condense.

        Two floors, and the second is the one that costs money. A digest is
        a FIXED size - the prompt asks for two or three sentences, or two to
        six points - so how much it condenses is decided by the passage
        rather than by the model. Measured over this corpus: against a
        passage under 400 characters the median summary was 99% of it and
        92% were refused as `not_condensed`; from 1,200 characters up, none
        were. The check was measuring passage length and calling it digest
        quality, and every one of those refusals had been paid for.
        """
        if self._digest is None or passage.claims < _DIGESTIBLE:
            return []
        if len(normalised(passage.text)) < self._digest_min_chars:
            return []
        return [
            self._checker.check(
                passage, candidate, self._digest.method, self._digest.provenance
            )
            for candidate in self._digest.extract(passage)
        ]
