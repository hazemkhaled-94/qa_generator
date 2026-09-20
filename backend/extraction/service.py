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
from extraction.models import CheckedFact, PassageToExtract, Twin
from extraction.repository import FactCatalog, PassageQueue
from extraction.validation import WRITTEN, FactChecker
from nlp.analysis import claim, normalised
from nlp.embedding import Embedder, cosine
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


#: How many rows one backfill batch embeds. The forward pass is the cost and
#: it is per batch, not per row.
_EMBED_BATCH = 64


def embed(catalog: FactCatalog, embedder: Embedder, within=None) -> int:
    """Writes the vectors onto passages and facts already stored.

    The replay for this gate, as `recap` is for the cap: it calls no served
    model, because a vector is read off a statement that is already written.
    What it does not do is apply the gate - a fact stored before the column
    existed was accepted, and refusing it now would rewrite a verdict the
    corpus was measured under. Re-extract to have it judged.

    Args:
        catalog: Where the rows are read and written.
        embedder: The model the vectors come from.
        within: A condition narrowing which rows are read, or None.

    Returns:
        How many rows were embedded, passages and facts together.
    """
    done = 0
    for read, write in (
        (catalog.unembedded_passages, catalog.embed_passages),
        (catalog.unembedded, catalog.embed_facts),
    ):
        while batch := read(_EMBED_BATCH, within):
            vectors = embedder.embed_all([normalised(text) for _, text in batch])
            done += write(
                [(row, vector) for (row, _), vector in zip(batch, vectors, strict=True)]
            )
            log.info("embedded %d row(s)", done)
    return done


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


def _partner(
    head: PassageToExtract, offered: Sequence[PassageToExtract]
) -> PassageToExtract | None:
    """Whichever unused passage the head most nearly meets, or None.

    Cosine over `passages.embedding`, and the one preferring another
    document loses nothing: the interleave has already put the documents in
    turn, so a head's neighbours are mostly from elsewhere anyway.

    Falls back to the first offered where nothing carries a vector, which is
    the adjacency this replaced and is what a corpus extracted before the
    column existed still gets.
    """
    if not offered:
        return None
    if head.embedding is None:
        return offered[0]
    scored = [
        (cosine(list(head.embedding), list(one.embedding)), -one.id, one)
        for one in offered
        if one.embedding is not None
    ]
    if not scored:
        return offered[0]
    return max(scored)[2]


def _paired(
    interleaved: list[PassageToExtract], size: int
) -> list[list[PassageToExtract]]:
    """Groups each passage with the ones it most nearly meets.

    What a bridge is asked for is a claim no single passage states, and two
    passages with nothing between them have no such claim: the prompt says
    returning none is correct, and a group of strangers is a call spent
    being told so. Sharing a topic is the corpus's own statement that two
    passages are related, and it is a weak one - a topic holds eighty
    passages and a subject is narrower than that.

    So the topic narrows and the vectors pair, which is the same two-stage
    shape question generation uses to pair passages for a wide sample.

    Deterministic: the head order is the interleave's, and a tie between two
    equally near passages goes to the lower id.
    """
    remaining = list(interleaved)
    groups: list[list[PassageToExtract]] = []
    while len(remaining) >= size:
        group = [remaining.pop(0)]
        while len(group) < size:
            found = _partner(group[-1], remaining)
            if found is None:
                break
            remaining.remove(found)
            group.append(found)
        if len(group) == size:
            groups.append(group)
    return groups


def grouped(
    passages: list[PassageToExtract], wanted: int, size: int
) -> list[list[PassageToExtract]]:
    """Splits one topic's passages into the groups a bridge is read from.

    Documents are taken in turn, each passage is paired with the one it most
    nearly meets, and the groups are strided over the whole topic rather
    than taken from its start.

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
    whole = _paired(interleaved, size)
    # Evenly spaced from the first group to the last, rather than a stride.
    # A stride of len(whole) // wanted floors to 1 for any topic holding
    # fewer than twice as many groups as were asked for, which is the
    # ordinary case at twelve a topic - it then takes the first twelve of
    # twenty-three and the back half of the topic is never read. Ceiling
    # instead returns ten groups where twelve were wanted.
    chosen = min(wanted, len(whole))
    step = (len(whole) - 1) / (chosen - 1) if chosen > 1 else 0
    return [whole[round(at * step)] for at in range(chosen)]


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
        within: A condition selecting the topics to read, or None.

    Returns:
        How many bridge facts were stored, refused ones included.
    """
    log.info("replacing %d bridge fact(s)", catalog.clear_bridges(within))
    written = 0
    for topic_id, passages in catalog.by_topic(within):
        # The same gate the per-passage stage applies, so the two cannot
        # drift. Without it a group is spent on a table of contents or a
        # heading: `by_topic` asks only that a passage was segmented, and
        # the index this corpus renders as a code block yielded 95 facts
        # from four passages before the gate existed.
        readable = [one for one in passages if skipped(one) is None]
        for offered in grouped(readable, groups_per_topic, size):
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

#: A cell holding an identifier rather than a statement: a learning-objective
#: code, a level, a tick. Anchored, so a cell containing one of these among
#: words is not one.
_CODE_CELL = re.compile(
    r"""^(?:
        [^\W\d_]{0,6}[-. ]?\d+(?:\.\d+)*   # TM-2.2.1, Abschnitt 4, 1.7
      | [A-Z]{1,5}-?[A-Z]*\d+              # TA-BO1, FL4
      | [XxKk]\d?                          # K2, X
      | [✓✗X-]                   # tick, cross, dash
    )$""",
    re.VERBOSE,
)

#: The share of a table's filled cells that may be identifiers before the grid
#: stops being a statement of anything. A traceability matrix, a release-note
#: table and an abbreviation list are all mostly codes; measured over this
#: corpus, no table carrying prose reaches half.
#:
#:     content tables    0.155 mean, none above 0.5
#:     matrices etc.     0.424 mean, 28 of 68 above it
#:
#: Set where nothing good is refused rather than where everything bad is
#: caught: a table wrongly read yields a fact per cell, and a table wrongly
#: skipped yields nothing, but the first is the one that reaches a question.
_CODE_CELLS = 0.5


def _code_grid(passage: PassageToExtract) -> bool:
    """Whether a table's cells are identifiers rather than statements."""
    cells = [
        text
        for grid in passage.table_cells
        for cell in grid.get("cells", ())
        if (text := (cell.get("text") or "").strip())
    ]
    if not cells:
        return False
    codes = sum(1 for one in cells if _CODE_CELL.match(one))
    return codes / len(cells) > _CODE_CELLS


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
        # The one furniture check a table is not exempt from. The exemptions
        # below exist because a grid carries no sentences and so no claims,
        # which the signals they read are counted in; this one reads the
        # cells themselves and so says something about a table.
        if _code_grid(passage):
            return "a grid of identifiers rather than of statements"
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


def _unread_prose(passage: PassageToExtract) -> bool:
    """Whether a table passage also carries lines no reader would reach.

    The chunker labels a merged passage `table` if ANY item in it is one,
    and numbers the whole of it by rendered line rather than by sentence -
    `lines_of` records `predicates: 0` for every line, so `claims` is zero
    for every table and the prose merged into one is invisible to each of
    the signals above. The cell reader then walks the grid and nothing
    reads the rest.

    Measured over this corpus: 161 of 188 table passages carry lines
    outside their grids, about 145,000 characters. Most of it is rendered
    rows and `|---|` separators, and the rest is ordinary German prose -
    "In diesem Abschnitt werden die geschäftlichen Nutzen aufgeführt...".
    A finite verb is what tells the two apart, so it is asked here rather
    than read off a column that was written as zero.
    """
    if not passage.table_cells:
        return False
    gridded = {
        cell.get("line")
        for grid in passage.table_cells
        for cell in grid.get("cells", ())
    }
    return any(
        one.index not in gridded and claim(one.text, passage.language).predicates
        for one in passage.sentences
    )


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

    A statement a deterministic reader composed does not compete. The cap is
    EXTRACTION_MIN_OTHER_SHARE read the other way round - the share of a
    passage's facts that may be something other than atomic - and what makes
    a floor on the others a cap on these is that a passage yields a FIXED
    number of digests beside them. A table yields none: its rendered rows
    carry no finite verb, so `_DIGESTIBLE` is never met and there is no
    other kind for the share to be. Applying the cap there is arithmetic
    about a quantity that is zero, and it threw away 56 of a 60-cell grid -
    the cheapest and most checkable facts in the corpus - by grid position.
    """
    if cap is None:
        return set()
    competing = [
        (position, fact)
        for position, fact in enumerate(facts)
        if fact.kind == FactKind.ATOMIC
        and fact.validated
        and fact.extraction_method == WRITTEN
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


def _duplicate(fact: CheckedFact, twin: Twin, threshold: float) -> CheckedFact:
    """Marks one fact as a near twin of a fact already stored."""
    return replace(
        fact,
        validated=False,
        rejection_code=Rejection.DUPLICATE,
        validation_error=(
            f"says what fact {twin.fact_id} already says at cosine "
            f"{twin.similarity:.3f}, over the {threshold:.2f} "
            f"EXTRACTION_DUPLICATE_COSINE allows: {twin.statement!r}"
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
        embedder: Embedder | None = None,
        duplicate_cosine: float = 0.0,
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
            embedder: The embedding model, or None to write no vectors and
                run no dedup gate.
            duplicate_cosine: How alike two statements may be before the
                second is refused. 0 turns the gate off and leaves the
                vectors being written.
        """
        super().__init__(repository)
        self._repository: PassageQueue = repository
        self._extractors = extractors
        self._checker = checker
        self._digest = digest
        self._atomic_cap = atomic_cap
        self._digest_min_chars = digest_min_chars
        self._embedder = embedder
        self._duplicate_cosine = duplicate_cosine

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
                facts = self._deduplicated(facts, current)
                stored = self._repository.store(
                    passage.id, facts, self._embedded(passage)
                )
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

        facts = [
            self._checker.check(passage, candidate, reader.method, reader.provenance)
            for reader in self._readers(passage, current)
            for candidate in reader.extract(passage)
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

    def _embedded(self, passage: PassageToExtract) -> list[float] | None:
        """The passage's own vector, or None when this deployment writes none."""
        if self._embedder is None:
            return None
        return self._embedder.embed(normalised(passage.text))

    def _deduplicated(self, facts: list[CheckedFact], current) -> list[CheckedFact]:
        """Embeds every fact and refuses the ones already in the corpus.

        Two probes per candidate, because neither alone is enough. The index
        holds what earlier passages stored; `written` holds what this passage
        has already kept, without which one passage would happily store the
        same statement twice - a summary and an atomic fact often say the
        same thing about a short passage.

        Only validated facts compete and only they are refused: a fact a
        check already threw out is not a duplicate of anything, and comparing
        it would make the corpus's first copy depend on which passage was
        read first.
        """
        if self._embedder is None:
            return facts
        vectors = self._embedder.embed_all([fact.statement for fact in facts])
        threshold = self._duplicate_cosine
        written: list[tuple[str, list[float]]] = []
        checked: list[CheckedFact] = []
        refused = 0
        for fact, vector in zip(facts, vectors, strict=True):
            fact = replace(fact, embedding=vector)
            if threshold <= 0 or not fact.validated:
                checked.append(fact)
                continue
            twin = self._nearest(vector, written)
            if twin is not None and twin.similarity >= threshold:
                refused += 1
                checked.append(_duplicate(fact, twin, threshold))
                continue
            written.append((fact.statement, vector))
            checked.append(fact)
        if refused:
            current.set_attribute("extract.duplicates", refused)
        return checked

    def _nearest(
        self, vector: list[float], written: Sequence[tuple[str, list[float]]]
    ) -> Twin | None:
        """The closest stored fact, this passage's own kept facts included."""
        twin = self._repository.nearest_fact(vector)
        for statement, other in written:
            score = cosine(vector, other)
            if twin is None or score > twin.similarity:
                twin = Twin(fact_id=0, statement=statement, similarity=score)
        return twin

    def _readers(self, passage: PassageToExtract, current) -> list:
        """Which extractors read one passage, in the order they are applied.

        Usually the one its block type routes to. A passage that merged a
        table with the prose around it gets both: the cell reader for the
        grid and the model for the rest.
        """
        routed = self._extractors.for_block_type(passage.block_type)
        default = self._extractors.default
        if routed is default or not _unread_prose(passage):
            return [routed]
        current.set_attribute("extract.prose_beside_a_table", True)
        return [routed, default]

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
