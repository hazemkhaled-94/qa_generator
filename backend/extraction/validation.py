"""Checking a proposed fact against the passages it came from.

Which checks a candidate faces is decided by its kind. See README.md for the
matrix and for what each one is protecting against.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial

from database.qa_generator import FactKind, Rejection
from extraction.models import (
    BULLET,
    MIN_POINTS,
    CandidateFact,
    CheckedFact,
    PassageToExtract,
    Provenance,
)
from nlp.analysis import VERSION, claim, normalised, vocabulary
from nlp.models import Claim
from nlp.pipelines import name as pipeline_name
from settings import decimal

#: The method whose statements a model writes. A deterministic reader
#: composes its statement from the grid, so it is neither a sentence nor
#: expected to read like one.
_WRITTEN = "llm"

#: A failed check, as its code and the measurement behind it.
Failure = tuple[str, str]
Check = Callable[["Judged"], Failure | None]


@dataclass(frozen=True)
class Judged:
    """One candidate resolved against its passages and parsed.

    Attributes:
        statement: The candidate's statement, stripped.
        evidence: The passage text the citation resolved to.
        sentence_ids: Which sentences of the anchor passage that was.
        start: Offset of the evidence in the anchor passage's text.
        end: Offset one past its last character.
        evidence_predicates: Finite verbs across the cited sentences.
        claim: What the statement itself asserts.
        evidence_vocabulary: Every form a word of the evidence appears in,
            over every passage the candidate rests on.
        passage_ids: Every passage it rests on, anchor first.
    """

    statement: str
    evidence: str
    sentence_ids: list[int]
    start: int
    end: int
    evidence_predicates: int
    claim: Claim
    evidence_vocabulary: frozenset[str]
    passage_ids: tuple[int, ...] = field(default=())

    @property
    def added(self) -> tuple[str, ...]:
        """Units the statement asserts that do not occur in its evidence."""
        return tuple(
            unit for unit in self.claim.units if unit not in self.evidence_vocabulary
        )


def _copied(judged: Judged) -> Failure | None:
    """Refuses a statement that is its evidence copied.

    Evidence carrying exactly one claim is exempt: it already is the fact.
    """
    if normalised(judged.statement) != normalised(judged.evidence):
        return None
    if judged.evidence_predicates == 1:
        return None
    return (
        Rejection.COPIED,
        (
            f"the statement is its evidence copied rather than written, and "
            f"that evidence carries {judged.evidence_predicates} claims, so it "
            "restates the passage instead of drawing one claim out of it"
        ),
    )


def _asserts(judged: Judged) -> Failure | None:
    """Refuses a statement carrying no finite verb."""
    if judged.claim.predicates:
        return None
    return (
        Rejection.ASSERTS_NOTHING,
        (
            "the statement has no finite verb, so it names something rather "
            "than asserting anything about it"
        ),
    )


def _atomic(judged: Judged) -> Failure | None:
    """Refuses a statement carrying more than one claim."""
    found = judged.claim.predicates
    if found <= 1:
        return None
    return (
        Rejection.NOT_ATOMIC,
        (
            f"the statement carries {found} claims; a fact carries one, and "
            "the rest belong to facts of their own"
        ),
    )


def _supported(judged: Judged) -> Failure | None:
    """Refuses a statement asserting what the cited text does not say."""
    added = judged.added
    if not added:
        return None
    return (
        Rejection.UNSUPPORTED_ADDITION,
        f"the statement asserts {', '.join(added)}, which the cited text does not say",
    )


def _self_contained(judged: Judged) -> Failure | None:
    """Refuses a statement a reader cannot understand without the passage."""
    references = judged.claim.references
    if not references:
        return None
    return (
        Rejection.UNRESOLVED_REFERENCE,
        (
            f"the statement leaves {', '.join(references)} unresolved, so a "
            "question drawn from it cannot be answered on its own"
        ),
    )


def _condensed(judged: Judged, share: float) -> Failure | None:
    """Refuses a digest no shorter than the passage it stands in for.

    Args:
        judged: The candidate under check.
        share: The longest a digest may be, as a share of its evidence.

    Returns:
        The refusal and its measurement, or None.
    """
    written, source = (
        len(normalised(judged.statement)),
        len(normalised(judged.evidence)),
    )
    if not source or written <= share * source:
        return None
    return (
        Rejection.NOT_CONDENSED,
        (
            f"the digest is {written} characters against {source} of passage, "
            f"which is {written / source:.0%} of it and above the {share:.0%} a "
            "reader gains anything by reading instead"
        ),
    )


def _listed(judged: Judged) -> Failure | None:
    """Refuses an outline holding fewer than two points.

    An outline is checked on its shape and not on its grammar: a point is
    written as a fragment, and a parser reads a fragment as having no verb.
    """
    points = sum(1 for line in judged.statement.splitlines() if line.startswith(BULLET))
    if points >= MIN_POINTS:
        return None
    return (
        Rejection.NOT_LISTED,
        (
            f"the outline holds {points} point(s); a list of fewer than "
            f"{MIN_POINTS} is a label"
        ),
    )


def _bridging(judged: Judged) -> Failure | None:
    """Refuses a bridge resting on fewer than two passages."""
    if len(judged.passage_ids) >= 2:
        return None
    return (
        Rejection.NOT_BRIDGING,
        (
            f"the statement rests on {len(judged.passage_ids)} passage(s); a "
            "bridge carries a claim no single passage states"
        ),
    )


#: Applied to a statement a deterministic reader composed, whatever its kind.
#: Nothing else is asked of one: it is neither written nor a sentence.
_COMPOSED: tuple[Check, ...] = (_copied,)


class FactChecker:
    """Judges a candidate against the passages it was drawn from.

    A candidate that passes carries the span its citation covers. One that
    fails comes back marked invalid, with the code that refused it and
    everything the checks read.
    """

    def __init__(self, digest_share: float | None = None) -> None:
        """Initialises the checker.

        Args:
            digest_share: The longest a summary or an outline may be, as a
                share of its passage. Read from EXTRACTION_DIGEST_MAX_SHARE
                when not given.
        """
        share = (
            digest_share
            if digest_share is not None
            else decimal("EXTRACTION_DIGEST_MAX_SHARE")
        )
        condensed = partial(_condensed, share=share)
        self._by_kind: dict[str, tuple[Check, ...]] = {
            FactKind.ATOMIC: (_copied, _asserts, _atomic, _supported, _self_contained),
            FactKind.SUMMARY: (_asserts, _supported, condensed),
            FactKind.OUTLINE: (_listed, _supported, condensed),
            FactKind.BRIDGE: (
                _asserts,
                _atomic,
                _supported,
                _self_contained,
                _bridging,
            ),
        }

    def check(
        self,
        passage: PassageToExtract,
        candidate: CandidateFact,
        method: str,
        provenance: Provenance | None = None,
    ) -> CheckedFact:
        """Checks one candidate drawn from one passage.

        Args:
            passage: The passage it was drawn from.
            candidate: What the extractor proposed.
            method: llm or deterministic.
            provenance: What produced it, recorded on the fact.

        Returns:
            The candidate as a storable fact, validated or refused.
        """
        cited = (
            passage.sentences
            if candidate.kind in (FactKind.SUMMARY, FactKind.OUTLINE)
            else self._cited(passage, candidate)
        )
        if not cited:
            return _absent(passage, candidate, method, provenance)

        return self._verdict(
            passage,
            candidate,
            self._judge(passage, candidate, list(cited)),
            method,
            provenance,
        )

    def check_bridge(
        self,
        offered: Sequence[PassageToExtract],
        candidate: CandidateFact,
        provenance: Provenance | None = None,
    ) -> CheckedFact:
        """Checks one candidate drawn from a group of passages.

        Args:
            offered: The passages the model was shown, in the order it saw
                them. A candidate names them by position.
            candidate: What the extractor proposed.
            provenance: What produced it, recorded on the fact.

        Returns:
            The candidate as a storable fact, anchored to the first passage
            it rests on.
        """
        rested = [
            offered[position]
            for position in sorted(set(candidate.passages))
            if 0 <= position < len(offered)
        ]
        if not rested:
            return _absent(
                offered[0] if offered else _NOWHERE, candidate, _WRITTEN, provenance
            )

        anchor = rested[0]
        judged = Judged(
            statement=candidate.statement.strip(),
            evidence=anchor.text,
            sentence_ids=[sentence.index for sentence in anchor.sentences],
            start=0,
            end=len(anchor.text),
            evidence_predicates=sum(
                sentence.predicates for one in rested for sentence in one.sentences
            ),
            claim=claim(candidate.statement.strip(), anchor.language),
            evidence_vocabulary=frozenset().union(
                *(vocabulary(one.text, one.language) for one in rested)
            ),
            passage_ids=tuple(one.id for one in rested),
        )
        return self._verdict(anchor, candidate, judged, _WRITTEN, provenance)

    def _verdict(
        self,
        passage: PassageToExtract,
        candidate: CandidateFact,
        judged: Judged,
        method: str,
        provenance: Provenance | None,
    ) -> CheckedFact:
        """Runs the checks this kind and method call for, first failure wins."""
        checks = self._by_kind[candidate.kind] if method == _WRITTEN else _COMPOSED
        for check in checks:
            failed = check(judged)
            if failed:
                return _fact(passage, judged, candidate, method, provenance, failed)
        return _fact(passage, judged, candidate, method, provenance, None)

    @staticmethod
    def _cited(passage: PassageToExtract, candidate: CandidateFact) -> list | None:
        """Resolves the cited sentence indices, or None if any is not one."""
        ids = sorted(set(candidate.sentences))
        if not ids or any(i < 0 or i >= len(passage.sentences) for i in ids):
            return None
        return [passage.sentences[i] for i in ids]

    @staticmethod
    def _judge(
        passage: PassageToExtract, candidate: CandidateFact, cited: list
    ) -> Judged:
        """Reads what the statement and its cited sentences each assert."""
        start, end = cited[0].start, cited[-1].end
        evidence = passage.text[start:end]
        return Judged(
            statement=candidate.statement.strip(),
            evidence=evidence,
            sentence_ids=[sentence.index for sentence in cited],
            start=start,
            end=end,
            evidence_predicates=sum(sentence.predicates for sentence in cited),
            claim=claim(candidate.statement.strip(), passage.language),
            evidence_vocabulary=vocabulary(evidence, passage.language),
            passage_ids=(passage.id,),
        )


#: Stands in for a passage when a bridge candidate names none at all.
_NOWHERE = PassageToExtract(
    id=0, text="", section_path=None, block_type=None, language=None
)


def _fact(
    passage: PassageToExtract,
    judged: Judged,
    candidate: CandidateFact,
    method: str,
    provenance: Provenance | None,
    failed: Failure | None,
) -> CheckedFact:
    """Assembles one checked fact."""
    code, detail = failed if failed else (None, None)
    return CheckedFact(
        passage_id=passage.id,
        statement=judged.statement,
        evidence_text=judged.evidence,
        evidence_sentence_ids=judged.sentence_ids,
        evidence_start=judged.start,
        evidence_end=judged.end,
        extraction_method=method,
        validated=code is None,
        rejection_code=code,
        validation_error=detail,
        kind=candidate.kind,
        passage_ids=list(judged.passage_ids)
        if candidate.kind == FactKind.BRIDGE
        else [],
        statement_predicates=judged.claim.predicates,
        evidence_predicates=judged.evidence_predicates,
        units_statement=list(judged.claim.units),
        units_added=list(judged.added),
        unresolved_references=list(judged.claim.references),
        extraction_model=provenance.model if provenance else None,
        prompt_version=provenance.prompt_version if provenance else None,
        extraction_temperature=provenance.temperature if provenance else None,
        spacy_model=pipeline_name(passage.language),
        spacy_version=VERSION,
    )


def _absent(
    passage: PassageToExtract,
    candidate: CandidateFact,
    method: str,
    provenance: Provenance | None,
) -> CheckedFact:
    """Builds a fact whose citation names nothing that was offered."""
    if candidate.kind == FactKind.BRIDGE:
        named, detail = candidate.passages, "no passage of the group it was shown"
    else:
        named = candidate.sentences
        detail = f"the passage has {len(passage.sentences)}"
    return CheckedFact(
        passage_id=passage.id,
        statement=candidate.statement.strip(),
        evidence_text="",
        evidence_sentence_ids=list(candidate.sentences),
        evidence_start=0,
        evidence_end=0,
        extraction_method=method,
        validated=False,
        rejection_code=Rejection.EVIDENCE_ABSENT,
        validation_error=(
            f"the citation names {', '.join(str(i) for i in named) or '(none)'}, "
            f"and {detail}"
        ),
        kind=candidate.kind,
        extraction_model=provenance.model if provenance else None,
        prompt_version=provenance.prompt_version if provenance else None,
        extraction_temperature=provenance.temperature if provenance else None,
        spacy_model=pipeline_name(passage.language),
        spacy_version=VERSION,
    )
