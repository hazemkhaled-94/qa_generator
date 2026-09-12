"""Checking a proposed fact against the passage it came from.

Every check is structural rather than lexical: the citation is an index, so
it cannot be half-right, and what a statement asserts is read off its parse
rather than guessed from how many words it shares with its source.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from database.qa_generator import Rejection
from extraction.models import (
    CandidateFact,
    CheckedFact,
    PassageToExtract,
    Provenance,
)
from nlp.analysis import VERSION, claim, vocabulary
from nlp.models import Claim
from nlp.pipelines import name as pipeline_name

_WHITESPACE = re.compile(r"\s+")

#: The method whose statements a model writes. A deterministic reader
#: composes its statement from the grid, so it is neither a sentence nor
#: expected to read like one.
_WRITTEN = "llm"


def normalised(text: str) -> str:
    """Collapses spacing and case, for comparing two strings as words."""
    return _WHITESPACE.sub(" ", text).strip().casefold()


@dataclass(frozen=True)
class Judged:
    """One candidate resolved against its passage and parsed."""

    statement: str
    evidence: str
    sentence_ids: list[int]
    start: int
    end: int
    evidence_predicates: int
    claim: Claim
    evidence_vocabulary: frozenset[str]

    @property
    def added(self) -> tuple[str, ...]:
        """Units the statement asserts that do not occur in its evidence."""
        return tuple(
            unit for unit in self.claim.units if unit not in self.evidence_vocabulary
        )


def _copied(judged: Judged) -> tuple[str, str] | None:
    """The model filled both fields with one string."""
    if normalised(judged.statement) != normalised(judged.evidence):
        return None
    return (
        Rejection.COPIED,
        (
            "the statement is its evidence copied rather than written, so it "
            "restates the passage instead of drawing a claim out of it"
        ),
    )


def _atomic(judged: Judged) -> tuple[str, str] | None:
    """A fact carries exactly one claim."""
    found = judged.claim.predicates
    if found == 1:
        return None
    if found == 0:
        return (
            Rejection.NOT_ATOMIC,
            (
                "the statement has no finite verb, so it names something "
                "rather than asserting anything about it"
            ),
        )
    return (
        Rejection.NOT_ATOMIC,
        (
            f"the statement carries {found} claims; a fact carries one, and "
            "the rest belong to facts of their own"
        ),
    )


def _supported(judged: Judged) -> tuple[str, str] | None:
    """Nothing is asserted that the cited sentences do not say."""
    added = judged.added
    if not added:
        return None
    return (
        Rejection.UNSUPPORTED_ADDITION,
        f"the statement asserts {', '.join(added)}, which the cited text does not say",
    )


def _self_contained(judged: Judged) -> tuple[str, str] | None:
    """A reader who cannot see the passage understands the statement."""
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


#: Applied to every candidate, in order.
_ALWAYS: tuple[Callable[[Judged], tuple[str, str] | None], ...] = (_copied,)

#: Applied only to a statement a model wrote.
_WRITTEN_ONLY: tuple[Callable[[Judged], tuple[str, str] | None], ...] = (
    _atomic,
    _supported,
    _self_contained,
)


class FactChecker:
    """Judges a candidate against the passage it was drawn from.

    A candidate that passes carries the span its cited sentences cover. One
    that fails is not discarded: it comes back marked invalid, with the code
    that rejected it and everything the checks read.
    """

    def check(
        self,
        passage: PassageToExtract,
        candidate: CandidateFact,
        method: str,
        provenance: Provenance | None = None,
    ) -> CheckedFact:
        """Checks one candidate and turns it into a storable fact."""
        cited = self._cited(passage, candidate)
        if cited is None:
            return _absent(passage, candidate, method, provenance)

        judged = self._judge(passage, candidate, cited)
        for check in (*_ALWAYS, *(_WRITTEN_ONLY if method == _WRITTEN else ())):
            failed = check(judged)
            if failed:
                return _fact(passage, judged, method, provenance, failed)
        return _fact(passage, judged, method, provenance, None)

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
        )


def _fact(
    passage: PassageToExtract,
    judged: Judged,
    method: str,
    provenance: Provenance | None,
    failed: tuple[str, str] | None,
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
    """Builds a fact whose citation names no sentence of the passage."""
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
            f"the citation names sentence(s) "
            f"{', '.join(str(i) for i in candidate.sentences) or '(none)'}, "
            f"and the passage has {len(passage.sentences)}"
        ),
        extraction_model=provenance.model if provenance else None,
        prompt_version=provenance.prompt_version if provenance else None,
        extraction_temperature=provenance.temperature if provenance else None,
        spacy_model=pipeline_name(passage.language),
        spacy_version=VERSION,
    )
