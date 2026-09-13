"""The things extraction passes around."""

from __future__ import annotations

from dataclasses import dataclass, field

from nlp.models import Sentence


@dataclass(frozen=True)
class Provenance:
    """What produced a fact, carried onto the fact itself."""

    model: str | None = None
    prompt_version: str | None = None
    temperature: float | None = None


@dataclass(frozen=True)
class PassageToExtract:
    """One passage handed to an extractor.

    `sentences` are what a fact cites: chunking numbered them, so a citation
    is an index rather than a quote to search for.
    """

    id: int
    text: str
    section_path: str | None
    block_type: str | None
    language: str | None
    sentences: list[Sentence] = field(default_factory=list)
    table_cells: list[dict] = field(default_factory=list)

    @property
    def claims(self) -> int:
        """How many of its sentences assert anything."""
        return sum(1 for sentence in self.sentences if sentence.predicates)


@dataclass(frozen=True)
class CandidateFact:
    """A statement an extractor proposes, before it has been checked."""

    statement: str
    sentences: tuple[int, ...]


@dataclass(frozen=True)
class CheckedFact:
    """A checked fact, ready for the facts table."""

    passage_id: int
    statement: str
    evidence_text: str
    evidence_sentence_ids: list[int]
    evidence_start: int
    evidence_end: int
    extraction_method: str
    validated: bool
    rejection_code: str | None
    validation_error: str | None
    statement_predicates: int = 0
    evidence_predicates: int = 0
    units_statement: list[str] = field(default_factory=list)
    units_added: list[str] = field(default_factory=list)
    unresolved_references: list[str] = field(default_factory=list)
    extraction_model: str | None = None
    prompt_version: str | None = None
    extraction_temperature: float | None = None
    spacy_model: str | None = None
    spacy_version: str | None = None


@dataclass(frozen=True)
class StoredFact:
    """One fact as it is read back out, with where it came from."""

    id: int
    statement: str
    evidence_text: str
    extraction_method: str
    validated: bool
    rejection_code: str | None
    validation_error: str | None
    statement_predicates: int
    evidence_predicates: int
    units_added: list[str]
    unresolved_references: list[str]
    #: The passage this was drawn from, which is the unit extraction queues
    #: over: a page offering to read one fact's passage again needs its id.
    passage_id: int
    doc_sha256: str
    ordinal: int
    page_from: int | None


@dataclass(frozen=True)
class FactQuality:
    """How well extraction did, over whatever the caller filtered to.

    `mean_evidence_predicates` against `mean_statement_predicates` is how far
    the passage was decomposed: equal means the model restated a sentence
    rather than drawing one claim out of it.
    """

    total: int
    validated: int
    mean_statement_chars: float
    mean_evidence_chars: float
    mean_statement_predicates: float
    mean_evidence_predicates: float
    facts_per_passage: float
    rejected: dict[str, int]
