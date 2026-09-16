"""The things extraction passes around."""

from __future__ import annotations

from dataclasses import dataclass, field

from database.qa_generator import FactKind
from nlp.models import Sentence

#: What each line of a stored outline opens with.
BULLET = "- "

#: The fewest points an outline may hold.
MIN_POINTS = 2


@dataclass(frozen=True)
class Provenance:
    """What produced a fact.

    Attributes:
        model: The served model that wrote the statement, or None.
        prompt_version: Version of the prompt that asked for it, or None.
        temperature: Sampling temperature used, or None.
    """

    model: str | None = None
    prompt_version: str | None = None
    temperature: float | None = None


@dataclass(frozen=True)
class PassageToExtract:
    """One passage handed to an extractor.

    Attributes:
        id: Its primary key.
        text: What it says. Every evidence offset is into this string.
        section_path: Its heading trail, or None.
        block_type: The parser's label for it. Routes it to a reader.
        language: ISO 639-1, or None. Selects the spaCy pipeline.
        sentences: Its sentences, numbered. What a citation names.
        table_cells: The cell grids behind a table passage.
        doc_sha256: The document it was cut from, or None.
    """

    id: int
    text: str
    section_path: str | None
    block_type: str | None
    language: str | None
    sentences: list[Sentence] = field(default_factory=list)
    table_cells: list[dict] = field(default_factory=list)
    doc_sha256: str | None = None

    @property
    def claims(self) -> int:
        """How many of its sentences assert anything."""
        return sum(1 for sentence in self.sentences if sentence.predicates)


@dataclass(frozen=True)
class Cited:
    """Which sentences of one offered passage a bridge rests on.

    Attributes:
        position: That passage's position in the offered group, from 0.
        sentences: Indices of its sentences the claim rests on.
    """

    position: int
    sentences: tuple[int, ...]


@dataclass(frozen=True)
class Citation:
    """Which sentences of one passage a fact rests on, and the span they cover.

    Attributes:
        passage_id: The passage.
        sentence_ids: Which of its sentences, in order.
        start: Offset of the span in that passage's text.
        end: Offset one past its last character.
    """

    passage_id: int
    sentence_ids: list[int]
    start: int
    end: int


@dataclass(frozen=True)
class CandidateFact:
    """A statement an extractor proposes, before it has been checked.

    Attributes:
        statement: What the extractor wrote.
        sentences: Indices of the passage sentences it cites. Empty on a
            bridge, which cites each passage separately.
        kind: Which checks it is held to.
        passages: What a bridge rests on, one entry per passage it cites.
            Empty on every other kind.
    """

    statement: str
    sentences: tuple[int, ...]
    kind: str = FactKind.ATOMIC
    passages: tuple[Cited, ...] = ()


@dataclass(frozen=True)
class CheckedFact:
    """A checked fact, ready for the facts table.

    Attributes:
        passage_id: The passage it is anchored to.
        statement: What the extractor wrote.
        evidence_text: The cited passage text, resolved from the citation.
        evidence_sentence_ids: Which sentences of the anchor that was.
        evidence_start: Offset of the evidence in the anchor's text.
        evidence_end: Offset one past its last character.
        extraction_method: llm or deterministic.
        validated: Whether every check its kind faces passed.
        rejection_code: Which one did not, or None.
        validation_error: That failure in words, or None.
        kind: atomic, summary, outline or bridge.
        citations: What a bridge rests on, one entry per passage it cites,
            anchor first. Empty on every other kind, which rests on the
            anchor alone and carries its span in the columns above.
        statement_predicates: Finite verbs in the statement.
        evidence_predicates: Finite verbs across the cited sentences.
        units_statement: The numbers and proper nouns the statement asserts.
        units_added: Those of them the evidence does not carry.
        unresolved_references: Pronouns pointing outside the statement.
        extraction_model: The model that wrote it, or None.
        prompt_version: The prompt that asked for it, or None.
        extraction_temperature: Sampling temperature used, or None.
        spacy_model: The pipeline that judged it.
        spacy_version: That pipeline's version.
    """

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
    kind: str = FactKind.ATOMIC
    citations: list[Citation] = field(default_factory=list)
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
    """One fact as it is read back out, with where it came from.

    Attributes:
        id: Its primary key.
        statement: What the extractor wrote.
        evidence_text: The cited passage text.
        kind: atomic, summary, outline or bridge.
        extraction_method: llm or deterministic.
        validated: Whether every check its kind faces passed.
        rejection_code: Which one did not, or None.
        validation_error: That failure in words, or None.
        statement_predicates: Finite verbs in the statement.
        evidence_predicates: Finite verbs across the cited sentences.
        units_added: Units the evidence does not carry.
        unresolved_references: Pronouns pointing outside the statement.
        passage_id: The passage it is anchored to.
        doc_sha256: That passage's document.
        ordinal: That passage's position in it.
        page_from: The page it starts on, or None.
    """

    id: int
    statement: str
    evidence_text: str
    kind: str
    extraction_method: str
    validated: bool
    rejection_code: str | None
    validation_error: str | None
    statement_predicates: int
    evidence_predicates: int
    units_added: list[str]
    unresolved_references: list[str]
    passage_id: int
    doc_sha256: str
    ordinal: int
    page_from: int | None


@dataclass(frozen=True)
class FactQuality:
    """How well extraction did, over whatever the caller filtered to.

    Attributes:
        total: Facts matching the filter, refused ones included.
        validated: How many of them passed.
        mean_statement_chars: Mean length of a statement.
        mean_evidence_chars: Mean length of the text one cites.
        mean_statement_predicates: Mean finite verbs in a statement.
        mean_evidence_predicates: Mean finite verbs in the text one cites.
            Against the figure above, how far the passage was decomposed.
        facts_per_passage: Facts drawn from each passage that yielded any.
        rejected: How many failed, by the code that refused them.
        kinds: How many of each kind.
    """

    total: int
    validated: int
    mean_statement_chars: float
    mean_evidence_chars: float
    mean_statement_predicates: float
    mean_evidence_predicates: float
    facts_per_passage: float
    rejected: dict[str, int]
    kinds: dict[str, int] = field(default_factory=dict)
