"""The things extraction passes around."""

from __future__ import annotations

from dataclasses import dataclass, field

from database.qa_generator import FactKind
from nlp.models import Sentence

#: The method whose statements a model writes, as `extraction_method`
#: records it. A deterministic reader composes its statement from the grid,
#: so it is neither a sentence nor expected to read like one.
#:
#: Here rather than beside the checks that read it: the repository asks the
#: same question in SQL, and the api holds a repository. A catalogue must
#: not import the module that loads a pipeline.
WRITTEN = "llm"

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
        embedding: Its vector, or None where nothing has embedded it.
    """

    id: int
    text: str
    section_path: str | None
    block_type: str | None
    language: str | None
    sentences: list[Sentence] = field(default_factory=list)
    table_cells: list[dict] = field(default_factory=list)
    doc_sha256: str | None = None
    #: Its vector, where the corpus has been embedded. Read by the bridge
    #: pass to pair a passage with the one it most nearly meets, and None
    #: for a corpus extracted before the column existed.
    embedding: tuple[float, ...] | None = None

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
        sentence_ids: Which of its sentences, in order, or None when the
            claim named no sentence this passage has. The record of what the
            claim rests on; the offsets below are the range covering them.
        start: Offset of the span in that passage's text, or None with the
            sentences. The range from the first cited sentence to the last,
            which is what a reader sees highlighted - so it covers a
            sentence between two cited ones that was not itself cited. The
            checks read `sentence_ids`, never this.
        end: Offset one past its last character, or None with the sentences.
    """

    passage_id: int
    sentence_ids: list[int] | None = None
    start: int | None = None
    end: int | None = None


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
        statement: What the extractor wrote.
        evidence_text: Every cited span, joined in the order below.
        extraction_method: llm or deterministic.
        validated: Whether every check its kind faces passed.
        rejection_code: Which one did not, or None.
        validation_error: That failure in words, or None.
        kind: atomic, summary, outline or bridge.
        citations: Every passage it rests on and where in each, in the order
            the model was shown them. One entry for an atomic fact, a summary
            and an outline, two or more for a bridge. A refused fact carries
            the passages it was read from with no span.
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
        embedding: The statement's vector, or None when this deployment does
            not embed. Written whatever the verdict: a rejected fact carries
            the vector the gate rejected it on.
    """

    statement: str
    evidence_text: str
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
    embedding: list[float] | None = None


@dataclass(frozen=True)
class Twin:
    """The stored fact a candidate is closest to.

    Attributes:
        fact_id: Which fact it is, so a rejection can name it.
        statement: What that fact says.
        similarity: Cosine likeness in [-1, 1]. 1 is the same vector.
    """

    fact_id: int
    statement: str
    similarity: float


@dataclass(frozen=True)
class FactSource:
    """One passage a stored fact rests on, and where that passage sits.

    Attributes:
        passage_id: The passage.
        doc_sha256: Its document.
        ordinal: Its position in that document, in reading order.
        page_from: The page it starts on, or None.
        position: Where it sat in what the model was shown, from 0.
    """

    passage_id: int
    doc_sha256: str
    ordinal: int
    page_from: int | None
    position: int


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
        passages: Every passage it rests on, in position order.
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
    passages: list[FactSource] = field(default_factory=list)


@dataclass(frozen=True)
class FactQuality:
    """How well extraction did, over whatever the caller filtered to.

    The counts are over everything the filter selects and the means over
    the accepted facts alone: the first is the rate at which the model
    fails, and the second is what it does when it does not.

    Attributes:
        total: Facts matching the filter, refused ones included.
        validated: How many of them passed.
        mean_statement_chars: Mean length of an accepted statement.
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
