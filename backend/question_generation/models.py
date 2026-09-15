"""The things question generation passes around."""

from __future__ import annotations

from dataclasses import dataclass, field

from database.qa_generator import Difficulty


def difficulty_of(*, documents: int, passages: int) -> str:
    """Names how far evidence spread over this many documents and passages is.

    One function, because two callers read it off different things: the
    service off the facts it is about to write a question from, and a
    re-check off the facts a stored question still has. A question whose
    evidence moved under it is one whose difficulty stopped being true, and
    the two readings have to be the same reading to notice.
    """
    if documents > 1:
        return Difficulty.CROSS_DOCUMENT
    if passages > 1:
        return Difficulty.CROSS_PASSAGE
    return Difficulty.SINGLE_PASSAGE


@dataclass(frozen=True)
class TopicToCover:
    """One topic claimed off the queue, as the subject of some questions."""

    id: int
    language: str | None
    label: str | None
    include_in_coverage: bool


@dataclass(frozen=True)
class SourceFact:
    """One validated fact a question may be written from.

    Carries the passage text rather than only the statement: the verifier is
    shown the passage, because what is being tested is whether the answer is
    in the corpus and not whether it is in a sentence somebody rewrote.
    """

    id: int
    statement: str
    passage_id: int
    passage_text: str
    doc_sha256: str
    #: Not optional: the repository selects only passages whose language
    #: chunking could read, because a question has to be written in one and
    #: the column holding it is NOT NULL.
    language: str
    #: The heading trail the passage sits under, which is most of what says
    #: whose rule or which year a question is about. NULL on a passage the
    #: parser found no heading above.
    section_path: str | None = None


@dataclass(frozen=True)
class FactGroup:
    """The facts one question is written from.

    One fact is an ordinary question; two or more spanning two documents is a
    cross-document one. Nothing here decides which - the spread of the facts
    does, which is what makes `difficulty` measurable.
    """

    facts: tuple[SourceFact, ...]

    @property
    def difficulty(self) -> str:
        """How far this group's evidence is spread."""
        return difficulty_of(
            documents=len({fact.doc_sha256 for fact in self.facts}),
            passages=len({fact.passage_id for fact in self.facts}),
        )

    @property
    def language(self) -> str:
        """The language the question is written in.

        The first fact's. A group is dealt from passages of one topic, and a
        topic is fitted over one language's vocabulary, so the facts of a
        group agree; the first is as good as a vote.
        """
        return self.facts[0].language

    @property
    def passages(self) -> tuple[str, ...]:
        """Each distinct passage behind this group, in citation order.

        What the verifier is shown, and all it is shown: the question is
        being asked of the corpus, not of a retriever.
        """
        seen: dict[int, str] = {}
        for fact in self.facts:
            seen.setdefault(fact.passage_id, fact.passage_text)
        return tuple(seen.values())

    @property
    def context(self) -> tuple[tuple[str, str], ...]:
        """Each distinct passage with its heading trail, for the writer.

        The heading is carried separately and labelled, rather than run
        together with the text, for the reason extraction fences one: a model
        shown the two as one block asks about the heading.
        """
        seen: dict[int, tuple[str, str]] = {}
        for fact in self.facts:
            heading = f"Under: {fact.section_path}\n" if fact.section_path else ""
            seen.setdefault(fact.passage_id, (heading, fact.passage_text))
        return tuple(seen.values())

    @property
    def statements(self) -> tuple[str, ...]:
        """What each fact of this group asserts, for the free gates."""
        return tuple(fact.statement for fact in self.facts)


@dataclass(frozen=True)
class Candidate:
    """A question as the model wrote it, before any gate has read it."""

    question_text: str
    target_answer: str | None
    answerable: bool
    group: FactGroup


@dataclass(frozen=True)
class CheckedQuestion:
    """A candidate that has been through the gates, ready for the table."""

    question_text: str
    target_answer: str | None
    answerable: bool
    difficulty: str
    language: str
    status: str
    rejected_reason: str | None
    fact_ids: tuple[int, ...]
    embedding: list[float] | None = None

    @property
    def accepted(self) -> bool:
        """Whether every gate let this question through."""
        return self.rejected_reason is None


@dataclass(frozen=True)
class Neighbour:
    """The nearest already-accepted question to the one being checked.

    Declared here rather than beside the gates that read it, because the
    catalogue is what produces one and the catalogue is what the API holds:
    a value object in the module that loads torch would drag torch into the
    process that serves JSON.
    """

    question_text: str
    answerable: bool
    similarity: float


@dataclass(frozen=True)
class JudgedQuestion:
    """A stored question and everything a re-check needs, read in one query.

    `facts_validated` and the two spreads are read now rather than when the
    question was written: re-judging a fact can reject one a question rests
    on, and re-extracting one document can take a cross-document question's
    second citation away without taking the question, which leaves a stored
    difficulty that is no longer true.
    """

    id: int
    question_text: str
    target_answer: str | None
    answerable: bool
    difficulty: str | None
    language: str
    status: str
    embedding: list[float] | None
    statements: tuple[str, ...]
    facts_validated: bool
    documents: int
    passages: int


@dataclass(frozen=True)
class StoredQuestion:
    """One question as it is read back out, with what it rests on."""

    id: int
    question_text: str
    target_answer: str | None
    answerable: bool
    difficulty: str | None
    language: str
    status: str
    rejected_reason: str | None
    created_at: str | None
    #: How many facts it cites, and the documents and topics they reach.
    facts: int
    documents: list[str]
    topics: list[str]


@dataclass(frozen=True)
class QuestionSource:
    """One fact a question cites, and where it came from.

    `validated` is read now rather than trusted: re-judging the facts can
    reject one a question was written from, and the page has to be able to
    say so.
    """

    fact_id: int
    statement: str
    evidence_text: str
    validated: bool
    passage_id: int
    doc_sha256: str
    ordinal: int


@dataclass(frozen=True)
class QuestionDetail:
    """One question in full, with the facts it was written from."""

    question: StoredQuestion
    sources: list[QuestionSource] = field(default_factory=list)


@dataclass(frozen=True)
class QuestionQuality:
    """How well generation did, over whatever the caller filtered to.

    `rejected` is keyed by gate, which is what says whether the model is
    writing questions the corpus answers or questions it merely looks like
    it should.
    """

    total: int
    accepted: int
    draft: int
    unanswerable: int
    topics_covered: int
    topics_in_coverage: int
    mean_question_chars: float
    rejected: dict[str, int]
    difficulty: dict[str, int]
