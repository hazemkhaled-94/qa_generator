"""The things question generation passes around."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from database.qa_generator import (
    Difficulty,
    DocumentScope,
    FactKind,
    PassageScope,
    TopicScope,
)
from question_generation.types import TypeSpec, spec

#: A long answer, in characters. Above it a question counts as harder: a
#: chatbot has to produce more of the right thing, and a grader has more to
#: disagree about. QUESTIONS_LONG_ANSWER_CHARS sets it; this is the fallback
#: for a re-check reading a row whose settings are not to hand.
LONG_ANSWER_CHARS = 60


def band(score: int) -> str:
    """The difficulty band a point total falls in.

    Written once, because the service bands a question it is about to store
    and a re-check bands the same question read back; two spellings of this
    would be two difficulties for one row.
    """
    if score >= 3:
        return Difficulty.HARD
    if score >= 2:
        return Difficulty.MEDIUM
    return Difficulty.EASY


@dataclass(frozen=True)
class Criteria:
    """What kind of question this is, read off the facts it cites.

    Three scopes and a length, none of them judged. Each says something
    different about what a chatbot has to do: reach more than one passage,
    reach more than one document, bridge two subjects, or produce more than
    a value. `difficulty` is their total, banded.
    """

    passage_scope: str
    document_scope: str
    topic_scope: str
    answer_chars: int | None
    difficulty: str
    follows: bool = False
    #: The boundary the answer length was read against, kept so `score` can
    #: reproduce the band from the object rather than from a setting that may
    #: since have changed.
    long_answer: int = LONG_ANSWER_CHARS

    @property
    def score(self) -> int:
        """How many of the things that make a question harder are true.

        Every input, `follows` included, so the band can be recomputed from
        this object alone and a row that disagrees with its own score is
        something a test can catch.
        """
        return sum(
            (
                self.passage_scope == PassageScope.MULTI,
                self.document_scope == DocumentScope.CROSS,
                self.topic_scope == TopicScope.MULTI,
                (self.answer_chars or 0) >= self.long_answer,
                self.follows,
            )
        )


def criteria_of(
    *,
    passages: int,
    documents: int,
    topics: int,
    answer_chars: int | None,
    follows: bool = False,
    long_answer: int = LONG_ANSWER_CHARS,
) -> Criteria:
    """Reads the three scopes and the difficulty band off one question.

    One function, because three callers read it off different things: the
    service off the facts it is about to write from, the writer off what it
    reported using, and a re-check off the facts a stored question still
    has. A question whose evidence moved under it is one whose criteria
    stopped being true, and the readings have to be the same reading for
    anything to notice.

    Each of the four is worth a point, and so is following another question:
    a chatbot answering a follow-up has to carry the thread. Nothing here is
    weighted, because a weighting is an opinion and the point of deriving
    this rather than judging it is that nobody has to hold one.

    Three points is hard, not four. `cross_document` implies
    `multi_passage` - two documents are two passages - so the most the three
    scopes can total is three, and a threshold of four would have made the
    hardest thing a retriever faces, a question spanning two documents and
    two subjects, only medium. Hard has to be reachable by evidence spread
    alone rather than only by a long answer or a follow-up.
    """
    passage_scope = PassageScope.MULTI if passages > 1 else PassageScope.SINGLE
    document_scope = DocumentScope.CROSS if documents > 1 else DocumentScope.SINGLE
    topic_scope = TopicScope.MULTI if topics > 1 else TopicScope.SINGLE

    read = Criteria(
        passage_scope=passage_scope,
        document_scope=document_scope,
        topic_scope=topic_scope,
        answer_chars=answer_chars,
        difficulty=Difficulty.EASY,
        follows=follows,
        long_answer=long_answer,
    )
    return replace(read, difficulty=band(read.score))


@dataclass(frozen=True)
class TopicToCover:
    """One topic claimed off the queue, as the subject of some questions."""

    id: int
    language: str | None
    label: str | None
    include_in_coverage: bool


@dataclass(frozen=True)
class SourcePassage:
    """One passage a fact rests on, as question generation reads it.

    Carries the text rather than only an id: the verifier is shown the
    passage, because what is being tested is whether the answer is in the
    corpus and not whether it is in a sentence somebody rewrote.
    """

    id: int
    text: str
    doc_sha256: str
    #: Not optional: the repository selects only passages whose language
    #: chunking could read, because a question has to be written in one and
    #: the column holding it is NOT NULL.
    language: str
    #: The heading trail it sits under, which is most of what says whose rule
    #: or which year a question is about. NULL where the parser found none.
    section_path: str | None = None
    #: The topic it counts towards - its strongest, not all of them. Two
    #: passages with different ones make a multi-topic question.
    topic_id: int | None = None
    #: The title of its document. Carried so a gate can refuse a question
    #: that quotes it: naming where the answer lives is the one thing a
    #: question must not do.
    document_title: str | None = None
    #: Where it sits in its document. Passages are numbered in reading
    #: order, so two close ordinals are two parts of one section.
    ordinal: int = 0
    #: Its content lemmas, which is what says whether two passages are about
    #: related things. The same vocabulary the topics were fitted over, so
    #: nothing here re-reads any text.
    lemmas: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceFact:
    """One validated fact a question may be written from.

    A bridge rests on several passages and every other kind on one, so
    `passages` holds them all, anchor first. The anchor is what the fact is
    listed, filtered and ordered by; the whole tuple is what it is answered
    from.
    """

    id: int
    statement: str
    passages: tuple[SourcePassage, ...]
    #: Which reading produced it. A bridge is asked about alone, because it
    #: already spans the passages a wide question needs.
    kind: str = FactKind.ATOMIC
    #: The numbers, dates and amounts this fact asserts, as extraction read
    #: them. A fact carrying one is what a checkable question is written from.
    units: tuple[str, ...] = ()

    @property
    def anchor(self) -> SourcePassage:
        """The passage this fact is anchored to."""
        return self.passages[0]

    @property
    def passage_id(self) -> int:
        """The anchor's id."""
        return self.anchor.id

    @property
    def passage_text(self) -> str:
        """The anchor's text."""
        return self.anchor.text

    @property
    def doc_sha256(self) -> str:
        """The anchor's document."""
        return self.anchor.doc_sha256

    @property
    def language(self) -> str:
        """The anchor's language."""
        return self.anchor.language

    @property
    def section_path(self) -> str | None:
        """The anchor's heading trail."""
        return self.anchor.section_path

    @property
    def topic_id(self) -> int | None:
        """The topic the anchor counts towards."""
        return self.anchor.topic_id

    @property
    def document_title(self) -> str | None:
        """The anchor's document title."""
        return self.anchor.document_title

    @property
    def ordinal(self) -> int:
        """Where the anchor sits in its document."""
        return self.anchor.ordinal

    @property
    def lemmas(self) -> tuple[str, ...]:
        """The anchor's content lemmas."""
        return self.anchor.lemmas

    @property
    def spans(self) -> bool:
        """Whether this fact rests on more than one passage."""
        return len(self.passages) > 1


@dataclass(frozen=True)
class FactGroup:
    """The facts one question is written from.

    One fact is an ordinary question; two or more spanning two documents is a
    cross-document one. Nothing here decides which - the spread of the facts
    does, which is what makes `difficulty` measurable.
    """

    facts: tuple[SourceFact, ...]

    @property
    def resting(self) -> tuple[SourcePassage, ...]:
        """Every distinct passage this group's facts rest on, anchors first.

        Over every passage of every fact, not one per fact: a bridge rests on
        two, and a question written from it needs both.
        """
        seen: dict[int, SourcePassage] = {}
        for fact in self.facts:
            for passage in fact.passages:
                seen.setdefault(passage.id, passage)
        return tuple(seen.values())

    def criteria(
        self,
        answer_chars: int | None = None,
        *,
        follows: bool = False,
        long_answer: int = LONG_ANSWER_CHARS,
    ) -> Criteria:
        """What kind of question this group's facts make."""
        resting = self.resting
        return criteria_of(
            passages=len(resting),
            documents=len({passage.doc_sha256 for passage in resting}),
            topics=len({one.topic_id for one in resting if one.topic_id}),
            answer_chars=answer_chars,
            follows=follows,
            long_answer=long_answer,
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
        return tuple(passage.text for passage in self.resting)

    @property
    def context(self) -> tuple[tuple[str, str], ...]:
        """Each distinct passage with its heading trail, for the writer.

        The heading is carried separately and labelled, rather than run
        together with the text, for the reason extraction fences one: a model
        shown the two as one block asks about the heading.
        """
        return tuple(
            (
                f"Under: {passage.section_path}\n" if passage.section_path else "",
                passage.text,
            )
            for passage in self.resting
        )

    @property
    def statements(self) -> tuple[str, ...]:
        """What each fact of this group asserts, for the free gates."""
        return tuple(fact.statement for fact in self.facts)

    @property
    def titles(self) -> tuple[str, ...]:
        """The distinct document titles behind this group.

        What a question must not quote. A question naming the file its answer
        is in has already done the retrieving it was written to measure.

        Over every passage, so a bridge's second document is covered too.
        """
        return tuple(
            dict.fromkeys(
                passage.document_title
                for passage in self.resting
                if passage.document_title
            )
        )


@dataclass(frozen=True)
class Candidate:
    """A question as the model wrote it, before any gate has read it."""

    question_text: str
    target_answer: str | None
    answerable: bool
    group: FactGroup
    #: The question and answer of everything earlier in this thread, oldest
    #: first. Empty on a root question. A follow-up is written with these in
    #: front of the model and judged with them in front of the verifier,
    #: because relying on them is what makes it a follow-up.
    thread: tuple[tuple[str, str | None], ...] = ()
    #: What this was asked to be. The gates read the form off it, and the row
    #: records the type, so a set can be filtered to the reasons or the
    #: comparisons.
    spec: TypeSpec = field(default_factory=lambda: spec(None))
    #: The band the plan aimed for, stored beside the one the question turned
    #: out to be.
    planned_difficulty: str = Difficulty.EASY

    @property
    def follows(self) -> bool:
        """Whether this question is a follow-up to another."""
        return bool(self.thread)

    @property
    def thread_position(self) -> int:
        """Where this sits in its thread: 1 is a root."""
        return len(self.thread) + 1


@dataclass(frozen=True)
class CheckedQuestion:
    """A candidate that has been through the gates, ready for the table."""

    question_text: str
    target_answer: str | None
    answerable: bool
    criteria: Criteria
    language: str
    status: str
    rejected_reason: str | None
    fact_ids: tuple[int, ...]
    embedding: list[float] | None = None
    thread_position: int = 1
    question_type: str | None = None
    answer_form: str | None = None
    planned_difficulty: str | None = None

    @property
    def accepted(self) -> bool:
        """Whether every gate let this question through."""
        return self.rejected_reason is None

    @property
    def follows(self) -> bool:
        """Whether this question is a follow-up to another."""
        return self.thread_position > 1


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
    passage_scope: str | None
    document_scope: str | None
    topic_scope: str | None
    language: str
    status: str
    embedding: list[float] | None
    statements: tuple[str, ...]
    facts_validated: bool
    documents: int
    passages: int
    topics: int
    follows_id: int | None
    question_type: str | None = None
    answer_form: str | None = None


@dataclass(frozen=True)
class StoredQuestion:
    """One question as it is read back out, with what it rests on."""

    id: int
    question_text: str
    target_answer: str | None
    answerable: bool
    difficulty: str | None
    passage_scope: str | None
    document_scope: str | None
    topic_scope: str | None
    answer_chars: int | None
    language: str
    status: str
    rejected_reason: str | None
    created_at: str | None
    #: How many facts it cites, and the documents and topics they reach.
    facts: int
    documents: list[str]
    topics: list[str]
    #: Where it sits in its thread, and what it follows. 1 and None on a root.
    thread_position: int = 1
    follows_id: int | None = None
    #: What it was asked to be, and what shape of answer that wanted.
    question_type: str | None = None
    answer_form: str | None = None
    planned_difficulty: str | None = None


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
    #: The whole thread this question sits in, oldest first, the question
    #: itself included. One entry for a root nothing follows.
    thread: list[StoredQuestion] = field(default_factory=list)


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
    followups: int
    topics_covered: int
    topics_in_coverage: int
    mean_question_chars: float
    mean_answer_chars: float
    rejected: dict[str, int]
    difficulty: dict[str, int]
    #: One entry per value of each scope column, so the page can say how much
    #: of the set is cross-document, multi-topic and multi-passage without
    #: three more requests.
    passage_scope: dict[str, int]
    document_scope: dict[str, int]
    topic_scope: dict[str, int]
    #: What was asked for and what came out. `planned_difficulty` beside
    #: `difficulty` says how often a band the plan aimed for was reached.
    question_type: dict[str, int]
    answer_form: dict[str, int]
    planned_difficulty: dict[str, int]
    #: Questions whose band is the one the plan asked for.
    planned_met: int
