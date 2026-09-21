"""The queue this stage claims topics from, and what it writes back.

The worker's half of the database access. Reading the questions back is the
:mod:`question_generation.catalog`'s, which the API serves and which never
claims a row.
"""

from __future__ import annotations

from typing import Any, ClassVar

from sqlalchemy import false as sa_false
from sqlalchemy import func, insert, select
from sqlalchemy.orm import InstrumentedAttribute, aliased

from database.qa_generator import (
    Document,
    Fact,
    FactKind,
    FactPassage,
    Passage,
    PassageTopic,
    Question,
    QuestionFact,
    QuestionStatus,
    Status,
    Topic,
)
from database.qa_generator.passage_topics import DOMINANT
from question_generation.models import (
    CheckedQuestion,
    SourceFact,
    SourcePassage,
    TopicToCover,
)
from stages import Columns, RowQueue

#: A topic rather than a request to refit. `topics` carries both queues, and
#: this is what keeps this one off the fit's row: without it `start` would
#: queue the asking as though it were a subject, and the topic worker and
#: this one would fight over it.
#:
#: Public because the catalogue counts coverage over the same rows.
IS_TOPIC = Topic.topic_index.is_not(None)

#: The next topic to write questions for. This stage's own column and the
#: guard above. FOR UPDATE SKIP LOCKED lets a second worker take the next.
_NEXT_PENDING = (
    select(Topic.id)
    .where(IS_TOPIC, Topic.question_status == Status.PENDING)
    .order_by(Topic.id)
    .with_for_update(skip_locked=True)
    .limit(1)
    .scalar_subquery()
)

#: A fact some accepted question already rests on. Skipping these is what
#: makes a second run cheap: a refit returns every topic to `new`, and
#: without it the writer would be paid for once per duplicate before the
#: dedup gate could throw the duplicate away.
_ALREADY_ASKED = (
    select(QuestionFact.fact_id)
    .join(Question, Question.id == QuestionFact.question_id)
    .where(QuestionFact.fact_id == Fact.id, Question.status == QuestionStatus.ACCEPTED)
    .exists()
)


def repeated_across_documents(threshold: float):
    """Whether this passage recurs, nearly unchanged, in another document.

    The reading of "this is the document's furniture, not its subject". A
    corpus repeats its own boilerplate: a copyright notice, a table of
    contents, an accreditation clause, a revision table, a
    learning-objective preamble. Every one of those is real text, a fact
    extracts from it cleanly, and a question written about it passes every
    gate here - because nothing is wrong with it except that nobody wants
    to know. Measured over one corpus of eight documents, 25% of a balanced
    release rested on exactly this.

    What separates furniture from subject matter is not a word. A list of
    section names is a different list per corpus, and a list of words is
    worse: `Norm`, `Standard` and `Bericht` are the furniture of one corpus
    and the subject of another, and reading them as document words fired on
    58% of this one. What generalises is the repetition itself - boilerplate
    is what a publisher puts in every document, so it is what appears twice.

    Measured against the same corpus: of the 234 passages with a
    cross-document twin above 0.95, 71% sat in a document's first or last
    twelve pages, against 17% of everything else.

    Three ways this abstains, all of them correct rather than missing:
    a corpus of one document has no other document to repeat into, a
    passage with no embedding cannot be compared, and a threshold of 0
    turns the reading off.

    ponytail: one k-NN probe per candidate passage, which is what pgvector
    can do without a stored score. Fine at the 1,495 passages this was
    measured on and linear in the corpus; past roughly 100k passages this
    wants the nearest-foreign-neighbour distance materialised on `passages`
    at chunk time, and this predicate reading that column instead.
    """
    if threshold <= 0:
        return sa_false()
    twin = aliased(Passage)
    return (
        select(twin.id)
        .where(
            twin.doc_sha256 != Passage.doc_sha256,
            twin.embedding.is_not(None),
            Passage.embedding.is_not(None),
            (1 - twin.embedding.cosine_distance(Passage.embedding)) >= threshold,
        )
        .limit(1)
        .exists()
    )


#: The kinds of fact a question may be written from, by the name the setting
#: takes. All four, and what each is good for:
#:
#:   atomic   one claim in one sentence, which is the shape every prompt here
#:            assumes and the shape the citation gates read
#:   summary  a paragraph standing in for a whole passage. What a definition
#:            or a procedure is written from: those ask what something IS or
#:            HOW it is done, which is a reading of the passage rather than
#:            one sentence of it. Still a checked fact - the same
#:            unsupported_addition and unresolved_reference gates judge it -
#:            and the verifier is shown the passage it condenses, so a
#:            question resting on one is recoverable like any other
#:   outline  newline-separated `- ` bullets. An enumeration written from one
#:            has a real set behind it, which is what that type needs and
#:            what a single atomic claim cannot give it. The bullets are
#:            indented under their own number when the sample is rendered,
#:            so they no longer break the numbering the writer cites by
#:   bridge   one claim resting on the passages in fact_passages. Asked
#:            about alone: it already spans the passages a wide question
#:            needs, and the verifier is shown all of them. A bridge drawn
#:            before prompt version 2 recorded no citation and is skipped,
#:            because there is nothing to show the verifier
#:
#: QUESTIONS_FACT_KINDS is what decides; this is what it is checked against.
ASKABLE = (FactKind.ATOMIC, FactKind.SUMMARY, FactKind.OUTLINE, FactKind.BRIDGE)


#: One validated fact, selected through the passage it opens on. Written
#: once because two queries select it: a topic's own facts and the facts of
#: the passages that bridge it to another topic.
#:
#: The opening passage and no other, so a fact comes back once. It is what
#: the topic, the document and the language are asked of; the passages the
#: fact actually rests on are _RESTING below. A bridge is read from one
#: topic's group, so every passage of one carries the same dominant topic
#: and which of them is asked makes no difference to what is selected.
_SOURCE = (
    select(
        Fact.id,
        Fact.statement,
        Fact.kind,
        Fact.units_statement,
        Fact.embedding,
    )
    .select_from(Fact)
    .join(FactPassage, (FactPassage.fact_id == Fact.id) & (FactPassage.position == 0))
    .join(Passage, Passage.id == FactPassage.passage_id)
    .join(Document, Document.sha256 == Passage.doc_sha256)
    .join(DOMINANT, DOMINANT.c.passage_id == Passage.id)
)

#: The passages a fact rests on, with everything the writer and the gates
#: read off one. Ordered by position, so they come as the model was shown
#: them.
#:
#: A separate query rather than an aggregate on _SOURCE: two call sites share
#: that select, and joining the passages onto it would multiply every row and
#: change what its callers count.
_RESTING = (
    select(
        FactPassage.fact_id,
        FactPassage.position,
        FactPassage.sentence_ids,
        Passage.id.label("passage_id"),
        Passage.text,
        Passage.doc_sha256,
        Passage.language,
        Passage.section_path,
        Passage.ordinal,
        Passage.lemmas,
        Passage.embedding,
        DOMINANT.c.topic_id,
        Document.title,
    )
    .select_from(FactPassage)
    .join(Passage, Passage.id == FactPassage.passage_id)
    .join(Document, Document.sha256 == Passage.doc_sha256)
    .join(DOMINANT, DOMINANT.c.passage_id == Passage.id, isouter=True)
    .order_by(FactPassage.fact_id, FactPassage.position)
)


def _vector(stored: Any) -> tuple[float, ...] | None:
    """One embedding column as a tuple, or None where nothing wrote it.

    A tuple rather than the list pgvector hands back, because a SourceFact is
    frozen and hashed by the sampler.
    """
    return None if stored is None else tuple(float(one) for one in stored)


def _passage(row: Any) -> SourcePassage:
    """Reads one passage out of either query above."""
    return SourcePassage(
        id=row.passage_id,
        text=row.text,
        doc_sha256=row.doc_sha256,
        language=row.language,
        section_path=row.section_path,
        topic_id=row.topic_id,
        document_title=row.title,
        ordinal=row.ordinal,
        lemmas=tuple(row.lemmas or ()),
        embedding=_vector(row.embedding),
    )


def _fact(row: Any, resting: tuple[SourcePassage, ...]) -> SourceFact:
    """Reads one row of _SOURCE back as a fact to write from.

    Args:
        row: The selected row.
        resting: Every passage it rests on, in the order the model saw them.

    Returns:
        The fact.
    """
    return SourceFact(
        id=row.id,
        statement=row.statement,
        kind=row.kind,
        passages=resting,
        units=tuple(row.units_statement or ()),
        embedding=_vector(row.embedding),
    )


class QuestionQueue(RowQueue):
    """Reads the question queue and records what became of each topic."""

    columns = Columns(
        entity=Topic,
        key=Topic.id,
        status=Topic.question_status,
        error=Topic.question_error,
        claimed_at=Topic.question_claimed_at,
    )
    #: Only the topics. `topics` also holds the fit requests, which are not a
    #: unit of anything here; every inherited operation carries this.
    base: ClassVar[Any] = IS_TOPIC
    #: This stage queues over topics, and a topic is the only thing it can be
    #: narrowed to: a topic is already the smallest subject there is.
    scopes: ClassVar[dict[str, InstrumentedAttribute]] = {"topic": Topic.id}
    done = Status.GENERATED
    next_pending = _NEXT_PENDING

    def __init__(
        self,
        lease=None,
        kinds: tuple[str, ...] = ASKABLE,
        version: str | None = None,
        boilerplate_cosine: float = 0.0,
    ) -> None:
        """Binds to the session factory, with the kinds of fact it may offer.

        `kinds` is QUESTIONS_FACT_KINDS. `atomic` alone by default, which is
        the only shape every prompt and gate here was written for.

        `version` is the configuration every question it writes was written
        under.

        `boilerplate_cosine` is QUESTIONS_BOILERPLATE_COSINE. 0 by default,
        so a caller that says nothing gets every passage: this excludes
        material, and a default that quietly dropped some would be the
        wrong way round.
        """
        super().__init__(lease)
        self._kinds = kinds
        self._version = version
        self._boilerplate = boilerplate_cosine

    def claim(self) -> TopicToCover | None:
        """Takes the next topic off the queue."""
        claimed = self._claim(
            Topic.id, Topic.language, Topic.label, Topic.include_in_coverage
        )
        if claimed is None:
            return None
        return TopicToCover(
            id=claimed.id,
            language=claimed.language,
            label=claimed.label,
            include_in_coverage=claimed.include_in_coverage,
        )

    def facts(self, topic_id: int) -> list[SourceFact]:
        """The validated facts this topic is the subject of.

        A fact reaches its topic through its passage, and the passage's
        strongest topic is the one it counts towards: a passage belongs a
        little to many topics, and writing a question for every one of them
        would ask the same thing under a dozen subjects.

        Facts an accepted question already rests on are left out, so this is
        what is still missing rather than everything there ever was.
        """
        with self._session() as session:
            rows = session.execute(
                _SOURCE.where(
                    DOMINANT.c.topic_id == topic_id,
                    Fact.validated,
                    # A fact of a kind this stage can use. Without it, the
                    # first re-extraction that writes outlines hands the
                    # writer a bulleted blob as though it were one claim.
                    Fact.kind.in_(self._kinds),
                    # A question is written in a language, and the column
                    # holding it is NOT NULL: a passage too short for the
                    # detector has no language to write one in.
                    Passage.language.is_not(None),
                    ~_ALREADY_ASKED,
                    # The corpus's own furniture, which passes every gate
                    # and is worth nobody's time.
                    ~repeated_across_documents(self._boilerplate),
                ).order_by(Fact.id)
            ).all()
        return self._read(rows)

    def _read(self, rows: list[Any]) -> list[SourceFact]:
        """Turns selected rows into facts, each with the passages it rests on.

        A fact one of whose passages carries no citation is dropped: a bridge
        drawn before the prompt said where in each passage it rests, and a
        refused fact that resolved nowhere. Neither can be shown to the
        verifier as what a question was answered from.

        Args:
            rows: What one of the selects above returned.

        Returns:
            The facts, in the order the rows came.
        """
        resting = self._resting([row.id for row in rows])
        return [_fact(row, resting[row.id]) for row in rows if row.id in resting]

    def _resting(self, fact_ids: list[int]) -> dict[int, tuple[SourcePassage, ...]]:
        """Reads the passages those facts rest on, in the order shown.

        Args:
            fact_ids: The facts to read.

        Returns:
            Per fact, its passages. A fact any of whose passages carries no
            citation, or has no language, is absent: neither can be written
            from.
        """
        found: dict[int, list[SourcePassage]] = {}
        refused: set[int] = set()
        with self._session() as session:
            for row in session.execute(
                _RESTING.where(FactPassage.fact_id.in_(fact_ids))
            ):
                if not row.sentence_ids or row.language is None:
                    refused.add(row.fact_id)
                    continue
                found.setdefault(row.fact_id, []).append(_passage(row))
        return {
            fact_id: tuple(passages)
            for fact_id, passages in found.items()
            if fact_id not in refused
        }

    def bridging(self, topic_id: int) -> list[SourceFact]:
        """The facts of passages that bridge this topic to another.

        A bridge passage belongs to some other topic more strongly than to
        this one, but carries this one above the weight floor - so the corpus
        itself says the two subjects meet there. Pairing a bridge with an own
        passage is what makes a multi-topic question available, and it is a
        far better reason to put two passages together than that they came
        from different files.

        Facts an accepted question already rests on are skipped, as in
        `facts`: a bridge is still a fact to be asked about once.
        """
        with self._session() as session:
            rows = session.execute(
                _SOURCE.join(PassageTopic, PassageTopic.passage_id == Passage.id)
                .where(
                    # Carries this topic, but is not strongest in it.
                    PassageTopic.topic_id == topic_id,
                    DOMINANT.c.topic_id != topic_id,
                    Fact.validated,
                    Fact.kind.in_(self._kinds),
                    Passage.language.is_not(None),
                    ~_ALREADY_ASKED,
                )
                .order_by(Fact.id)
            ).all()
        return self._read(rows)

    def store(self, topic_id: int, threads: list[list[CheckedQuestion]]) -> int:
        """Writes one topic's threads and finishes it, in one transaction.

        A thread is a root question and its follow-ups, in the order they
        were asked. They are written in that order and each follow-up is
        linked to the row before it, which is why this takes threads rather
        than a flat list: a follow-up needs its parent's id, and the parent
        does not have one until it is inserted.

        Appends rather than replaces, which is what every other stage does:
        questions are append-only, a rejected one is the drop-rate evidence,
        and a second run over the same topic writes only what the facts it
        skipped did not already cover.
        """
        written = 0
        with self._session.begin() as session:
            for thread in threads:
                follows: int | None = None
                for position, question in enumerate(thread, 1):
                    question_id = session.scalar(
                        insert(Question)
                        .values(
                            question_text=question.question_text,
                            target_answer=question.target_answer,
                            answer_explanation=question.answer_explanation,
                            answerable=question.answerable,
                            difficulty=question.criteria.difficulty,
                            planned_difficulty=question.planned_difficulty,
                            question_type=question.question_type,
                            cognitive_level=question.cognitive_level,
                            answer_form=question.answer_form,
                            passage_scope=question.criteria.passage_scope,
                            document_scope=question.criteria.document_scope,
                            topic_scope=question.criteria.topic_scope,
                            answer_chars=question.criteria.answer_chars,
                            language=question.language,
                            embedding=question.embedding,
                            status=question.status,
                            rejected_reason=question.rejected_reason,
                            status_changed_at=func.now(),
                            settings_version=self._version,
                            follows_id=follows,
                            thread_position=position,
                        )
                        .returning(Question.id)
                    )
                    session.execute(
                        insert(QuestionFact),
                        [
                            {"question_id": question_id, "fact_id": fact_id}
                            for fact_id in question.fact_ids
                        ],
                    )
                    follows = question_id
                    written += 1
            self._finish(topic_id, session=session)
        return written

    def counts(self) -> dict[str, int]:
        """Reports what this service owns, for the status panel."""
        with self._session() as session:
            total = session.scalar(select(func.count()).select_from(Question)) or 0
            accepted = (
                session.scalar(
                    select(func.count())
                    .select_from(Question)
                    .where(Question.status == QuestionStatus.ACCEPTED)
                )
                or 0
            )
        return {**self.counts_by_status(), "questions": total, "accepted": accepted}
