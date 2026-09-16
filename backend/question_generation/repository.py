"""Database access for the question generation service."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import datetime
from typing import Any, ClassVar

from sqlalchemy import Select, func, insert, select, update
from sqlalchemy.orm import InstrumentedAttribute

from database.qa_generator import (
    Document,
    Fact,
    FactKind,
    Passage,
    PassageTopic,
    Question,
    QuestionFact,
    QuestionStatus,
    Status,
    Topic,
)
from database.qa_generator.passage_topics import DOMINANT
from database.qa_generator.repository import Repository, matching
from question_generation.models import (
    CheckedQuestion,
    JudgedQuestion,
    Neighbour,
    QuestionDetail,
    QuestionQuality,
    QuestionSource,
    SourceFact,
    StoredQuestion,
    TopicToCover,
)
from stages import Columns, RowQueue

#: A topic rather than a request to refit. `topics` carries both queues, and
#: this is what keeps this one off the fit's row: without it `start` would
#: queue the asking as though it were a subject, and the topic worker and
#: this one would fight over it.
_IS_TOPIC = Topic.topic_index.is_not(None)

#: The next topic to write questions for. This stage's own column and the
#: guard above. FOR UPDATE SKIP LOCKED lets a second worker take the next.
_NEXT_PENDING = (
    select(Topic.id)
    .where(_IS_TOPIC, Topic.question_status == Status.PENDING)
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

#: Which columns the search box looks in, by the name the API accepts.
SEARCH_FIELDS = {
    "question": (Question.question_text,),
    "answer": (Question.target_answer,),
    "both": (Question.question_text, Question.target_answer),
}

DEFAULT_FIELD = "both"

#: How many questions to hold at once while re-checking them.
_BATCH = 500


def _joined(query: Select) -> Select:
    """Joins everything a question is filtered by, placed by and named by.

    Outer throughout. A question reaches its document and its topic through
    its facts, and a question whose last fact has just gone is one the
    trigger is about to delete: it should not disappear from a count before
    it disappears from the table.

    Every join here multiplies a question by its facts, so anything counted
    over this has to be counted distinctly - which is why the figures are
    measured over :func:`_ids` instead.
    """
    return (
        query.select_from(Question)
        .join(QuestionFact, QuestionFact.question_id == Question.id, isouter=True)
        .join(Fact, Fact.id == QuestionFact.fact_id, isouter=True)
        .join(Passage, Passage.id == Fact.passage_id, isouter=True)
        .join(DOMINANT, DOMINANT.c.passage_id == Passage.id, isouter=True)
        .join(Topic, Topic.id == DOMINANT.c.topic_id, isouter=True)
    )


#: The columns a caller may narrow to, by the name the API takes. Every one is
#: also a column of the quality report, so a reader can ask for the reasons,
#: the cross-document ones or the hard ones alone.
SCOPES = {
    "passage_scope": Question.passage_scope,
    "document_scope": Question.document_scope,
    "topic_scope": Question.topic_scope,
    "difficulty": Question.difficulty,
    "planned_difficulty": Question.planned_difficulty,
    "question_type": Question.question_type,
    "answer_form": Question.answer_form,
}


def _filtered(query, document, topic, search, status, answerable, field, **scopes):
    """Applies every filter a listing and its figures have to agree about.

    One place, so the page of rows and the quality panel above it cannot be
    measured over different sets. The query must already be joined.
    """
    if document:
        query = query.where(Passage.doc_sha256 == document)
    if topic:
        query = query.where(DOMINANT.c.topic_id == topic)
    for name, column in SCOPES.items():
        if scopes.get(name):
            query = query.where(column == scopes[name])
    if scopes.get("follows") is not None:
        query = query.where(
            Question.follows_id.is_not(None)
            if scopes["follows"]
            else Question.follows_id.is_(None)
        )
    if search:
        query = query.where(
            matching(
                search, *SEARCH_FIELDS.get(field or "", SEARCH_FIELDS[DEFAULT_FIELD])
            )
        )
    if status:
        query = query.where(Question.status == status)
    if answerable is not None:
        query = query.where(Question.answerable == answerable)
    return query


def _ids(document, topic, search, status, answerable, field, **scopes) -> Select:
    """The questions a filter selects, as one id apiece.

    Every figure is measured over this rather than over the joined query,
    because that one carries a question once per fact: counting distinctly
    would fix the counts and leave the mean length weighted by how many
    facts a question happens to cite.
    """
    return _filtered(
        _joined(select(Question.id)),
        document,
        topic,
        search,
        status,
        answerable,
        field,
        **scopes,
    ).distinct()


def _listing() -> Select:
    """One question per row, with what it cites and where that sits."""
    return (
        _joined(
            select(
                Question.id,
                Question.question_text,
                Question.target_answer,
                Question.answerable,
                Question.difficulty,
                Question.passage_scope,
                Question.document_scope,
                Question.topic_scope,
                Question.answer_chars,
                Question.language,
                Question.status,
                Question.rejected_reason,
                Question.created_at,
                Question.thread_position,
                Question.follows_id,
                Question.question_type,
                Question.answer_form,
                Question.planned_difficulty,
                func.count(func.distinct(QuestionFact.fact_id)).label("facts"),
                func.array_agg(func.distinct(Passage.doc_sha256)).label("documents"),
                func.array_agg(func.distinct(Topic.label)).label("topics"),
            )
        )
        .group_by(Question.id)
        .order_by(Question.id)
    )


def _when(moment: datetime | None) -> str | None:
    """Renders a timestamp for the wire."""
    return moment.isoformat() if moment else None


def _present(values: Sequence[Any] | None) -> list[str]:
    """Drops the NULLs an outer join left in an aggregate."""
    return sorted({str(value) for value in values or () if value is not None})


#: The kinds of fact a question may be written from, by the name the setting
#: takes. Only what this stage can honestly use:
#:
#:   atomic   one claim in one sentence, which is the shape every prompt here
#:            assumes and the shape the citation gates read
#:   summary  a paragraph standing in for a whole passage. A question "answered
#:            by" one rests on a paraphrase rather than on a checked claim
#:   outline  newline-separated `- ` bullets. Interpolated into a numbered
#:            prompt it breaks the numbering the writer cites facts by
#:   bridge   one claim resting on the passages in fact_passages. The right
#:            seed for a cross-passage question and the one that needs work:
#:            SourceFact carries a single passage, so the verifier would be
#:            shown the anchor alone and could never recover the answer
#:
#: QUESTIONS_FACT_KINDS is what decides; this is what it is checked against.
ASKABLE = (FactKind.ATOMIC,)


#: One validated fact with everything selection and the writer read off it.
#: Written once because two queries select it: a topic's own facts and the
#: facts of the passages that bridge it to another topic.
_SOURCE = (
    select(
        Fact.id,
        Fact.statement,
        Fact.units_statement,
        Passage.id.label("passage_id"),
        Passage.text,
        Passage.doc_sha256,
        Passage.language,
        Passage.section_path,
        Passage.ordinal,
        Passage.lemmas,
        DOMINANT.c.topic_id,
        Document.title,
    )
    .select_from(Fact)
    .join(Passage, Passage.id == Fact.passage_id)
    .join(Document, Document.sha256 == Passage.doc_sha256)
    .join(DOMINANT, DOMINANT.c.passage_id == Passage.id)
)


def _fact(row: Any) -> SourceFact:
    """Reads one row of the query above back as a fact to write from."""
    return SourceFact(
        id=row.id,
        statement=row.statement,
        passage_id=row.passage_id,
        passage_text=row.text,
        doc_sha256=row.doc_sha256,
        language=row.language,
        section_path=row.section_path,
        topic_id=row.topic_id,
        document_title=row.title,
        ordinal=row.ordinal,
        lemmas=tuple(row.lemmas or ()),
        units=tuple(row.units_statement or ()),
    )


def _stored(row: Any) -> StoredQuestion:
    """Reads one row of the listing back as a question."""
    return StoredQuestion(
        id=row.id,
        question_text=row.question_text,
        target_answer=row.target_answer,
        answerable=row.answerable,
        difficulty=row.difficulty,
        passage_scope=row.passage_scope,
        document_scope=row.document_scope,
        topic_scope=row.topic_scope,
        answer_chars=row.answer_chars,
        language=row.language,
        status=row.status,
        rejected_reason=row.rejected_reason,
        created_at=_when(row.created_at),
        facts=row.facts,
        documents=_present(row.documents),
        topics=_present(row.topics),
        thread_position=row.thread_position,
        follows_id=row.follows_id,
        question_type=row.question_type,
        answer_form=row.answer_form,
        planned_difficulty=row.planned_difficulty,
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
    base: ClassVar[Any] = _IS_TOPIC
    #: This stage queues over topics, and a topic is the only thing it can be
    #: narrowed to: a topic is already the smallest subject there is.
    scopes: ClassVar[dict[str, InstrumentedAttribute]] = {"topic": Topic.id}
    done = Status.GENERATED
    next_pending = _NEXT_PENDING

    def __init__(self, lease=None, kinds: tuple[str, ...] = ASKABLE) -> None:
        """Binds to the session factory, with the kinds of fact it may offer.

        `kinds` is QUESTIONS_FACT_KINDS. `atomic` alone by default, which is
        the only shape every prompt and gate here was written for.
        """
        super().__init__(lease)
        self._kinds = kinds

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
                ).order_by(Fact.id)
            ).all()
        return [_fact(row) for row in rows]

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
        return [_fact(row) for row in rows]

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
                            answerable=question.answerable,
                            difficulty=question.criteria.difficulty,
                            planned_difficulty=question.planned_difficulty,
                            question_type=question.question_type,
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


class QuestionCatalog(Repository):
    """Reads back the questions, and records what a person decided about one.

    Separate from the queue: the API serves these and never claims a row.
    """

    def nearest(
        self, embedding: list[float], before: int | None = None
    ) -> Neighbour | None:
        """The accepted question closest to this vector, or None if there is none.

        `before` looks only at questions written earlier. A re-check needs
        it: without it, two questions that are duplicates of each other each
        find the other and both are rejected, which leaves the benchmark
        with neither rather than with one.

        The HNSW index serves the ordering and the filter on status is
        applied over it, which at this corpus's size costs nothing; a corpus
        large enough for that to under-return is one to raise
        `hnsw.ef_search` for.
        """
        distance = Question.embedding.cosine_distance(embedding)
        query = (
            select(Question.question_text, Question.answerable, distance.label("far"))
            .where(
                Question.status == QuestionStatus.ACCEPTED,
                Question.embedding.is_not(None),
            )
            .order_by(distance)
            .limit(1)
        )
        if before is not None:
            query = query.where(Question.id < before)
        with self._session() as session:
            found = session.execute(query).one_or_none()
        if found is None:
            return None
        return Neighbour(
            question_text=found.question_text,
            answerable=found.answerable,
            # pgvector answers in cosine distance; the gates read likeness.
            similarity=1.0 - float(found.far),
        )

    def judged(self, within=None) -> Iterator[JudgedQuestion]:
        """Streams every stored question with what a re-check needs.

        One query rather than one per question: whether the facts under it
        still hold, how far they are spread now, and what they say.
        """
        query = (
            _joined(
                select(
                    Question.id,
                    Question.question_text,
                    Question.target_answer,
                    Question.answerable,
                    Question.difficulty,
                    Question.passage_scope,
                    Question.document_scope,
                    Question.topic_scope,
                    Question.follows_id,
                    Question.language,
                    Question.status,
                    Question.embedding,
                    Question.question_type,
                    Question.answer_form,
                    func.coalesce(func.bool_and(Fact.validated), False).label("holds"),
                    func.count(func.distinct(Passage.doc_sha256)).label("documents"),
                    func.count(func.distinct(Fact.passage_id)).label("passages"),
                    func.count(func.distinct(DOMINANT.c.topic_id)).label("topics"),
                    func.array_agg(Fact.statement).label("statements"),
                )
            )
            .group_by(Question.id)
            .order_by(Question.id)
            .execution_options(yield_per=_BATCH)
        )
        if within is not None:
            query = query.where(within)
        with self._session() as session:
            for row in session.execute(query):
                yield JudgedQuestion(
                    id=row.id,
                    question_text=row.question_text,
                    target_answer=row.target_answer,
                    answerable=row.answerable,
                    difficulty=row.difficulty,
                    passage_scope=row.passage_scope,
                    document_scope=row.document_scope,
                    topic_scope=row.topic_scope,
                    follows_id=row.follows_id,
                    language=row.language,
                    status=row.status,
                    embedding=list(row.embedding)
                    if row.embedding is not None
                    else None,
                    statements=tuple(
                        one for one in row.statements or () if one is not None
                    ),
                    facts_validated=bool(row.holds),
                    documents=row.documents,
                    passages=row.passages,
                    topics=row.topics,
                    question_type=row.question_type,
                    answer_form=row.answer_form,
                )

    def reject(self, verdicts: list[tuple[int, str]]) -> int:
        """Records what a re-check found, one gate code per question."""
        if not verdicts:
            return 0
        with self._session.begin() as session:
            for question_id, code in verdicts:
                session.execute(
                    update(Question)
                    .where(Question.id == question_id)
                    .values(
                        status=QuestionStatus.REJECTED,
                        rejected_reason=code,
                        status_changed_at=func.now(),
                    )
                )
        return len(verdicts)

    def decide(self, question_id: int, status: str) -> StoredQuestion | None:
        """Records what a person decided about one question.

        Clears the gate's reason when a person accepts: the row is no longer
        rejected, so a reason for its rejection would be a stale one. A
        person rejecting names no gate, which is why the column allows NULL.
        """
        with self._session.begin() as session:
            changed = session.execute(
                update(Question)
                .where(Question.id == question_id)
                .values(
                    status=status, rejected_reason=None, status_changed_at=func.now()
                )
            ).rowcount
        if not changed:
            return None
        return self.one(question_id)

    def one(self, question_id: int) -> StoredQuestion | None:
        """Reads one question, as the listing shows it."""
        with self._session() as session:
            row = session.execute(
                _listing().where(Question.id == question_id)
            ).one_or_none()
        return _stored(row) if row else None

    def page(
        self,
        document: str | None = None,
        topic: int | None = None,
        limit: int = 50,
        offset: int = 0,
        search: str | None = None,
        status: str | None = None,
        answerable: bool | None = None,
        field: str | None = None,
        **scopes: Any,
    ) -> tuple[int, list[StoredQuestion]]:
        """Reads one page of questions and the total behind it."""
        listing = _filtered(
            _listing(), document, topic, search, status, answerable, field, **scopes
        )
        counting = select(func.count()).select_from(
            _ids(
                document, topic, search, status, answerable, field, **scopes
            ).subquery()
        )
        with self._session() as session:
            total = session.scalar(counting) or 0
            rows = session.execute(listing.limit(limit).offset(offset)).all()
        return total, [_stored(row) for row in rows]

    def _thread(self, question: StoredQuestion) -> list[StoredQuestion]:
        """The whole thread this question sits in, oldest first.

        Walked from the root rather than from the question, so a follow-up
        shows what was asked before it as well as after: reading a follow-up
        without its parent is reading half a conversation, and the parent is
        what makes it answerable.
        """
        root = question
        while root.follows_id is not None:
            found = self.one(root.follows_id)
            if found is None:  # pragma: no cover - the cascade prevents it
                break
            root = found

        thread = [root]
        while True:
            with self._session() as session:
                nxt = session.scalar(
                    select(Question.id).where(Question.follows_id == thread[-1].id)
                )
            if nxt is None:
                break
            found = self.one(nxt)
            if found is None:  # pragma: no cover
                break
            thread.append(found)
        return thread

    def detail(self, question_id: int) -> QuestionDetail | None:
        """Reads one question with the facts it was written from."""
        found = self.one(question_id)
        if found is None:
            return None
        with self._session() as session:
            sources = session.execute(
                select(
                    Fact.id,
                    Fact.statement,
                    Fact.evidence_text,
                    Fact.validated,
                    Fact.passage_id,
                    Passage.doc_sha256,
                    Passage.ordinal,
                )
                .select_from(QuestionFact)
                .join(Fact, Fact.id == QuestionFact.fact_id)
                .join(Passage, Passage.id == Fact.passage_id)
                .where(QuestionFact.question_id == question_id)
                .order_by(Passage.doc_sha256, Passage.ordinal, Fact.id)
            ).all()
        return QuestionDetail(
            question=found,
            thread=self._thread(found),
            sources=[
                QuestionSource(
                    fact_id=row.id,
                    statement=row.statement,
                    evidence_text=row.evidence_text,
                    validated=row.validated,
                    passage_id=row.passage_id,
                    doc_sha256=row.doc_sha256,
                    ordinal=row.ordinal,
                )
                for row in sources
            ],
        )

    def quality(
        self,
        document: str | None = None,
        topic: int | None = None,
        search: str | None = None,
        status: str | None = None,
        answerable: bool | None = None,
        field: str | None = None,
        **scopes: Any,
    ) -> QuestionQuality:
        """Measures how generation is doing, under the listing's filter.

        Rejections are grouped on the gate that fired, which is what says
        whether the model is writing questions the corpus answers or
        questions that merely look as though it should.
        """
        # One question per row from here on, so a plain count is the right
        # count and the mean is not weighted by how many facts each cites.
        selected = Question.id.in_(
            _ids(document, topic, search, status, answerable, field, **scopes)
        )
        totals = select(
            func.count().label("total"),
            func.count()
            .filter(Question.status == QuestionStatus.ACCEPTED)
            .label("accepted"),
            func.count().filter(Question.status == QuestionStatus.DRAFT).label("draft"),
            func.count().filter(~Question.answerable).label("unanswerable"),
            func.count().filter(Question.follows_id.is_not(None)).label("followups"),
            func.count()
            .filter(Question.difficulty == Question.planned_difficulty)
            .label("planned_met"),
            func.coalesce(func.avg(Question.answer_chars), 0.0).label("answer"),
            func.coalesce(func.avg(func.length(Question.question_text)), 0.0).label(
                "chars"
            ),
        ).where(selected)
        grouped = {
            "rejected": Question.rejected_reason,
            "difficulty": Question.difficulty,
            "planned_difficulty": Question.planned_difficulty,
            "question_type": Question.question_type,
            "answer_form": Question.answer_form,
            "passage_scope": Question.passage_scope,
            "document_scope": Question.document_scope,
            "topic_scope": Question.topic_scope,
        }
        with self._session() as session:
            row = session.execute(totals).one()
            spread = {
                name: dict(
                    session.execute(
                        select(column, func.count())
                        .where(selected, column.is_not(None))
                        .group_by(column)
                        .order_by(func.count().desc())
                    ).all()
                )
                for name, column in grouped.items()
            }
            # Corpus-wide rather than filtered: coverage is about the topics
            # that exist, and a filter that selects no question of a topic
            # cannot say that topic is uncovered.
            covered = (
                session.scalar(
                    select(func.count())
                    .select_from(Topic)
                    .where(
                        _IS_TOPIC,
                        Topic.include_in_coverage,
                        Topic.question_status == Status.GENERATED,
                    )
                )
                or 0
            )
            in_coverage = (
                session.scalar(
                    select(func.count())
                    .select_from(Topic)
                    .where(_IS_TOPIC, Topic.include_in_coverage)
                )
                or 0
            )

        return QuestionQuality(
            total=row.total,
            accepted=row.accepted,
            draft=row.draft,
            unanswerable=row.unanswerable,
            followups=row.followups,
            topics_covered=covered,
            topics_in_coverage=in_coverage,
            mean_question_chars=round(float(row.chars), 1),
            mean_answer_chars=round(float(row.answer), 1),
            rejected=spread["rejected"],
            difficulty=spread["difficulty"],
            passage_scope=spread["passage_scope"],
            document_scope=spread["document_scope"],
            topic_scope=spread["topic_scope"],
            question_type=spread["question_type"],
            answer_form=spread["answer_form"],
            planned_difficulty=spread["planned_difficulty"],
            planned_met=row.planned_met,
        )
