"""Reading the questions back, and recording what a person decided about one.

The API's half of the database access, plus the two operations that run over
everything stored rather than over one claimed topic: the release draw and
the re-check. Claiming a topic is the :mod:`question_generation.queue`'s.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import (
    ARRAY,
    Select,
    Text,
    and_,
    case,
    cast,
    distinct,
    func,
    select,
    update,
)

from database.qa_generator import (
    Document,
    Fact,
    FactPassage,
    Passage,
    Question,
    QuestionFact,
    QuestionStatus,
    Status,
    Topic,
)
from database.qa_generator.passage_topics import DOMINANT
from database.qa_generator.repository import Repository, matching
from question_generation.balance import Row
from question_generation.models import (
    Citation,
    JudgedQuestion,
    Neighbour,
    QuestionDetail,
    QuestionQuality,
    QuestionSource,
    StoredQuestion,
)
from question_generation.queue import IS_TOPIC
from question_generation.types import CHECKED

#: Which columns the search box looks in, by the name the API accepts.
SEARCH_FIELDS = {
    "question": (Question.question_text,),
    "answer": (Question.target_answer, Question.answer_explanation),
    "both": (
        Question.question_text,
        Question.target_answer,
        Question.answer_explanation,
    ),
}

DEFAULT_FIELD = "both"

#: How many questions to hold at once while re-checking them.
_BATCH = 500

#: Only passages the corpus makes a checkable claim from. Without it a probe
#: comes back with tables of contents and indexes, which carry every term in
#: the material and sit near every question in the embedding space, and
#: answer none of them: they are the same passages extraction skips as
#: navigation. Shared by both `elsewhere` probes, so the lemma one and the
#: vector one cannot come to disagree about what a passage is.
_ASSERTS = (
    select(FactPassage.passage_id)
    .join(Fact, Fact.id == FactPassage.fact_id)
    .where(Fact.validated)
    .where(FactPassage.passage_id == Passage.id)
    .exists()
)


def _joined(query: Select) -> Select:
    """Joins everything a question is filtered by, placed by and named by.

    Outer throughout. A question reaches its document and its topic through
    its facts, and a question whose last fact has just gone is one the
    trigger is about to delete: it should not disappear from a count before
    it disappears from the table.

    A fact reaches its passages through `fact_passages`, so a bridge reaches
    both of its passages here: the documents a question is filtered by,
    listed under and scored against are the ones its answer actually needs.

    Every join here multiplies a question by its facts, and a bridge by its
    passages on top, so anything counted over this has to be counted
    distinctly - which is why the figures are measured over :func:`_ids`
    instead.
    """
    return (
        query.select_from(Question)
        .join(QuestionFact, QuestionFact.question_id == Question.id, isouter=True)
        .join(Fact, Fact.id == QuestionFact.fact_id, isouter=True)
        .join(FactPassage, FactPassage.fact_id == Fact.id, isouter=True)
        .join(Passage, Passage.id == FactPassage.passage_id, isouter=True)
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
    "cognitive_level": Question.cognitive_level,
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
                Question.answer_explanation,
                Question.answerable,
                Question.difficulty,
                Question.passage_scope,
                Question.document_scope,
                Question.topic_scope,
                Question.answer_chars,
                Question.language,
                Question.status,
                Question.rejected_reason,
                Question.reviewed_verdict,
                Question.created_at,
                Question.thread_position,
                Question.follows_id,
                Question.question_type,
                Question.cognitive_level,
                Question.answer_form,
                Question.planned_difficulty,
                Question.prompt_version,
                Question.trace_id,
                Question.span_id,
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


def _stored(row: Any) -> StoredQuestion:
    """Reads one row of the listing back as a question."""
    return StoredQuestion(
        id=row.id,
        question_text=row.question_text,
        target_answer=row.target_answer,
        answer_explanation=row.answer_explanation,
        answerable=row.answerable,
        difficulty=row.difficulty,
        passage_scope=row.passage_scope,
        document_scope=row.document_scope,
        topic_scope=row.topic_scope,
        answer_chars=row.answer_chars,
        language=row.language,
        status=row.status,
        rejected_reason=row.rejected_reason,
        reviewed_verdict=row.reviewed_verdict,
        created_at=_when(row.created_at),
        facts=row.facts,
        documents=_present(row.documents),
        topics=_present(row.topics),
        thread_position=row.thread_position,
        follows_id=row.follows_id,
        question_type=row.question_type,
        cognitive_level=row.cognitive_level,
        answer_form=row.answer_form,
        planned_difficulty=row.planned_difficulty,
        prompt_version=row.prompt_version,
        trace_id=row.trace_id,
        span_id=row.span_id,
    )


#: How much of the corpus the ACCEPTED questions reach.
#:
#: Four counts in one statement rather than four round trips, and every one
#: of them corpus-wide: coverage is a property of the material, so a filter
#: that selects no question of a passage cannot say that passage is
#: unasked.
#:
#: `passages_with_facts` counts only passages a VALIDATED fact rests on,
#: because that is the most this stage could ever ask about. Measuring
#: against every passage would report extraction's refusals as question
#: generation's gap.
_ASKED_PASSAGES = (
    select(FactPassage.passage_id)
    .join(QuestionFact, QuestionFact.fact_id == FactPassage.fact_id)
    .join(Question, Question.id == QuestionFact.question_id)
    .where(Question.status == QuestionStatus.ACCEPTED)
    .distinct()
    .scalar_subquery()
)

_COVERAGE = select(
    select(func.count()).select_from(Passage).scalar_subquery().label("passages_total"),
    select(func.count(distinct(FactPassage.passage_id)))
    .select_from(FactPassage)
    .join(Fact, Fact.id == FactPassage.fact_id)
    .where(Fact.validated)
    .scalar_subquery()
    .label("passages_with_facts"),
    select(func.count())
    .select_from(_ASKED_PASSAGES.subquery())
    .scalar_subquery()
    .label("passages_asked"),
    select(func.count())
    .select_from(Fact)
    .where(Fact.validated)
    .scalar_subquery()
    .label("facts_validated"),
    select(func.count(distinct(QuestionFact.fact_id)))
    .select_from(QuestionFact)
    .join(Question, Question.id == QuestionFact.question_id)
    .where(Question.status == QuestionStatus.ACCEPTED)
    .scalar_subquery()
    .label("facts_asked"),
)


class QuestionCatalog(Repository):
    """Reads back the questions, and records what a person decided about one.

    Separate from the queue: the API serves these and never claims a row.
    """

    def releasable(self, within=None) -> list[Row]:
        """Every accepted question a release may be chosen from.

        Only the three columns the choosing reads. A question with no
        difficulty or no type is left out rather than given a bucket: it
        would be chosen into a quota it cannot be counted against, and the
        report would then disagree with the set it describes.
        """
        query = select(
            Question.id,
            Question.answerable,
            Question.difficulty,
            Question.question_type,
        ).where(
            Question.status == QuestionStatus.ACCEPTED,
            Question.difficulty.is_not(None),
            Question.question_type.is_not(None),
        )
        if within is not None:
            query = query.where(within)
        with self._session() as session:
            return [
                Row(
                    id=row.id,
                    answerable=row.answerable,
                    difficulty=row.difficulty,
                    question_type=row.question_type,
                )
                for row in session.execute(query.order_by(Question.id))
            ]

    def release(self, ids: Sequence[int]) -> tuple[uuid.UUID, int]:
        """Marks these questions as the release, and clears the last one.

        One release at a time, because the column holds one id and the
        question a report is about is "what would ship today". A previous
        release is not history worth keeping here: every question it held
        is still in the table, and the draw is reproducible from the
        settings that made it.
        """
        drawn = uuid.uuid4()
        with self._session.begin() as session:
            session.execute(
                update(Question)
                .where(Question.release_id.is_not(None))
                .values(release_id=None)
            )
            if not ids:
                return drawn, 0
            return drawn, session.execute(
                update(Question)
                .where(Question.id.in_(list(ids)))
                .values(release_id=drawn)
            ).rowcount

    def elsewhere(
        self,
        lemmas: Sequence[str],
        language: str,
        skip: Sequence[int],
        limit: int,
        embedding: Sequence[float] | None = None,
    ) -> list[str]:
        """The passages outside this question's own that talk about it most.

        What the `answerable_elsewhere` gate reads. The verifier is shown
        only the passages a question cites, which is right for measuring the
        dataset and wrong for one claim: an unanswerable question is a claim
        about the WHOLE corpus, and a passage nobody cited may answer it. A
        question labelled unanswerable that the material does answer marks a
        correct chatbot wrong, which is the failure the gates exist for.

        Ranked by cosine over `passages.embedding` when the question carries
        a vector and the corpus has been embedded, and by shared content
        lemmas otherwise.

        The vectors matter more here than anywhere else this pair of
        measures is used. A lemma probe finds the passages that REPEAT the
        question's words, and an unanswerable question is written by moving
        a fact just out of reach - so the passage that would answer it is
        the one phrased differently, which is the one a lemma probe ranks
        last. This gate exists to catch a question the corpus answers after
        all, and it was retrieving on the signal least likely to find one.

        Either way it is a retrieval and not a proof: it finds where to
        look, and the model still reads the passages and decides.

        Args:
            lemmas: The question's content lemmas.
            language: Only passages written in this one.
            skip: Passages the question already cites, which the first
                verifier pass has read.
            limit: The most passages to hand back.
            embedding: The question's vector, when one was computed.

        Returns:
            Their text, nearest first. Empty when the question names nothing
            the corpus does.
        """
        if embedding is not None:
            found = self._nearest_passages(language, skip, limit, list(embedding))
            if found:
                return found
        if not lemmas:
            return []
        wanted = cast(list(lemmas), ARRAY(Text))
        # The overlap counted by intersecting the two arrays. The `&&`
        # filter beside it is what the GIN index on lemmas serves, so this
        # only ever orders the rows that filter already kept.
        overlap = func.cardinality(
            func.array(
                select(func.unnest(Passage.lemmas))
                .intersect(select(func.unnest(wanted)))
                .scalar_subquery()
            )
        )
        query = (
            select(Passage.text)
            .where(
                Passage.language == language,
                Passage.lemmas.op("&&")(wanted),
                _ASSERTS,
            )
            .order_by(overlap.desc(), Passage.id)
            .limit(limit)
        )
        if skip:
            query = query.where(Passage.id.notin_(skip))
        with self._session() as session:
            return [row.text for row in session.execute(query)]

    def _nearest_passages(
        self, language: str, skip: Sequence[int], limit: int, embedding: list[float]
    ) -> list[str]:
        """The embedded passages of one language nearest this vector.

        Empty when the corpus carries no vectors, which is what sends
        `elsewhere` back to its lemma probe rather than to nothing.
        """
        distance = Passage.embedding.cosine_distance(embedding)
        query = (
            select(Passage.text)
            .where(
                Passage.language == language,
                Passage.embedding.is_not(None),
                # The same filter the lemma probe applies, and for the same
                # reason: a contents page carries every term in the material
                # and sits near every question in the space, and answers
                # none of them.
                _ASSERTS,
            )
            .order_by(distance)
            .limit(limit)
        )
        if skip:
            query = query.where(Passage.id.notin_(skip))
        with self._session() as session:
            return [row.text for row in session.execute(query)]

    def nearest(
        self,
        embedding: list[float],
        before: int | None = None,
        excluding: Sequence[int] = (),
    ) -> Neighbour | None:
        """The accepted question closest to this vector, or None if there is none.

        `before` looks only at questions written earlier. A re-check needs
        it: without it, two questions that are duplicates of each other each
        find the other and both are rejected, which leaves the benchmark
        with neither rather than with one.

        `excluding` drops questions a caller has decided to reject but has
        not written yet. A re-check batches its verdicts, and without this
        the answer depends on whether the batch happened to have been
        flushed - the same pass over the same rows rejecting a different
        set at a different batch size.

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
        if excluding:
            query = query.where(Question.id.notin_(list(excluding)))
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
                    func.count(func.distinct(Passage.id)).label("passages"),
                    func.count(func.distinct(DOMINANT.c.topic_id)).label("topics"),
                    func.array_agg(func.distinct(Fact.statement)).label("statements"),
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

        Three columns, and they say different things. `status` is what the
        question now IS, `rejected_reason` stays whatever gate stopped it,
        and `reviewed_verdict` is what the person said.

        The reason used to be cleared here, on the grounds that a reason for
        a rejection somebody overturned is a stale one. It is not stale, it
        is the other half of a disagreement: without it, a person accepting
        what a gate refused erased the only record that the two had ever
        disagreed, and the agreement rate a review sample exists to produce
        was not a query. A person rejecting still names no gate, which is
        why the column allows NULL on rows no gate ever touched.

        Putting a question back to `draft` is the one decision that is not a
        verdict - it says "I have not decided" - so it clears the verdict
        rather than recording itself as one, which the column could not hold
        anyway.
        """
        decided = status in (QuestionStatus.ACCEPTED, QuestionStatus.REJECTED)
        with self._session.begin() as session:
            changed = session.execute(
                update(Question)
                .where(Question.id == question_id)
                .values(
                    status=status,
                    reviewed_verdict=status if decided else None,
                    reviewed_at=func.now() if decided else None,
                    status_changed_at=func.now(),
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
                    # A bridge comes back once per passage it rests on, and
                    # the document and the ordinal beside it are that
                    # passage's.
                    Passage.id.label("passage_id"),
                    Passage.doc_sha256,
                    Passage.ordinal,
                )
                .select_from(QuestionFact)
                .join(Fact, Fact.id == QuestionFact.fact_id)
                .join(FactPassage, FactPassage.fact_id == Fact.id)
                .join(Passage, Passage.id == FactPassage.passage_id)
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

    def citations(
        self,
        document: str | None = None,
        topic: int | None = None,
        search: str | None = None,
        status: str | None = None,
        answerable: bool | None = None,
        field: str | None = None,
        **scopes: Any,
    ) -> list[Citation]:
        """What every question a filter selects was written from.

        The same rows :meth:`detail` reads one question at a time, for a
        whole filter in one query. An export runs over thousands of
        questions, and asking per question is that many round trips for a
        join the database does once.

        A question comes back once per (fact, passage) pair, so a bridge
        resting on two passages is two rows. That is the grain a reader
        checking an answer against its source wants: the document and the
        page beside each half of what was cited.

        The chosen ids are JOINED as a derived table rather than put in an
        `IN`, and `correlate(None)` is why the difference matters. `_ids`
        is joined over the same four tables this query selects from, so as
        a WHERE subquery SQLAlchemy correlated those tables to the outer
        query, emptied the subquery's own FROM and turned a filter on
        1,229 rows into six million.
        """
        chosen = (
            _ids(document, topic, search, status, answerable, field, **scopes)
            .correlate(None)
            .subquery()
        )
        with self._session() as session:
            rows = session.execute(
                select(
                    QuestionFact.question_id,
                    Fact.id.label("fact_id"),
                    Fact.statement,
                    Fact.evidence_text,
                    Fact.validated,
                    Passage.doc_sha256,
                    Passage.ordinal,
                    Passage.page_from,
                    Document.title,
                )
                .select_from(QuestionFact)
                .join(chosen, chosen.c.id == QuestionFact.question_id)
                .join(Fact, Fact.id == QuestionFact.fact_id)
                .join(FactPassage, FactPassage.fact_id == Fact.id)
                .join(Passage, Passage.id == FactPassage.passage_id)
                .join(Document, Document.sha256 == Passage.doc_sha256, isouter=True)
                .order_by(QuestionFact.question_id, Passage.doc_sha256, Passage.ordinal)
            ).all()
        return [
            Citation(
                question_id=row.question_id,
                fact_id=row.fact_id,
                statement=row.statement,
                evidence_text=row.evidence_text,
                validated=row.validated,
                document=row.title or row.doc_sha256[:12],
                doc_sha256=row.doc_sha256,
                ordinal=row.ordinal,
                page=row.page_from,
            )
            for row in rows
        ]

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
            # How much of `cognitive_level` is a measurement. The level is
            # derived from the type, and only a few types have a label a
            # rule settles; the rest declare one nothing checked.
            func.count(
                case(
                    (
                        and_(
                            Question.status == QuestionStatus.ACCEPTED,
                            Question.question_type.in_(tuple(CHECKED)),
                        ),
                        1,
                    )
                )
            ).label("levels_checked"),
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
            "cognitive_level": Question.cognitive_level,
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
                        IS_TOPIC,
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
                    .where(IS_TOPIC, Topic.include_in_coverage)
                )
                or 0
            )
            reach = session.execute(_COVERAGE).one()

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
            cognitive_level=spread["cognitive_level"],
            answer_form=spread["answer_form"],
            planned_difficulty=spread["planned_difficulty"],
            planned_met=row.planned_met,
            cognitive_level_checked=row.levels_checked,
            passages_total=reach.passages_total,
            passages_with_facts=reach.passages_with_facts,
            passages_asked=reach.passages_asked,
            facts_validated=reach.facts_validated,
            facts_asked=reach.facts_asked,
        )
