"""Database access for the extraction service."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import timedelta
from itertools import groupby
from typing import ClassVar, cast

from sqlalchemy import (
    ColumnElement,
    Select,
    Table,
    bindparam,
    delete,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.orm import InstrumentedAttribute, aliased

from database.qa_generator import (
    Document,
    Fact,
    FactKind,
    FactPassage,
    Passage,
    Rejection,
    Status,
)
from database.qa_generator.passage_topics import DOMINANT
from database.qa_generator.repository import Repository, matching
from extraction.models import (
    WRITTEN,
    CandidateFact,
    CheckedFact,
    Cited,
    FactQuality,
    FactSource,
    PassageToExtract,
    StoredFact,
    Twin,
)
from nlp.models import Sentence
from stages import Columns, RowQueue

#: The next passage to read. This stage's own column and nothing else: it
#: does not join documents to ask what chunking did. FOR UPDATE SKIP LOCKED
#: lets a second worker take the following row.
_NEXT_PENDING = (
    select(Passage.id)
    .where(Passage.extract_status == Status.PENDING)
    .order_by(Passage.id)
    .with_for_update(skip_locked=True)
    .limit(1)
    .scalar_subquery()
)

#: Which columns the search box looks in, by the name the API accepts.
SEARCH_FIELDS = {
    "statement": (Fact.statement,),
    "evidence": (Fact.evidence_text,),
    "both": (Fact.statement, Fact.evidence_text),
}

DEFAULT_FIELD = "both"

#: How many facts to hold at once while re-judging them.
_BATCH = 500

#: The facts table itself, for the bulk update a re-judgement writes.
_FACTS = cast("Table", Fact.__table__)

#: The link table itself, for the same reason.
_LINKS = cast("Table", FactPassage.__table__)

#: The passages table itself, for the bulk update the backfill writes.
_PASSAGES = cast("Table", Passage.__table__)

#: What one passage of a bridge group is read as. The same columns `claim`
#: reads, so a group and a queued passage are the same object.
_PASSAGE_COLUMNS = (
    Passage.id,
    Passage.text,
    Passage.section_path,
    Passage.block_type,
    Passage.language,
    Passage.sentences,
    Passage.table_cells,
    Passage.doc_sha256,
    Passage.embedding,
)


def _sentences(text: str, stored: list[dict] | None) -> list[Sentence]:
    """Rebuilds a passage's numbered sentences from the stored offsets."""
    return [
        Sentence(
            index=entry["i"],
            start=entry["start"],
            end=entry["end"],
            text=text[entry["start"] : entry["end"]],
            predicates=entry.get("predicates", 0),
        )
        for entry in stored or []
    ]


def _passage(row, language: str | None) -> PassageToExtract:
    """Reads one row of `_PASSAGE_COLUMNS` as a passage to extract from.

    Args:
        row: The selected columns.
        language: The document's language, which is the fallback.

    Returns:
        The passage, reading in its own language where it has one. Chunking
        detects the language per passage and segmented this one under it, so
        the document's label would judge a statement with one pipeline
        against sentence counts another produced. One file carries a German
        report and its English summary, which is what the column is for.
    """
    return PassageToExtract(
        id=row.id,
        text=row.text,
        section_path=row.section_path,
        block_type=row.block_type,
        language=row.language or language,
        sentences=_sentences(row.text, row.sentences),
        table_cells=row.table_cells or [],
        doc_sha256=row.doc_sha256,
        embedding=tuple(row.embedding) if row.embedding is not None else None,
    )


#: The link a listing is placed and ordered by: the passage the fact opens
#: on. Aliased, because the document filter reaches the table again to ask
#: about every other passage the fact rests on.
_OPENS = aliased(FactPassage)


def _resting(condition) -> ColumnElement[bool]:
    """Matches a fact resting on any passage the condition selects.

    Any rather than the one it opens on: a bridge across two documents
    belongs to both, and narrowing to the second is asking for the facts its
    passages support. Written as a subquery on the fact rather than as a join
    so the narrowing selects facts whole - joining it would drop the half of
    a bridge that sits outside the narrowing.
    """
    return Fact.id.in_(
        select(FactPassage.fact_id)
        .join(Passage, Passage.id == FactPassage.passage_id)
        .where(condition)
    )


def _filtered(query, document, search, method, field, kind=None):
    """Applies the document, text, method and kind filters to a fact query.

    One place, so a listing and its count cannot disagree about what they are
    looking at. The query must already be joined by :func:`_joined`.
    """
    if document:
        query = query.where(_resting(Passage.doc_sha256 == document))
    if search:
        query = query.where(
            matching(
                search, *SEARCH_FIELDS.get(field or "", SEARCH_FIELDS[DEFAULT_FIELD])
            )
        )
    if method:
        query = query.where(Fact.extraction_method == method)
    if kind:
        query = query.where(Fact.kind == kind)
    return query


def _joined(query: Select) -> Select:
    """Joins the passage a fact is ordered by: the one it opens on.

    One row per fact, not one per passage it rests on: a listing that
    multiplied a bridge by its passages would page through it twice.
    """
    return (
        query.select_from(Fact)
        .join(_OPENS, (_OPENS.fact_id == Fact.id) & (_OPENS.position == 0))
        .join(Passage, Passage.id == _OPENS.passage_id)
    )


#: One fact's passages, in the order the model was shown them. Read for the
#: page of facts a listing returned rather than joined into it, which would
#: give a bridge one row per passage.
_SOURCES = (
    select(
        FactPassage.fact_id,
        FactPassage.passage_id,
        FactPassage.position,
        Passage.doc_sha256,
        Passage.ordinal,
        Passage.page_from,
    )
    .select_from(FactPassage)
    .join(Passage, Passage.id == FactPassage.passage_id)
    .order_by(FactPassage.fact_id, FactPassage.position)
)


class PassageQueue(RowQueue):
    """Reads the extraction queue and records what became of each passage.

    Holds the settings version, which every fact it writes records. Built
    without one, it writes NULL.
    """

    columns = Columns(
        entity=Passage,
        key=Passage.id,
        status=Passage.extract_status,
        error=Passage.extract_error,
        claimed_at=Passage.extract_claimed_at,
    )
    #: This stage queues over passages, so it answers for one of those and for
    #: every passage of one document - which is the unit a person thinks in.
    scopes: ClassVar[dict[str, InstrumentedAttribute]] = {
        "document": Passage.doc_sha256,
        "passage": Passage.id,
    }
    done = Status.EXTRACTED
    next_pending = _NEXT_PENDING

    def __init__(
        self, lease: timedelta | None = None, version: str | None = None
    ) -> None:
        """Binds to the session factory, with a lease and a settings version."""
        super().__init__(lease)
        self._version = version

    def claim(self) -> PassageToExtract | None:
        """Takes the next unread passage off the queue.

        Returns:
            The passage, or None when nothing is queued.
        """
        claimed = self._claim(*_PASSAGE_COLUMNS)
        if claimed is None:
            return None
        if claimed.language:
            return _passage(claimed, None)
        # Only for a passage too short to tell its own language. Asked here
        # rather than joined into the claim, which is an UPDATE.
        with self._session() as session:
            language = session.scalar(
                select(Document.language).where(Document.sha256 == claimed.doc_sha256)
            )
        return _passage(claimed, language)

    def store(
        self,
        passage_id: int,
        facts: list[CheckedFact],
        embedding: list[float] | None = None,
    ) -> int:
        """Replaces one passage's facts, in one transaction, and finishes it.

        Scoped to the passage, so two workers on two passages of the same
        document do not delete each other's results. Bridge facts resting
        here are left alone: they are the bridge pass's to write and to
        replace.

        Args:
            passage_id: The passage that was read.
            facts: Everything it yielded, refused facts included.
            embedding: The passage's own vector, or None to leave the column
                as it is. Written here rather than by chunking because this
                is the worker that has the model loaded.

        Returns:
            How many facts were written.
        """
        with self._session.begin() as session:
            session.execute(
                delete(Fact).where(
                    Fact.kind != FactKind.BRIDGE,
                    Fact.id.in_(
                        select(FactPassage.fact_id).where(
                            FactPassage.passage_id == passage_id
                        )
                    ),
                )
            )
            _write(session, facts, self._version)
            if embedding is not None:
                session.execute(
                    update(Passage)
                    .where(Passage.id == passage_id)
                    .values(embedding=embedding)
                )
            self._finish(passage_id, session=session)
        return len(facts)

    def nearest_fact(
        self, embedding: list[float], within: Sequence[int] = ()
    ) -> Twin | None:
        """The validated fact closest to this vector, or None if there is none.

        `within` narrows the search to a set of passages. Empty searches the
        whole corpus, which is what dedup wants: a duplicate written from
        another document is still a duplicate, and at this corpus's size the
        HNSW index answers either in single-digit milliseconds.
        """
        distance = Fact.embedding.cosine_distance(embedding)
        query = (
            select(Fact.id, Fact.statement, distance.label("far"))
            .where(Fact.validated, Fact.embedding.is_not(None))
            .order_by(distance)
            .limit(1)
        )
        if within:
            query = query.where(
                Fact.id.in_(
                    select(FactPassage.fact_id).where(
                        FactPassage.passage_id.in_(within)
                    )
                )
            )
        with self._session() as session:
            found = session.execute(query).one_or_none()
        if found is None:
            return None
        # pgvector answers in cosine distance; the gate reads likeness.
        return Twin(
            fact_id=found.id,
            statement=found.statement,
            similarity=1.0 - float(found.far),
        )

    def counts(self) -> dict[str, int]:
        """Reports what this service owns, for the status panel."""
        with self._session() as session:
            total = session.scalar(select(func.count()).select_from(Fact))
            valid = session.scalar(
                select(func.count()).select_from(Fact).where(Fact.validated)
            )
        return {**self.counts_by_status(), "facts": total, "validated": valid}


def _write(session, facts: list[CheckedFact], version: str | None = None) -> int:
    """Writes facts and the passages each rests on, in the caller's transaction.

    Args:
        session: The open transaction.
        facts: Checked facts of any kind, refused ones included.
        version: The configuration these were extracted under, recorded on
            each of them.

    Returns:
        How many facts were written.
    """
    if not facts:
        return 0
    # sort_by_parameter_order: the links below are matched to the facts by
    # position, so the ids have to come back in the order they went in rather
    # than in whatever order the insert chose.
    ids = session.scalars(
        insert(Fact).returning(Fact.id, sort_by_parameter_order=True),
        [_row(fact, version) for fact in facts],
    ).all()
    links = [
        {
            "fact_id": fact_id,
            "passage_id": cited.passage_id,
            "position": position,
            "sentence_ids": cited.sentence_ids,
            "evidence_start": cited.start,
            "evidence_end": cited.end,
        }
        for fact_id, fact in zip(ids, facts, strict=True)
        for position, cited in enumerate(fact.citations)
    ]
    if links:
        session.execute(insert(FactPassage), links)
    return len(facts)


def _row(fact: CheckedFact, version: str | None = None) -> dict:
    """Turns one checked fact into the columns the facts table holds."""
    return {
        "settings_version": version,
        "kind": fact.kind,
        "statement": fact.statement,
        "evidence_text": fact.evidence_text,
        "extraction_method": fact.extraction_method,
        "validated": fact.validated,
        "rejection_code": fact.rejection_code,
        "validation_error": fact.validation_error,
        "statement_predicates": fact.statement_predicates,
        "evidence_predicates": fact.evidence_predicates,
        "units_statement": fact.units_statement,
        "units_added": fact.units_added,
        "unresolved_references": fact.unresolved_references,
        "extraction_model": fact.extraction_model,
        "prompt_version": fact.prompt_version,
        "extraction_temperature": fact.extraction_temperature,
        "spacy_model": fact.spacy_model,
        "spacy_version": fact.spacy_version,
        "embedding": fact.embedding,
    }


class FactCatalog(Repository):
    """Reads back the facts extraction produced, and writes the bridges.

    Separate from the queue: the API serves these and never claims a row.

    Holds the settings version for the two operations that write a verdict,
    a bridge pass and a re-judgement. The API builds this without one.
    """

    def __init__(self, version: str | None = None) -> None:
        """Binds to the session factory, with the settings version to record."""
        super().__init__()
        self._version = version

    def by_topic(self, within=None) -> Iterator[tuple[int, list[PassageToExtract]]]:
        """Streams each topic's passages, in document and reading order.

        A passage belongs to the topic it carries most strongly, which is
        what puts two of them in the same group.

        Args:
            within: A condition selecting passages, or None for the corpus.
                Every topic holding one of them is read WHOLE, which is not
                the same as reading the passages it selects. `clear_bridges`
                deletes any bridge resting on a selected passage, and a
                bridge rests on two - so narrowing the read to one document
                would delete the cross-document bridges and then be unable
                to write them again, having kept only one side of each. The
                cross-document bridge is the one worth having.

        Yields:
            One topic's id and its passages.
        """
        query = (
            select(
                *_PASSAGE_COLUMNS,
                # Labelled, because `_PASSAGE_COLUMNS` now carries the
                # passage's own and two `language` keys in one row is one
                # key too few.
                Document.language.label("document_language"),
                DOMINANT.c.topic_id,
            )
            .select_from(Passage)
            .join(Document, Document.sha256 == Passage.doc_sha256)
            .join(DOMINANT, DOMINANT.c.passage_id == Passage.id)
            .where(Passage.sentences.is_not(None))
            .order_by(DOMINANT.c.topic_id, Passage.doc_sha256, Passage.ordinal)
        )
        if within is not None:
            query = query.where(
                DOMINANT.c.topic_id.in_(
                    select(DOMINANT.c.topic_id)
                    .join(Passage, Passage.id == DOMINANT.c.passage_id)
                    .where(within)
                )
            )

        with self._session() as session:
            held: list[PassageToExtract] = []
            current: int | None = None
            for row in session.execute(query):
                if row.topic_id != current:
                    if held:
                        yield cast("int", current), held
                    current, held = row.topic_id, []
                held.append(_passage(row, row.document_language))
            if held:
                yield cast("int", current), held

    def recap(self, cap: int, reason: str, within=None) -> int:
        """Refuses the atomic facts already stored above the cap.

        The counterpart of `extract-revalidate` and the same bargain: what
        the model wrote is the record of one extraction and is kept, only
        the verdict moves. No model is called, so a corpus extracted before
        EXTRACTION_MIN_OTHER_SHARE existed is re-balanced in seconds rather
        than re-read over hours.

        Ranked exactly as `over_cap` ranks a passage in flight - the facts
        a model wrote, asserting a number, a date or a name first, ties by
        id - so a passage re-extracted later keeps the same facts this
        leaves.

        Args:
            cap: The most validated atomic facts one passage may keep.
            reason: What to record on the ones refused.
            within: A condition on Passage narrowing which are read, or None.

        Returns:
            How many facts were refused.
        """
        ranked = (
            select(
                Fact.id,
                func.row_number()
                .over(
                    partition_by=FactPassage.passage_id,
                    order_by=(
                        func.coalesce(func.cardinality(Fact.units_statement), 0) == 0,
                        Fact.id,
                    ),
                )
                .label("rank"),
            )
            .select_from(Fact)
            .join(
                FactPassage,
                (FactPassage.fact_id == Fact.id) & (FactPassage.position == 0),
            )
            .where(
                Fact.kind == FactKind.ATOMIC,
                Fact.validated,
                # As `over_cap` ranks one in flight: a statement composed
                # from a grid does not compete. The cap exists to leave room
                # for the digests, and a table yields none.
                Fact.extraction_method == WRITTEN,
            )
        )
        if within is not None:
            ranked = ranked.where(_resting(within))
        over = ranked.subquery()
        with self._session.begin() as session:
            return session.execute(
                update(Fact)
                .where(Fact.id.in_(select(over.c.id).where(over.c.rank > cap)))
                .values(
                    validated=False,
                    rejection_code=Rejection.OVER_CAP,
                    validation_error=reason,
                )
            ).rowcount

    def unembedded(self, limit: int, within=None) -> list[tuple[int, str]]:
        """The next facts carrying no vector, as (id, statement).

        Args:
            limit: How many to read at once.
            within: A condition on Passage narrowing which are read, or None.

        Returns:
            Up to `limit` of them, empty when every fact is embedded.
        """
        query = select(Fact.id, Fact.statement).where(Fact.embedding.is_(None))
        if within is not None:
            query = query.where(_resting(within))
        with self._session() as session:
            return [
                (row.id, row.statement)
                for row in session.execute(query.order_by(Fact.id).limit(limit))
            ]

    def embed_facts(self, vectors: list[tuple[int, list[float]]]) -> int:
        """Writes a vector onto each of the given facts."""
        if not vectors:
            return 0
        with self._session.begin() as session:
            # The table rather than the entity, as `rejudge` does: an
            # executemany against the mapped class is read as an ORM bulk
            # update by primary key, which refuses a WHERE of its own.
            return session.execute(
                update(_FACTS).where(_FACTS.c.id == bindparam("row")),
                [{"row": fact_id, "embedding": one} for fact_id, one in vectors],
            ).rowcount

    def unembedded_passages(self, limit: int, within=None) -> list[tuple[int, str]]:
        """The next passages carrying no vector, as (id, text)."""
        query = select(Passage.id, Passage.text).where(Passage.embedding.is_(None))
        if within is not None:
            query = query.where(within)
        with self._session() as session:
            return [
                (row.id, row.text)
                for row in session.execute(query.order_by(Passage.id).limit(limit))
            ]

    def embed_passages(self, vectors: list[tuple[int, list[float]]]) -> int:
        """Writes a vector onto each of the given passages."""
        if not vectors:
            return 0
        with self._session.begin() as session:
            return session.execute(
                update(_PASSAGES).where(_PASSAGES.c.id == bindparam("row")),
                [{"row": passage_id, "embedding": one} for passage_id, one in vectors],
            ).rowcount

    def clear_bridges(self, within=None) -> int:
        """Deletes the bridge facts a fresh pass is about to replace.

        Args:
            within: A condition on Passage narrowing which bridges go, or
                None for every one of them.

        Returns:
            How many were deleted.
        """
        chosen = select(Fact.id).where(Fact.kind == FactKind.BRIDGE)
        if within is not None:
            chosen = chosen.where(_resting(within))
        with self._session.begin() as session:
            return session.execute(delete(Fact).where(Fact.id.in_(chosen))).rowcount

    def add_bridges(self, facts: list[CheckedFact]) -> int:
        """Writes bridge facts and the passages each rests on.

        Args:
            facts: Checked bridge facts, refused ones included.

        Returns:
            How many were written.
        """
        with self._session.begin() as session:
            return _write(session, facts, self._version)

    def judged(
        self, within=None
    ) -> Iterator[tuple[int, list[PassageToExtract], CandidateFact, str]]:
        """Streams every stored fact beside the passages it was drawn from.

        Yields the fact's id, those passages, what the extractor proposed and
        the method that proposed it, which is everything a re-judgement
        needs: the model is not called again.

        Args:
            within: A condition narrowing which facts are read, or None.

        Yields:
            One fact's id, passages, candidate and method.
        """
        query = (
            select(
                Fact.id.label("fact_id"),
                Fact.kind,
                Fact.statement,
                Fact.extraction_method,
                FactPassage.sentence_ids,
                *_PASSAGE_COLUMNS,
                Document.language.label("document_language"),
            )
            .select_from(Fact)
            .join(FactPassage, FactPassage.fact_id == Fact.id)
            .join(Passage, Passage.id == FactPassage.passage_id)
            .join(Document, Document.sha256 == Passage.doc_sha256)
            # Never a fact the cap refused. `over_cap` is the only verdict
            # here that no check reaches: it says the passage had no room
            # left, which is a fact about the passage's budget and not about
            # this claim. A re-judgement asks the checks what they make of a
            # statement, they make nothing of that, and the row would come
            # back validated - quietly undoing EXTRACTION_MIN_OTHER_SHARE
            # across the corpus. `make extract-recap` is what re-applies it.
            .where(Fact.rejection_code.is_distinct_from(Rejection.OVER_CAP))
            # One row per passage, consecutive per fact, so a fact is read off
            # the run of rows carrying its id.
            .order_by(Fact.id, FactPassage.position)
            .execution_options(yield_per=_BATCH)
        )
        if within is not None:
            query = query.where(_resting(within))

        with self._session() as session:
            for fact_id, found in groupby(
                session.execute(query), key=lambda row: row.fact_id
            ):
                rows = list(found)
                yield (
                    fact_id,
                    [_passage(row, row.document_language) for row in rows],
                    CandidateFact(
                        statement=rows[0].statement,
                        sentences=tuple(rows[0].sentence_ids or ()),
                        kind=rows[0].kind,
                        passages=tuple(
                            Cited(
                                position=position,
                                sentences=tuple(row.sentence_ids or ()),
                            )
                            for position, row in enumerate(rows)
                        ),
                    ),
                    rows[0].extraction_method,
                )

    def rejudge(self, verdicts: list[tuple[int, CheckedFact]]) -> int:
        """Writes back what the checks read, leaving what the model wrote.

        The statement, the kind, the method and the model's own provenance
        are the record of one extraction and are never rewritten here. The
        spans in fact_passages are: they are what the checks resolved, and a
        fact keeping yesterday's offsets beside today's verdict cites text it
        was not judged on.

        Args:
            verdicts: Each fact's id beside its new verdict.

        Returns:
            How many rows were written.
        """
        if not verdicts:
            return 0
        with self._session.begin() as session:
            # The table rather than the entity: an executemany against the
            # mapped class is read as an ORM bulk update by primary key, which
            # wants the key among the values being set.
            session.execute(
                update(_FACTS).where(_FACTS.c.id == bindparam("row")),
                [
                    {
                        "row": fact_id,
                        "evidence_text": checked.evidence_text,
                        "validated": checked.validated,
                        "rejection_code": checked.rejection_code,
                        "validation_error": checked.validation_error,
                        "statement_predicates": checked.statement_predicates,
                        "evidence_predicates": checked.evidence_predicates,
                        "units_statement": checked.units_statement,
                        "units_added": checked.units_added,
                        "unresolved_references": checked.unresolved_references,
                        "spacy_model": checked.spacy_model,
                        "spacy_version": checked.spacy_version,
                        # Rewritten with the verdict it was reached under.
                        "settings_version": self._version,
                    }
                    for fact_id, checked in verdicts
                ],
            )
            # Cleared first: a passage the checks no longer resolve against
            # keeps no span, and its row is left saying so rather than saying
            # what last year's parse found.
            session.execute(
                update(_LINKS)
                .where(_LINKS.c.fact_id.in_([fact_id for fact_id, _ in verdicts]))
                .values(sentence_ids=None, evidence_start=None, evidence_end=None)
            )
            spans = [
                {
                    "row": fact_id,
                    "passage": cited.passage_id,
                    "sentence_ids": cited.sentence_ids,
                    "evidence_start": cited.start,
                    "evidence_end": cited.end,
                }
                for fact_id, checked in verdicts
                for cited in checked.citations
            ]
            if spans:
                session.execute(
                    update(_LINKS).where(
                        _LINKS.c.fact_id == bindparam("row"),
                        _LINKS.c.passage_id == bindparam("passage"),
                    ),
                    spans,
                )
        return len(verdicts)

    def passages_of(self, fact_id: int) -> list[int]:
        """The passages one fact rests on, in the order the model saw them.

        Args:
            fact_id: The fact to read.

        Returns:
            Their ids; one for an atomic fact, a summary and an outline, two
            or more for a bridge.
        """
        with self._session() as session:
            return list(
                session.scalars(
                    select(FactPassage.passage_id)
                    .where(FactPassage.fact_id == fact_id)
                    .order_by(FactPassage.position)
                )
            )

    def page(
        self,
        document: str | None = None,
        limit: int = 100,
        offset: int = 0,
        search: str | None = None,
        method: str | None = None,
        field: str | None = None,
        kind: str | None = None,
    ) -> tuple[int, list[StoredFact]]:
        """Reads one page of facts and the total behind it.

        Args:
            document: Only facts resting on a passage of this document.
            limit: How many rows to return.
            offset: How many to skip.
            search: Text to match, in the columns `field` names.
            method: Only facts drawn this way.
            field: Which columns `search` looks in.
            kind: Only facts of this kind.

        Returns:
            The total matching the filter, and one page of it.
        """
        listing = _filtered(
            _joined(
                select(
                    Fact.id,
                    Fact.statement,
                    Fact.evidence_text,
                    Fact.kind,
                    Fact.extraction_method,
                    Fact.validated,
                    Fact.rejection_code,
                    Fact.validation_error,
                    Fact.statement_predicates,
                    Fact.evidence_predicates,
                    Fact.units_added,
                    Fact.unresolved_references,
                )
            ).order_by(Passage.doc_sha256, Passage.ordinal, Fact.id),
            document,
            search,
            method,
            field,
            kind,
        )
        counting = _filtered(
            _joined(select(func.count())), document, search, method, field, kind
        )
        with self._session() as session:
            total = session.scalar(counting) or 0
            rows = session.execute(listing.limit(limit).offset(offset)).all()
            sources = self._sources(session, [row.id for row in rows])
        return total, [
            StoredFact(**row._asdict(), passages=sources.get(row.id, []))
            for row in rows
        ]

    @staticmethod
    def _sources(session, fact_ids: list[int]) -> dict[int, list[FactSource]]:
        """Reads the passages one page of facts rests on, by fact id."""
        if not fact_ids:
            return {}
        found: dict[int, list[FactSource]] = {}
        for row in session.execute(_SOURCES.where(FactPassage.fact_id.in_(fact_ids))):
            found.setdefault(row.fact_id, []).append(
                FactSource(
                    passage_id=row.passage_id,
                    doc_sha256=row.doc_sha256,
                    ordinal=row.ordinal,
                    page_from=row.page_from,
                    position=row.position,
                )
            )
        return found

    def quality(
        self,
        document: str | None = None,
        search: str | None = None,
        method: str | None = None,
        field: str | None = None,
        kind: str | None = None,
    ) -> FactQuality:
        """Measures how well extraction is doing, under the listing's filter.

        Rejections are grouped on the code rather than the message: a message
        carrying a measurement gives one bucket per measurement.

        The counts are over everything the filter selects, refused facts
        included - that rate is the measurement. The means are over the
        accepted ones only. They are there to answer whether the model is
        decomposing a passage or restating it, and a fact refused as
        `evidence_absent` carries no evidence at all: it enters both
        evidence means as a zero and pulls the ratio the README reads as
        "claims per statement against claims per cited sentence" towards a
        number describing the failures rather than the work.

        Args:
            document: Only facts resting on a passage of this document.
            search: Text to match, in the columns `field` names.
            method: Only facts drawn this way.
            field: Which columns `search` looks in.
            kind: Only facts of this kind.

        Returns:
            The figures over everything the filter selects.
        """
        totals = _filtered(
            _joined(
                select(
                    func.count().label("total"),
                    func.count().filter(Fact.validated).label("validated"),
                    func.count(func.distinct(_OPENS.passage_id)).label("passages"),
                    func.coalesce(
                        func.avg(func.length(Fact.statement)).filter(Fact.validated),
                        0.0,
                    ).label("statement_chars"),
                    func.coalesce(
                        func.avg(func.length(Fact.evidence_text)).filter(
                            Fact.validated
                        ),
                        0.0,
                    ).label("evidence_chars"),
                    func.coalesce(
                        func.avg(Fact.statement_predicates).filter(Fact.validated), 0.0
                    ).label("statement_predicates"),
                    func.coalesce(
                        func.avg(Fact.evidence_predicates).filter(Fact.validated), 0.0
                    ).label("evidence_predicates"),
                )
            ),
            document,
            search,
            method,
            field,
            kind,
        )
        reasons = _filtered(
            _joined(select(Fact.rejection_code, func.count()))
            .where(Fact.rejection_code.is_not(None))
            .group_by(Fact.rejection_code)
            .order_by(func.count().desc()),
            document,
            search,
            method,
            field,
            kind,
        )
        kinds = _filtered(
            _joined(select(Fact.kind, func.count())).group_by(Fact.kind),
            document,
            search,
            method,
            field,
            kind,
        )
        with self._session() as session:
            row = session.execute(totals).one()
            rejected = dict(session.execute(reasons).all())
            held = dict(session.execute(kinds).all())
        return FactQuality(
            total=row.total,
            validated=row.validated,
            mean_statement_chars=round(float(row.statement_chars), 1),
            mean_evidence_chars=round(float(row.evidence_chars), 1),
            mean_statement_predicates=round(float(row.statement_predicates), 2),
            mean_evidence_predicates=round(float(row.evidence_predicates), 2),
            facts_per_passage=round(row.total / row.passages, 1)
            if row.passages
            else 0.0,
            rejected=rejected,
            kinds=held,
        )
