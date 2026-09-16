"""Database access for the extraction service."""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar, cast

from sqlalchemy import (
    Select,
    Table,
    bindparam,
    delete,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.orm import InstrumentedAttribute

from database.qa_generator import (
    Document,
    Fact,
    FactKind,
    FactPassage,
    Passage,
    Status,
)
from database.qa_generator.passage_topics import DOMINANT
from database.qa_generator.repository import Repository, matching
from extraction.models import (
    CandidateFact,
    CheckedFact,
    Cited,
    FactQuality,
    PassageToExtract,
    StoredFact,
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

#: What one passage of a bridge group is read as. The same columns `claim`
#: reads, so a group and a queued passage are the same object.
_PASSAGE_COLUMNS = (
    Passage.id,
    Passage.text,
    Passage.section_path,
    Passage.block_type,
    Passage.sentences,
    Passage.table_cells,
    Passage.doc_sha256,
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
    """Reads one row of `_PASSAGE_COLUMNS` as a passage to extract from."""
    return PassageToExtract(
        id=row.id,
        text=row.text,
        section_path=row.section_path,
        block_type=row.block_type,
        language=language,
        sentences=_sentences(row.text, row.sentences),
        table_cells=row.table_cells or [],
        doc_sha256=row.doc_sha256,
    )


def _filtered(query, document, search, method, field, kind=None):
    """Applies the document, text, method and kind filters to a fact query.

    One place, so a listing and its count cannot disagree about what they are
    looking at. The query must already join Passage.
    """
    if document:
        query = query.where(Passage.doc_sha256 == document)
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
    """Joins the passage every fact filter and ordering needs."""
    return query.select_from(Fact).join(Passage, Fact.passage_id == Passage.id)


class PassageQueue(RowQueue):
    """Reads the extraction queue and records what became of each passage."""

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

    def claim(self) -> PassageToExtract | None:
        """Takes the next unread passage off the queue.

        Returns:
            The passage, or None when nothing is queued.
        """
        claimed = self._claim(*_PASSAGE_COLUMNS)
        if claimed is None:
            return None
        with self._session() as session:
            language = session.scalar(
                select(Document.language).where(Document.sha256 == claimed.doc_sha256)
            )
        return _passage(claimed, language)

    def store(self, passage_id: int, facts: list[CheckedFact]) -> int:
        """Replaces one passage's facts, in one transaction, and finishes it.

        Scoped to the passage, so two workers on two passages of the same
        document do not delete each other's results. Bridge facts anchored
        here are left alone: they are the bridge pass's to write and to
        replace.

        Args:
            passage_id: The passage that was read.
            facts: Everything it yielded, refused facts included.

        Returns:
            How many facts were written.
        """
        with self._session.begin() as session:
            session.execute(
                delete(Fact).where(
                    Fact.passage_id == passage_id, Fact.kind != FactKind.BRIDGE
                )
            )
            if facts:
                session.execute(insert(Fact), [_row(fact) for fact in facts])
            self._finish(passage_id, session=session)
        return len(facts)

    def counts(self) -> dict[str, int]:
        """Reports what this service owns, for the status panel."""
        with self._session() as session:
            total = session.scalar(select(func.count()).select_from(Fact))
            valid = session.scalar(
                select(func.count()).select_from(Fact).where(Fact.validated)
            )
        return {**self.counts_by_status(), "facts": total, "validated": valid}


def _row(fact: CheckedFact) -> dict:
    """Turns one checked fact into the columns the facts table holds."""
    return {
        "passage_id": fact.passage_id,
        "kind": fact.kind,
        "statement": fact.statement,
        "evidence_text": fact.evidence_text,
        "evidence_sentence_ids": fact.evidence_sentence_ids,
        "evidence_start": fact.evidence_start,
        "evidence_end": fact.evidence_end,
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
    }


class FactCatalog(Repository):
    """Reads back the facts extraction produced, and writes the bridges.

    Separate from the queue: the API serves these and never claims a row.
    """

    def by_topic(self, within=None) -> Iterator[tuple[int, list[PassageToExtract]]]:
        """Streams each topic's passages, in document and reading order.

        A passage belongs to the topic it carries most strongly, which is
        what puts two of them in the same group.

        Args:
            within: A condition narrowing which passages are read, or None.

        Yields:
            One topic's id and its passages.
        """
        query = (
            select(*_PASSAGE_COLUMNS, Document.language, DOMINANT.c.topic_id)
            .select_from(Passage)
            .join(Document, Document.sha256 == Passage.doc_sha256)
            .join(DOMINANT, DOMINANT.c.passage_id == Passage.id)
            .where(Passage.sentences.is_not(None))
            .order_by(DOMINANT.c.topic_id, Passage.doc_sha256, Passage.ordinal)
        )
        if within is not None:
            query = query.where(within)

        with self._session() as session:
            held: list[PassageToExtract] = []
            current: int | None = None
            for row in session.execute(query):
                if row.topic_id != current:
                    if held:
                        yield cast("int", current), held
                    current, held = row.topic_id, []
                held.append(_passage(row, row.language))
            if held:
                yield cast("int", current), held

    def clear_bridges(self, within=None) -> int:
        """Deletes the bridge facts a fresh pass is about to replace.

        Args:
            within: A condition on Passage narrowing which bridges go, or
                None for every one of them.

        Returns:
            How many were deleted.
        """
        anchored = (
            select(Fact.id)
            .select_from(Fact)
            .join(Passage, Passage.id == Fact.passage_id)
            .where(Fact.kind == FactKind.BRIDGE)
        )
        if within is not None:
            anchored = anchored.where(within)
        with self._session.begin() as session:
            return session.execute(delete(Fact).where(Fact.id.in_(anchored))).rowcount

    def add_bridges(self, facts: list[CheckedFact]) -> int:
        """Writes bridge facts and the passages each rests on.

        Args:
            facts: Checked bridge facts, refused ones included.

        Returns:
            How many were written.
        """
        if not facts:
            return 0
        with self._session.begin() as session:
            # sort_by_parameter_order: the links below are matched to the
            # facts by position, so the ids have to come back in the order
            # they went in rather than in whatever order the insert chose.
            ids = session.scalars(
                insert(Fact).returning(Fact.id, sort_by_parameter_order=True),
                [_row(fact) for fact in facts],
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
                Fact.evidence_sentence_ids,
                Fact.extraction_method,
                *_PASSAGE_COLUMNS,
                Document.language,
            )
            .select_from(Fact)
            .join(Passage, Fact.passage_id == Passage.id)
            .join(Document, Document.sha256 == Passage.doc_sha256)
            .order_by(Fact.id)
            .execution_options(yield_per=_BATCH)
        )
        if within is not None:
            query = query.where(within)

        bridges = self._bridge_groups()
        with self._session() as session:
            for row in session.execute(query):
                group = bridges.get(row.fact_id)
                yield (
                    row.fact_id,
                    [passage for passage, _ in group]
                    if group
                    else [_passage(row, row.language)],
                    CandidateFact(
                        statement=row.statement,
                        sentences=tuple(row.evidence_sentence_ids or ()),
                        kind=row.kind,
                        passages=tuple(
                            Cited(position=position, sentences=cited)
                            for position, (_, cited) in enumerate(group or ())
                        ),
                    ),
                    row.extraction_method,
                )

    def _bridge_groups(
        self,
    ) -> dict[int, list[tuple[PassageToExtract, tuple[int, ...]]]]:
        """Reads every bridge's passages and citations, in the order shown.

        Held whole rather than streamed, and never narrowed: a bridge is one
        call per topic group, so there are orders of magnitude fewer of these
        than of facts, and a narrowing on the fact would drop the half of a
        group that sits in another document.

        Returns:
            Per fact id, one (passage, cited sentence indices) per passage.
            The indices are empty on a bridge drawn before prompt version 2,
            which recorded none.
        """
        query = (
            select(
                FactPassage.fact_id,
                FactPassage.sentence_ids,
                *_PASSAGE_COLUMNS,
                Document.language,
            )
            .select_from(FactPassage)
            .join(Passage, Passage.id == FactPassage.passage_id)
            .join(Document, Document.sha256 == Passage.doc_sha256)
            .order_by(FactPassage.fact_id, FactPassage.position)
        )

        groups: dict[int, list[tuple[PassageToExtract, tuple[int, ...]]]] = {}
        with self._session() as session:
            for row in session.execute(query):
                groups.setdefault(row.fact_id, []).append(
                    (_passage(row, row.language), tuple(row.sentence_ids or ()))
                )
        return groups

    def rejudge(self, verdicts: list[tuple[int, CheckedFact]]) -> int:
        """Writes back what the checks read, leaving what the model wrote.

        The statement, the kind, the method and the model's own provenance
        are the record of one extraction and are never rewritten here.

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
                        "evidence_sentence_ids": checked.evidence_sentence_ids,
                        "evidence_start": checked.evidence_start,
                        "evidence_end": checked.evidence_end,
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
                    }
                    for fact_id, checked in verdicts
                ],
            )
        return len(verdicts)

    def passages_of(self, fact_id: int) -> list[int]:
        """The passages one bridge fact rests on, anchor first.

        Args:
            fact_id: The fact to read.

        Returns:
            Their ids in the order the model was shown them; empty for a
            fact resting on its anchor alone.
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
            document: Only facts of this document's passages.
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
                    Fact.passage_id,
                    Passage.doc_sha256,
                    Passage.ordinal,
                    Passage.page_from,
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
        return total, [StoredFact(**row._asdict()) for row in rows]

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

        Args:
            document: Only facts of this document's passages.
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
                    func.count(func.distinct(Fact.passage_id)).label("passages"),
                    func.coalesce(func.avg(func.length(Fact.statement)), 0.0).label(
                        "statement_chars"
                    ),
                    func.coalesce(func.avg(func.length(Fact.evidence_text)), 0.0).label(
                        "evidence_chars"
                    ),
                    func.coalesce(func.avg(Fact.statement_predicates), 0.0).label(
                        "statement_predicates"
                    ),
                    func.coalesce(func.avg(Fact.evidence_predicates), 0.0).label(
                        "evidence_predicates"
                    ),
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
