"""Database access for the extraction service."""

from __future__ import annotations

from typing import ClassVar

from sqlalchemy import Select, delete, func, insert, select
from sqlalchemy.orm import InstrumentedAttribute

from database.qa_generator import Document, Fact, Passage, Status
from database.qa_generator.repository import Repository, matching
from extraction.models import (
    CheckedFact,
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


def _filtered(query, document, search, method, field):
    """Applies the document, text and method filters to a fact query.

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
        """Takes the next unread passage off the queue."""
        claimed = self._claim(
            Passage.id,
            Passage.text,
            Passage.section_path,
            Passage.block_type,
            Passage.sentences,
            Passage.table_cells,
            Passage.doc_sha256,
        )
        if claimed is None:
            return None
        with self._session() as session:
            language = session.scalar(
                select(Document.language).where(Document.sha256 == claimed.doc_sha256)
            )
        return PassageToExtract(
            id=claimed.id,
            text=claimed.text,
            section_path=claimed.section_path,
            block_type=claimed.block_type,
            language=language,
            sentences=[
                Sentence(
                    index=entry["i"],
                    start=entry["start"],
                    end=entry["end"],
                    text=claimed.text[entry["start"] : entry["end"]],
                    predicates=entry.get("predicates", 0),
                )
                for entry in claimed.sentences or []
            ],
            table_cells=claimed.table_cells or [],
        )

    def store(self, passage_id: int, facts: list[CheckedFact]) -> int:
        """Replaces one passage's facts, in one transaction, and finishes it.

        Scoped to the passage, so two workers on two passages of the same
        document do not delete each other's results.
        """
        with self._session.begin() as session:
            session.execute(delete(Fact).where(Fact.passage_id == passage_id))
            if facts:
                session.execute(
                    insert(Fact),
                    [
                        {
                            "passage_id": fact.passage_id,
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
                        for fact in facts
                    ],
                )
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


class FactCatalog(Repository):
    """Reads back the facts extraction produced.

    Separate from the queue: the API serves these and never claims a row.
    """

    def page(
        self,
        document: str | None = None,
        limit: int = 100,
        offset: int = 0,
        search: str | None = None,
        method: str | None = None,
        field: str | None = None,
    ) -> tuple[int, list[StoredFact]]:
        """Reads one page of facts and the total behind it."""
        listing = _filtered(
            _joined(
                select(
                    Fact.id,
                    Fact.statement,
                    Fact.evidence_text,
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
        )
        counting = _filtered(
            _joined(select(func.count())), document, search, method, field
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
    ) -> FactQuality:
        """Measures how well extraction is doing, under the listing's filter.

        Rejections are grouped on the code rather than the message: a message
        carrying a measurement gives one bucket per measurement.
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
        )
        with self._session() as session:
            row = session.execute(totals).one()
            rejected = dict(session.execute(reasons).all())
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
        )
