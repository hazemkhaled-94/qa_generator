"""One artefact's chain: everything it is joined to, in pipeline order.

Walks upwards from the artefact asked about - a question to its facts, a
fact to its passages, a passage to its document - and reports each stage's
artefacts under the stage that produced them. "What came of this" is the
listing every stage page already answers with a filter, and is not here.

A step is capped and carries its own total, because the fan-out is not
symmetric: a question rests on a handful of facts, and one topic holds
every passage of the corpus that scored above the weight floor.

The one reader that crosses every service. It sits here rather than in a
package of its own because the api is already the composition root and the
only layer above all seven, and because it adds no dependency: these are
the same SQLAlchemy models the service repositories read.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, select

from database.qa_generator import (
    Assessment,
    Document,
    Fact,
    FactPassage,
    IngestEvent,
    Passage,
    PassageTopic,
    Question,
    QuestionFact,
    Topic,
    sessions,
)
from question_generation.types import GATES

#: Every artefact kind, in the order a corpus moves through them, with the
#: stage that produces it and the position that stage holds in the pipeline.
#: The position is what orders a chain in every surface that shows one.
KINDS: tuple[tuple[str, str, int], ...] = (
    ("document", "parsing", 2),
    ("passage", "chunking", 3),
    ("fact", "extraction", 4),
    ("topic", "topic_modelling", 5),
    ("question", "question_generation", 6),
)

#: The stage and the position, by kind.
WHERE_FROM = {kind: (stage, position) for kind, stage, position in KINDS}

#: How much of a passage or a statement one artefact carries.
_LABEL = 240

#: How many artefacts one step lists. A topic holds every passage that
#: scored above the weight floor, which for this corpus is hundreds.
_PER_STEP = 20


@dataclass(frozen=True)
class Artifact:
    """One artefact in a chain, as the stage that produced it left it."""

    kind: str
    id: str
    label: str
    #: What the stage decided about it: a status, or a verdict.
    verdict: str | None
    #: Why, where it was refused.
    reason: str | None
    #: What the evaluation phase made of it, where it has run.
    judge: str | None
    run_id: str | None
    trace_id: str | None
    span_id: str | None
    at: datetime | None


@dataclass(frozen=True)
class Step:
    """One stage of the pipeline, and what this chain holds at it."""

    #: Where the stage sits in the pipeline, counting ingestion as 1.
    position: int
    stage: str
    kind: str
    #: How many artefacts of this kind the chain reaches.
    total: int
    #: The first `_PER_STEP` of them.
    artifacts: list[Artifact]


@dataclass(frozen=True)
class Gate:
    """One gate that read a question, at its fixed position."""

    position: int
    name: str
    passed: bool


@dataclass(frozen=True)
class Lineage:
    """Everything one artefact is joined to, in pipeline order."""

    kind: str
    id: str
    #: One entry per stage that produced something in this chain, in the
    #: order the pipeline runs them. A stage this chain reaches nothing at
    #: is absent rather than empty.
    steps: list[Step]
    #: The gates that read it, in the order they read it. Questions only.
    gates: list[Gate]
    #: Named where the row records no sequence, which is every question
    #: written before the column existed.
    gates_recorded: bool


def _text(value: str | None, fallback: str = "") -> str:
    """One artefact's label, cut to a length a table can hold."""
    cleaned = " ".join((value or fallback).split())
    return cleaned[:_LABEL] + "…" if len(cleaned) > _LABEL else cleaned


def _judgements(session, kinds: dict[str, list]) -> dict[tuple[str, str], str]:
    """What the judge said about each artefact, by kind and id."""
    column = {
        "fact": Assessment.fact_id,
        "topic": Assessment.topic_id,
        "question": Assessment.question_id,
    }
    found: dict[tuple[str, str], str] = {}
    for kind, ids in kinds.items():
        if kind not in column or not ids:
            continue
        rows = session.execute(
            select(column[kind], Assessment.approved).where(
                column[kind].in_(ids), Assessment.approved.is_not(None)
            )
        ).all()
        for row_id, approved in rows:
            found[(kind, str(row_id))] = "approved" if approved else "refused"
    return found


def _ancestors(session, kind: str, row_id: str, key: int | None) -> dict[str, list]:
    """Resolves one artefact to every artefact it was produced from."""
    questions = [key] if kind == "question" and key is not None else []
    facts = [key] if kind == "fact" and key is not None else []
    topics = [key] if kind == "topic" and key is not None else []
    passages = [key] if kind == "passage" and key is not None else []
    documents = [row_id] if kind == "document" else []

    if questions:
        facts = list(
            session.scalars(
                select(QuestionFact.fact_id).where(
                    QuestionFact.question_id.in_(questions)
                )
            ).all()
        )
    if facts:
        passages = list(
            session.scalars(
                select(FactPassage.passage_id)
                .where(FactPassage.fact_id.in_(facts))
                .distinct()
            ).all()
        )
    if topics and not passages:
        passages = list(
            session.scalars(
                select(PassageTopic.passage_id)
                .where(PassageTopic.topic_id.in_(topics))
                .distinct()
            ).all()
        )
    if passages:
        documents = list(
            session.scalars(
                select(Passage.doc_sha256).where(Passage.id.in_(passages)).distinct()
            ).all()
        )
        if not topics:
            topics = list(
                session.scalars(
                    select(PassageTopic.topic_id)
                    .where(PassageTopic.passage_id.in_(passages))
                    .distinct()
                ).all()
            )
    return {
        "document": documents,
        "passage": passages,
        "fact": facts,
        "topic": topics,
        "question": questions,
    }


def _documents(session, ids: list) -> list[Artifact]:
    """The documents, with the upload that put each one there."""
    if not ids:
        return []
    uploaded = (
        select(
            IngestEvent.sha256.label("sha256"),
            func.min(IngestEvent.submitted_at).label("at"),
        )
        .where(IngestEvent.sha256.in_(ids))
        .group_by(IngestEvent.sha256)
        .subquery()
    )
    rows = session.execute(
        select(Document, uploaded.c.at)
        .outerjoin(uploaded, uploaded.c.sha256 == Document.sha256)
        .where(Document.sha256.in_(ids))
        .order_by(Document.sha256)
    ).all()
    return [
        Artifact(
            kind="document",
            id=row.sha256,
            label=_text(row.title, row.sha256[:12]),
            verdict=row.parse_status,
            reason=row.parse_error,
            judge=None,
            run_id=row.parse_run_id,
            trace_id=row.parse_trace_id,
            span_id=row.parse_span_id,
            at=at,
        )
        for row, at in rows
    ]


def _passages(session, ids: list) -> list[Artifact]:
    """The passages, in reading order within each document."""
    if not ids:
        return []
    rows = session.scalars(
        select(Passage)
        .where(Passage.id.in_(ids))
        .order_by(Passage.doc_sha256, Passage.ordinal)
    ).all()
    return [
        Artifact(
            kind="passage",
            id=str(row.id),
            label=_text(row.text),
            verdict=row.extract_status,
            reason=row.extract_error,
            judge=None,
            run_id=row.run_id,
            trace_id=row.trace_id,
            span_id=row.span_id,
            at=None,
        )
        for row in rows
    ]


def _facts(session, ids: list, judged: dict) -> list[Artifact]:
    """The facts, oldest first."""
    if not ids:
        return []
    rows = session.scalars(
        select(Fact).where(Fact.id.in_(ids)).order_by(Fact.created_at, Fact.id)
    ).all()
    return [
        Artifact(
            kind="fact",
            id=str(row.id),
            label=_text(row.statement),
            verdict="validated" if row.validated else (row.rejection_code or "refused"),
            reason=row.validation_error,
            judge=judged.get(("fact", str(row.id))),
            run_id=row.run_id,
            trace_id=row.trace_id,
            span_id=row.span_id,
            at=row.created_at,
        )
        for row in rows
    ]


def _topics(session, ids: list, judged: dict) -> list[Artifact]:
    """The topics, by language and then by index."""
    if not ids:
        return []
    rows = session.scalars(
        select(Topic)
        .where(Topic.id.in_(ids))
        .order_by(Topic.language, Topic.topic_index)
    ).all()
    return [
        Artifact(
            kind="topic",
            id=str(row.id),
            label=_text(row.label, ", ".join(row.top_terms or [])),
            verdict=row.status,
            reason=row.error,
            judge=judged.get(("topic", str(row.id))),
            run_id=row.run_id,
            trace_id=row.trace_id,
            span_id=row.span_id,
            at=row.fitted_at,
        )
        for row in rows
    ]


def _questions(session, ids: list, judged: dict) -> list[Artifact]:
    """The questions, oldest first."""
    if not ids:
        return []
    rows = session.scalars(
        select(Question)
        .where(Question.id.in_(ids))
        .order_by(Question.created_at, Question.id)
    ).all()
    return [
        Artifact(
            kind="question",
            id=str(row.id),
            label=_text(row.question_text),
            verdict=row.status,
            reason=row.rejected_reason,
            judge=judged.get(("question", str(row.id))),
            run_id=row.run_id,
            trace_id=row.trace_id,
            span_id=row.span_id,
            at=row.created_at,
        )
        for row in rows
    ]


def _gates(session, row_id: str) -> tuple[list[Gate], bool]:
    """Which gates read one question, and whether the row records them.

    The checker returns on the first failure, so a gate that refused the
    question is the last one it ran. Every gate before it passed, and the
    gates after it never read the question at all.
    """
    row = session.execute(
        select(Question.gates_ran, Question.status).where(Question.id == int(row_id))
    ).first()
    if row is None or row.gates_ran is None:
        return [], False
    ran = list(row.gates_ran)
    refused = row.status == "rejected" and bool(ran)
    return [
        Gate(
            position=GATES.index(name) + 1 if name in GATES else len(GATES) + 1,
            name=name,
            passed=not (refused and index == len(ran) - 1),
        )
        for index, name in enumerate(ran)
    ], True


def of(kind: str, row_id: str) -> Lineage | None:
    """Everything one artefact is joined to, in pipeline order.

    Args:
        kind: document, passage, fact, topic or question.
        row_id: The artefact's key - a digest for a document, an id for
            everything else.

    Returns:
        The chain, or None when nothing holds that id.
    """
    if kind not in WHERE_FROM:
        return None
    # A document is keyed by its digest; everything else by a row id, and a
    # row id that is not a number names nothing rather than raising.
    key: int | None = None
    if kind != "document":
        if not row_id.isdigit():
            return None
        key = int(row_id)
    with sessions()() as session:
        found = _ancestors(session, kind, row_id, key)
        if not found[kind]:
            return None
        judged = _judgements(session, found)
        held = {
            "document": _documents(session, found["document"][:_PER_STEP]),
            "passage": _passages(session, found["passage"][:_PER_STEP]),
            "fact": _facts(session, found["fact"][:_PER_STEP], judged),
            "topic": _topics(session, found["topic"][:_PER_STEP], judged),
            "question": _questions(session, found["question"][:_PER_STEP], judged),
        }
        if not any(held.values()):
            return None
        gates, recorded = _gates(session, row_id) if kind == "question" else ([], False)

    steps = [
        Step(
            position=position,
            stage=stage,
            kind=at,
            total=len(found[at]),
            artifacts=held[at],
        )
        for at, stage, position in KINDS
        if held[at]
    ]
    return Lineage(
        kind=kind, id=row_id, steps=steps, gates=gates, gates_recorded=recorded
    )
