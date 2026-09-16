"""Rows to test against, with only the columns the schema insists on.

Everything optional is left out, so a test that cares about a column sets it
and a reader can tell which ones a test is actually about.
"""

from __future__ import annotations

import hashlib
from typing import Any

from database.qa_generator import (
    Document,
    Fact,
    FactPassage,
    IngestEvent,
    Passage,
    PassageTopic,
    Question,
    QuestionFact,
    Topic,
)


def digest(seed: str = "a") -> str:
    """Builds a distinct digest from a short seed.

    A real SHA-256, not a repeated letter: the routes check a path
    parameter against `[0-9a-f]{64}` before it becomes an object key, so a
    stand-in that is not hex is refused as malformed rather than unknown.
    """
    return hashlib.sha256(seed.encode()).hexdigest()


def document(sha256: str | None = None, **columns: Any) -> Document:
    """A stored document, parsed and chunked by nothing yet."""
    return Document(
        sha256=sha256 or digest(),
        mime_type="application/pdf",
        **columns,
    )


def passage(doc_sha256: str, ordinal: int = 1, **columns: Any) -> Passage:
    """One passage of a document."""
    return Passage(
        doc_sha256=doc_sha256,
        ordinal=ordinal,
        text=columns.pop("text", "The device weighs 4 kg."),
        doc_item_refs=columns.pop("doc_item_refs", []),
        **columns,
    )


def fact(passage_id: int, *rests_on: int, **columns: Any) -> Fact:
    """One checked fact, validated unless a test says otherwise.

    Args:
        passage_id: The passage it rests on, written at position 0.
        *rests_on: Further passages, for a bridge, at the positions they are
            given in. Each cites sentence 0 over an empty span.
        **columns: Anything else the row takes. `evidence_sentence_ids`,
            `evidence_start` and `evidence_end` name the first passage's
            citation, which is where they used to live.

    Returns:
        The fact, with its link rows attached.
    """
    statement = columns.pop("statement", "The device weighs 4 kg.")
    return Fact(
        passage_links=[
            FactPassage(
                passage_id=passage_id,
                position=0,
                sentence_ids=columns.pop("evidence_sentence_ids", [0]),
                evidence_start=columns.pop("evidence_start", 0),
                evidence_end=columns.pop("evidence_end", len(statement)),
            ),
            *(
                FactPassage(
                    passage_id=one,
                    position=position,
                    sentence_ids=[0],
                    evidence_start=0,
                    evidence_end=0,
                )
                for position, one in enumerate(rests_on, start=1)
            ),
        ],
        statement=statement,
        evidence_text=columns.pop("evidence_text", statement),
        extraction_method=columns.pop("extraction_method", "llm"),
        # Stated rather than defaulted: the verdict and the code have to
        # agree, and the column defaults to false while the code defaults
        # to NULL.
        validated=columns.pop("validated", True),
        units_statement=columns.pop("units_statement", []),
        units_added=columns.pop("units_added", []),
        unresolved_references=columns.pop("unresolved_references", []),
        **columns,
    )


def question(**columns: Any) -> Question:
    """One generated question."""
    return Question(
        question_text=columns.pop("question_text", "What does the device weigh?"),
        answerable=columns.pop("answerable", True),
        language=columns.pop("language", "en"),
        **columns,
    )


def link(question_id: int, fact_id: int) -> QuestionFact:
    """The link a question is kept alive by."""
    return QuestionFact(question_id=question_id, fact_id=fact_id)


def topic(**columns: Any) -> Topic:
    """A request row, which is what a topic table starts as."""
    return Topic(top_terms=columns.pop("top_terms", []), **columns)


def fitted(topic_index: int = 0, **columns: Any) -> Topic:
    """A topic a fit produced, rather than a request for one.

    The distinction is the whole reason question generation carries a base
    predicate: both live in this table and only one of them is a subject.
    """
    return Topic(
        topic_index=topic_index,
        language=columns.pop("language", "en"),
        top_terms=columns.pop("top_terms", ["device", "weight"]),
        status=columns.pop("status", "modelled"),
        **columns,
    )


def membership(passage_id: int, topic_id: int, weight: float = 0.9) -> PassageTopic:
    """How strongly one passage belongs to one topic."""
    return PassageTopic(passage_id=passage_id, topic_id=topic_id, weight=weight)


def event(**columns: Any) -> IngestEvent:
    """One upload attempt, whatever became of it."""
    return IngestEvent(
        submitted_filename=columns.pop("submitted_filename", "report.pdf"),
        size_bytes=columns.pop("size_bytes", 1024),
        outcome=columns.pop("outcome", "stored"),
        **columns,
    )
