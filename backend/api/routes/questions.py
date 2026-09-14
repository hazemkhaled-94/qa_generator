"""The question generation stage, and the questions it produced.

One router for both, unlike every other stage, whose queue is under its own
verb and whose output is under a noun - `/extraction` and `/facts`. Here the
queue and the output share the word, so they share the router: the five queue
routes are declared first by `stage_router`, and the read routes are added
after it, which is what keeps `/questions/status` from being read as
`/questions/{question_id}` with a question id of `status`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fastapi import Query
from pydantic import BaseModel

from api.dependencies import question_catalog, questions_queue
from api.errors import ApiError, ErrorBody
from api.routes.stage import stage_router
from database.qa_generator import QuestionStatus
from question_generation.models import (
    QuestionDetail,
    QuestionQuality,
    StoredQuestion,
)

#: The columns `q` may look in, and the states a question may be filtered to
#: or moved to. Literals rather than free strings, so the OpenAPI document
#: lists them and the frontend's pickers cannot drift from what is accepted.
SearchField = Literal["question", "answer", "both"]
Decision = Literal["draft", "accepted", "rejected"]

router = stage_router(name="questions", repository=questions_queue)


@dataclass(frozen=True)
class QuestionPage:
    """One page of questions, and the total behind it."""

    total: int
    questions: list[StoredQuestion]


class QuestionVerdict(BaseModel):
    """What a person decided about one question.

    Only the three states the column holds. A person accepting clears the
    gate's reason, because the row is no longer rejected and a reason for a
    rejection that was overturned is a stale one.
    """

    status: Decision


@router.get("")
def questions(
    document: str | None = None,
    topic: int | None = None,
    q: str | None = Query(default=None, max_length=200),
    field: SearchField = "both",
    status: Decision | None = None,
    answerable: bool | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> QuestionPage:
    """Lists generated questions, including those a gate rejected."""
    total, rows = question_catalog.page(
        document, topic, limit, offset, q, status, answerable, field
    )
    return QuestionPage(total=total, questions=rows)


@router.get("/quality")
def quality(
    document: str | None = None,
    topic: int | None = None,
    q: str | None = Query(default=None, max_length=200),
    field: SearchField = "both",
    status: Decision | None = None,
    answerable: bool | None = None,
) -> QuestionQuality:
    """Reports how generation is doing, under the same filter.

    The numbers that say whether the questions are questions: how many
    cleared every gate, which gate stopped the rest, how the evidence is
    spread, and how much of the corpus's subject matter is covered at all.
    """
    return question_catalog.quality(document, topic, q, status, answerable, field)


@router.get("/{question_id}", responses={404: {"model": ErrorBody}})
def question(question_id: int) -> QuestionDetail:
    """Reads one question with the facts it was written from.

    Raises:
        ApiError: 404 `unknown_question` if no question has that id.
    """
    found = question_catalog.detail(question_id)
    if found is None:
        raise ApiError(404, "unknown_question", "no question with that id")
    return found


@router.patch("/{question_id}", responses={404: {"model": ErrorBody}})
def decide(question_id: int, verdict: QuestionVerdict) -> StoredQuestion:
    """Accepts or rejects one question.

    A rejected question keeps its row whoever rejected it: the share that
    was thrown away is the evidence behind the coverage report, and a
    deleted row is a drop nobody can count.

    Raises:
        ApiError: 404 `unknown_question` if no question has that id.
    """
    decided = question_catalog.decide(question_id, QuestionStatus(verdict.status))
    if decided is None:
        raise ApiError(404, "unknown_question", "no question with that id")
    return decided
