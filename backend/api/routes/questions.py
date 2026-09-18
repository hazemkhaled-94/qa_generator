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
from question_generation.config import Settings as QuestionSettings
from question_generation.models import (
    QuestionDetail,
    QuestionQuality,
    StoredQuestion,
)
from settings.store import resolved

#: The columns `q` may look in, and the states a question may be filtered to
#: or moved to. Literals rather than free strings, so the OpenAPI document
#: lists them and the frontend's pickers cannot drift from what is accepted.
SearchField = Literal["question", "answer", "both"]
Decision = Literal["draft", "accepted", "rejected"]

#: The three criteria a question is classified by, and the band they feed.
#: Each is a filter on the listing and a column of the quality report, so a
#: reader can ask for the cross-document ones or the hard ones alone.
PassageScope = Literal["single_passage", "multi_passage"]
DocumentScope = Literal["single_document", "cross_document"]
TopicScope = Literal["single_topic", "multi_topic"]
Band = Literal["easy", "medium", "hard"]

#: What a question asks for, and what shape of answer that wants. Requested
#: before the question is written, so a reader can ask for the reasons or the
#: comparisons alone and a report can say which kinds came out.
QuestionType = Literal[
    "factoid",
    "definition",
    "entity",
    "enumeration",
    "condition",
    "reason",
    "procedure",
    "consequence",
    "comparison",
    "aggregation",
    "temporal",
]
AnswerForm = Literal["value", "list", "explanation"]

router = stage_router(name="questions", repository=questions_queue)


@dataclass(frozen=True)
class GenerationPlan:
    """What generation was asked to write, as the settings say.

    Served so a reader can put the mix that was asked for beside the mix that
    came out. Every figure here is a setting, not a measurement.
    """

    types: dict[str, int]
    difficulty: dict[str, int]
    followup_types: list[str]
    per_topic: int
    unanswerable_share: float
    followup_share: float
    max_followups: int
    answer_chars: dict[str, list[int]]


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
    passage_scope: PassageScope | None = None,
    document_scope: DocumentScope | None = None,
    topic_scope: TopicScope | None = None,
    difficulty: Band | None = None,
    planned_difficulty: Band | None = None,
    question_type: QuestionType | None = None,
    answer_form: AnswerForm | None = None,
    follows: bool | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> QuestionPage:
    """Lists generated questions, including those a gate rejected.

    `follows` narrows to the follow-ups or to the roots: true for questions
    asked after another, false for the ones that start a thread.

    `difficulty` is the band the question turned out to be and
    `planned_difficulty` is the one the plan asked for; the two disagreeing is
    what says how often a wide sample produced a wide question.
    """
    total, rows = question_catalog.page(
        document,
        topic,
        limit,
        offset,
        q,
        status,
        answerable,
        field,
        passage_scope=passage_scope,
        document_scope=document_scope,
        topic_scope=topic_scope,
        difficulty=difficulty,
        planned_difficulty=planned_difficulty,
        question_type=question_type,
        answer_form=answer_form,
        follows=follows,
    )
    return QuestionPage(total=total, questions=rows)


@router.get("/plan")
def plan() -> GenerationPlan:
    """Reports what generation is configured to write.

    The mix of types and bands, the shares, and the answer bounds each form
    is held to. Read per request, stored values included: what the next run
    will do rather than what the last one did.
    """
    settings = QuestionSettings.load(resolved())
    return GenerationPlan(
        types=settings.type_mix,
        difficulty=settings.difficulty_mix,
        followup_types=list(settings.followup_types),
        per_topic=settings.per_topic,
        unanswerable_share=settings.unanswerable_share,
        followup_share=settings.followup_share,
        max_followups=settings.max_followups,
        answer_chars={
            form: list(bounds) for form, bounds in settings.answer_chars.items()
        },
    )


@router.get("/quality")
def quality(
    document: str | None = None,
    topic: int | None = None,
    q: str | None = Query(default=None, max_length=200),
    field: SearchField = "both",
    status: Decision | None = None,
    answerable: bool | None = None,
    passage_scope: PassageScope | None = None,
    document_scope: DocumentScope | None = None,
    topic_scope: TopicScope | None = None,
    difficulty: Band | None = None,
    planned_difficulty: Band | None = None,
    question_type: QuestionType | None = None,
    answer_form: AnswerForm | None = None,
    follows: bool | None = None,
) -> QuestionQuality:
    """Reports how generation is doing, under the same filter.

    The numbers that say whether the questions are questions: how many
    cleared every gate, which gate stopped the rest, which kinds were written,
    how the three criteria are spread, and how much of the corpus's subject
    matter is covered at all.
    """
    return question_catalog.quality(
        document,
        topic,
        q,
        status,
        answerable,
        field,
        passage_scope=passage_scope,
        document_scope=document_scope,
        topic_scope=topic_scope,
        difficulty=difficulty,
        planned_difficulty=planned_difficulty,
        question_type=question_type,
        answer_form=answer_form,
        follows=follows,
    )


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
