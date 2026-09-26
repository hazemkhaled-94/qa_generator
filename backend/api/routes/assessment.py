"""The assessment phase, and the verdicts it produced.

One router for both, like `/questions` and unlike the other four stages:
the queue and its output share the word, so the five queue routes come
first from `stage_router` and the read routes are added after, which is
what keeps `/assessment/status` from being read as an id.

Nothing here can change a verdict. There is no PATCH, deliberately - a
judge's opinion is a record of what a model said, and a person who
disagrees with it has somewhere to say so already: `reviewed_verdict`,
through `review/` or the Facts and Questions pages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fastapi import APIRouter

from api.dependencies import assessment_catalog, assessment_queue
from api.params import Limit, Offset, SearchText
from api.routes.stage import StageAction, answered, stage_router
from assessment.config import Settings as AssessmentSettings
from assessment.models import AssessmentQuality, StoredAssessment
from assessment.templates import PROMPT_VERSION, TEMPLATES, metrics_of
from settings.store import resolved

#: The artefacts a verdict may be about, and the metrics it may be refused
#: by. Literals rather than free strings, so the OpenAPI document lists
#: them and the frontend's pickers cannot drift from what is accepted.
Kind = Literal["fact", "topic", "question"]
Metric = Literal["hallucination", "relevance", "qa_correctness", "summarization"]

#: Where a search looks. The artefact is "which fact was this about";
#: the explanation is "what did the judge say", and that is the half no
#: other listing in this API can search.
SearchField = Literal["artifact", "explanation", "both"]

#: This phase's own operation, and the only one of the seven a stage has
#: beyond the shared verbs that the API can answer at all: it moves rows and
#: loads nothing. The other six re-derive what a stage computed - vocabulary,
#: embeddings, the gates - and each of them needs the model or the language
#: pipeline that produced it, which `tests/static/test_api_stays_light.py`
#: keeps out of this process. Those stay on the command line.
#:
#: A router of its own, registered ahead of the queue routes in
#: `routes/__init__.py`: `POST /assessment/{action}` would otherwise match
#: `/assessment/enrol` first and refuse `enrol` as an unknown verb.
enrol_router = APIRouter(prefix="/assessment", tags=["assessment"])


@enrol_router.post("/enrol", status_code=202)
def enrol() -> StageAction:
    """Creates an assessment for every artefact that has none, queuing none.

    The dry run: nothing upstream enrols these rows, so this is what turns
    "judge the corpus" into a number before any of it is paid for. The rows
    land `new`, which no worker looks at, and `GET /assessment/status` then
    says how many model calls `start` would cost.
    """
    return answered("assessment", "enrol", assessment_queue.enrol())


router = stage_router(name="assessment", repository=assessment_queue)


@dataclass(frozen=True)
class AssessmentPage:
    """One page of verdicts, and the total behind it."""

    total: int
    assessments: list[StoredAssessment]


@dataclass(frozen=True)
class JudgePlan:
    """What the phase is configured to do, as the settings say.

    Served so a reader can see what will be judged and by whom before
    anything is. Every figure here is a setting, not a measurement.
    """

    enabled: bool
    judge_model: str | None
    kinds: list[str]
    sample: int
    prompt_version: str
    #: Which metrics each kind is asked, by kind. Read off the templates
    #: rather than listed here, so adding one shows up without an edit.
    metrics: dict[str, list[str]]


@router.get("/plan")
def plan() -> JudgePlan:
    """Reports what the assessment phase is configured to do.

    Read per request, stored overrides included: what the next run will
    judge rather than what the last one did.
    """
    settings = AssessmentSettings.load(resolved())
    return JudgePlan(
        enabled=settings.enabled,
        judge_model=settings.judge_model,
        kinds=list(settings.kinds),
        sample=settings.sample,
        prompt_version=PROMPT_VERSION,
        metrics={kind: list(metrics_of(kind)) for kind in TEMPLATES},
    )


@router.get("/quality")
def quality(kind: Kind | None = None) -> AssessmentQuality:
    """Reports what the judge made of the corpus.

    How much was judged, how much it approved, how each metric did, and -
    the one figure anybody should act on - how many artefacts the pipeline
    KEPT and the judge refused.
    """
    return assessment_catalog.quality(kind)


@router.get("")
def assessments(
    kind: Kind | None = None,
    approved: bool | None = None,
    metric: Metric | None = None,
    disagreements: bool = False,
    q: SearchText = None,
    field: SearchField = "both",
    limit: Limit = 50,
    offset: Offset = 0,
) -> AssessmentPage:
    """Lists the verdicts, newest first.

    `q` searches the artefact itself or what the judge said about it.
    Searching the explanations is the half nothing else here can do: until
    this phase there were no model opinions stored to search, so "which
    answers did it call unsupported, and in what words" had no query.

    `metric` narrows to the artefacts one named metric did NOT approve,
    which is how a reader asks "what did the hallucination check catch".

    `disagreements` narrows to the artefacts the pipeline kept and the
    judge refused. One direction only: an artefact the pipeline rejected
    and the judge approved is usually a rejection for something the judge
    was never asked about, and reporting it would bury the queue that
    matters.
    """
    total, rows = assessment_catalog.page(
        kind, approved, metric, disagreements, q, field, limit, offset
    )
    return AssessmentPage(total=total, assessments=rows)
