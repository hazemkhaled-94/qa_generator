"""The corpus as one thing, for a caller who does not want six calls.

Every stage already answers its own five verbs, and nothing here replaces
them. What this adds is the two controls a stage route cannot have, because
neither is about one stage:

* **Run it end to end.** Stages have to be sequenced - chunking needs a
  parsed document - and sequencing means waiting for one to drain before
  starting the next. A request cannot hold that wait: extraction over this
  corpus was measured in hours. So `run` asks the orchestrator, which is the
  component whose whole job is deciding when a stage should run, and returns
  the run it started. See `api/orchestrator.py`.

* **Act on all of them at once.** After a model outage the failures are
  spread over five queues, and visiting five routes to retry them is five
  chances to forget one.

`GET /pipeline` is the one call that answers "what is the state of this
corpus" - every queue, the run in flight, and whether anything is set to
start one by itself. It is what a caller who is not using the frontend polls.

**Without an orchestrator this degrades rather than fails.** `DAGSTER_URL`
unset, or a webserver that is down, leaves `run` refused with a 503 that says
so while `start`, `stop`, `retry` and `status` go on working: those are queue
verbs and the queue is the backend's own.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter

from api.dependencies import (
    assessment_queue,
    chunking_queue,
    extraction_queue,
    orchestrator,
    parsing_queue,
    questions_queue,
    topics_queue,
)
from api.errors import ApiError, ErrorBody
from api.orchestrator import Automation, Orchestration, Refused, Run, Unavailable
from api.routes.stage import StageStatus, answered, status_of

log = logging.getLogger(__name__)

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

#: Every stage, in the order a document moves through them. The same order
#: `make corpus` runs them in and the same one the asset graph chains.
STAGES = (
    ("parsing", parsing_queue),
    ("chunking", chunking_queue),
    ("extraction", extraction_queue),
    ("topics", topics_queue),
    ("questions", questions_queue),
    ("assessment", assessment_queue),
)

#: Which of them `start` queues. Topic modelling is absent because a fit is
#: asked for rather than started, and asking replaces every topic there is -
#: which is a decision, not a fan-out. Assessment is absent because the phase
#: is off unless `.env` turns it on, and starting it here would be this route
#: overruling that. Both are reachable on their own: POST /topics/discover
#: and POST /assessment/start.
STARTABLE = ("parsing", "chunking", "extraction", "questions")

#: The verbs that mean something over the whole corpus, and the three the
#: methods below are named for. Narrower than the stage router's `Action` on
#: purpose: `rerun` over every stage at once would rebuild the corpus from
#: the files up, and `reclaim` can take a row from a worker that is still on
#: it. Both are real operations and both belong to one stage at a time.
Fanned = Literal["start", "stop", "retry"]

#: Answered when `run` is asked of a deployment that has no orchestrator.
_NO_ORCHESTRATOR = 503


@dataclass(frozen=True)
class StageMoved:
    """What one verb moved in one stage."""

    stage: str
    rows: int


@dataclass(frozen=True)
class PipelineAction:
    """What one verb moved across every stage it reached.

    `rows` is the total and `stages` is the breakdown, because "started 412"
    and "started 412, all of them parsing" are different things to know.
    """

    action: str
    rows: int
    stages: list[StageMoved]
    detail: str
    #: Set by `stop` when it also stopped a run in flight.
    run: Run | None = None
    #: The run every row this verb queued was stamped with - the same value
    #: `StageAction.run` carries for one stage, minted once here so a
    #: corpus-wide verb is one run across every queue it touched. Named
    #: apart from `run` above because that one is a Dagster run and this is
    #: an id written on rows. Absent for `stop`, which queues nothing.
    queued_as: str | None = None


@dataclass(frozen=True)
class Pipeline:
    """The state of the corpus, as one answer.

    Every stage's queue, plus what the orchestrator is doing. A caller
    polling this needs nothing else to know where a corpus has got to.
    """

    stages: list[StageStatus]
    orchestration: Orchestration
    #: True while any stage has a row queued or in progress. The question
    #: "is this corpus still moving" is not the same as "is a Dagster run
    #: going": `make extract` and the Start button move rows without one.
    working: bool
    failed: int


@dataclass(frozen=True)
class AutomationChange:
    """Which triggers to switch. Omitted, or null, means leave it alone."""

    on_arrival: bool | None = None
    nightly: bool | None = None


def _queues() -> list[StageStatus]:
    """Every stage's queue depth, in pipeline order."""
    return [status_of(name, queue) for name, queue in STAGES]


def _fan(
    action: Fanned, names: tuple[str, ...], trigger: str | None = None
) -> tuple[list[StageMoved], int]:
    """Runs one queue verb over several stages and records what each moved.

    Looked up by name rather than through a dict of bound methods: topic
    modelling answers `stop` and `retry` and has no `start` at all, and
    building that dict would read the attribute for every verb whichever one
    was asked for. The three verbs are named for the methods they call.

    One `trigger` for every stage it reaches, because this is one act: a
    person who pressed Start once should not have to find five ids to see
    what it did. `stop` passes none - it queues nothing.
    """
    moved = []
    for name, queue in STAGES:
        if name not in names:
            continue
        rows = getattr(queue, action)(**({"trigger": trigger} if trigger else {}))
        # Through the stage router's own logger, so a corpus-wide verb leaves
        # the same line per stage that pressing the button on that stage's
        # page would have left. Without this a fan-out would be invisible in
        # the one place somebody looks to find out who moved a row.
        answered(name, action, rows, run=trigger)
        moved.append(StageMoved(stage=name, rows=rows))
    return moved, sum(one.rows for one in moved)


@router.get("")
def state() -> Pipeline:
    """Reports every stage's queue and what the orchestrator is doing.

    One call rather than six. Never fails because the orchestrator is
    absent: `orchestration.available` is false and the queues still answer.
    """
    stages = _queues()
    return Pipeline(
        stages=stages,
        orchestration=orchestrator.state(),
        working=any(
            one.working or one.rows.get("pending", 0) or one.rows.get("in_progress", 0)
            for one in stages
        ),
        failed=sum(one.rows.get("failed", 0) for one in stages),
    )


@router.post(
    "/run",
    status_code=202,
    responses={
        409: {"model": ErrorBody},
        _NO_ORCHESTRATOR: {"model": ErrorBody},
    },
)
def run() -> Run:
    """Takes the whole corpus through every stage, in order.

    Returns at once with the run that was started; it does not wait for it.
    The work happens where it always happens - in the stage workers, one row
    at a time - and what the run adds is somebody to start each stage when
    the one before it has drained.

    Incremental, not a rebuild: each stage queues only what has never been
    asked for, so a run after one upload takes that document alone.

    Raises:
        ApiError: 409 `already_running` if a run is already going, since two
            would race each other through the same queues. 503
            `no_orchestrator` if this deployment has none, or its webserver
            is not answering.
    """
    live = orchestrator.state().running
    if live is not None:
        raise ApiError(
            409,
            "already_running",
            f"run {live.id} is already taking this corpus through. Stop it "
            f"first, or wait for it.",
        )
    try:
        return orchestrator.launch()
    except Unavailable as missing:
        raise ApiError(
            _NO_ORCHESTRATOR,
            "no_orchestrator",
            f"{missing} Every stage can still be run on its own: "
            f"POST /parsing/start, then /chunking/start, and so on.",
        ) from None
    except Refused as refusal:
        raise ApiError(_NO_ORCHESTRATOR, "orchestrator_refused", str(refusal)) from None


@router.post("/start", status_code=202)
def start() -> PipelineAction:
    """Queues every row that is ready to be worked, in every stage.

    One wave, not a sequence: a stage can only queue what the stage before
    it has already produced, so this moves the corpus as far as it can go
    right now and nothing further. Calling it again after the workers drain
    moves it the next stretch. `run` is the one that does that waiting.

    Topic modelling and the assessment phase are left alone - a fit replaces
    every topic and the judge costs model calls a deployment opts into, so
    both are asked for rather than swept into a fan-out.
    """
    queued_as = uuid4().hex
    moved, rows = _fan("start", STARTABLE, queued_as)
    return PipelineAction(
        action="start",
        rows=rows,
        stages=moved,
        queued_as=queued_as,
        detail=(
            f"{rows} row(s) queued; the workers will pick them up. Call this "
            f"again when they have drained to take the corpus its next "
            f"stretch, or POST /pipeline/run to have that waiting done for you."
            if rows
            else "nothing is ready to be queued. Either every stage is up to "
            "date, or the stage before has not produced anything yet."
        ),
    )


@router.post("/stop", status_code=202)
def stop() -> PipelineAction:
    """Takes back everything queued, and stops the run that queued it.

    Both halves, because either alone leaves the corpus moving: emptying the
    queues while a run is going means the run queues them again at its next
    stage, and stopping the run alone leaves every already-queued row to be
    worked.

    Whatever a worker holds right now finishes. Nothing here interrupts an
    item in progress - `reclaim` is the only verb that does.
    """
    stopped = None
    live = orchestrator.state().running
    if live is not None:
        try:
            if orchestrator.terminate(live.id):
                stopped = live
        except (Unavailable, Refused) as refusal:
            # Not fatal: the queues are the backend's own and emptying them
            # is most of what Stop means. Said in the detail rather than
            # raised, so a Dagster that is down cannot stop a person
            # stopping their pipeline.
            log.warning("pipeline: could not stop run %s: %s", live.id, refusal)

    moved, rows = _fan("stop", tuple(name for name, _ in STAGES))
    return PipelineAction(
        action="stop",
        rows=rows,
        stages=moved,
        run=stopped,
        detail=(
            f"{rows} row(s) taken off the queue"
            + (f", and run {stopped.id} stopped" if stopped else "")
            + ". Whatever a worker holds right now finishes."
        ),
    )


@router.post("/retry", status_code=202)
def retry() -> PipelineAction:
    """Returns every failed row of every stage to the queue.

    What a model outage leaves behind is failures spread over five queues,
    and this is the one call that clears them. It queues them; it does not
    work them, and it starts nothing that has not been asked for.
    """
    queued_as = uuid4().hex
    moved, rows = _fan("retry", tuple(name for name, _ in STAGES), queued_as)
    return PipelineAction(
        action="retry",
        rows=rows,
        stages=moved,
        queued_as=queued_as,
        detail=(
            f"{rows} failed row(s) returned to the queue."
            if rows
            else "nothing has failed."
        ),
    )


@router.get("/runs", responses={_NO_ORCHESTRATOR: {"model": ErrorBody}})
def runs(limit: int = 20) -> list[Run]:
    """The recent runs of the corpus job, newest first.

    Raises:
        ApiError: 503 `no_orchestrator` if this deployment has none.
    """
    try:
        return orchestrator.runs(limit=limit)
    except (Unavailable, Refused) as missing:
        raise ApiError(_NO_ORCHESTRATOR, "no_orchestrator", str(missing)) from None


@router.get("/automation")
def automation() -> Automation:
    """Whether anything is set to start a run without being asked.

    Never fails: a deployment with no orchestrator answers
    `available: false`, which is the honest answer to "is anything watching".
    """
    return orchestrator.state().automation


@router.put("/automation", responses={_NO_ORCHESTRATOR: {"model": ErrorBody}})
def automate(change: AutomationChange) -> Automation:
    """Switches the two triggers on or off.

    `on_arrival` starts a run when documents are sitting unasked-for, which
    is what makes the pipeline unattended. `nightly` runs the whole corpus at
    02:00 UTC, which is worth having because a topic fit goes stale on every
    new document. Both ship off: a stack that starts working the moment it
    comes up is one nobody chose.

    Omitting either, or sending null, leaves it as it is.

    Raises:
        ApiError: 503 `no_orchestrator` if there is none to switch.
    """
    try:
        return orchestrator.automate(
            on_arrival=change.on_arrival, nightly=change.nightly
        )
    except (Unavailable, Refused) as missing:
        raise ApiError(_NO_ORCHESTRATOR, "no_orchestrator", str(missing)) from None
