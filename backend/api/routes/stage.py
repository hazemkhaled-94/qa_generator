"""The queue surface every row-based pipeline stage shares.

Reads and writes the queue and never works it. The work happens in the
stage's worker, which claims rows on its own, and never in the process
serving JSON - a conversion running here held it for sixteen minutes.

Nothing here runs because something else finished: `start` is what makes a
row claimable and `stop` is what makes it `new` again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from fastapi import APIRouter

from stages import QueueState


class StageRepository(Protocol):
    """What a stage's queue must be able to report and repair."""

    def queue_state(self) -> QueueState:
        """Reports the depth of this queue and whether a worker is on it."""
        ...

    def start(self) -> int:
        """Queues the rows never asked for."""
        ...

    def stop(self) -> int:
        """Takes back the rows not started yet."""
        ...

    def retry(self) -> int:
        """Returns failed rows to the queue."""
        ...

    def reset(self) -> int:
        """Returns every row to the queue, finished ones included."""
        ...


@dataclass(frozen=True)
class StageStatus:
    """How much work a stage has waiting.

    `rows` is named for what it counts and not for what it counts of:
    extraction queues over passages and topic modelling over fits.
    """

    stage: str
    working: bool
    rows: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class StageQueued:
    """The answer to starting or stopping a stage."""

    stage: str
    queued: int
    detail: str


@dataclass(frozen=True)
class StageRetry:
    """The result of returning failed rows to a queue."""

    stage: str
    retried: int


@dataclass(frozen=True)
class StageRerun:
    """The answer to a request to run a stage again from the start."""

    stage: str
    reset: int
    detail: str


def status_of(name: str, repository: StageRepository) -> StageStatus:
    """Reports how much work one stage has waiting.

    Shared with the stages that build their own router, so every /status
    answers the same shape whatever built it.
    """
    state = repository.queue_state()
    return StageStatus(stage=name, working=state.working, rows=state.rows)


def stage_router(*, name: str, repository: StageRepository) -> APIRouter:
    """Builds the five routes a row-based stage exposes."""
    router = APIRouter(prefix=f"/{name}", tags=[name])

    @router.get("/status")
    def status() -> StageStatus:
        """Reports how much work this stage has waiting."""
        return status_of(name, repository)

    @router.post("/start", status_code=202)
    def start() -> StageQueued:
        """Queues everything this stage has not been asked to do yet.

        Returns at once: the worker claims the rows on its next poll.
        """
        queued = repository.start()
        return StageQueued(
            stage=name,
            queued=queued,
            detail=f"{queued} queued; the worker will pick them up."
            if queued
            else "nothing new to do.",
        )

    @router.post("/stop")
    def stop() -> StageQueued:
        """Takes back everything this stage has not started yet.

        The row a worker is holding is left to finish.
        """
        queued = repository.stop()
        return StageQueued(
            stage=name,
            queued=queued,
            detail=f"{queued} taken off the queue; the item in progress will finish."
            if queued
            else "nothing was queued.",
        )

    @router.post("/retry")
    def retry() -> StageRetry:
        """Returns every failed row of this stage to the queue."""
        return StageRetry(stage=name, retried=repository.retry())

    @router.post("/rerun", status_code=202)
    def rerun() -> StageRerun:
        """Queues every row of this stage again, finished ones included.

        Safe while a worker is running: rows a worker holds are skipped.
        """
        reset = repository.reset()
        return StageRerun(
            stage=name,
            reset=reset,
            detail=f"{reset} queued; the worker will pick them up.",
        )

    return router
