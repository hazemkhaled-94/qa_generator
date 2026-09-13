"""The queue surface every row-based pipeline stage shares.

Reads and writes the queue and never works it. The work happens in the
stage's worker, which claims rows on its own, and never in the process
serving JSON - a conversion running here held it for sixteen minutes.

Nothing here runs because something else finished: `start` is what makes a
row claimable and `stop` is what makes it `new` again.

Each verb comes twice: once for the whole queue, and once under
`/{scope}/{value}` for the part of it a scope names - one document, one
passage. The narrowed form is the same operation against fewer rows, so a
page that offers per-item controls needs nothing here that the corpus-wide
one does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from fastapi import APIRouter

from api.errors import ApiError, ErrorBody
from stages import QueueState

#: The verbs a narrowed action accepts. A Literal rather than a free string,
#: so the OpenAPI document lists them and anything else is refused before it
#: reaches a queue.
Action = Literal["start", "stop", "retry", "rerun"]

#: How each verb reads once it has happened, for the message the page shows.
_DONE = {
    "start": "queued",
    "stop": "taken off the queue",
    "retry": "returned to the queue",
    "rerun": "queued again",
}


class StageRepository(Protocol):
    """What a stage's queue must be able to report and repair.

    Every operation takes an optional `within`, which narrows it to the rows
    a condition selects instead of the whole queue.
    """

    #: Which columns this stage may be narrowed to, by the name a route takes.
    scopes: dict[str, Any]

    def narrow(self, scope: str, value: str) -> Any:
        """Builds the condition one scope's value selects."""
        ...

    def queue_state(self, within: Any = None) -> QueueState:
        """Reports the depth of this queue and whether a worker is on it."""
        ...

    def start(self, within: Any = None) -> int:
        """Queues the rows never asked for."""
        ...

    def stop(self, within: Any = None) -> int:
        """Takes back the rows not started yet."""
        ...

    def retry(self, within: Any = None) -> int:
        """Returns failed rows to the queue."""
        ...

    def reset(self, within: Any = None) -> int:
        """Returns every row to the queue, finished ones included."""
        ...


@dataclass(frozen=True)
class StageStatus:
    """How much work a stage has waiting.

    `rows` is named for what it counts and not for what it counts of:
    extraction queues over passages and topic modelling over fits.

    `scope` and `value` say what the counts are of: absent, the whole queue.
    """

    stage: str
    working: bool
    rows: dict[str, int] = field(default_factory=dict)
    scope: str | None = None
    value: str | None = None


@dataclass(frozen=True)
class StageAction:
    """What one narrowed verb moved.

    One shape for all four, because narrowed they differ only in which rows
    they select and what the count then means.
    """

    stage: str
    scope: str
    value: str
    action: str
    rows: int
    detail: str


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


def status_of(
    name: str,
    repository: StageRepository,
    *,
    scope: str | None = None,
    value: str | None = None,
) -> StageStatus:
    """Reports how much work one stage has waiting.

    Shared with the stages that build their own router, so every /status
    answers the same shape whatever built it.

    Raises:
        ApiError: 404 or 400, when a scope is given and is not one this stage
            accepts or the value is not what its column holds.
    """
    within = None if scope is None else _narrowed(repository, scope, value or "")
    state = repository.queue_state(within)
    return StageStatus(
        stage=name, working=state.working, rows=state.rows, scope=scope, value=value
    )


def _narrowed(repository: StageRepository, scope: str, value: str):
    """Builds the condition one scope and value select, refusing either.

    Raises:
        ApiError: 404 `unknown_scope` if this stage cannot be narrowed that
            way, 400 `invalid_value` if the value is not what the column
            holds.
    """
    try:
        return repository.narrow(scope, value)
    except KeyError:
        accepted = ", ".join(repository.scopes) or "nothing"
        raise ApiError(
            404, "unknown_scope", f"this stage narrows to {accepted}, not {scope!r}"
        ) from None
    except ValueError:
        raise ApiError(
            400, "invalid_value", f"{value!r} is not a valid {scope}"
        ) from None


def stage_router(*, name: str, repository: StageRepository) -> APIRouter:
    """Builds the five corpus-wide routes and their narrowed pair."""
    router = APIRouter(prefix=f"/{name}", tags=[name])
    refusals = {code: {"model": ErrorBody} for code in (400, 404)}

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

    @router.get("/{scope}/{value}/status", responses=refusals)
    def scoped_status(scope: str, value: str) -> StageStatus:
        """Reports how much work this stage has waiting for one item.

        Raises:
            ApiError: 404 `unknown_scope`, 400 `invalid_value`.
        """
        return status_of(name, repository, scope=scope, value=value)

    @router.post("/{scope}/{value}/{action}", status_code=202, responses=refusals)
    def scoped_action(scope: str, value: str, action: Action) -> StageAction:
        """Runs one queue verb against a single item.

        The same operation as the route without a scope, against the rows
        that scope selects. Nothing runs here either: the worker picks up
        whatever has become claimable.

        Raises:
            ApiError: 404 `unknown_scope`, 400 `invalid_value`.
        """
        within = _narrowed(repository, scope, value)
        rows = {
            "start": repository.start,
            "stop": repository.stop,
            "retry": repository.retry,
            "rerun": repository.reset,
        }[action](within)
        return StageAction(
            stage=name,
            scope=scope,
            value=value,
            action=action,
            rows=rows,
            detail=f"{rows} row(s) {_DONE[action]}."
            if rows
            else f"nothing to {action} for that {scope}.",
        )

    return router
