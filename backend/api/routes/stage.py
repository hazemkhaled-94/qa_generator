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
from stages import QueueState, Unnarrowable

#: The verbs a narrowed action accepts. A Literal rather than a free string,
#: so the OpenAPI document lists them and anything else is refused before it
#: reaches a queue.
Action = Literal["start", "stop", "retry", "rerun"]

#: What each refusal the queue can raise is answered with.
_REFUSED = {"unknown_scope": 404, "invalid_value": 400}

#: How each verb reads once it has happened, and how it reads when it moved
#: nothing. Both here, so the whole-queue route and the narrowed one cannot
#: describe the same operation differently.
_DONE = {
    "start": ("queued; the worker will pick them up", "nothing new to do"),
    "stop": (
        "taken off the queue; the item in progress finishes",
        "nothing was queued",
    ),
    "retry": ("returned to the queue", "nothing has failed"),
    "rerun": ("queued again; the worker will pick them up", "there is nothing to redo"),
}


class StageRepository(Protocol):
    """What a stage's queue must be able to report and repair.

    Every operation takes an optional `within`, which narrows it to the rows
    a condition selects instead of the whole queue.
    """

    #: Which columns this stage may be narrowed to, by the name a route takes.
    scopes: dict[str, Any]

    def narrowed(self, scope: str, value: str) -> Any:
        """Builds the condition one scope's value selects, refusing either."""
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
    """What one verb moved.

    One shape for every verb, narrowed or not: they differ only in which rows
    they select and what the count then means, which `action` and `detail`
    say. `scope` and `value` are absent when the verb ran over the whole
    queue.
    """

    stage: str
    action: str
    rows: int
    detail: str
    scope: str | None = None
    value: str | None = None


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

    The wording is the queue's; this only chooses the status it answers with.

    Raises:
        ApiError: 404 `unknown_scope` if this stage cannot be narrowed that
            way, 400 `invalid_value` if the value is not what the column
            holds.
    """
    try:
        return repository.narrowed(scope, value)
    except Unnarrowable as exc:
        raise ApiError(_REFUSED[exc.code], exc.code, str(exc)) from None


def answered(
    name: str,
    action: Action,
    rows: int,
    *,
    scope: str | None = None,
    value: str | None = None,
) -> StageAction:
    """Says what one verb moved, in the words every route uses for it.

    Separate from :func:`acted` for topic modelling, which owns its routes
    because a fit is asked for rather than started, and so counts its own
    rows - but should still describe `stop` the way every other stage does.
    """
    done, nothing = _DONE[action]
    return StageAction(
        stage=name,
        action=action,
        rows=rows,
        detail=(
            f"{rows} row(s) {done}."
            if rows
            else f"{nothing}{f' for that {scope}' if scope else ''}."
        ),
        scope=scope,
        value=value,
    )


def acted(
    name: str,
    repository: StageRepository,
    action: Action,
    *,
    within: Any = None,
    scope: str | None = None,
    value: str | None = None,
) -> StageAction:
    """Runs one queue verb and says what it moved.

    Shared by the whole-queue route and the narrowed one, so the same verb
    cannot describe itself two ways depending on which was called.
    """
    rows = {
        "start": repository.start,
        "stop": repository.stop,
        "retry": repository.retry,
        "rerun": repository.reset,
    }[action](within)
    return answered(name, action, rows, scope=scope, value=value)


def stage_router(*, name: str, repository: StageRepository) -> APIRouter:
    """Builds the five corpus-wide routes and their narrowed pair."""
    router = APIRouter(prefix=f"/{name}", tags=[name])
    refusals = {code: {"model": ErrorBody} for code in (400, 404)}

    @router.get("/status")
    def status() -> StageStatus:
        """Reports how much work this stage has waiting."""
        return status_of(name, repository)

    @router.post("/{action}", status_code=202)
    def whole_queue(action: Action) -> StageAction:
        """Runs one queue verb over everything this stage owns.

        Returns at once: none of these does the work. `start` makes rows
        claimable, `stop` makes them `new` again, `retry` clears a failure
        and `rerun` queues finished rows too, skipping whatever a worker
        holds right now. The worker picks them up on its next poll.
        """
        return acted(name, repository, action)

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
        that scope selects.

        Raises:
            ApiError: 404 `unknown_scope`, 400 `invalid_value`.
        """
        return acted(
            name,
            repository,
            action,
            within=_narrowed(repository, scope, value),
            scope=scope,
            value=value,
        )

    return router
