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

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Protocol

from fastapi import APIRouter

from api.errors import ApiError, ErrorBody
from stages import QueueState, Unnarrowable

log = logging.getLogger(__name__)

#: The verbs a narrowed action accepts. A Literal rather than a free string,
#: so the OpenAPI document lists them and anything else is refused before it
#: reaches a queue.
#:
#: `reclaim` is here and was not, and the reason it was not no longer holds.
#: It used to be CLI-only because "the API cannot know a worker is gone" -
#: but neither can the command line, and a person driving this over HTTP was
#: left with the one stuck row none of the other verbs reaches. What the
#: danger actually needs is a narrowing, which is what `confirm` below
#: enforces on the corpus-wide form.
Action = Literal["start", "stop", "retry", "rerun", "reclaim"]

#: Every verb `answered` can put into words, which is `Action` plus the one
#: a single stage has. `enrol` is the assessment phase's: nothing upstream
#: creates an assessment, so that stage alone can be asked to create the
#: rows without queuing any of them.
#:
#: Wider than `Action` on purpose. `Action` is what `stage_router` puts in
#: `POST /{action}`, so a verb added there is a verb EVERY stage advertises
#: and no other stage has an `enrol` to offer. A route that owns itself
#: describes what it did through this instead.
Verb = Literal["start", "stop", "retry", "rerun", "reclaim", "enrol"]

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
    "reclaim": (
        "taken back from the worker that held them and queued again",
        "no row is held",
    ),
    #: Assessment's own, phrased here so every verb reads the same way
    #: wherever it is answered. See `routes/assessment.py`.
    "enrol": ("enrolled, and queued for nobody", "every artefact is enrolled"),
}


class StageQueueState(Protocol):
    """What a stage's queue must be able to report.

    Separate from :class:`StageRepository` because topic modelling answers
    /status without answering the four verbs: it has no row until a fit is
    asked for, so it has nothing to start or to reset.
    """

    #: Which columns this stage may be narrowed to, by the name a route takes.
    scopes: ClassVar[dict[str, Any]]

    def narrowed(self, scope: str, value: str) -> Any:
        """Builds the condition one scope's value selects, refusing either."""
        ...

    def queue_state(self, within: Any = None) -> QueueState:
        """Reports the depth of this queue and whether a worker is on it."""
        ...


class StageRepository(StageQueueState, Protocol):
    """What a stage's queue must be able to report and repair.

    Every operation takes an optional `within`, which narrows it to the rows
    a condition selects instead of the whole queue.
    """

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

    def reclaim(self, within: Any = None) -> int:
        """Returns a row a dead worker still holds."""
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
    repository: StageQueueState,
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


def _narrowed(repository: StageQueueState, scope: str, value: str):
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
    action: Verb,
    rows: int,
    *,
    scope: str | None = None,
    value: str | None = None,
) -> StageAction:
    """Says what one verb moved, in the words every route uses for it.

    Separate from :func:`acted` for topic modelling, which owns its routes
    because a fit is asked for rather than started, and so counts its own
    rows - but should still describe `stop` the way every other stage does.

    This is where a queue verb asked for over HTTP is recorded. Every stage
    reaches it, narrowed or not, and the command line logs its own: without
    a line here, a corpus that started moving because somebody pressed a
    button left nothing behind saying so.
    """
    done, nothing = _DONE[action]
    log.info(
        "%s: %s moved %d row(s)%s",
        name,
        action,
        rows,
        f" for {scope}={value}" if scope else "",
        extra={
            "stage": name,
            "queue.action": action,
            "queue.rows": rows,
            **({"queue.scope": scope, "queue.value": value} if scope else {}),
        },
    )
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
        "reclaim": repository.reclaim,
    }[action](within)
    return answered(name, action, rows, scope=scope, value=value)


def stage_router(*, name: str, repository: StageRepository) -> APIRouter:
    """Builds the five corpus-wide routes and their narrowed pair."""
    router = APIRouter(prefix=f"/{name}", tags=[name])
    refusals: dict[int | str, dict[str, Any]] = {
        code: {"model": ErrorBody} for code in (400, 404)
    }

    @router.get("/status")
    def status() -> StageStatus:
        """Reports how much work this stage has waiting."""
        return status_of(name, repository)

    @router.post("/{action}", status_code=202, responses=refusals)
    def whole_queue(action: Action, confirm: bool = False) -> StageAction:
        """Runs one queue verb over everything this stage owns.

        Returns at once: none of these does the work. `start` makes rows
        claimable, `stop` makes them `new` again, `retry` clears a failure
        and `rerun` queues finished rows too, skipping whatever a worker
        holds right now. The worker picks them up on its next poll.

        Raises:
            ApiError: 400 `confirm_required` for an unnarrowed `reclaim`.
                Nothing here can tell a dead claim from a live one, so this
                is the one verb that can hand a row a live worker is on to a
                second one. Narrowed to a single item it is the ordinary way
                out of a killed worker; over a whole stage it is safe only
                once that stage is stopped, which is a thing only the person
                running it knows.
        """
        if action == "reclaim" and not confirm:
            raise ApiError(
                400,
                "confirm_required",
                "reclaim over a whole stage can take a row from a worker "
                "that is still on it. Narrow it to one item, or pass "
                "?confirm=true once the stage is stopped.",
            )
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
