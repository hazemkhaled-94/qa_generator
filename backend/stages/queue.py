"""The queue every pipeline stage runs on.

A stage owns a status column, an error column and the timestamp of its claim.
Claiming, failing, sweeping an abandoned claim and requeueing are written here
once against those declarations.

A row arrives `new` and no worker looks at it until something asks for it, so
a stage claims on its own status column and on nothing else. The one thing
read beside it is `RowQueue.ready`, which says which rows may be QUEUED - see
there for why chunking is the only stage that needs it.

Every queue operation takes an optional `within`, which narrows it to part of
the queue - one document, one passage - instead of all of it. A stage says
which narrowings it accepts by declaring `scopes`; the route and the command
line both build theirs through :meth:`StageQueue.narrow`, so neither learns a
column name.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, ClassVar

from sqlalchemy import func, or_, select, update
from sqlalchemy.orm import InstrumentedAttribute

from database.qa_generator import Status
from database.qa_generator.repository import Repository

#: Recorded against a row whose worker never came back.
ABANDONED = "the worker did not finish; the run was interrupted"


class Unnarrowable(Exception):
    """A narrowing a stage cannot make, and why.

    One exception rather than the KeyError and ValueError `narrow` raises,
    because the route and the command line answer the same two refusals and
    would otherwise each spell out the same two messages. `code` is what the
    API answers with; the command line reads only the message.
    """

    def __init__(self, code: str, detail: str) -> None:
        """Initialises the refusal."""
        super().__init__(detail)
        self.code = code


@dataclass(frozen=True)
class Columns:
    """The pair of columns a stage owns, plus the time of its claim."""

    entity: type
    key: InstrumentedAttribute
    status: InstrumentedAttribute
    error: InstrumentedAttribute
    claimed_at: InstrumentedAttribute


@dataclass(frozen=True)
class QueueState:
    """How much work a stage has waiting, and whether anyone is on it."""

    working: bool
    rows: dict[str, int]


class StageQueue(Repository):
    """One stage's view of its queue.

    Subclasses declare which columns are theirs; every method below is written
    against those declarations rather than against a column name.
    """

    #: The three columns this stage owns. Held in a value object rather than
    #: as three class attributes: a mapped column is a descriptor, so naming
    #: one directly on this class would make `self.status` try to read the
    #: repository as if it were a row.
    columns: ClassVar[Columns]
    #: Which columns a caller may narrow an operation to, by the name the
    #: route and the command line take. Empty means this stage answers for
    #: the whole queue only, which is true of anything fitted all at once.
    scopes: ClassVar[dict[str, InstrumentedAttribute]] = {}
    #: Which rows of the table are this queue's at all, when the table holds
    #: rows that are not. Every operation below carries it, so a stage cannot
    #: start, sweep or count a row that is none of its business: topics holds
    #: both the fit requests and the topics, and only the topics are a unit
    #: of question generation.
    base: ClassVar[Any] = None
    #: The status this stage sets when it finishes a row.
    done: ClassVar[Status]
    #: How long a claim may go unrefreshed before a later run treats it as
    #: abandoned. A worker holding a row refreshes it every
    #: `stages.service.HEARTBEAT_SECONDS`, so this bounds how long a DEAD
    #: worker's row stays claimed and not how long the work may take. Ten
    #: missed beats.
    lease: timedelta = timedelta(minutes=5)
    #: Selects the key of the next row this stage should take.
    next_pending: ClassVar[Any]

    _REQUIRED = ("columns", "done", "next_pending")

    def __init_subclass__(cls, *, abstract: bool = False, **kwargs: Any) -> None:
        """Refuses a stage that has not said which columns are its own.

        `abstract=True` exempts a base that declares no columns of its own.
        """
        super().__init_subclass__(**kwargs)
        if abstract:
            return
        missing = [name for name in StageQueue._REQUIRED if not hasattr(cls, name)]
        if missing:
            raise TypeError(
                f"{cls.__name__} is a StageQueue but does not declare "
                f"{', '.join(missing)}."
            )

    def __init__(self, lease: timedelta | None = None) -> None:
        """Binds to the session factory, optionally with a longer lease."""
        super().__init__()
        if lease is not None:
            self.lease = lease
        #: The key of the row this instance holds, for `touch` to refresh.
        #: Written by the thread that works the row and read by the one that
        #: beats; a lone reference, so neither sees a half-written value.
        self._held: Any = None

    def _where(self, *conditions: Any) -> tuple[Any, ...]:
        """This queue's own rows, narrowed by whatever the caller gave.

        `where()` takes no None, and every operation below is written once
        for the narrowed and the whole-queue case.
        """
        return tuple(
            condition for condition in (self.base, *conditions) if condition is not None
        )

    def _claim(self, *returning: InstrumentedAttribute):
        """Takes the next pending row and marks it in progress."""
        # A short transaction, not a held row lock: the work runs for seconds
        # or minutes. The claim is timestamped so a later run can tell it from
        # a live one.
        with self._session.begin() as session:
            claimed = session.execute(
                update(self.columns.entity)
                .where(self.columns.key == self.next_pending)
                .values(
                    {
                        self.columns.status: Status.IN_PROGRESS,
                        self.columns.claimed_at: func.now(),
                    }
                )
                .returning(*returning)
            ).one_or_none()
        # Every caller returns its key column first, which is what `touch`
        # refreshes. Read by name rather than by position, so a stage that
        # returns its columns in another order still beats.
        self._held = None if claimed is None else getattr(claimed, self.columns.key.key)
        return claimed

    def _finish(self, key: Any, *, session: Any = None, **values: Any) -> None:
        """Marks a row done for this stage, releasing the claim."""
        statement = (
            update(self.columns.entity)
            .where(self.columns.key == key)
            .values(
                {
                    self.columns.claimed_at: None,
                    self.columns.status: self.done,
                    self.columns.error: None,
                    **values,
                }
            )
        )
        self._held = None
        if session is not None:
            session.execute(statement)
            return
        with self._session.begin() as opened:
            opened.execute(statement)

    def fail(self, key: Any, error: str) -> None:
        """Records a row this stage could not process, releasing the claim."""
        self._held = None
        with self._session.begin() as session:
            session.execute(
                update(self.columns.entity)
                .where(self.columns.key == key)
                .values(
                    {
                        self.columns.claimed_at: None,
                        self.columns.status: Status.FAILED,
                        self.columns.error: error[:2000],
                    }
                )
            )

    def touch(self) -> bool:
        """Refreshes the claim on the row this instance is working.

        What separates a live claim from a dead one. `claimed_at` records
        when work started and nothing else, so a sweep could only ever ask
        "could this still be running?" - and the honest answer is the worst
        case the work might take, which is days for a stage whose unit is a
        whole topic. Beating turns that into "is anyone still on it?", which
        the lease can answer in minutes however long the work runs.

        Refuses to refresh a row this queue no longer holds: `reclaim` moves
        an interrupted row back to pending, and a beat still in flight must
        not drag it back to claimed.
        """
        key = self._held
        if key is None:
            return False
        with self._session.begin() as session:
            return bool(
                session.execute(
                    update(self.columns.entity)
                    .where(
                        *self._where(
                            self.columns.key == key,
                            self.columns.status == Status.IN_PROGRESS,
                        )
                    )
                    .values({self.columns.claimed_at: func.now()})
                ).rowcount
            )

    def abandon(self) -> int:
        """Fails every row whose claim has outlived the lease."""
        with self._session.begin() as session:
            return session.execute(
                update(self.columns.entity)
                .where(
                    *self._where(
                        self.columns.status == Status.IN_PROGRESS,
                        # A NULL claim counts as abandoned: such a row matches
                        # no other operation and would sit in_progress forever.
                        or_(
                            self.columns.claimed_at.is_(None),
                            self.columns.claimed_at < func.now() - self.lease,
                        ),
                    )
                )
                .values(
                    {
                        self.columns.claimed_at: None,
                        self.columns.status: Status.FAILED,
                        self.columns.error: ABANDONED,
                    }
                )
            ).rowcount

    def narrow(self, scope: str, value: str) -> Any:
        """Builds the condition one scope's value selects.

        Raises:
            KeyError: If this stage accepts no such scope.
            ValueError: If the value is not what that column holds.
        """
        column = self.scopes[scope]
        return column == column.type.python_type(value)

    def narrowed(self, scope: str, value: str) -> Any:
        """Builds that condition, refusing a scope or a value with a reason.

        The wording is here rather than at each caller: the route and the
        command line refuse the same two things.

        Raises:
            Unnarrowable: If this stage accepts no such scope, or the value
                is not what that column holds.
        """
        try:
            return self.narrow(scope, value)
        except KeyError:
            accepted = ", ".join(self.scopes) or "nothing"
            raise Unnarrowable(
                "unknown_scope", f"this stage narrows to {accepted}, not {scope!r}"
            ) from None
        except ValueError:
            raise Unnarrowable(
                "invalid_value", f"{value!r} is not a valid {scope}"
            ) from None

    def stop(self, within: Any = None) -> int:
        """Takes back every row this stage has not started yet."""
        with self._session.begin() as session:
            return session.execute(
                update(self.columns.entity)
                .where(*self._where(self.columns.status == Status.PENDING, within))
                .values({self.columns.status: Status.NEW})
            ).rowcount

    def retry(self, within: Any = None) -> int:
        """Returns every failed row to the queue, clearing its error."""
        return self._requeue(self.columns.status == Status.FAILED, within)

    def reclaim(self, within: Any = None) -> int:
        """Returns a row a dead worker still holds, without waiting it out.

        The gap between `retry` and `rerun`, and the only verb that moves
        an `in_progress` row. `retry` takes the failed, `rerun` explicitly
        skips what a worker holds, and `stop` takes back only what has not
        begun - so a worker killed mid-row left that row unreachable until
        its lease ran out. Question generation's lease is derived from the
        work a topic costs and computes to 67 days at the settings this
        was written under, which is not a wait.

        `abandon` is the automatic version and stays the normal one: it
        sweeps on a lease, because a row a LIVE worker holds must not be
        given to a second. This is the deliberate one, and the reason it
        is worth having separately is that a person knows the worker is
        gone and the lease cannot.

        **Narrow it.** Nothing here can tell a dead claim from a live one,
        so run it against a named row when a worker may still be up, and
        over the stage only when none is. Two workers on one row is what
        the lease exists to prevent, and this is the verb that can cause
        it.
        """
        return self._requeue(self.columns.status == Status.IN_PROGRESS, within)

    def _requeue(self, *where: Any) -> int:
        """Moves the rows a condition selects to pending, clearing the error."""
        with self._session.begin() as session:
            return session.execute(
                update(self.columns.entity)
                .where(*self._where(*where))
                .values({self.columns.status: Status.PENDING, self.columns.error: None})
            ).rowcount

    def queue_state(self, within: Any = None) -> QueueState:
        """Reports the queue depth and whether a worker is on it."""
        # One query for both: a page polls this every few seconds.
        with self._session() as session:
            rows = session.execute(
                select(
                    self.columns.status,
                    func.count(),
                    func.count().filter(
                        self.columns.claimed_at >= func.now() - self.lease
                    ),
                )
                .where(*self._where(within))
                .group_by(self.columns.status)
                .order_by(self.columns.status)
            ).all()

        return QueueState(
            working=any(
                status == Status.IN_PROGRESS and live for status, _, live in rows
            ),
            rows={status: total for status, total, _ in rows},
        )

    def counts_by_status(self, within: Any = None) -> dict[str, int]:
        """Counts rows in each state of this stage."""
        return self.queue_state(within).rows


class RowQueue(StageQueue, abstract=True):
    """A queue over rows that exist before the stage is asked to run them.

    Separate from :class:`StageQueue` because `start` and `reset` presuppose
    that: topic modelling has no row until a fit is requested, so it extends
    the base without them rather than answering them with something else.
    """

    #: Which of this stage's rows have an input to work on. None means every
    #: one does, which is true wherever the row is created by the stage
    #: before - a passage exists because chunking made it, a topic because a
    #: fit did. Only chunking needs this: it queues over `documents`, and a
    #: document exists from the moment it is uploaded, so `start` without it
    #: queues documents that have not been parsed and whose only possible
    #: outcome is a failure on the missing parsed object.
    #:
    #: Carried by `start` and `reset` and by nothing else. Claiming, failing,
    #: sweeping and `retry` still select on this stage's own status column
    #: alone: this says which rows may ENTER the queue, not which a worker
    #: may take, and a row already queued is worked whatever this says.
    ready: ClassVar[Any] = None

    def start(self, within: Any = None) -> int:
        """Queues the rows this stage has never been asked to do.

        Skips rows the stage before this one has not produced an input for.
        """
        return self._requeue(self.columns.status == Status.NEW, within, self.ready)

    def reset(self, within: Any = None) -> int:
        """Returns every row to the queue, finished ones included.

        Skips rows a worker holds right now, and rows with no input to redo.
        """
        return self._requeue(
            self.columns.status != Status.IN_PROGRESS, within, self.ready
        )
