"""The queue every pipeline stage runs on.

A stage owns a status column, an error column and the timestamp of its claim.
Claiming, failing, sweeping an abandoned claim and requeueing are written here
once against those declarations.

A row arrives `new` and no worker looks at it until something asks for it, so
a stage selects on its own status column and on nothing else.
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
    #: The status this stage sets when it finishes a row.
    done: ClassVar[Status]
    #: How long a claim may go unfinished before a later run treats it as
    #: abandoned. Overridden per instance by a worker that knows what one row
    #: can actually cost.
    lease: timedelta = timedelta(hours=1)
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

    def _claim(self, *returning: InstrumentedAttribute):
        """Takes the next pending row and marks it in progress."""
        # A short transaction, not a held row lock: the work runs for seconds
        # or minutes. The claim is timestamped so a later run can tell it from
        # a live one.
        with self._session.begin() as session:
            return session.execute(
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
        if session is not None:
            session.execute(statement)
            return
        with self._session.begin() as opened:
            opened.execute(statement)

    def fail(self, key: Any, error: str) -> None:
        """Records a row this stage could not process, releasing the claim."""
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

    def abandon(self) -> int:
        """Fails every row whose claim has outlived the lease."""
        with self._session.begin() as session:
            return session.execute(
                update(self.columns.entity)
                .where(
                    self.columns.status == Status.IN_PROGRESS,
                    # A NULL claim counts as abandoned: such a row matches no
                    # other operation and would sit in_progress forever.
                    or_(
                        self.columns.claimed_at.is_(None),
                        self.columns.claimed_at < func.now() - self.lease,
                    ),
                )
                .values(
                    {
                        self.columns.claimed_at: None,
                        self.columns.status: Status.FAILED,
                        self.columns.error: ABANDONED,
                    }
                )
            ).rowcount

    def stop(self) -> int:
        """Takes back every row this stage has not started yet."""
        with self._session.begin() as session:
            return session.execute(
                update(self.columns.entity)
                .where(self.columns.status == Status.PENDING)
                .values({self.columns.status: Status.NEW})
            ).rowcount

    def retry(self) -> int:
        """Returns every failed row to the queue, clearing its error."""
        return self._requeue(self.columns.status == Status.FAILED)

    def _requeue(self, where) -> int:
        """Moves the rows a condition selects to pending, clearing the error."""
        with self._session.begin() as session:
            return session.execute(
                update(self.columns.entity)
                .where(where)
                .values({self.columns.status: Status.PENDING, self.columns.error: None})
            ).rowcount

    def queue_state(self) -> QueueState:
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
                .group_by(self.columns.status)
                .order_by(self.columns.status)
            ).all()

        return QueueState(
            working=any(
                status == Status.IN_PROGRESS and live for status, _, live in rows
            ),
            rows={status: total for status, total, _ in rows},
        )

    def counts_by_status(self) -> dict[str, int]:
        """Counts rows in each state of this stage."""
        return self.queue_state().rows


class RowQueue(StageQueue, abstract=True):
    """A queue over rows that exist before the stage is asked to run them.

    Separate from :class:`StageQueue` because `start` and `reset` presuppose
    that: topic modelling has no row until a fit is requested, so it extends
    the base without them rather than answering them with something else.
    """

    def start(self) -> int:
        """Queues the rows this stage has never been asked to do."""
        return self._requeue(self.columns.status == Status.NEW)

    def reset(self) -> int:
        """Returns every row to the queue, finished ones included.

        Skips rows a worker holds right now.
        """
        return self._requeue(self.columns.status != Status.IN_PROGRESS)
