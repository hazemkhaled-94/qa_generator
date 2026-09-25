"""The drain loop every pipeline stage runs.

A stage claims one row, works it, and records what became of it. Every stage
annotates its span with `stage.name` and `stage.outcome`, so one trace query
finds everything that failed whichever stage failed it.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, ClassVar

from opentelemetry.trace import Span

from stages.queue import StageQueue

log = logging.getLogger(__name__)

#: How often a worker says it is still on the row it holds. Ten of these fit
#: inside `StageQueue.lease`, so a beat may be missed for a slow query or a
#: paused process without the row being swept out from under it.
HEARTBEAT_SECONDS = 30.0


def _never() -> bool:
    """Reports that no stop has been asked for."""
    return False


class StageService(ABC):
    """One stage's worker: claim a row, work it, record the outcome."""

    #: The stage's name, recorded on every span and log line.
    name: ClassVar[str]
    #: What this stage counts, singular: document, passage or fit.
    unit: ClassVar[str]

    _REQUIRED = ("name", "unit")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Refuses a stage that has not said what it is called."""
        super().__init_subclass__(**kwargs)
        if ABC in cls.__bases__:
            return
        missing = [name for name in StageService._REQUIRED if not hasattr(cls, name)]
        if missing:
            raise TypeError(
                f"{cls.__name__} is a StageService but does not declare "
                f"{', '.join(missing)}."
            )

    def __init__(self, repository: StageQueue) -> None:
        """Initialises the service with its queue."""
        self._repository = repository

    @abstractmethod
    def process_next(self) -> Any | None:
        """Claims one row and works it, returning its key or None."""

    @contextmanager
    def _beating(self) -> Iterator[None]:
        """Says this worker is still alive for as long as the body runs.

        One thread for the drain rather than one per row: the queue beats
        whichever row it holds, and holds none between rows.
        """
        stop = threading.Event()

        def beat() -> None:
            """Refreshes the held claim until the drain is done with it."""
            while not stop.wait(HEARTBEAT_SECONDS):
                try:
                    self._repository.touch()
                except Exception:
                    # Logged, not raised: this thread failing must not take
                    # the work down with it, and a missed beat costs nothing
                    # until the lease runs out.
                    log.warning("%s: heartbeat failed", self.name, exc_info=True)

        thread = threading.Thread(
            target=beat, name=f"{self.name}-heartbeat", daemon=True
        )
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=HEARTBEAT_SECONDS)

    def drain(self, stopping: Callable[[], bool] = _never) -> int:
        """Works every queued row, sweeping abandoned claims first."""
        abandoned = self._repository.abandon()
        if abandoned:
            log.warning(
                "%s: failed %d %s(s) left claimed by an earlier run",
                self.name,
                abandoned,
                self.unit,
            )
        processed = 0
        with self._beating():
            while not stopping():
                if self.process_next() is None:
                    break
                processed += 1
        if processed:
            log.info("%s: %d %s(s)", self.name, processed, self.unit)
        return processed

    def _fail(self, key: Any, error: str, current: Span) -> None:
        """Marks a row failed and annotates the span."""
        current.set_attribute("stage.name", self.name)
        current.set_attribute("stage.outcome", "failed")
        self._repository.fail(key, error)
        log.warning("%s: failed %s %s: %s", self.name, self.unit, key, error)

    def _done(self, current: Span) -> None:
        """Annotates the span of a row this stage finished."""
        current.set_attribute("stage.name", self.name)
        current.set_attribute("stage.outcome", self._repository.done)
