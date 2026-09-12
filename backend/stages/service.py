"""The drain loop every pipeline stage runs.

A stage claims one row, works it, and records what became of it. Every stage
annotates its span with `stage.name` and `stage.outcome`, so one trace query
finds everything that failed whichever stage failed it.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

from opentelemetry.trace import Span

from stages.queue import StageQueue

log = logging.getLogger(__name__)


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
