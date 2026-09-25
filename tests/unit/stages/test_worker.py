"""Ending a worker's loop.

A worker killed mid-item leaves its row claimed until the lease expires.
"""

from __future__ import annotations

import os
import signal
from typing import Any

from stages.service import StageService
from stages.worker import Shutdown, watch


def test_a_signal_is_recorded_rather_than_raised() -> None:
    """The flag goes up; the process stays."""
    flag = Shutdown()
    assert not flag.requested()

    flag._request(signal.SIGTERM, None)
    assert flag.requested()


def test_sigterm_ends_the_loop_after_the_current_drain() -> None:
    """The drain in progress finishes and no further drain begins."""
    drains: list[bool] = []

    def drain(stopping) -> int:
        """Records what the loop was told, and asks to stop on the second."""
        drains.append(stopping())
        if len(drains) == 2:
            os.kill(os.getpid(), signal.SIGTERM)
        return 1  # work found, so the loop does not sleep

    watch(drain, 0.01)
    assert drains == [False, False], drains


class _Repository:
    """A queue that counts how often it was asked for work."""

    done = "extracted"

    def __init__(self) -> None:
        """Initialises the counter."""
        self.claims = 0

    def abandon(self) -> int:
        """Reports that nothing was left claimed."""
        return 0

    def touch(self) -> bool:
        """Holds nothing to refresh."""
        return False


class _Service(StageService):
    """A stage whose work is to count that it was asked."""

    name = "check"
    unit = "row"

    def process_next(self) -> Any | None:
        """Claims one row and does nothing with it."""
        self._repository.claims += 1
        return None


def test_a_stopping_worker_claims_nothing() -> None:
    """A drain that starts already stopping asks the queue for no row."""
    service = _Service(_Repository())

    assert service.drain(stopping=lambda: True) == 0
    assert service._repository.claims == 0, "a stopping worker claimed another row"


def test_a_draining_worker_claims_once_before_an_empty_queue_ends_it() -> None:
    """An ordinary drain asks for a row, and stops when there is none."""
    running = _Service(_Repository())

    assert running.drain() == 0
    assert running._repository.claims == 1, "a draining worker never claimed"
