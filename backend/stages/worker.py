"""The watch loop every stage worker runs.

ponytail: a poll, not LISTEN/NOTIFY. At this interval an idle worker costs one
cheap query every few seconds; NOTIFY on the status column is the upgrade if
that ever matters.
"""

from __future__ import annotations

import logging
import signal
import threading
from collections.abc import Callable
from types import FrameType

log = logging.getLogger(__name__)


class Shutdown:
    """Records that the process has been asked to stop, and wakes the loop."""

    def __init__(self) -> None:
        """Initialises an unrequested shutdown."""
        # An Event rather than a bool, so a worker asleep between polls wakes
        # at once instead of serving out the rest of its interval.
        self._event = threading.Event()

    def install(self) -> Shutdown:
        """Catches the signals a supervisor stops a container with."""
        for received in (signal.SIGTERM, signal.SIGINT):
            signal.signal(received, self._request)
        return self

    def _request(self, number: int, frame: FrameType | None) -> None:
        """Records the request."""
        del frame
        log.info(
            "%s received: finishing the current item, then stopping",
            signal.Signals(number).name,
        )
        self._event.set()

    def requested(self) -> bool:
        """Whether a stop has been asked for."""
        return self._event.is_set()

    def wait(self, seconds: float) -> None:
        """Sleeps, unless and until a stop is requested."""
        self._event.wait(seconds)


def watch(drain: Callable[[Callable[[], bool]], int], interval: float) -> None:
    """Drains the queue until the process is asked to stop."""
    shutdown = Shutdown().install()
    log.info("watching the queue, polling every %.0fs", interval)

    while not shutdown.requested():
        try:
            if drain(shutdown.requested) == 0:
                shutdown.wait(interval)
        except Exception:
            # Logged rather than raised: exiting here would stop the pipeline
            # until somebody noticed, and the lease recovers the held row.
            log.exception("drain failed; retrying after %.0fs", interval)
            shutdown.wait(interval)

    log.info("stopped with nothing claimed")
