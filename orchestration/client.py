"""The stage routes, as the orchestrator calls them.

HTTP and nothing else. The orchestrator holds no database URL and no object
store credentials on purpose: the queue has three faces already - the Start
button, the make targets and these routes - and a fourth that reached past
them into the tables would be a second way of moving a row, disagreeing with
the other three the first time one of them changed.

Every call here is one a person could make with curl. That is the test of
whether this stays an orchestrator rather than becoming a second backend.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import requests

log = logging.getLogger(__name__)

#: How long one call may take. Generous for a POST, because `rerun` over a
#: large corpus is one UPDATE over every row of a table; none of them does
#: the work, so none of them waits for a model.
_TIMEOUTS = {"GET": 15.0, "POST": 120.0}


class StageFailed(Exception):
    """A stage finished its queue with rows left failed."""


@dataclass(frozen=True)
class Queue:
    """How much work one stage has, as /status reports it."""

    stage: str
    working: bool
    rows: dict[str, int]

    @property
    def outstanding(self) -> int:
        """Rows a worker still has to take, or is holding right now."""
        return self.rows.get("pending", 0) + self.rows.get("in_progress", 0)

    @property
    def failed(self) -> int:
        """Rows this stage could not process."""
        return self.rows.get("failed", 0)

    @property
    def idle(self) -> bool:
        """Whether there is nothing left for a worker to do.

        Not the same as finished: a queue nobody has started is idle too,
        every row sitting `new`. That distinction is the caller's, because
        `new` is what `start` is for.
        """
        return self.outstanding == 0 and not self.working


class Backend:
    """The backend's stage surface."""

    def __init__(self, base_url: str | None = None, session=None) -> None:
        """Initialises the client against BACKEND_URL."""
        self._base_url = (base_url or os.environ["BACKEND_URL"]).rstrip("/")
        self._session = session or requests.Session()

    def status(self, stage: str) -> Queue:
        """Reports how much work one stage has waiting."""
        body = self._call("GET", f"/{stage}/status")
        return Queue(stage=stage, working=body["working"], rows=body["rows"])

    def act(self, stage: str, action: str) -> int:
        """Runs one queue verb over everything a stage owns.

        Returns:
            How many rows it moved. None of them does the work: the verb
            moves rows between statuses and whichever worker is watching
            picks up what became claimable.
        """
        moved = self._call("POST", f"/{stage}/{action}")["rows"]
        log.info("%s: %s moved %d row(s)", stage, action, moved)
        return moved

    def discover(self) -> int:
        """Asks for a topic fit over the whole corpus.

        Topic modelling's own verb. It has no `new` rows and so no `start`:
        every topic is estimated jointly over one vocabulary, so asking is
        what creates the work.
        """
        fit = self._call("POST", "/topics/discover")["fit"]
        log.info("topics: fit %s queued", fit)
        return fit

    def drain(self, stage: str, *, timeout: float, poll: float) -> Queue:
        """Waits until a stage has nothing claimable left.

        Polls rather than subscribes, which is what the workers do to the
        queue as well. The interval is the caller's: a stage whose unit
        costs a model call is not worth asking about every second.

        Raises:
            TimeoutError: If the queue has not drained in `timeout`
                seconds. Never a reason to fail the rows - they are still
                queued and a worker is still on them - so what this means
                is that the run gave up watching, not that the work did.
        """
        deadline = time.monotonic() + timeout
        while True:
            queue = self.status(stage)
            if queue.idle:
                return queue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"{stage} still has {queue.outstanding} row(s) after "
                    f"{timeout:.0f}s. The work is queued and the worker has "
                    f"it; only this run stopped waiting."
                )
            time.sleep(poll)

    def _call(self, method: str, path: str) -> dict:
        """Sends one request and returns the parsed body.

        Raises:
            requests.exceptions.RequestException: On a network failure or
                an error status, whose body carries the reason.
        """
        response = self._session.request(
            method, f"{self._base_url}{path}", timeout=_TIMEOUTS[method]
        )
        response.raise_for_status()
        return response.json()
