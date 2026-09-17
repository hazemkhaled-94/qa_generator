"""The stage routes as the orchestrator reads them.

Against a stub rather than a running api: what is worth pinning here is
how a queue is read - when it counts as drained, when it has failed rows -
and that is arithmetic on the body, not HTTP.
"""

from __future__ import annotations

import pytest
import requests

from orchestration.client import Backend, Queue

BASE = "http://api:8000"


class Session:
    """Answers each path with a body, and records what was asked."""

    def __init__(self, replies: dict[str, list[dict]]) -> None:
        """Initialises with one queue of replies per path."""
        self._replies = {path: list(bodies) for path, bodies in replies.items()}
        self.asked: list[tuple[str, str]] = []

    def request(self, method: str, url: str, timeout: float):
        """Answers one call."""
        del timeout
        path = url[len(BASE) :]
        self.asked.append((method, path))
        bodies = self._replies[path]
        # The last reply stands for every call after it, which is what
        # lets a drain be given "busy, busy, idle" and then be left alone.
        body = bodies.pop(0) if len(bodies) > 1 else bodies[0]
        return Reply(body)


class Reply:
    """One response."""

    def __init__(self, body: dict) -> None:
        """Initialises with the parsed body."""
        self._body = body

    def raise_for_status(self) -> None:
        """Accepts every status; the refusal path is tested separately."""

    def json(self) -> dict:
        """The body."""
        return self._body


def queue(**rows: int) -> dict:
    """A /status body with these counts and nobody working."""
    return {"stage": "parsing", "working": False, "rows": rows}


def test_a_queue_with_nothing_claimable_is_idle() -> None:
    """Which is what an asset waits for."""
    assert Queue("parsing", working=False, rows={"parsed": 8}).idle


def test_rows_nobody_asked_for_do_not_make_a_queue_busy() -> None:
    """`new` waits on a person, not on a worker.

    An asset that treated `new` as outstanding would wait out its whole
    timeout on a corpus nobody had started.
    """
    assert Queue("parsing", working=False, rows={"new": 400}).idle


def test_a_worker_holding_a_row_makes_a_queue_busy() -> None:
    """Even with nothing pending: the row it holds is not done."""
    state = Queue("parsing", working=True, rows={"in_progress": 1})

    assert not state.idle
    assert state.outstanding == 1


def test_failed_rows_do_not_hold_a_queue_open() -> None:
    """A failure is recorded and waits for `retry`, which is a decision.

    Waiting for them would mean every run after a single bad document
    sitting until its timeout.
    """
    state = Queue("parsing", working=False, rows={"failed": 3, "parsed": 5})

    assert state.idle
    assert state.failed == 3


def test_drain_polls_until_the_queue_empties() -> None:
    """And returns the state it stopped on."""
    session = Session(
        {
            "/parsing/status": [
                queue(pending=4),
                queue(pending=1),
                queue(parsed=8),
            ]
        }
    )
    backend = Backend(BASE, session)

    state = backend.drain("parsing", timeout=5, poll=0)

    assert state.rows == {"parsed": 8}
    assert len(session.asked) == 3


def test_drain_gives_up_rather_than_waiting_for_ever() -> None:
    """Giving up watching is not failing the rows, and says so."""
    session = Session({"/parsing/status": [queue(pending=4)]})

    with pytest.raises(TimeoutError, match="only this run stopped waiting"):
        Backend(BASE, session).drain("parsing", timeout=0, poll=0)


def test_a_verb_reports_what_it_moved() -> None:
    """The count is the route's, not a guess from the status after it."""
    session = Session({"/parsing/start": [{"stage": "parsing", "rows": 12}]})

    assert Backend(BASE, session).act("parsing", "start") == 12
    assert session.asked == [("POST", "/parsing/start")]


def test_a_fit_is_asked_for_rather_than_started() -> None:
    """Topic modelling has no `start`: asking is what creates the work."""
    session = Session({"/topics/discover": [{"fit": 7, "detail": "queued"}]})

    assert Backend(BASE, session).discover() == 7
    assert session.asked == [("POST", "/topics/discover")]


def test_the_backend_url_comes_from_the_environment(monkeypatch) -> None:
    """One address, given by compose, and no default anywhere."""
    monkeypatch.setenv("BACKEND_URL", "http://api:8000/")

    # Trailing slash stripped, or every path would carry a double slash.
    assert Backend(session=Session({})).status.__self__._base_url == "http://api:8000"


def test_an_error_status_is_raised_rather_than_read(monkeypatch) -> None:
    """A refusal must not be parsed as a queue with no rows."""

    class Refusing(Session):
        """Raises the way requests does on an error status."""

        def request(self, method, url, timeout):
            """Refuses."""
            raise requests.exceptions.HTTPError("503 Service Unavailable")

    with pytest.raises(requests.exceptions.HTTPError):
        Backend(BASE, Refusing({})).status("parsing")
