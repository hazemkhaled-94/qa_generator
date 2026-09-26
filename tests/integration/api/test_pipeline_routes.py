"""The corpus as one thing, over HTTP.

Two halves. The queue verbs are the backend's own and are tested against a
real database: what `start` queues, what `stop` takes back, what `retry`
returns. The orchestrated half is tested against a stand-in, because
launching the real `corpus` job would take the corpus through every stage
and call a model several thousand times.

The stand-in matters less than it looks: what the API sends Dagster is
proved against a live webserver in `tests/smoke/test_orchestrator_queries.py`
and what each reply means in `tests/unit/api/test_orchestrator.py`. What is
left for here is the part those cannot see - which HTTP status each outcome
becomes, and that a missing orchestrator takes nothing else down with it.
"""

from __future__ import annotations

import pytest
from seed import digest, document, passage
from sqlalchemy.orm import Session

from api.orchestrator import Automation, Orchestration, Refused, Run, Unavailable
from database.qa_generator import Status

pytestmark = pytest.mark.integration


class FakeOrchestrator:
    """An orchestrator that answers what a test told it to."""

    def __init__(
        self,
        *,
        available: bool = True,
        running: Run | None = None,
        launches: Run | Exception | None = None,
        terminates: bool | Exception = True,
    ) -> None:
        """Holds the answers, and records what it was asked to do."""
        self._available = available
        self._running = running
        self._launches = launches or Run(id="new-run", status="QUEUED")
        self._terminates = terminates
        self.launched = 0
        self.terminated: list[str] = []

    def state(self, *, recent: int = 5) -> Orchestration:
        """What the orchestrator is doing."""
        if not self._available:
            return Orchestration(detail="DAGSTER_URL is not set")
        return Orchestration(
            available=True,
            running=self._running,
            recent=[one for one in (self._running,) if one],
            automation=Automation(available=True),
        )

    def launch(self) -> Run:
        """Starts the corpus job."""
        if not self._available:
            raise Unavailable("DAGSTER_URL is not set")
        self.launched += 1
        if isinstance(self._launches, Exception):
            raise self._launches
        return self._launches

    def terminate(self, run_id: str) -> bool:
        """Stops one run."""
        self.terminated.append(run_id)
        if isinstance(self._terminates, Exception):
            raise self._terminates
        return self._terminates

    def runs(self, *, limit: int = 20) -> list[Run]:
        """Recent runs."""
        if not self._available:
            raise Unavailable("DAGSTER_URL is not set")
        return [one for one in (self._running,) if one]

    def automate(self, *, on_arrival, nightly) -> Automation:
        """Switches the triggers."""
        if not self._available:
            raise Unavailable("DAGSTER_URL is not set")
        return Automation(
            available=True, on_arrival=bool(on_arrival), nightly=bool(nightly)
        )


@pytest.fixture
def orchestrator(monkeypatch):
    """Replaces the orchestrator the pipeline routes hold.

    The routes name it at module level rather than through `Depends`, which
    is what the rest of this API does and what keeps the wiring real; so
    this patches the name rather than overriding a dependency.
    """

    def install(**kwargs) -> FakeOrchestrator:
        fake = FakeOrchestrator(**kwargs)
        monkeypatch.setattr("api.routes.pipeline.orchestrator", fake)
        return fake

    return install


@pytest.fixture
def corpus(client, engine):
    """A corpus part way through: one document parsed, one not."""
    with Session(engine) as session:
        session.add_all(
            [
                document(digest("a"), parse_status=Status.PARSED),
                document(digest("b"), parse_status=Status.NEW),
            ]
        )
        session.flush()
        session.add_all([passage(digest("a"), ordinal=1)])
        session.commit()
    return client


# ── Reading ────────────────────────────────────────────────────────────────


def test_one_call_answers_every_stage_and_the_orchestrator(corpus, orchestrator):
    """What a caller who is not using the frontend polls."""
    orchestrator()

    answered = corpus.get("/pipeline")

    assert answered.status_code == 200
    body = answered.json()
    assert [one["stage"] for one in body["stages"]] == [
        "parsing",
        "chunking",
        "extraction",
        "topics",
        "questions",
        "assessment",
    ]
    assert body["orchestration"]["available"] is True
    assert body["working"] is False
    assert body["failed"] == 0


def test_the_queues_still_answer_when_there_is_no_orchestrator(corpus, orchestrator):
    """`orchestration/` is optional, and its absence is not a broken API."""
    orchestrator(available=False)

    body = corpus.get("/pipeline").json()

    assert body["orchestration"]["available"] is False
    assert body["orchestration"]["automation"]["available"] is False
    assert len(body["stages"]) == 6, "every queue still reports"


def test_work_in_flight_is_reported_whoever_set_it_going(corpus, orchestrator):
    """`make extract` and the Start button move rows without a run."""
    orchestrator()
    corpus.post("/parsing/start")

    assert corpus.get("/pipeline").json()["working"] is True


def test_failures_are_totalled_across_every_stage(client, engine, orchestrator):
    """One number, because they are spread over five queues."""
    orchestrator()
    with Session(engine) as session:
        session.add(document(digest("a"), parse_status=Status.FAILED))
        session.commit()

    assert client.get("/pipeline").json()["failed"] == 1


# ── Starting one wave ──────────────────────────────────────────────────────


def test_start_queues_what_is_ready_and_skips_what_is_not(corpus, orchestrator):
    """The guard that makes a fan-out safe at all.

    Chunking queues over `documents`, and a document exists from upload. The
    unparsed one must not be queued: its only possible outcome is a failure
    on the missing parsed object.
    """
    orchestrator()

    answered = corpus.post("/pipeline/start")

    assert answered.status_code == 202
    moved = {one["stage"]: one["rows"] for one in answered.json()["stages"]}
    assert moved["parsing"] == 1, "only the one parsing has not done yet"
    assert moved["chunking"] == 1, "only the parsed one can be chunked"
    assert moved["extraction"] == 1, "the passage the parsed one already has"


def test_start_leaves_the_fit_and_the_judge_alone(corpus, orchestrator):
    """A refit replaces every topic and the judge costs model calls.

    Both are asked for rather than swept into a fan-out, so neither is in
    the breakdown at all.
    """
    orchestrator()

    moved = {one["stage"] for one in corpus.post("/pipeline/start").json()["stages"]}

    assert "topics" not in moved and "assessment" not in moved


def test_starting_a_corpus_with_nothing_ready_says_so(client, orchestrator):
    """The empty answer is a sentence, not a zero."""
    orchestrator()

    body = client.post("/pipeline/start").json()

    assert body["rows"] == 0
    assert "nothing is ready" in body["detail"]


def test_start_needs_no_orchestrator(corpus, orchestrator):
    """These are queue verbs, and the queue is the backend's own."""
    orchestrator(available=False)

    assert corpus.post("/pipeline/start").status_code == 202


# ── Stopping and retrying ──────────────────────────────────────────────────


def test_stop_empties_every_queue_and_stops_the_run_that_filled_it(
    corpus, orchestrator
):
    """Either half alone leaves the corpus moving."""
    live = Run(id="run-7", status="STARTED")
    fake = orchestrator(running=live)
    corpus.post("/pipeline/start")

    answered = corpus.post("/pipeline/stop")

    assert answered.status_code == 202
    body = answered.json()
    assert body["rows"] > 0
    assert body["run"]["id"] == "run-7"
    assert fake.terminated == ["run-7"]
    assert corpus.get("/pipeline").json()["working"] is False


def test_stopping_with_no_run_in_flight_still_empties_the_queues(corpus, orchestrator):
    """Most of what Stop means is the queues."""
    fake = orchestrator(running=None)
    corpus.post("/pipeline/start")

    body = corpus.post("/pipeline/stop").json()

    assert body["run"] is None
    assert body["rows"] > 0
    assert fake.terminated == []


def test_a_run_that_cannot_be_stopped_does_not_stop_the_queues_emptying(
    corpus, orchestrator
):
    """A Dagster that is down must not stop a person stopping their pipeline."""
    orchestrator(running=Run(id="run-7", status="STARTED"), terminates=Refused("no"))
    corpus.post("/pipeline/start")

    answered = corpus.post("/pipeline/stop")

    assert answered.status_code == 202
    assert answered.json()["rows"] > 0
    assert corpus.get("/pipeline").json()["working"] is False


def test_retry_returns_every_stage_s_failures(client, engine, orchestrator):
    """What a model outage leaves behind, in one call."""
    orchestrator()
    with Session(engine) as session:
        session.add(
            document(
                digest("a"),
                parse_status=Status.FAILED,
                chunk_status=Status.FAILED,
            )
        )
        session.commit()

    body = client.post("/pipeline/retry").json()

    assert body["rows"] == 2
    assert {one["stage"] for one in body["stages"] if one["rows"]} == {
        "parsing",
        "chunking",
    }
    assert client.get("/pipeline").json()["failed"] == 0


def test_retry_with_nothing_failed_says_so(client, orchestrator):
    """And starts nothing that was never asked for."""
    orchestrator()

    body = client.post("/pipeline/retry").json()

    assert body["rows"] == 0 and body["detail"] == "nothing has failed."


# ── Running it end to end ──────────────────────────────────────────────────


def test_run_answers_at_once_with_the_run_it_started(client, orchestrator):
    """202: it does not wait for the corpus to go through."""
    fake = orchestrator()

    answered = client.post("/pipeline/run")

    assert answered.status_code == 202
    assert answered.json()["id"] == "new-run"
    assert fake.launched == 1


def test_a_second_run_is_refused_rather_than_racing_the_first(client, orchestrator):
    """Two runs would take the same queues through at once."""
    fake = orchestrator(running=Run(id="run-7", status="STARTED"))

    answered = client.post("/pipeline/run")

    assert answered.status_code == 409
    assert answered.json()["code"] == "already_running"
    assert "run-7" in answered.json()["detail"]
    assert fake.launched == 0


def test_run_without_an_orchestrator_says_what_to_use_instead(client, orchestrator):
    """503, and the refusal names the per-stage route that still works."""
    orchestrator(available=False)

    answered = client.post("/pipeline/run")

    assert answered.status_code == 503
    body = answered.json()
    assert body["code"] == "no_orchestrator"
    assert "/parsing/start" in body["detail"]


def test_an_orchestrator_that_refuses_is_reported_in_its_own_words(
    client, orchestrator
):
    """The message names which of a job or a location was missing."""
    orchestrator(launches=Refused("Could not find Pipeline corpus"))

    answered = client.post("/pipeline/run")

    assert answered.status_code == 503
    assert "Could not find Pipeline" in answered.json()["detail"]


def test_the_run_history_is_served_and_refused_together(client, orchestrator):
    """Present with an orchestrator, 503 without one."""
    orchestrator(running=Run(id="run-7", status="STARTED"))
    assert client.get("/pipeline/runs").json()[0]["id"] == "run-7"

    orchestrator(available=False)
    assert client.get("/pipeline/runs").status_code == 503


# ── Automation ─────────────────────────────────────────────────────────────


def test_automation_reads_as_off_when_there_is_nothing_watching(client, orchestrator):
    """The honest answer to "is anything going to start this by itself"."""
    orchestrator(available=False)

    body = client.get("/pipeline/automation").json()

    assert body == {
        "available": False,
        "on_arrival": False,
        "nightly": False,
        "detail": None,
    }


def test_a_trigger_can_be_switched_on_over_http(client, orchestrator):
    """`make auto` without make."""
    orchestrator()

    answered = client.put("/pipeline/automation", json={"on_arrival": True})

    assert answered.status_code == 200
    assert answered.json()["on_arrival"] is True


def test_switching_a_trigger_without_an_orchestrator_is_refused(client, orchestrator):
    """There is nothing to switch, and saying so beats answering False."""
    orchestrator(available=False)

    answered = client.put("/pipeline/automation", json={"nightly": True})

    assert answered.status_code == 503
    assert answered.json()["code"] == "no_orchestrator"
