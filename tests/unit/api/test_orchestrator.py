"""Dagster as the API calls it, without a Dagster.

Every reply here is one the real webserver gave: the shapes were taken off a
running 1.13 instance rather than written from the documentation, which is
the only way to be sure a union branch is spelt the way the server spells it.

What these cover is the half that is ours - which reply means started, which
means refused, and which means "this deployment has no orchestrator" - and
not GraphQL itself. The queries are separately proved against a live server
by `tests/smoke/test_orchestrator_queries.py`.
"""

from __future__ import annotations

import json

import pytest
import requests

from api.orchestrator import (
    JOB,
    LOCATION,
    REPOSITORY,
    SENSOR,
    Dagster,
    Refused,
    Unavailable,
)


class FakeResponse:
    """One canned GraphQL reply, at a status code."""

    def __init__(self, body: dict | None, status: int = 200) -> None:
        """Holds what the webserver would have answered.

        A body of None is a reply that is not GraphQL at all - a proxy's
        HTML error page - which `json()` raises on the way requests does.
        """
        self._body = body
        self.status_code = status

    def json(self) -> dict:
        """The parsed body, or the failure a non-JSON reply gives."""
        if self._body is None:
            raise ValueError("not json")
        return self._body


class FakeSession:
    """A session that answers from a script and records what it was asked.

    A reply may be a body, an exception to raise from `post`, or a
    `FakeResponse` where the status code matters.
    """

    def __init__(self, *replies: dict | Exception | FakeResponse) -> None:
        """Queues the replies, in the order the calls will come."""
        self.replies = list(replies)
        self.sent: list[dict] = []

    def post(self, url: str, json: dict, timeout: float):
        """Answers the next scripted reply."""
        self.sent.append({"url": url, **json})
        reply = self.replies.pop(0) if self.replies else {"data": {}}
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, FakeResponse) else FakeResponse(reply)


def runs(*statuses: str) -> dict:
    """A `runsOrError` carrying one run per status given."""
    return {
        "__typename": "Runs",
        "results": [
            {
                "runId": f"run-{index}",
                "status": status,
                "startTime": 1.0,
                "endTime": None,
            }
            for index, status in enumerate(statuses)
        ],
    }


def repository(*, sensor: str = "STOPPED", schedule: str = "STOPPED") -> dict:
    """A `repositoryOrError` with the two triggers in the states named."""
    return {
        "__typename": "Repository",
        "sensors": [
            {"name": SENSOR, "sensorState": {"id": "sensor-id", "status": sensor}},
            {"name": "progress", "sensorState": {"id": "other", "status": "RUNNING"}},
        ],
        "schedules": [
            {
                "name": "nightly_corpus",
                "scheduleState": {"id": "schedule-id", "status": schedule},
            }
        ],
    }


def dagster(*replies: dict | Exception) -> tuple[Dagster, FakeSession]:
    """A client wired to a scripted session."""
    session = FakeSession(*replies)
    return Dagster("http://dagster:3000", session=session), session


# ── Without an orchestrator ────────────────────────────────────────────────


def test_no_url_is_a_deployment_without_an_orchestrator() -> None:
    """Not an error: every queue verb still works, and this says so."""
    client = Dagster("", session=FakeSession())

    assert client.configured is False
    state = client.state()
    assert state.available is False
    assert state.running is None
    assert "DAGSTER_URL" in (state.detail or "")


def test_a_webserver_that_is_down_reads_as_unavailable_rather_than_raising() -> None:
    """What a page polls must not fail because the orchestrator is out."""
    client, _ = dagster(requests.exceptions.ConnectionError("refused"))

    state = client.state()

    assert state.available is False
    assert state.automation.available is False
    assert "not answering" in (state.detail or "")


def test_a_verb_against_a_missing_orchestrator_raises_unavailable() -> None:
    """`state` swallows it; anything that would decide something does not."""
    client = Dagster("", session=FakeSession())

    with pytest.raises(Unavailable):
        client.launch()


def test_a_reply_that_is_not_json_is_unavailable_not_a_crash() -> None:
    """A proxy answering HTML is a webserver that is not up."""
    client, _ = dagster(FakeResponse(None, status=502))

    assert client.state().available is False


def test_an_error_dagster_answers_with_a_500_is_still_an_answer() -> None:
    """The one that decided this is read by body rather than by status.

    `stopSensor` against an id no stored state has comes back 500 carrying
    a perfectly good GraphQL error. Reading the status first would report
    a live orchestrator as down and send a reader to the wrong logs.
    """
    client, _ = dagster(
        FakeResponse(
            {
                "data": {
                    "stopSensor": {
                        "__typename": "PythonError",
                        "message": "CheckError: Expected non-None value",
                    }
                }
            },
            status=500,
        ),
    )

    with pytest.raises(Refused, match="CheckError"):
        client._switch("sensor", SENSOR, "bad-id", on=False)


# ── Reading ────────────────────────────────────────────────────────────────


def test_state_reads_the_run_in_flight_and_the_two_triggers() -> None:
    """One round trip answers what a caller polls for."""
    client, session = dagster(
        {
            "data": {
                "runsOrError": runs("STARTED", "SUCCESS"),
                "repositoryOrError": repository(sensor="RUNNING"),
            }
        }
    )

    state = client.state()

    assert state.available is True
    assert state.running is not None and state.running.id == "run-0"
    assert [one.id for one in state.recent] == ["run-0", "run-1"]
    assert state.automation.on_arrival is True
    assert state.automation.nightly is False
    assert len(session.sent) == 1, "the panel polls this; it is one call"


@pytest.mark.parametrize("status", ["SUCCESS", "FAILURE", "CANCELED"])
def test_a_finished_run_is_not_a_run_in_flight(status: str) -> None:
    """Only a live run is what `run` refuses to start a second beside."""
    client, _ = dagster(
        {"data": {"runsOrError": runs(status), "repositoryOrError": repository()}}
    )

    assert client.state().running is None


@pytest.mark.parametrize("status", ["QUEUED", "NOT_STARTED", "STARTING", "CANCELING"])
def test_every_unfinished_status_counts_as_running(status: str) -> None:
    """A run Dagster has accepted but not started is still a run."""
    client, _ = dagster(
        {"data": {"runsOrError": runs(status), "repositoryOrError": repository()}}
    )

    assert client.state().running is not None


def test_a_trigger_this_orchestrator_does_not_have_reads_as_off() -> None:
    """An older `orchestration/` should report what it has, not fail."""
    client, _ = dagster(
        {
            "data": {
                "runsOrError": runs(),
                "repositoryOrError": {
                    "__typename": "Repository",
                    "sensors": [],
                    "schedules": [],
                },
            }
        }
    )

    automation = client.state().automation

    assert automation.available is True
    assert automation.on_arrival is False and automation.nightly is False


# ── Deciding ───────────────────────────────────────────────────────────────


def test_launch_names_the_job_and_the_code_location_that_holds_it() -> None:
    """The selector is what `configs/dagster/workspace.yaml` loads."""
    client, session = dagster(
        {
            "data": {
                "launchRun": {
                    "__typename": "LaunchRunSuccess",
                    "run": {"runId": "new-run", "status": "QUEUED"},
                }
            }
        }
    )

    launched = client.launch()

    assert launched.id == "new-run"
    assert launched.live is True
    selector = session.sent[0]["variables"]["params"]["selector"]
    assert selector == {
        "jobName": JOB,
        "repositoryName": REPOSITORY,
        "repositoryLocationName": LOCATION,
    }


def test_a_job_the_orchestrator_cannot_find_is_refused_with_its_own_words() -> None:
    """The message names which of a job, a location or a run was missing."""
    client, _ = dagster(
        {
            "data": {
                "launchRun": {
                    "__typename": "PipelineNotFoundError",
                    "message": "Could not find Pipeline corpus",
                }
            }
        }
    )

    with pytest.raises(Refused, match="Could not find Pipeline"):
        client.launch()


def test_a_failure_branch_with_no_message_is_named_by_its_type() -> None:
    """`RunConfigValidationInvalid` carries no message, and is not asked for one."""
    client, _ = dagster(
        {"data": {"launchRun": {"__typename": "RunConfigValidationInvalid"}}}
    )

    with pytest.raises(Refused, match="RunConfigValidationInvalid"):
        client.launch()


def test_graphql_errors_are_refusals_rather_than_empty_answers() -> None:
    """A query the server rejected must never read as "nothing happened"."""
    client, _ = dagster({"errors": [{"message": "Cannot query field 'nope'"}]})

    with pytest.raises(Refused, match="Cannot query field"):
        client.launch()


def test_terminate_reports_the_run_it_stopped() -> None:
    """True is "this call stopped it"."""
    client, _ = dagster(
        {
            "data": {
                "terminateRuns": {
                    "__typename": "TerminateRunsResult",
                    "terminateRunResults": [{"__typename": "TerminateRunSuccess"}],
                }
            }
        }
    )

    assert client.terminate("run-0") is True


def test_terminating_a_run_that_already_finished_is_not_a_failure() -> None:
    """A second press of Stop, and a run that ended while the page polled."""
    client, _ = dagster(
        {
            "data": {
                "terminateRuns": {
                    "__typename": "TerminateRunsResult",
                    "terminateRunResults": [
                        {
                            "__typename": "TerminateRunFailure",
                            "message": "Run has already finished.",
                        }
                    ],
                }
            }
        }
    )

    assert client.terminate("run-0") is False


def test_a_run_that_could_not_be_stopped_for_another_reason_is_refused() -> None:
    """Anything but "already finished" is a Stop that did not happen."""
    client, _ = dagster(
        {
            "data": {
                "terminateRuns": {
                    "__typename": "TerminateRunsResult",
                    "terminateRunResults": [
                        {
                            "__typename": "TerminateRunFailure",
                            "message": "no run coordinator",
                        }
                    ],
                }
            }
        }
    )

    with pytest.raises(Refused, match="no run coordinator"):
        client.terminate("run-0")


# ── Automation ─────────────────────────────────────────────────────────────


def test_switching_a_trigger_on_sends_the_selector_and_reads_back_the_state() -> None:
    """Start takes a selector, which is Dagster's own asymmetry."""
    client, session = dagster(
        {"data": {"repositoryOrError": repository()}},
        {"data": {"startSensor": {"__typename": "Sensor"}}},
        {"data": {"repositoryOrError": repository(sensor="RUNNING")}},
    )

    automation = client.automate(on_arrival=True, nightly=None)

    assert automation.on_arrival is True
    started = json.dumps(session.sent[1])
    assert SENSOR in started and "sensorName" in started


def test_switching_a_trigger_off_sends_the_id_of_its_stored_state() -> None:
    """Stop takes an id, and the id comes from the read before it."""
    client, session = dagster(
        {"data": {"repositoryOrError": repository(sensor="RUNNING")}},
        {"data": {"stopSensor": {"__typename": "StopSensorMutationResult"}}},
        {"data": {"repositoryOrError": repository()}},
    )

    automation = client.automate(on_arrival=False, nightly=None)

    assert automation.on_arrival is False
    assert session.sent[1]["variables"] == {"id": "sensor-id"}


def test_setting_a_trigger_to_what_it_already_is_sends_no_mutation() -> None:
    """Idempotent, because a page redraws this every few seconds."""
    client, session = dagster(
        {"data": {"repositoryOrError": repository(sensor="RUNNING")}},
        {"data": {"repositoryOrError": repository(sensor="RUNNING")}},
    )

    client.automate(on_arrival=True, nightly=None)

    assert len(session.sent) == 2, "one read, one read back, and no mutation"


def test_a_trigger_left_out_is_left_alone() -> None:
    """None means "do not touch", which is what a one-switch form sends."""
    client, session = dagster(
        {
            "data": {
                "repositoryOrError": repository(sensor="RUNNING", schedule="RUNNING")
            }
        },
        {
            "data": {
                "repositoryOrError": repository(sensor="RUNNING", schedule="RUNNING")
            }
        },
    )

    automation = client.automate(on_arrival=None, nightly=None)

    assert automation.on_arrival is True and automation.nightly is True
    assert len(session.sent) == 2


def test_a_refused_switch_says_so_rather_than_reporting_the_old_state() -> None:
    """A switch that did not move must not answer as if it had."""
    client, _ = dagster(
        {"data": {"repositoryOrError": repository()}},
        {
            "data": {
                "startSensor": {
                    "__typename": "UnauthorizedError",
                    "message": "not permitted",
                }
            }
        },
    )

    with pytest.raises(Refused, match="not permitted"):
        client.automate(on_arrival=True, nightly=None)
