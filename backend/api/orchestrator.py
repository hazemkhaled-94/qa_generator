"""Dagster, as the API asks it to decide something.

The one place the backend calls the orchestrator rather than the other way
round, and the direction is worth being precise about. `orchestration/` still
holds every decision: the asset graph is what knows a stage waits for the one
before it, the `corpus` job is what runs them in order, and the sensor is what
watches for uploads. Nothing of that is duplicated here. This asks for the
job that already exists, by name, and reads back what Dagster says about it.

**Why it has to be asked rather than done here.** Starting a corpus means
starting a stage, waiting for the workers to drain it, and starting the next -
and something has to hold that wait. The shell holds it for `make corpus` and
a Dagster run holds it for everything else. A request cannot: the first stage
alone was measured in hours, and `api/routes/stage.py` refuses to do a stage's
work in the process serving JSON for the same reason.

**It is optional, and says so.** `DAGSTER_URL` unset, or a webserver that is
not up, answers `available: false` rather than raising. Deleting
`orchestration/` from a deployment still leaves every queue verb working on
every surface; what it takes away is the one control that runs them in order,
and that control then reports itself missing instead of failing.

GraphQL because that is the only remote surface Dagster has. The queries are
pinned against 1.13 and `tests/unit/api/test_orchestrator.py` records the
shapes they expect.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import requests

from settings.env import optional

log = logging.getLogger(__name__)

#: The code location and repository `configs/dagster/workspace.yaml` loads.
#: `__repository__` is what Dagster calls the one a bare `Definitions` makes.
LOCATION = "qa_generator"
REPOSITORY = "__repository__"

#: The job that runs every stage in the order a document moves through them.
#: Defined in `orchestration/__init__.py`; named here and built nowhere.
JOB = "corpus"

#: The trigger that starts a run when documents are sitting `new`, and the
#: clock that starts one nightly. Both ship stopped.
SENSOR = "arrivals"
SCHEDULE = "nightly_corpus"

#: Short: every call is a local HTTP round trip to a webserver that answers
#: from its own database, and a page polls this. A slow orchestrator must
#: read as "not answering" quickly rather than hold a request open.
_TIMEOUT = 10.0

#: Run states Dagster reports for a run that has not finished. A run in any
#: of them is why `POST /pipeline/run` refuses to start a second.
LIVE = ("QUEUED", "NOT_STARTED", "STARTING", "STARTED", "MANAGED", "CANCELING")


class Unavailable(Exception):
    """Dagster is not configured, or not answering."""


class Refused(Exception):
    """Dagster answered, and said no."""


@dataclass(frozen=True)
class Run:
    """One run of the corpus job, as Dagster records it."""

    id: str
    status: str
    started_at: float | None = None
    ended_at: float | None = None

    @property
    def live(self) -> bool:
        """Whether this run is still going."""
        return self.status in LIVE


@dataclass(frozen=True)
class Automation:
    """Whether anything is set to start a run without being asked.

    Two switches rather than one because they answer different questions:
    the sensor is "when a document arrives", the schedule is "at 02:00".
    """

    #: False when Dagster is not configured or not answering, in which case
    #: both switches read False and neither can be set.
    available: bool = False
    on_arrival: bool = False
    nightly: bool = False
    detail: str | None = None


@dataclass(frozen=True)
class Orchestration:
    """What the orchestrator is doing, as one answer.

    `available` False is not an error: it is a deployment without
    `orchestration/`, and every queue verb still works on every surface.
    """

    available: bool = False
    running: Run | None = None
    recent: list[Run] = field(default_factory=list)
    automation: Automation = field(default_factory=Automation)
    detail: str | None = None


class Dagster:
    """The orchestrator's GraphQL surface, as the API calls it.

    Built once and held, like every other dependency. The session is reused,
    so a page polling `/pipeline` costs one connection rather than one per
    poll.
    """

    def __init__(self, url: str | None = None, session: Any = None) -> None:
        """Binds to DAGSTER_URL, which may be unset.

        Args:
            url: The webserver's base address. Absent, DAGSTER_URL is read,
                and absent from there too this reports itself unavailable.
            session: A requests session to reuse, for tests.
        """
        self._url = (url or optional("DAGSTER_URL") or "").rstrip("/")
        self._session = session or requests.Session()

    @property
    def configured(self) -> bool:
        """Whether this deployment named an orchestrator at all."""
        return bool(self._url)

    def _query(self, document: str, **variables: Any) -> dict:
        """Sends one GraphQL document and returns its `data`.

        The status code is deliberately not what decides between the two.
        Dagster answers a mutation it could not carry out with a 500 whose
        body is a perfectly good GraphQL error - `stopSensor` against an id
        no stored state has does exactly that - and reading the status first
        turns "the orchestrator said no" into "the orchestrator is down",
        which sends a reader to look at the wrong thing. A body that parses
        is an answer; only a body that does not is an outage.

        Raises:
            Unavailable: If there is no URL, the webserver cannot be reached,
                or it answered with something that is not a GraphQL body.
            Refused: If the query ran and Dagster reported an error. The
                message is Dagster's, because it is the one that says which
                of a job, a sensor or a run it could not find.
        """
        if not self._url:
            raise Unavailable(
                "DAGSTER_URL is not set, so this deployment has no "
                "orchestrator. Every stage still answers its own routes."
            )
        try:
            response = self._session.post(
                f"{self._url}/graphql",
                json={"query": document, "variables": variables},
                timeout=_TIMEOUT,
            )
        except requests.exceptions.RequestException as unreachable:
            raise Unavailable(
                f"the orchestrator at {self._url} is not answering: {unreachable}"
            ) from unreachable
        try:
            body = response.json()
        except ValueError as unreadable:
            raise Unavailable(
                f"the orchestrator at {self._url} is not answering: it replied "
                f"{response.status_code} with something that is not GraphQL"
            ) from unreadable
        if body.get("errors"):
            raise Refused(
                "; ".join(error.get("message", "") for error in body["errors"]).strip()
                or "the orchestrator refused the request"
            )
        return body.get("data") or {}

    # ── Reading ────────────────────────────────────────────────────────────

    def state(self, *, recent: int = 5) -> Orchestration:
        """Everything the orchestrator has to say, in one round trip.

        Never raises. A deployment without an orchestrator and one whose
        webserver is down are both `available: false` with the reason in
        `detail`, because this is what a page polls and what a caller asks
        before deciding whether to offer a Run button at all.
        """
        try:
            data = self._query(_STATE, limit=recent, **_selector())
        except (Unavailable, Refused) as missing:
            return Orchestration(detail=str(missing))

        runs = [_run(one) for one in _results(data.get("runsOrError"))]
        triggers = _triggers(data.get("repositoryOrError") or {})
        return Orchestration(
            available=True,
            running=next((one for one in runs if one.live), None),
            recent=runs,
            automation=Automation(available=True, **triggers),
        )

    def runs(self, *, limit: int = 20) -> list[Run]:
        """The corpus job's recent runs, newest first.

        Raises:
            Unavailable: If the orchestrator is not configured or not up.
        """
        data = self._query(_RUNS, limit=limit, filter={"pipelineName": JOB})
        return [_run(one) for one in _results(data.get("runsOrError"))]

    # ── Deciding ───────────────────────────────────────────────────────────

    def launch(self) -> Run:
        """Starts the corpus job and returns the run it created.

        Raises:
            Unavailable: If the orchestrator is not configured or not up.
            Refused: If Dagster would not start it.
        """
        data = self._query(
            _LAUNCH,
            params={
                "selector": {
                    "jobName": JOB,
                    "repositoryName": REPOSITORY,
                    "repositoryLocationName": LOCATION,
                },
                "mode": "default",
            },
        )
        answer = data.get("launchRun") or {}
        if answer.get("__typename") != "LaunchRunSuccess":
            raise Refused(_why(answer, f"the orchestrator would not start {JOB}"))
        launched = answer["run"]
        log.info("pipeline: run %s of %s launched", launched["runId"], JOB)
        return _run(launched)

    def terminate(self, run_id: str) -> bool:
        """Stops one run, and reports whether it was this call that did.

        Returns:
            True if the run was terminated, False if it had already finished
            - which is not a failure, and is what a second press of Stop
            answers with.

        Raises:
            Unavailable: If the orchestrator is not configured or not up.
            Refused: If Dagster would not stop it for any other reason.
        """
        data = self._query(_TERMINATE, runId=run_id)
        answer = data.get("terminateRuns") or {}
        if answer.get("__typename") != "TerminateRunsResult":
            raise Refused(_why(answer, f"the orchestrator would not stop {run_id}"))
        one = (answer.get("terminateRunResults") or [{}])[0]
        if one.get("__typename") == "TerminateRunSuccess":
            log.info("pipeline: run %s terminated", run_id)
            return True
        if one.get("__typename") == "TerminateRunFailure" and "finished" in (
            one.get("message") or ""
        ):
            return False
        raise Refused(_why(one, f"the orchestrator would not stop {run_id}"))

    def automate(self, *, on_arrival: bool | None, nightly: bool | None) -> Automation:
        """Switches the two triggers on or off, leaving None alone.

        Reads the current state first, because stopping either needs the id
        of its stored state and because setting a switch to what it already
        is should cost nothing.

        Raises:
            Unavailable: If the orchestrator is not configured or not up.
            Refused: If Dagster would not move one of them.
        """
        data = self._query(_TRIGGERS, **_selector())
        held = _triggers(data.get("repositoryOrError") or {}, ids=True)

        if on_arrival is not None and on_arrival != held["on_arrival"]:
            self._switch("sensor", SENSOR, held["sensor_id"], on=on_arrival)
        if nightly is not None and nightly != held["nightly"]:
            self._switch("schedule", SCHEDULE, held["schedule_id"], on=nightly)

        after = _triggers(
            self._query(_TRIGGERS, **_selector()).get("repositoryOrError") or {}
        )
        return Automation(available=True, **after)

    def _switch(self, kind: str, name: str, state_id: str, *, on: bool) -> None:
        """Starts or stops one trigger.

        Starting takes the selector and stopping takes the id of the stored
        state, which is Dagster's own asymmetry and not one worth hiding.
        """
        document, variables = (
            (_START[kind], {"selector": {**_selector(), f"{kind}Name": name}})
            if on
            else (_STOP[kind], {"id": state_id})
        )
        answer = self._query(document, **variables)
        result = answer.get(next(iter(answer), "")) or {}
        # Every failure branch of these four unions is named `*Error` and
        # carries a message; every success branch is neither. Checked both
        # ways so a union that grows a third shape is refused rather than
        # read as success.
        if "Error" in (result.get("__typename") or "") or result.get("message"):
            raise Refused(_why(result, f"the orchestrator would not switch {name}"))
        log.info("pipeline: %s %s %s", kind, name, "started" if on else "stopped")


def _selector() -> dict[str, str]:
    """The repository every query names."""
    return {"repositoryName": REPOSITORY, "repositoryLocationName": LOCATION}


def _results(answer: Any) -> list[dict]:
    """The runs out of a `runsOrError`, or none when it is an error."""
    if not isinstance(answer, dict) or answer.get("__typename") != "Runs":
        return []
    return answer.get("results") or []


def _run(row: dict) -> Run:
    """One run, as Dagster reports it."""
    return Run(
        id=row.get("runId", ""),
        status=row.get("status", "UNKNOWN"),
        started_at=row.get("startTime"),
        ended_at=row.get("endTime"),
    )


def _triggers(repository: dict, *, ids: bool = False) -> dict[str, Any]:
    """Whether each trigger is running, and optionally the id that stops it.

    A name Dagster does not know reads as off rather than raising: a
    deployment running an older `orchestration/` should report the sensor it
    has, not fail the whole panel over the one it does not.
    """
    sensors = {one["name"]: one for one in repository.get("sensors") or []}
    schedules = {one["name"]: one for one in repository.get("schedules") or []}
    sensor = sensors.get(SENSOR) or {}
    schedule = schedules.get(SCHEDULE) or {}
    found: dict[str, Any] = {
        "on_arrival": (sensor.get("sensorState") or {}).get("status") == "RUNNING",
        "nightly": (schedule.get("scheduleState") or {}).get("status") == "RUNNING",
    }
    if ids:
        found["sensor_id"] = (sensor.get("sensorState") or {}).get("id", "")
        found["schedule_id"] = (schedule.get("scheduleState") or {}).get("id", "")
    return found


def _why(answer: dict, fallback: str) -> str:
    """The reason out of whichever error shape Dagster answered with."""
    return answer.get("message") or answer.get("__typename") or fallback


#: What a run is read back as, everywhere one is.
_RUN_FIELDS = "runId status startTime endTime"

#: Whether each trigger is on, and the id that switches it off.
_TRIGGER_FIELDS = """
  sensors { name sensorState { id status } }
  schedules { name scheduleState { id status } }
"""

_STATE = f"""
query State($repositoryName: String!, $repositoryLocationName: String!, $limit: Int) {{
  runsOrError(filter: {{pipelineName: "{JOB}"}}, limit: $limit) {{
    __typename
    ... on Runs {{ results {{ {_RUN_FIELDS} }} }}
  }}
  repositoryOrError(repositorySelector: {{
    repositoryName: $repositoryName,
    repositoryLocationName: $repositoryLocationName
  }}) {{
    __typename
    ... on Repository {{ {_TRIGGER_FIELDS} }}
  }}
}}
"""

_RUNS = f"""
query Runs($filter: RunsFilter!, $limit: Int) {{
  runsOrError(filter: $filter, limit: $limit) {{
    __typename
    ... on Runs {{ results {{ {_RUN_FIELDS} }} }}
  }}
}}
"""

_TRIGGERS = f"""
query Triggers($repositoryName: String!, $repositoryLocationName: String!) {{
  repositoryOrError(repositorySelector: {{
    repositoryName: $repositoryName,
    repositoryLocationName: $repositoryLocationName
  }}) {{
    __typename
    ... on Repository {{ {_TRIGGER_FIELDS} }}
  }}
}}
"""

#: Only the failure branches that carry a `message` are asked for one:
#: `RunConfigValidationInvalid` and `InvalidSubsetError` do not have the
#: field, and asking anyway is a query the server rejects outright. `_why`
#: falls back to the `__typename`, which names those well enough.
_LAUNCH = f"""
mutation Launch($params: ExecutionParams!) {{
  launchRun(executionParams: $params) {{
    __typename
    ... on LaunchRunSuccess {{ run {{ {_RUN_FIELDS} }} }}
    ... on PythonError {{ message }}
    ... on PipelineNotFoundError {{ message }}
    ... on RunConflict {{ message }}
    ... on UnauthorizedError {{ message }}
  }}
}}
"""

_TERMINATE = """
mutation Terminate($runId: String!) {
  terminateRuns(runIds: [$runId]) {
    __typename
    ... on TerminateRunsResult {
      terminateRunResults {
        __typename
        ... on TerminateRunFailure { message }
        ... on RunNotFoundError { message }
      }
    }
    ... on PythonError { message }
  }
}
"""

#: Starting takes a selector; stopping takes the id of the stored state.
_START = {
    "sensor": """
mutation StartSensor($selector: SensorSelector!) {
  startSensor(sensorSelector: $selector) {
    __typename
    ... on PythonError { message }
    ... on UnauthorizedError { message }
  }
}
""",
    "schedule": """
mutation StartSchedule($selector: ScheduleSelector!) {
  startSchedule(scheduleSelector: $selector) {
    __typename
    ... on PythonError { message }
    ... on UnauthorizedError { message }
  }
}
""",
}

_STOP = {
    "sensor": """
mutation StopSensor($id: String!) {
  stopSensor(id: $id) {
    __typename
    ... on PythonError { message }
    ... on UnauthorizedError { message }
  }
}
""",
    "schedule": """
mutation StopSchedule($id: String!) {
  stopRunningSchedule(id: $id) {
    __typename
    ... on PythonError { message }
    ... on UnauthorizedError { message }
  }
}
""",
}
