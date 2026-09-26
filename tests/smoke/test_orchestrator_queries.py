"""The GraphQL the API sends, against the Dagster that has to accept it.

`tests/unit/api/test_orchestrator.py` proves what each reply means. It cannot
prove that the reply would ever arrive: a field renamed between Dagster
versions, or a union branch that does not carry `message`, is a query the
server rejects outright, and a fake session answers it happily forever.

So this sends every document to the running webserver and asserts only that
it was accepted. **Nothing here launches a run, stops one, or moves a
switch.** Each mutation is sent against a name that does not exist, which
exercises the parser and the whole union - a query with a bad field is
refused before Dagster ever looks at the name - while the only possible
outcome is a `NotFoundError`.

Skipped when there is no webserver, like the rest of the smoke layer.
"""

from __future__ import annotations

import pytest
import requests

from api import orchestrator

pytestmark = pytest.mark.smoke

#: Where `make up` publishes the webserver. The API reaches it over the
#: compose network as `http://dagster-webserver:3000`; from the host it is
#: this, and the queries are the same either way.
URL = "http://localhost:3000/graphql"

#: Names no code location defines, so every mutation below is a lookup that
#: fails after the query has been parsed and validated.
ABSENT = "no_such_thing_this_test_invented"


def _send(document: str, **variables) -> dict:
    """Sends one document, or skips if the webserver is not up.

    The status code is not what decides. Dagster answers a mutation it
    could not carry out with a 500 whose body is a valid GraphQL error, and
    that is an accepted query - which is the only thing this file asserts.
    Only a reply that is not GraphQL at all means there is nothing there.
    """
    try:
        response = requests.post(
            URL, json={"query": document, "variables": variables}, timeout=10
        )
    except requests.exceptions.RequestException as down:
        pytest.skip(f"no Dagster webserver at {URL}: {down}")
    try:
        return response.json()
    except ValueError:
        pytest.skip(f"no Dagster webserver at {URL}: replied {response.status_code}")


def _accepted(document: str, **variables) -> dict:
    """Asserts the server parsed and validated the document, and returns data."""
    body = _send(document, **variables)
    assert not body.get("errors"), (
        "Dagster rejected the query the API sends:\n"
        + "\n".join(error.get("message", "") for error in body["errors"])
        + "\n\nA field or a union branch has moved. Introspect the schema "
        "and correct backend/api/orchestrator.py."
    )
    return body.get("data") or {}


def _selector() -> dict:
    """The repository the queries name."""
    return {
        "repositoryName": orchestrator.REPOSITORY,
        "repositoryLocationName": orchestrator.LOCATION,
    }


def test_the_state_query_is_accepted_and_finds_this_code_location() -> None:
    """The one call the pipeline panel polls.

    Also the one that proves `LOCATION` and `REPOSITORY` are what this
    deployment actually loaded, rather than what they were when written.
    """
    data = _accepted(orchestrator._STATE, limit=1, **_selector())

    assert data["runsOrError"]["__typename"] == "Runs"
    assert data["repositoryOrError"]["__typename"] == "Repository", (
        "the API names a code location this Dagster does not have"
    )


def test_the_corpus_job_exists_under_the_name_the_api_launches() -> None:
    """A renamed job would fail only when somebody pressed Run."""
    data = _accepted(
        """
        query($repositoryName: String!, $repositoryLocationName: String!) {
          repositoryOrError(repositorySelector: {
            repositoryName: $repositoryName,
            repositoryLocationName: $repositoryLocationName
          }) { ... on Repository { jobs { name } } }
        }
        """,
        **_selector(),
    )

    named = {job["name"] for job in data["repositoryOrError"]["jobs"]}
    assert orchestrator.JOB in named, f"{orchestrator.JOB} is not among {named}"


def test_the_two_triggers_exist_under_the_names_the_api_switches() -> None:
    """`arrivals` and `nightly_corpus`, as `orchestration/` defines them."""
    data = _accepted(orchestrator._TRIGGERS, **_selector())

    repository = data["repositoryOrError"]
    assert orchestrator.SENSOR in {one["name"] for one in repository["sensors"]}
    assert orchestrator.SCHEDULE in {one["name"] for one in repository["schedules"]}


def test_the_runs_query_is_accepted() -> None:
    """What `GET /pipeline/runs` sends."""
    data = _accepted(
        orchestrator._RUNS, limit=1, filter={"pipelineName": orchestrator.JOB}
    )

    assert data["runsOrError"]["__typename"] in (
        "Runs",
        "InvalidPipelineRunsFilterError",
    )


def test_the_launch_mutation_is_accepted_and_nothing_is_launched() -> None:
    """Sent against a job that does not exist, so no run is created."""
    data = _accepted(
        orchestrator._LAUNCH,
        params={
            "selector": {
                "jobName": ABSENT,
                "repositoryName": orchestrator.REPOSITORY,
                "repositoryLocationName": orchestrator.LOCATION,
            },
            "mode": "default",
        },
    )

    assert data["launchRun"]["__typename"] == "PipelineNotFoundError"


def test_the_terminate_mutation_is_accepted_and_nothing_is_stopped() -> None:
    """Sent against a run id that cannot exist."""
    data = _accepted(
        orchestrator._TERMINATE, runId="00000000-0000-0000-0000-000000000000"
    )

    answer = data["terminateRuns"]
    assert answer["__typename"] == "TerminateRunsResult"
    assert answer["terminateRunResults"][0]["__typename"] == "RunNotFoundError"


@pytest.mark.parametrize("kind", ["sensor", "schedule"])
def test_the_start_mutations_are_accepted_and_nothing_is_switched_on(
    kind: str,
) -> None:
    """Sent against a trigger that does not exist, so nothing starts."""
    data = _accepted(
        orchestrator._START[kind], selector={**_selector(), f"{kind}Name": ABSENT}
    )

    assert data[f"start{kind.capitalize()}"]["__typename"].endswith("NotFoundError")


@pytest.mark.parametrize("kind", ["sensor", "schedule"])
def test_the_stop_mutations_are_accepted_and_nothing_is_switched_off(
    kind: str,
) -> None:
    """Sent against an id no stored state has.

    Dagster answers this one with a `PythonError` rather than a not-found,
    which is the reply and not the point: what is asserted is that the
    document was parsed and validated, which a rejected query never is.
    """
    body = _send(orchestrator._STOP[kind], id=f"{ABSENT}::{ABSENT}")

    assert not body.get("errors"), body.get("errors")
