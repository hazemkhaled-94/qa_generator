"""The assessment routes over HTTP, which is what the page actually calls.

Every one of these was reachable in a unit test and none of them had been
called through the application. The Configuration panel is why that
mattered: `GET /settings/assessment` answered 422 for a service the
catalogue described perfectly, because the route keeps a Literal of its
own and nobody held the two together.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_the_plan_says_whether_the_phase_is_on(client) -> None:
    """What the page reads first, and the only route that answers when off."""
    answered = client.get("/assessment/plan")

    assert answered.status_code == 200
    plan = answered.json()
    assert plan["enabled"] is False, "the test environment never judges"
    assert set(plan["metrics"]) == {"fact", "topic", "question"}
    assert plan["metrics"]["question"] == [
        "hallucination",
        "qa_correctness",
        "relevance",
    ]
    assert plan["prompt_version"]


def test_the_queue_answers_its_status(client) -> None:
    """The five verbs come from the same factory every other stage uses."""
    answered = client.get("/assessment/status")

    assert answered.status_code == 200
    assert answered.json()["stage"] == "assessment"


def test_the_listing_is_empty_rather_than_absent(client) -> None:
    """A corpus nobody has judged is a page, not a 404."""
    answered = client.get("/assessment")

    assert answered.status_code == 200
    assert answered.json() == {"total": 0, "assessments": []}


def test_the_report_answers_with_nothing_judged(client) -> None:
    """Every figure is zero and none of them is missing."""
    answered = client.get("/assessment/quality")

    assert answered.status_code == 200
    quality = answered.json()
    assert quality["judged"] == 0
    assert quality["disagreements"] == 0
    assert quality["by_kind"] == {"fact": 0, "topic": 0, "question": 0}


@pytest.mark.parametrize("kind", ["fact", "topic", "question"])
def test_the_listing_narrows_to_one_kind(client, kind: str) -> None:
    """Which is how a reader asks for the questions alone."""
    answered = client.get("/assessment", params={"kind": kind})

    assert answered.status_code == 200


def test_a_kind_no_artefact_has_is_refused(client) -> None:
    """Before it reaches a query, because the Literal lists the three."""
    assert client.get("/assessment", params={"kind": "passage"}).status_code == 422


def test_the_configuration_panel_opens(client) -> None:
    """`GET /settings/assessment`, which the page draws its controls from.

    This is the regression. The catalogue described four settings under
    `assessment` and the route's own Literal did not list the service, so
    the panel answered 422 on one page and worked on the other six.
    """
    answered = client.get("/settings/assessment")

    assert answered.status_code == 200
    named = {one["name"] for one in answered.json()["settings"]}
    assert named == {
        "ASSESSMENT_ENABLED",
        "ASSESSMENT_JUDGE_MODEL",
        "ASSESSMENT_KINDS",
        "ASSESSMENT_SAMPLE",
    }


def test_the_queue_narrows_over_http(client) -> None:
    """`POST /assessment/kind/question/start`, which is the selective run."""
    answered = client.post("/assessment/kind/question/start")

    assert answered.status_code == 202
    assert answered.json()["scope"] == "kind"


def test_a_scope_this_stage_does_not_take_is_a_404(client) -> None:
    """The queue narrows to a kind and to nothing else."""
    answered = client.get("/assessment/topic/7/status")

    assert answered.status_code == 404
    assert answered.json()["code"] == "unknown_scope"
