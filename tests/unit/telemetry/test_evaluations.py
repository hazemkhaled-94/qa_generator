"""Posting a verdict to Phoenix, and never failing a run over it.

No network. What is worth testing without one is the shape of what would
be sent and, more importantly, what happens when nothing answers: a run
that is producing questions must not stop because the thing watching it
is down.
"""

from __future__ import annotations

import pytest

from telemetry.evaluations import Evaluations, Verdict, current_span_id

BASE = "http://phoenix.invalid:6006"


def _verdict(**over) -> Verdict:
    """One gate verdict, with the fields a caller varies."""
    return Verdict(
        **{
            "span_id": "0123456789abcdef",
            "name": "gate",
            "label": "accepted",
            "score": 1.0,
            "explanation": "Was kostet das?",
            **over,
        }
    )


def test_a_rule_is_code_and_a_model_is_an_llm() -> None:
    """The distinction this repository cares about most.

    A gate is a rule reading a parse. The phrasing judgements are a
    model's opinion. Shown in one view, the only thing that tells them
    apart is this field.
    """
    assert _verdict().payload()["annotator_kind"] == "CODE"
    assert _verdict(by_model=True).payload()["annotator_kind"] == "LLM"


def test_the_score_is_one_or_zero_so_the_mean_is_the_rate() -> None:
    """A project's mean over this annotation IS its acceptance rate."""
    accepted = _verdict(label="accepted", score=1.0).payload()
    refused = _verdict(label="duplicate", score=0.0).payload()

    assert accepted["result"]["score"] == 1.0
    assert refused["result"]["score"] == 0.0
    assert refused["result"]["label"] == "duplicate"


def test_a_verdict_with_no_score_carries_none() -> None:
    """Phoenix takes a label without one, and an absent key is not a 0."""
    assert "score" not in _verdict(score=None).payload()["result"]


def test_nothing_is_posted_without_somewhere_to_post_to(monkeypatch) -> None:
    """A deployment with no Phoenix records nothing and says nothing."""
    monkeypatch.delenv("PHOENIX_BASE_URL", raising=False)

    held = Evaluations()

    assert not held.enabled
    held.record(_verdict())
    assert held.flush() == 0


def test_one_name_holds_the_address_a_process_can_actually_reach(
    monkeypatch,
) -> None:
    """The container's address is set OVER this name, not beside it.

    Read as a fallback behind PHOENIX_CONTAINER_BASE_URL, a host command
    took the container's address - the Makefile sources .env, which holds
    both - and posted a run's verdicts to a hostname the host cannot
    resolve.
    """
    monkeypatch.setenv("PHOENIX_CONTAINER_BASE_URL", "http://phoenix:6006")
    monkeypatch.setenv("PHOENIX_BASE_URL", "http://localhost:6006")

    assert Evaluations()._base_url == "http://localhost:6006"


def test_a_host_command_signs_in_with_the_name_it_has(monkeypatch) -> None:
    """One credential under two names, and both are read.

    compose passes the secret as PHOENIX_API_KEY and `.env` calls it
    PHOENIX_ADMIN_SECRET; Phoenix compares the bearer token against that
    value directly, so they are the same thing.

    Reading only the first name, a host run against an authenticated
    Phoenix posted every batch unsigned, was refused, and gave up for the
    rest of the run on one warning.
    """
    monkeypatch.delenv("PHOENIX_API_KEY", raising=False)
    monkeypatch.setenv("PHOENIX_ADMIN_SECRET", "a-secret")
    signed: dict[str, object] = {}

    class _Client:
        def __init__(self, base_url: str, headers=None) -> None:
            signed["headers"] = headers

    monkeypatch.setattr("phoenix.client.Client", _Client)

    Evaluations(BASE)._connect()

    assert signed["headers"] == {"Authorization": "Bearer a-secret"}


def test_a_verdict_about_no_span_is_dropped_rather_than_invented() -> None:
    """Nothing was recording, so there is nothing to attach to."""
    held = Evaluations(BASE)
    held.record(_verdict(span_id=""))

    assert held.flush() == 0


def test_an_unreachable_phoenix_does_not_fail_the_run() -> None:
    """The bargain `traces.py` makes, made again here.

    The verdict is in Postgres either way. A run that is otherwise
    producing questions must not stop because the collector is down.
    """
    held = Evaluations(BASE, batch=1)

    held.record(_verdict())  # batch of 1, so this posts and fails

    assert held.flush() == 0


def test_it_stops_trying_after_the_first_failure() -> None:
    """A Phoenix that is down stays down for the length of a run.

    One warning, not one per hundred questions, which is the log nobody
    reads.
    """
    held = Evaluations(BASE, batch=1)
    held.record(_verdict())

    assert not held.enabled, "it should have given up after the first failure"

    held.record(_verdict())
    assert held.flush() == 0


def test_a_batch_is_held_until_it_is_full(monkeypatch) -> None:
    """One HTTP call per question would cost more than the gate it records."""
    held = Evaluations(BASE, batch=100)
    posted: list[int] = []
    monkeypatch.setattr(held, "_connect", lambda: None)

    for _ in range(99):
        held.record(_verdict())

    assert not posted, "nothing should have been posted yet"
    assert len(held._held) == 99


def test_no_span_is_recording_here() -> None:
    """Which is what every process without an exporter looks like."""
    assert current_span_id() is None


@pytest.mark.parametrize("field", ["run_id", "language", "answerable"])
def test_the_metadata_a_comparison_filters_on_survives(field: str) -> None:
    """Two runs are compared by filtering on these."""
    payload = _verdict(
        metadata={"run_id": "b-gemma4-12b", "language": "de", "answerable": True}
    ).payload()

    assert field in payload["metadata"]
