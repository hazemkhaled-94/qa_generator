"""The service catalogue, and what a probe of it reports.

Nothing here opens a socket to a real service. What is worth holding is the
reading: a port that refuses is down, a service with no port is neither up
nor down, and a link is offered only where one was configured.
"""

from __future__ import annotations

import pytest

from api import services


def test_every_service_is_named_once() -> None:
    """The name is the hostname, so two rows with one name is a typo."""
    names = [one.name for one in services.CATALOGUE]

    assert len(names) == len(set(names)), "a service is in the catalogue twice"


def test_every_service_says_what_it_is_for() -> None:
    """A row nobody can read is a row nobody acts on."""
    assert all(one.purpose.strip() for one in services.CATALOGUE)


def test_a_worker_serves_no_port() -> None:
    """The distinction the page is built on.

    A worker reads a queue. Probing it would refuse, and reporting that as
    down would say something untrue about a healthy process.
    """
    workers = [one for one in services.CATALOGUE if one.name.endswith("-worker")]

    assert workers, "the catalogue lost the workers"
    assert all(one.port is None for one in workers)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", {}),
        ("api=http://localhost:8000", {"api": "http://localhost:8000"}),
        (
            "api=http://a, grafana=http://b",
            {"api": "http://a", "grafana": "http://b"},
        ),
        # What the folded scalar in compose produces: a newline becomes a
        # space, so every pair after the first arrives with one in front.
        (
            "  api=http://a ,  grafana=http://b  ",
            {"api": "http://a", "grafana": "http://b"},
        ),
        # A name with no address is a service nobody published.
        ("api=, grafana=http://b", {"grafana": "http://b"}),
        ("nonsense", {}),
    ],
)
def test_addresses_are_read_off_the_environment(
    raw: str, expected: dict, monkeypatch
) -> None:
    """SERVICE_URLS is optional throughout, and half-written is not a crash."""
    monkeypatch.setenv("SERVICE_URLS", raw)

    assert services.addresses() == expected


def test_a_service_with_no_address_is_offered_no_link(monkeypatch) -> None:
    """An unreachable link is worse than none."""
    monkeypatch.setenv("SERVICE_URLS", "api=http://localhost:8000")
    monkeypatch.setattr(services, "CATALOGUE", (services.Service("redis", "x", 6379),))
    monkeypatch.setattr(services, "_listening", lambda one: (True, "up"))

    assert services.snapshot()[0].url is None


def test_a_refused_port_is_reported_down_with_the_address(monkeypatch) -> None:
    """The detail has to name what was tried, or nobody can check it."""
    monkeypatch.setenv("SERVICE_URLS", "")
    monkeypatch.setattr(
        services, "CATALOGUE", (services.Service("nowhere", "x", 65000),)
    )

    state = services.snapshot()[0]

    assert state.ok is False
    assert "nowhere:65000" in state.detail


def test_a_portless_service_is_neither_up_nor_down(monkeypatch) -> None:
    """None, not False. The page renders it as `no port`."""
    monkeypatch.setenv("SERVICE_URLS", "")
    monkeypatch.setattr(services, "CATALOGUE", (services.Service("a-worker", "x"),))

    state = services.snapshot()[0]

    assert state.ok is None
    assert "queue" in state.detail
