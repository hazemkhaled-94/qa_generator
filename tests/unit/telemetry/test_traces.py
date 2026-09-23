"""What produces a span and what does not.

No collector. What is worth testing without one is which instrumentors
`configure` reaches, because the two here were 2.9M of 3.9M spans in a
deployment where 47k were model calls.
"""

from __future__ import annotations

import pytest

from telemetry import traces


class _Recorder:
    """Stands in for an instrumentor, and remembers being applied."""

    calls: list[dict] = []

    def instrument(self, **kwargs) -> None:
        """Records one application."""
        type(self).calls.append(kwargs)


@pytest.fixture
def sqlalchemy(monkeypatch) -> type[_Recorder]:
    """A SQLAlchemyInstrumentor that records rather than instruments."""
    module = pytest.importorskip("opentelemetry.instrumentation.sqlalchemy")
    recorder = type("Recorder", (_Recorder,), {"calls": []})
    monkeypatch.setattr(module, "SQLAlchemyInstrumentor", recorder)
    return recorder


def test_database_statements_are_not_traced_by_default(monkeypatch, sqlalchemy):
    """The default, and the reason this gate exists."""
    monkeypatch.delenv(traces.TRACE_DATABASE, raising=False)
    traces.trace_engine(object())
    assert sqlalchemy.calls == []


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_trace_database_turns_them_back_on(monkeypatch, sqlalchemy, value):
    """Every spelling `settings/env.py` accepts."""
    monkeypatch.setenv(traces.TRACE_DATABASE, value)
    engine = object()
    traces.trace_engine(engine)
    assert sqlalchemy.calls == [{"engine": engine}]


@pytest.mark.parametrize("value", ["0", "false", "", "maybe"])
def test_anything_else_is_off(monkeypatch, sqlalchemy, value):
    """A boolean that guesses is worse than one that is simply off."""
    monkeypatch.setenv(traces.TRACE_DATABASE, value)
    traces.trace_engine(object())
    assert sqlalchemy.calls == []


def test_health_and_status_are_excluded_from_request_spans(monkeypatch):
    """The polled routes, which the frontend and dagster hit continuously."""
    module = pytest.importorskip("opentelemetry.instrumentation.fastapi")
    seen: list[dict] = []
    monkeypatch.setattr(
        module.FastAPIInstrumentor,
        "instrument_app",
        staticmethod(lambda app, **kwargs: seen.append(kwargs)),
    )
    traces.trace_app(object())
    excluded = seen[0]["excluded_urls"].split(",")
    assert "health" in excluded
    assert "status" in excluded
