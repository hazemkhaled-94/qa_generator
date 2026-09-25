"""What reaches Phoenix, and what is only ever a log line.

Phoenix holds the logic. A service, an HTTP request, an S3 GET and a SQL
statement are the system, and the system is what Grafana is for. These are
the two rules that keep the line: only a run exports, and only the logic is
instrumented.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from telemetry import traces


class _Recorder:
    """Stands in for an instrumentor, and remembers being applied."""

    calls: ClassVar[list[dict]] = []

    def instrument(self, **kwargs) -> None:
        """Records one application."""
        type(self).calls.append(kwargs)


@pytest.fixture
def exporters(monkeypatch) -> list[object]:
    """Every span processor `configure` installs on its provider."""
    added: list[object] = []
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://phoenix.invalid:4317")
    monkeypatch.setattr(
        traces.TracerProvider, "add_span_processor", lambda _self, p: added.append(p)
    )
    monkeypatch.setattr(traces.trace, "set_tracer_provider", lambda _p: None)
    return added


@pytest.fixture
def sqlalchemy(monkeypatch) -> type[_Recorder]:
    """A SQLAlchemyInstrumentor that records rather than instruments."""
    module = pytest.importorskip("opentelemetry.instrumentation.sqlalchemy")
    recorder = type("Recorder", (_Recorder,), {"calls": []})
    monkeypatch.setattr(module, "SQLAlchemyInstrumentor", recorder)
    return recorder


def test_a_process_without_a_run_exports_nothing(exporters) -> None:
    """The api, the frontend, the orchestrator and every host command.

    None of them is the logic Phoenix holds, and a project per API process
    would be a project per restart.
    """
    traces.configure("api")

    assert exporters == []


def test_a_run_exports(exporters) -> None:
    """A stage, which is the only thing that does."""
    traces.configure("extraction", run="abc123")

    assert len(exporters) == 1


def _resource(monkeypatch) -> dict:
    """Captures the resource attributes `configure` builds."""
    seen: dict = {}
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.setattr(traces.trace, "set_tracer_provider", lambda _p: None)
    monkeypatch.setattr(
        traces.Resource, "create", staticmethod(lambda a: seen.update(a) or None)
    )
    return seen


def test_a_named_run_is_filed_under_a_project_of_its_own(monkeypatch) -> None:
    """`<position>-<service>-<name>`, which is what an A/B is read off.

    Numbered, because Phoenix sorts its projects by name and a corpus moves
    extraction, topics, questions - which sorts the other way round.
    """
    seen = _resource(monkeypatch)

    traces.configure("extraction", run="abc123", named="a-gpt-4.1")

    assert seen[traces.PROJECT] == "4-extraction-a-gpt-4.1"


def test_a_run_nobody_named_appends_to_the_stage(monkeypatch) -> None:
    """Rather than taking a project of its own and never being found again.

    `run_id` mints a uuid per PROCESS and a worker restarts, so a project
    per unnamed run is a project per restart. One deployment reached five
    hundred of them, almost all holding a handful of spans.
    """
    seen = _resource(monkeypatch)

    traces.configure("extraction", run="8f2c1e")

    assert seen[traces.PROJECT] == "4-extraction"


def test_every_span_carries_the_run_whether_or_not_it_was_named(
    monkeypatch,
) -> None:
    """Which is what tells two runs apart inside a shared project.

    The project stopped being the thing that separates them, so something
    else has to be, and it is the same id the logs and the rows carry.
    """
    for named in (None, "a-gpt-4.1"):
        seen = _resource(monkeypatch)
        traces.configure("extraction", run="8f2c1e", named=named)
        assert seen["run.id"] == "8f2c1e"


def test_a_process_with_no_run_is_filed_nowhere(monkeypatch) -> None:
    """The api, the frontend and the orchestrator export nothing at all."""
    seen = _resource(monkeypatch)

    traces.configure("api")

    assert traces.PROJECT not in seen
    assert "run.id" not in seen


def test_nothing_instruments_http_or_the_object_store() -> None:
    """The 200k spans that were a worker fetching a file and polling a port.

    Both libraries stay installed and both stay uninstrumented: what they
    did is in the logs, per service, which is where the system belongs.
    """
    assert not hasattr(traces, "_INSTRUMENTORS")
    assert not hasattr(traces, "trace_app")


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


def test_a_model_call_carries_the_work_it_was_for(monkeypatch) -> None:
    """The stage, the passage, the topic - whatever `bind` is holding.

    Without it a `completion` is reachable only through the parent span it
    hangs under, so "every call this run made about that document" is a
    question the trace view cannot answer and the logs can.
    """
    import openinference.instrumentation as oi

    from telemetry import bind

    seen: dict = {}
    monkeypatch.setattr(oi, "using_attributes", lambda **kw: seen.update(kw) or _noop())

    with (
        bind({"stage": "extraction", "passage.id": 41}),
        traces.asking("_Answered", "8"),
    ):
        pass

    assert seen["tags"] == ["_Answered"]
    assert seen["prompt_template_version"] == "8"
    assert seen["metadata"] == {"stage": "extraction", "passage.id": "41"}


def _noop():
    """A context manager that does nothing."""
    from contextlib import nullcontext

    return nullcontext()
