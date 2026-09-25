"""Logging and tracing, configured identically in every process.

    from telemetry import configure, tracer, working

    configure("api")
    log = logging.getLogger(__name__)
    span = tracer(__name__)

    with working(span, "parse", {"document.sha256": sha}) as current:
        log.info("parsed")     # the line carries document.sha256 too

Neither logging nor tracing may prevent a process from starting: an
unreachable collector degrades to recording spans without exporting them,
and a log directory that cannot be written degrades to stdout alone.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace

from telemetry import logs, traces
from telemetry.logs import bind
from telemetry.traces import (
    asking,
    instrument_llm,
    trace_engine,
    tracer,
)

__all__ = [
    "asking",
    "bind",
    "configure",
    "instrument_llm",
    "trace_engine",
    "tracer",
    "working",
]


@contextmanager
def working(
    span: trace.Tracer, name: str, fields: Mapping[str, Any]
) -> Iterator[trace.Span]:
    """Opens a span and labels every log line under it with the same fields.

    One call because they are one fact said twice. A stage annotating its
    span with `document.sha256` and its lines with something else spelled
    differently is a trace that cannot be joined to the logs explaining it,
    and that is what two calls drift into.

    Yields the span, so a caller still records what it only learns while
    working - how many facts came back, why a passage was skipped.
    """
    with bind(fields), span.start_as_current_span(name) as current:
        for key, value in fields.items():
            current.set_attribute(key, value)
        yield current


def configure(
    service_name: str,
    level: str | None = None,
    run: str | None = None,
    named: str | None = None,
    tracing: bool = True,
) -> None:
    """Configures logging and tracing for this process.

    Call once, as early as possible: anything logged beforehand uses
    Python's default handler and carries no trace id.

    `run` is what separates the two stores. It names the Phoenix project
    this process's spans are filed under AND decides whether there are
    spans to file: a stage passes `settings.runs.run_id()` and exports; the
    api, the frontend, the orchestrator and every host command pass nothing
    and export nothing, because none of them is the logic Phoenix holds.

    The logs take it either way. `run.id` is on every line a stage writes,
    which is what lets Grafana show one run's lines across five processes
    and Phoenix's project name find the same run from the other side.

    `named` is the run's NAME where somebody chose one. It decides only
    which project the spans are filed under; see `traces.configure`.

    `tracing` off configures the logs and installs NO tracer provider,
    which is not the same as installing one that does not export. A
    provider can be set exactly once per process - OpenTelemetry ignores
    the second attempt and warns - so a caller that wants its logs before
    it knows whether it is going to work must leave tracing alone until it
    does. That is what a stage's preflight needs: the model call it makes
    to prove the model answers would otherwise be exported, and a run that
    then gives up has created a Phoenix project holding one failed call.
    """
    # Order matters: the shared format references the trace fields, and
    # tracing logs while setting itself up.
    logs.add_trace_fields()
    logs.configure(service_name, level, run)
    if tracing:
        traces.configure(service_name, run, named)
    logging.getLogger(__name__).info(
        "telemetry configured for %s%s", service_name, f", run {run}" if run else ""
    )
