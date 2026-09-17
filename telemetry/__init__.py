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
from telemetry.traces import trace_app, trace_engine, tracer

__all__ = ["bind", "configure", "trace_app", "trace_engine", "tracer", "working"]


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


def configure(service_name: str, level: str | None = None) -> None:
    """Configures logging and tracing for this process.

    Call once, as early as possible: anything logged beforehand uses
    Python's default handler and carries no trace id.
    """
    # Order matters: the shared format references the trace fields, and
    # tracing logs while setting itself up.
    logs.add_trace_fields()
    logs.configure(service_name, level)
    traces.configure(service_name)
    logging.getLogger(__name__).info("telemetry configured for %s", service_name)
