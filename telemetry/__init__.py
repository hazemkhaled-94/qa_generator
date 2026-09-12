"""Logging and tracing, configured identically in every process.

    from telemetry import configure, tracer

    configure("api")
    log = logging.getLogger(__name__)
    span = tracer(__name__)

Neither logging nor tracing may prevent a process from starting: an
unreachable collector degrades to recording spans without exporting them.
"""

from __future__ import annotations

import logging

from telemetry import logs, traces
from telemetry.traces import trace_app, trace_engine, tracer

__all__ = ["configure", "trace_app", "trace_engine", "tracer"]


def configure(service_name: str, level: str | None = None) -> None:
    """Configures logging and tracing for this process.

    Call once, as early as possible: anything logged beforehand uses
    Python's default handler and carries no trace id.
    """
    # Order matters: the shared format references the trace fields, and
    # tracing logs while setting itself up.
    logs.add_trace_fields()
    logs.configure(level)
    traces.configure(service_name)
    logging.getLogger(__name__).info("telemetry configured for %s", service_name)
