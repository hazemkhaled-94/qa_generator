"""Logging configuration shared by every process."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, cast

from opentelemetry import trace

_FORMAT = (
    "%(asctime)s %(levelname)-7s %(name)s "
    "[trace=%(otelTraceID)s span=%(otelSpanID)s] %(message)s"
)
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

#: Libraries that log once per operation at INFO.
_NOISY = ("botocore", "boto3", "urllib3", "watchdog", "sqlalchemy.engine")

#: Loggers that install handlers at import time. Left alone they bypass the
#: root handler and print in their own format.
_OPINIONATED = ("uvicorn", "uvicorn.error", "uvicorn.access")


def configure(level: str | None = None) -> None:
    """Installs the shared handler on the root logger, replacing any others.

    Quietens the libraries that log per operation and takes the loggers that
    install their own handlers back onto this one. Call `add_trace_fields`
    first: the format references the fields it provides.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))

    # force: replaces the root logger's handlers rather than adding to them.
    logging.basicConfig(
        handlers=[handler],
        level=level or os.getenv("LOG_LEVEL", "INFO").upper(),
        force=True,
    )

    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)

    for name in _OPINIONATED:
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True


def add_trace_fields() -> None:
    """Puts the active trace and span id on every log record, once.

    Both read as zeros outside a span. Reads the current span directly
    rather than through opentelemetry-instrumentation-logging, which does
    not populate these fields in the pinned version.
    """
    factory = logging.getLogRecordFactory()
    if getattr(factory, "_adds_trace_fields", False):
        return

    def record_factory(*args, **kwargs):
        record = factory(*args, **kwargs)
        context = trace.get_current_span().get_span_context()
        record.otelTraceID = format(context.trace_id, "032x")
        record.otelSpanID = format(context.span_id, "016x")
        return record

    # A marker on the factory rather than a module flag, so the guard above
    # stays correct if something else replaces it. The cast is what lets it
    # be written: a function carries no declared attributes.
    cast(Any, record_factory)._adds_trace_fields = True
    logging.setLogRecordFactory(record_factory)
