"""Logging configuration shared by every process.

One configuration, two renderings of the same record: text on stdout, which
is what `make logs` shows a person, and one JSON object per line in a file,
which is what the log shipper reads. Neither is a separate set of fields -
the JSON carries everything the text line does and the fields the text line
has no room for.

The JSON field names are ECS, which is what Elasticsearch's own template
maps: `log.level` and `service.name` arrive as keywords a dashboard can group
on, and `error.stack_trace` as text, with nothing to configure. A name of our
own would arrive as an unmapped string.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import socket
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
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

#: Where the JSON lines go, if anywhere. A container's own path from
#: compose, the bind-mounted host directory from .env, and unset - which is
#: any process given neither - means stdout only.
_LOG_DIR = "LOG_DIR"

#: One file may reach this before it rotates, and this many rotations are
#: kept. Sized so a worker logging steadily keeps about a day locally; the
#: shipper has already read the line long before it is deleted.
_MAX_BYTES = 50 * 1024 * 1024
_BACKUPS = 3

#: Everything a LogRecord carries by itself. What a caller passed as `extra`
#: is whatever is left, and that goes into the JSON beside the rest. Taken
#: from a throwaway record rather than typed out, so a new attribute in a
#: later Python does not start appearing as a field.
_STANDARD = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "asctime",
    "message",
    "taskName",
    "otelTraceID",
    "otelSpanID",
}

#: What `bind` has put on every record logged in this context. A ContextVar
#: rather than a module global because a thread or a task gets its own, so
#: two passages worked at once cannot label each other's lines. The default
#: is a read-only view: nothing here mutates what it reads - `bind` builds a
#: new mapping each time - and a plain {} shared by every context is one
#: careless `.update()` away from leaking a document id into every process.
_BOUND: ContextVar[Mapping[str, Any]] = ContextVar(
    "bound", default=MappingProxyType({})
)


@contextmanager
def bind(fields: Mapping[str, Any]) -> Iterator[None]:
    """Puts fields on every record logged inside this block.

    What the pipeline is working on, recorded once where it is claimed
    rather than spelled into each message beneath it. A document crosses
    five processes over hours, and without this there is nothing in
    Elasticsearch that says two lines are about the same one.

    Nests: an inner call adds to what an outer one bound rather than
    replacing it, and both are taken back on the way out.

        with bind({"stage": "parsing", "document.sha256": sha}):
            log.info("parsed")     # carries both

    A mapping rather than keyword arguments because the names are ECS and
    so mostly dotted, which no keyword can be. filebeat's `expand_keys`
    turns `document.sha256` into the nested field Elasticsearch maps rather
    than a string with a full stop in it, and the same spelling is what the
    span beside it is given, so one value reads the same in Grafana and in
    Phoenix.
    """
    token = _BOUND.set({**_BOUND.get(), **fields})
    try:
        yield
    finally:
        _BOUND.reset(token)


def bound() -> Mapping[str, Any]:
    """What `bind` holds in this context, for something other than a line.

    The model-call span reads it, so a call carries the work it was for in
    the same words the log line does.
    """
    return _BOUND.get()


class _Bound(logging.Filter):
    """Copies what `bind` holds onto each record on its way to a handler.

    A filter on the handler rather than work in the record factory, for two
    reasons. `Logger.makeRecord` raises KeyError if a caller's `extra`
    names a field the factory already set, so binding there turns an
    overlap into a lost log line and a traceback. And a filter on a logger
    is not consulted for records a child logger propagated, while one on a
    handler sees everything that reaches it - including the lines a library
    logs, which is the point.

    `setdefault`: a caller's `extra` is the more specific of the two, being
    passed at the line rather than around the block, so it wins.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Adds the bound fields and keeps the record."""
        for name, value in _BOUND.get().items():
            record.__dict__.setdefault(name, value)
        return True


class JsonFormatter(logging.Formatter):
    """Renders one record as a single JSON object, in ECS field names."""

    def __init__(self, service_name: str, run: str | None = None) -> None:
        """Initialises the formatter with what every line is tagged with."""
        super().__init__()
        self._service = service_name
        self._run = run or ""
        self._host = socket.gethostname()

    def format(self, record: logging.LogRecord) -> str:
        """Renders one record as one line."""
        fields: dict[str, Any] = {
            # RFC 3339 with a Z rather than +00:00. Both are valid, and the
            # shipper keeps this timestamp only if it can parse it - failing
            # to, it substitutes the time it read the line and says nothing.
            "@timestamp": datetime.fromtimestamp(record.created, UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "log.level": record.levelname.lower(),
            "log.logger": record.name,
            "message": record.getMessage(),
            "service.name": self._service,
            "host.name": self._host,
            "process.pid": record.process,
            "process.thread.name": record.threadName,
            "log.origin.file.name": record.pathname,
            "log.origin.file.line": record.lineno,
            "log.origin.function": record.funcName,
            "trace.id": getattr(record, "otelTraceID", ""),
            "span.id": getattr(record, "otelSpanID", ""),
            # Empty for a process that is not a run: the api, the frontend,
            # the orchestrator and every host command.
            "run.id": self._run,
        }

        if record.exc_info and record.exc_info[0] is not None:
            # The whole traceback, in one field. An exception that reaches a
            # log is the thing somebody will be reading; truncating it here
            # would mean going back to the container to find out what broke.
            fields["error.type"] = record.exc_info[0].__name__
            fields["error.message"] = str(record.exc_info[1])
            fields["error.stack_trace"] = self.formatException(record.exc_info)
        elif record.exc_text:
            fields["error.stack_trace"] = record.exc_text

        if record.stack_info:
            fields["log.origin.stack"] = self.formatStack(record.stack_info)

        fields.update(
            {
                key: value
                for key, value in record.__dict__.items()
                if key not in _STANDARD
            }
        )
        # default=str: a caller's `extra` may hold anything, and a log line
        # that raises while being written loses the event it was reporting.
        return json.dumps(fields, ensure_ascii=False, default=str)


def configure(
    service_name: str, level: str | None = None, run: str | None = None
) -> None:
    """Installs the shared handlers on the root logger, replacing any others.

    Quietens the libraries that log per operation and takes the loggers that
    install their own handlers back onto these. Call `add_trace_fields`
    first: the text format references the fields it provides.

    A JSON file handler is added when LOG_DIR names a writable directory,
    which is how a container is run and a host command is not.

    `run` goes on every shipped line as `run.id`, so one run's lines can be
    read together across the five processes it crosses.
    """
    handlers: list[logging.Handler] = [_stream()]
    shipped = _file(service_name, run)
    if shipped is not None:
        handlers.append(shipped)

    for handler in handlers:
        handler.addFilter(_Bound())

    # force: replaces the root logger's handlers rather than adding to them.
    logging.basicConfig(
        handlers=handlers,
        level=level or os.getenv("LOG_LEVEL", "INFO").upper(),
        force=True,
    )

    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)

    for name in _OPINIONATED:
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True


def _stream() -> logging.Handler:
    """Builds the handler a person reads, on stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
    return handler


def _file(service_name: str, run: str | None = None) -> logging.Handler | None:
    """Builds the handler the shipper reads, or None if there is nowhere.

    One file per PROCESS: the writer is what the name has to be unique per,
    because two processes rotating one file take each other's lines with
    them. A scaled stage is several containers over one volume, and the
    host directory is one machine over many runs, so neither the service
    nor the host alone is enough - `extraction-3f15a9823823-1.log` beside
    `extraction-hazems-mac-48213.log`.

    The cost is a file per run, and `make logs-prune` is what sweeps them.

    Never raises: an unwritable directory costs the shipped copy, and the
    line still reaches stdout. That is reported rather than swallowed - the
    warning is the first thing on the stream handler just installed.
    """
    directory = os.getenv(_LOG_DIR, "").strip()
    if not directory:
        return None
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            Path(directory)
            / f"{service_name}-{socket.gethostname()}-{os.getpid()}.log",
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUPS,
            encoding="utf-8",
        )
    except OSError as exc:
        logging.getLogger(__name__).warning(
            "%s=%s is not writable, so nothing is shipped from this process: %s: %s",
            _LOG_DIR,
            directory,
            type(exc).__name__,
            exc,
        )
        return None
    handler.setFormatter(JsonFormatter(service_name, run))
    return handler


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
