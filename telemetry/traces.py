"""OpenTelemetry tracing configuration."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

if TYPE_CHECKING:
    # Behind TYPE_CHECKING: this package is shared with the frontend, which
    # installs neither FastAPI nor SQLAlchemy.
    from fastapi import FastAPI
    from sqlalchemy.engine import Engine

log = logging.getLogger(__name__)

#: Applied only for libraries the process has, so one call serves every
#: process. SQLAlchemy and FastAPI are separate: both attach to an instance.
_INSTRUMENTORS = (
    ("opentelemetry.instrumentation.requests", "RequestsInstrumentor"),
    ("opentelemetry.instrumentation.botocore", "BotocoreInstrumentor"),
)


def configure(service_name: str) -> None:
    """Sets up the tracer provider and instruments what is installed.

    Exports over OTLP when OTEL_EXPORTER_OTLP_ENDPOINT is set, and
    otherwise records spans without sending them. Credentials come from
    OTEL_EXPORTER_OTLP_HEADERS and are read by the exporter itself.
    """
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("tracing export disabled: %s: %s", type(exc).__name__, exc)

    trace.set_tracer_provider(provider)
    _instrument()


def _instrument() -> None:
    """Applies every instrumentor whose library is importable."""
    for module_name, class_name in _INSTRUMENTORS:
        try:
            module = __import__(module_name, fromlist=[class_name])
        except ImportError:
            continue
        try:
            getattr(module, class_name)().instrument()
        except Exception as exc:  # noqa: BLE001
            log.warning("could not instrument %s: %s", class_name, exc)


def tracer(name: str) -> trace.Tracer:
    """Returns a tracer for one module."""
    return trace.get_tracer(name)


def trace_engine(engine: Engine) -> None:
    """Traces every statement issued through one SQLAlchemy engine.

    Does nothing if the instrumentation is not installed.
    """
    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    except ImportError:
        return
    try:
        SQLAlchemyInstrumentor().instrument(engine=engine)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not instrument the database engine: %s", exc)


def trace_app(app: FastAPI) -> None:
    """Traces every request handled by one FastAPI application.

    Continues the caller's trace from the incoming traceparent header.
    Attaches to the instance, so ordering does not matter: the global
    instrumentor replaces `fastapi.FastAPI` and does nothing for a module
    that imported the name first.
    """
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError:
        return
    try:
        FastAPIInstrumentor.instrument_app(app)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not instrument the application: %s", exc)
