"""OpenTelemetry tracing configuration."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
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
#:
#: litellm is deliberately not here; see :func:`instrument_llm`.
_INSTRUMENTORS = (
    ("opentelemetry.instrumentation.requests", "RequestsInstrumentor"),
    ("opentelemetry.instrumentation.botocore", "BotocoreInstrumentor"),
)

#: Whether the model client has been instrumented in this process. The
#: instrumentor warns when applied twice, and two stages build two clients.
_llm_instrumented = False

#: Set to trace every SQL statement and every pool connect. Off by default:
#: they were 2.8M of 3.9M spans, against 47k model calls.
TRACE_DATABASE = "TRACE_DATABASE"

#: Accepted spellings of true, as `settings/env.py` spells them.
_TRUE = frozenset({"1", "true", "yes", "on"})

#: Request paths that produce no span. Matched anywhere in the URL, so one
#: entry covers `/status` and `/{scope}/{value}/status` alike.
EXCLUDED_URLS = "health,status"


#: What Phoenix reads a span's project off. An OpenInference resource
#: attribute rather than an OTel one - Phoenix files a span under this and
#: falls back to `default` when it is absent, which is where every span in
#: this deployment has gone so far.
#:
#: Read off the package rather than written as a string, because it is
#: their name for their own concept; the fallback is the literal, for a
#: process that has the exporter and not the semantic conventions.
try:  # pragma: no cover - present wherever the instrumentor is
    from openinference.semconv.resource import ResourceAttributes

    PROJECT = ResourceAttributes.PROJECT_NAME
except ImportError:  # pragma: no cover
    PROJECT = "openinference.project.name"


def configure(service_name: str, run: str | None = None) -> None:
    """Sets up the tracer provider and instruments what is installed.

    Exports over OTLP when OTEL_EXPORTER_OTLP_ENDPOINT is set, and
    otherwise records spans without sending them. Credentials come from
    OTEL_EXPORTER_OTLP_HEADERS and are read by the exporter itself.

    `run` puts this process's spans in a Phoenix project of their own. What
    that buys is the half SQL cannot answer: the gate counts are columns and
    can be grouped by `questions.run_id`, but the call count, the token
    count, the latency and the spend are in the spans, and Phoenix compares
    two projects of those directly. Absent - which is every process that is
    not a stage - the spans go where they always went.
    """
    attributes = {"service.name": service_name}
    if run:
        # Named for the service too, so one run of the whole pipeline is
        # five projects that sort together rather than one heap in which
        # extraction's calls and question generation's are indistinguishable.
        attributes[PROJECT] = f"{service_name}-{run}"
    provider = TracerProvider(resource=Resource.create(attributes))

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


def instrument_llm() -> None:
    """Records every model call as a span carrying its prompt and its cost.

    Called by the module that builds the client rather than by
    :func:`configure`, and that is the whole reason this is a function of
    its own. Importing the instrumentor imports litellm, `configure` runs
    in every process, and the api is sized to serve JSON and deliberately
    loads neither it nor anything else that holds a model. Putting it in
    `_INSTRUMENTORS` would have loaded litellm into the api at start-up,
    and `tests/static/test_api_stays_light.py` would not have caught it:
    that reads imports as syntax, and the one there is a dynamic string.

    Does nothing the second time. Applies once per process; question
    generation builds two clients, a writer and a verifier, and the
    instrumentor warns when applied to a library already instrumented.
    """
    global _llm_instrumented
    if _llm_instrumented:
        return
    try:
        from openinference.instrumentation.litellm import LiteLLMInstrumentor
    except ImportError:
        # Warned rather than passed over in silence, unlike the optional
        # instrumentors above: a worker missing this still calls the model
        # and still works, and the only symptom is that the spans it was
        # meant to produce are the ones nobody notices are absent.
        log.warning(
            "openinference-instrumentation-litellm is not installed, so model "
            "calls are timed in the logs but carry no prompt or token counts"
        )
        return
    try:
        LiteLLMInstrumentor().instrument()
    except Exception as exc:  # noqa: BLE001
        log.warning("could not instrument the model client: %s", exc)
        return
    _llm_instrumented = True


@contextmanager
def asking(shape: str, version: str | None = None) -> Iterator[None]:
    """Labels the model-call spans opened inside with what was asked of it.

    The instrumentor names every call `completion`, so a run is two
    hundred identical spans and nothing on one says whether it wrote a
    question, judged its phrasing or read the answer back out. That is in
    the log as `llm.shape` and was in no trace, which is the reason
    `make spend-by-shape` reads log prose to answer what a judgement
    costs.

    `shape` becomes a **tag**, which Phoenix filters and groups on, and
    `version` becomes `llm.prompt_template.version`, which is the field it
    already has for exactly this. Neither is a span of our own: the
    instrumentor's span is the one carrying the prompt and the price, and
    a wrapper around it would be a second place for the same call.

    Does nothing where openinference is not installed - the frontend and
    the orchestrator have the tracer and not the instrumentor - and
    nothing where a caller names no version, which is a prompt that has
    not been given one.
    """
    try:
        from openinference.instrumentation import using_attributes
    except ImportError:
        yield
        return
    with using_attributes(tags=[shape], prompt_template_version=version or ""):
        yield


def tracer(name: str) -> trace.Tracer:
    """Returns a tracer for one module."""
    return trace.get_tracer(name)


def trace_engine(engine: Engine) -> None:
    """Traces every statement issued through one SQLAlchemy engine.

    Does nothing unless TRACE_DATABASE is set, and nothing if the
    instrumentation is not installed.
    """
    if os.getenv(TRACE_DATABASE, "").lower() not in _TRUE:
        return
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

    The paths in EXCLUDED_URLS produce no span.
    """
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError:
        return
    try:
        FastAPIInstrumentor.instrument_app(app, excluded_urls=EXCLUDED_URLS)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not instrument the application: %s", exc)
