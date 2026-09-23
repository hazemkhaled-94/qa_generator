"""OpenTelemetry tracing configuration.

Phoenix holds the LOGIC and nothing else: what a stage was asked, what a
model answered, which gate read it and what that cost. It does not hold
services. An HTTP request, a SQL statement, an S3 GET and a page the
frontend rendered are the system working, and the system is what the logs
are for - every one of them is in Elasticsearch, per service, in Grafana.

Two rules keep that line, and both are here:

- **Only a run exports.** A process that names a run is running the
  pipeline; one that does not is a service, a host command or a query, and
  it builds a provider that records and sends nothing.
- **Only the logic is instrumented.** The model client, and the spans the
  stages open themselves. No HTTP client, no request handler, no object
  store, and the database only when somebody asks for it by name.
"""

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
    # does not install SQLAlchemy.
    from sqlalchemy.engine import Engine

log = logging.getLogger(__name__)

#: Whether the model client has been instrumented in this process. The
#: instrumentor warns when applied twice, and two stages build two clients.
_llm_instrumented = False

#: Set to put a span on every SQL statement and every pool connect. Off by
#: default, and the one exception to the rule above: a statement is the
#: system, not the logic, and belongs in Phoenix only while somebody is
#: reading it. They were 2.8M of 3.9M spans.
TRACE_DATABASE = "TRACE_DATABASE"

#: Accepted spellings of true, as `settings/env.py` spells them.
_TRUE = frozenset({"1", "true", "yes", "on"})


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
    """Sets up the tracer provider, and exports only for a run.

    `run` does two things, and they are the same thing said twice: it puts
    this process's spans in a Phoenix project of their own,
    `<service>-<run>`, and it is what decides there is an exporter at all.
    A process that names no run is a service, a host command or a query
    against the database - none of them the logic Phoenix is for - and it
    gets a provider that records and sends nothing.

    That is the whole separation. Phoenix answers "what did this run
    produce, and what did it cost"; the logs answer everything else, for
    every process, run or no run.

    Exports to OTEL_EXPORTER_OTLP_ENDPOINT. Credentials come from
    OTEL_EXPORTER_OTLP_HEADERS and are read by the exporter itself.
    """
    attributes = {"service.name": service_name}
    if run:
        # Named for the service too, so one run of the whole pipeline is
        # five projects that sort together rather than one heap in which
        # extraction's calls and question generation's are indistinguishable.
        attributes[PROJECT] = f"{service_name}-{run}"
    provider = TracerProvider(resource=Resource.create(attributes))

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") if run else None
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

    **What the call was for** goes on as metadata, read off the same
    binding the log line beside it carries - the stage, the passage, the
    topic, the run. Without it a `completion` is filterable only through
    the parent span it happens to hang under, so "every call this run made
    about that document" is a question the trace view cannot answer and
    the logs can. One fact said twice, again.

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
    from telemetry.logs import bound

    with using_attributes(
        tags=[shape],
        prompt_template_version=version or "",
        metadata={key: str(value) for key, value in bound().items()},
    ):
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


