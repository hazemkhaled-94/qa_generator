"""Posting a verdict to Phoenix, against the span that reached it.

A gate verdict was in two places and neither could be compared. Postgres
answers how many of each gate fired - `question_generation.runs` is that
query - and it cannot say what a gate COST, because the calls, the tokens
and the latency are in the spans. An attribute on the span joins the two
for filtering, and that is where this started.

This is the other half. Phoenix files an ANNOTATION separately from the
span it is about, which is what puts a verdict in its Evaluations view: a
label, a score and an explanation per span, sortable, chartable and
comparable across two projects without anybody writing a query. An
attribute cannot do that, because Phoenix does not know an arbitrary
attribute is a judgement.

`annotator_kind` is **CODE** and not LLM, and that distinction is the one
this repository cares about most. A gate is a rule reading a parse; the
three phrasing judgements that are a model's opinion are posted as LLM,
so the two can be told apart in the one place they are shown together.

Best-effort throughout. Phoenix being unreachable must not fail a run that
is otherwise producing questions: everything here is caught and logged,
and the verdict is in Postgres either way. That is the same bargain
`traces.py` makes with an unreachable collector.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import trace

log = logging.getLogger(__name__)

#: Where Phoenix's HTTP API is, as the PROCESS reaches it. Not the same
#: address as the OTLP endpoint and not the same as the host's: a worker in
#: compose reaches `http://phoenix:6006` and the host reaches
#: `http://localhost:6006`, which is what PHOENIX_BASE_URL already holds for
#: `evaluation/`.
ENDPOINT = "PHOENIX_CONTAINER_BASE_URL"
FALLBACK = "PHOENIX_BASE_URL"

#: Phoenix's API key, where one is set.
API_KEY = "PHOENIX_API_KEY"

#: How many annotations are held before they are posted. One HTTP call per
#: question would cost more than the gate it is recording; a topic writes
#: about ten.
BATCH = 100


@dataclass(frozen=True)
class Verdict:
    """One judgement about one span.

    Attributes:
        span_id: The span this is about, as Phoenix names it - the
            16-character hex OTel writes.
        name: What the judgement is called. One name per question in the
            Evaluations view, so `gate` rather than the gate's own code.
        label: The answer, as a word. The gate that fired, or `accepted`.
        score: The answer as a number, where there is one. 1 for accepted
            and 0 for refused, so a project's mean IS the acceptance rate.
        explanation: Why. The reason the checker already composed.
        by_model: Whether a model decided this rather than a rule.
        metadata: Anything else worth filtering on.
    """

    span_id: str
    name: str
    label: str
    score: float | None = None
    explanation: str = ""
    by_model: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        """This verdict as the annotation Phoenix takes."""
        result: dict[str, Any] = {"label": self.label}
        if self.score is not None:
            result["score"] = self.score
        if self.explanation:
            result["explanation"] = self.explanation
        annotation: dict[str, Any] = {
            "name": self.name,
            "annotator_kind": "LLM" if self.by_model else "CODE",
            "span_id": self.span_id,
            "result": result,
        }
        if self.metadata:
            annotation["metadata"] = dict(self.metadata)
        return annotation


def current_span_id() -> str | None:
    """The span being recorded into, as Phoenix names it.

    None when nothing is recording, which is every process that configured
    telemetry without an exporter and every test that did not open a span.
    A verdict with no span to attach to is dropped rather than invented.
    """
    context = trace.get_current_span().get_span_context()
    if not context.is_valid:
        return None
    return format(context.span_id, "016x")


class Evaluations:
    """Holds verdicts and posts them to Phoenix in batches.

    Nothing is built until the first verdict arrives, so a worker whose
    stage records none - every stage but question generation, today -
    never constructs a client or reads a setting.
    """

    def __init__(self, base_url: str | None = None, batch: int = BATCH) -> None:
        """Names where Phoenix is. Nothing is connected until asked."""
        self._base_url = base_url or os.getenv(ENDPOINT) or os.getenv(FALLBACK)
        self._batch = max(batch, 1)
        self._held: list[dict[str, Any]] = []
        self._client: Any = None
        self._broken = False

    @property
    def enabled(self) -> bool:
        """Whether there is anywhere to post to."""
        return bool(self._base_url) and not self._broken

    def record(self, *verdicts: Verdict) -> None:
        """Holds some verdicts, posting once enough have piled up."""
        if not self.enabled:
            return
        self._held.extend(one.payload() for one in verdicts if one.span_id)
        if len(self._held) >= self._batch:
            self.flush()

    def flush(self) -> int:
        """Posts what is held, and reports how many went.

        Returns 0 and logs where Phoenix could not be reached. A run that
        is producing questions must not fail because the thing watching it
        is down.
        """
        if not self._held or not self.enabled:
            return 0
        held, self._held = self._held, []
        connected = self._connect()
        if connected is None:
            return 0
        try:
            connected.spans.log_span_annotations(span_annotations=held)
        except Exception as exc:  # noqa: BLE001 - any failure is the same answer
            # Once, not once per batch. A Phoenix that is down stays down
            # for the length of a run, and a warning per hundred questions
            # is the log nobody reads.
            self._broken = True
            log.warning(
                "could not record %d verdict(s) in Phoenix, and will not try "
                "again this run: %s: %s. The verdicts are in the database.",
                len(held),
                type(exc).__name__,
                exc,
            )
            return 0
        log.debug("recorded %d verdict(s) in Phoenix", len(held))
        return len(held)

    def _connect(self):
        """The Phoenix client, built once."""
        if self._client is not None:
            return self._client
        try:
            from phoenix.client import Client
        except ImportError:
            self._broken = True
            log.info(
                "arize-phoenix-client is not installed, so gate verdicts stay "
                "in the database and on the span and are not posted as "
                "annotations"
            )
            return None
        key = os.getenv(API_KEY)
        self._client = Client(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {key}"} if key else None,
        )
        return self._client


@contextmanager
def recording(base_url: str | None = None) -> Iterator[Evaluations]:
    """An `Evaluations` that posts whatever is left when the block ends.

    The flush is the reason this exists. A topic writing ninety questions
    leaves ninety annotations under the batch size, and without a flush the
    last topic of every run is the one missing from Phoenix.
    """
    held = Evaluations(base_url)
    try:
        yield held
    finally:
        held.flush()


def spans_of(verdicts: Iterable[Verdict]) -> Sequence[str]:
    """The spans some verdicts are about. For tests and for logging."""
    return [one.span_id for one in verdicts]
