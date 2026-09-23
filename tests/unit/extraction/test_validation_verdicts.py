"""The verdicts extraction files in Phoenix's Evaluations view.

The same arrangement question generation has for a gate: a summary saying
what became of the fact, and one annotation per rule that refused it, so a
rule's mean over a project IS its refusal rate and two runs compare rule
by rule without a query.
"""

from __future__ import annotations

import pytest

from database.qa_generator import FactKind, Rejection
from extraction.models import PassageToExtract


class _Held:
    """An Evaluations that keeps what it was given."""

    def __init__(self) -> None:
        """Starts enabled and holding nothing."""
        self.enabled = True
        self.recorded: list = []
        self.flushed = 0

    def record(self, *verdicts) -> None:
        """Keeps some verdicts."""
        self.recorded.extend(verdicts)

    def flush(self) -> int:
        """Counts one flush."""
        self.flushed += 1
        return 0


def _fact(**over):
    """One checked fact, with the fields a test varies."""
    from extraction.models import CheckedFact

    return CheckedFact(
        **{
            "statement": "Die Frist beträgt 48 Stunden.",
            "evidence_text": "Die Frist beträgt 48 Stunden.",
            "extraction_method": "llm",
            "validated": True,
            "rejection_code": None,
            "validation_error": None,
            "kind": FactKind.ATOMIC,
            "citations": [],
            **over,
        }
    )


@pytest.fixture(autouse=True)
def recording_spans(monkeypatch):
    """A tracer that mints real span ids and exports nothing.

    Without one every span is the no-op, its context is invalid and its id
    is empty - and an annotation with no span to attach to is dropped
    rather than invented, which would make these tests pass vacuously.

    Swapped on the module rather than installed globally: the global is a
    proxy that resolves through the same name, so putting it back leaves
    it delegating to itself and the next test to open a span recurses
    until the stack ends.
    """
    from opentelemetry.sdk.trace import TracerProvider

    from extraction import service

    monkeypatch.setattr(service, "span", TracerProvider().get_tracer(__name__))


@pytest.fixture
def service():
    """An extraction service holding a recording Evaluations."""
    from extraction.service import ExtractionService

    built = ExtractionService(
        repository=None,  # type: ignore[arg-type]
        extractors=None,  # type: ignore[arg-type]
        checker=None,  # type: ignore[arg-type]
    )
    built._evaluations = _Held()  # type: ignore[assignment]
    return built


PASSAGE = PassageToExtract(
    id=7,
    text="Die Frist beträgt 48 Stunden.",
    section_path=None,
    block_type="text",
    language="de",
    sentences=[],
)


def test_a_validated_fact_scores_one(service) -> None:
    """So a project's mean over `validation` IS its validation rate."""
    service._record(PASSAGE, [_fact()])

    (summary,) = service._evaluations.recorded
    assert summary.name == "validation"
    assert summary.label == "validated"
    assert summary.score == 1.0


def test_a_refused_fact_names_the_rule_twice(service) -> None:
    """Once as what became of it, once as a column of its own.

    The per-rule annotation is the whole point: without it, `not_atomic`
    moving while the total holds still is invisible.
    """
    service._record(
        PASSAGE,
        [
            _fact(
                validated=False,
                rejection_code=Rejection.NOT_ATOMIC,
                validation_error="it asserts two things",
            )
        ],
    )

    summary, per_rule = service._evaluations.recorded
    assert (summary.name, summary.label, summary.score) == (
        "validation",
        "not_atomic",
        0.0,
    )
    assert per_rule.name == "validation: not_atomic"
    assert per_rule.score == 0.0
    assert per_rule.explanation == "it asserts two things"


def test_every_fact_of_a_passage_is_recorded(service) -> None:
    """A passage has one span and however many facts, so each gets its own."""
    service._record(
        PASSAGE,
        [
            _fact(),
            _fact(validated=False, rejection_code=Rejection.DUPLICATE),
            _fact(kind=FactKind.SUMMARY),
        ],
    )

    names = [one.name for one in service._evaluations.recorded]
    assert names.count("validation") == 3
    assert "validation: duplicate" in names


def test_a_run_that_cannot_reach_phoenix_records_nothing(service) -> None:
    """And costs no span either. The run is what matters, not the watching."""
    service._evaluations.enabled = False

    service._record(PASSAGE, [_fact()])

    assert service._evaluations.recorded == []
    assert service._evaluations.flushed == 0
