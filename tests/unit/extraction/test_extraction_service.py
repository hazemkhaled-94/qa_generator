"""What the per-passage service does with one claimed passage.

A passage that yields nothing is not a failure and neither is a fact that
fails a check; the failures are the passage the model would not read and the
passage that raised something nobody expected.
"""

from __future__ import annotations

import pytest
from drivers import Extraction, passage, table_passage

from database.qa_generator import FactKind
from llm.client import ModelUnavailable

pytestmark = pytest.mark.nlp

ATOMIC = {"facts": [{"sentences": [0], "statement": "The device weighs 4 kg."}]}
DIGEST = {
    "summary": "The device weighs 4 kg.",
    "outline": ["Weighs 4 kg", "Runs for 12 hours"],
}
BOTH = (FactKind.SUMMARY, FactKind.OUTLINE)


def test_an_empty_queue_is_not_worked() -> None:
    """Nothing to claim, nothing to do."""
    run = Extraction()

    assert run.next() is None
    assert run.run() == 0
    assert run.model.calls == 0


def test_a_passage_is_read_and_its_facts_stored() -> None:
    """The id comes back, and the facts are written against it."""
    run = Extraction(passage(), facts=ATOMIC)

    assert run.next() == 1
    assert run.queue.kinds() == [FactKind.ATOMIC]
    assert run.queue.stored[1][0].statement == "The device weighs 4 kg."


def test_a_passage_asserting_nothing_never_reaches_the_model() -> None:
    """One call costs minutes; a heading returns its own text back."""
    run = Extraction(passage("Risiko", language="de"), facts=ATOMIC)
    run.next()

    assert run.model.calls == 0
    assert run.queue.stored == {1: []}, "read with nothing found, not failed"
    assert run.queue.failed == {}


def test_both_readings_are_written_for_one_passage() -> None:
    """The atomic pass and the digest pass, in that order."""
    run = Extraction(passage(), facts=ATOMIC, digest=DIGEST, kinds=BOTH)
    run.next()

    assert run.queue.kinds() == [
        FactKind.ATOMIC,
        FactKind.SUMMARY,
        FactKind.OUTLINE,
    ]
    assert run.model.calls == 1 and run.digest_model.calls == 1


def test_no_digest_reader_means_no_extra_call() -> None:
    """A deployment that writes neither pays for neither."""
    run = Extraction(passage(), facts=ATOMIC, digest=DIGEST)
    run.next()

    assert run.queue.kinds() == [FactKind.ATOMIC]
    assert run.digest_model.calls == 0


def test_a_passage_carrying_one_claim_is_not_digested() -> None:
    """It is already as short as its own summary."""
    run = Extraction(
        passage("The device weighs 4 kg."), facts=ATOMIC, digest=DIGEST, kinds=BOTH
    )
    run.next()

    assert run.queue.kinds() == [FactKind.ATOMIC]
    assert run.digest_model.calls == 0, "the passage carries one claim"


def test_a_model_that_will_not_answer_fails_the_passage() -> None:
    """The row carries the reason rather than being left in progress."""
    run = Extraction(passage(), facts=ModelUnavailable("Timeout: took too long"))
    run.next()

    assert run.queue.stored == {}
    assert "took too long" in run.queue.failed[1]


def test_a_digest_the_model_will_not_answer_fails_the_passage() -> None:
    """Both readings are one unit of work: half of one is not stored."""
    run = Extraction(
        passage(),
        facts=ATOMIC,
        digest=ModelUnavailable("Timeout: took too long"),
        kinds=BOTH,
    )
    run.next()

    assert run.queue.stored == {}
    assert "took too long" in run.queue.failed[1]


def test_an_unexpected_failure_is_named_by_its_type() -> None:
    """A row failed by something nobody planned for still says what it was."""
    run = Extraction(passage(), facts=ValueError("nonsense came back"))
    run.next()

    assert run.queue.failed[1] == "ValueError: nonsense came back"


def test_draining_works_every_passage_and_stops() -> None:
    """The loop ends on the first empty claim rather than spinning."""
    from dataclasses import replace

    first = passage()
    run = Extraction(first, replace(first, id=2), facts=ATOMIC)

    assert run.run() == 2
    assert sorted(run.queue.stored) == [1, 2]


def test_a_claim_an_earlier_run_left_behind_is_swept_first() -> None:
    """A drain begins by failing whatever no worker is on any more."""
    run = Extraction(passage(), facts=ATOMIC)
    run.queue.swept = 3

    assert run.run() == 1, "the sweep does not count as work"


def test_a_rejected_fact_is_stored_rather_than_dropped() -> None:
    """The share that failed is how extraction is judged."""
    invented = {"facts": [{"sentences": [0], "statement": "The device weighs 9 kg."}]}
    run = Extraction(passage(), facts=invented)
    run.next()

    stored = run.queue.stored[1]
    assert len(stored) == 1
    assert not stored[0].validated
    assert stored[0].rejection_code == "unsupported_addition"


def test_a_table_is_read_without_a_model() -> None:
    """The block type routes the passage to the deterministic reader."""
    run = Extraction(table_passage(), facts=ATOMIC)
    run.next()

    assert run.model.calls == 0, "a table needs no model"
    assert run.queue.stored[1], "and still yields facts"
    assert run.queue.stored[1][0].extraction_method == "deterministic"
