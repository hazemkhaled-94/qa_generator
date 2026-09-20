"""What the per-passage service does with one claimed passage.

A passage that yields nothing is not a failure and neither is a fact that
fails a check; the failures are the passage the model would not read and the
passage that raised something nobody expected.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from drivers import Extraction, passage, table_passage

from database.qa_generator import FactKind, Rejection
from extraction.models import Twin
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


class TestProseMergedWithATable:
    """The chunker labels a passage `table` if ANY item in it is one.

    It then numbers the whole of it by rendered line with no predicates
    recorded, so `claims` is zero for every table and prose merged into one
    is invisible. The cell reader walks the grid; nothing read the rest.
    Measured over this corpus: 161 of 188 table passages carry lines
    outside their grids, about 145,000 characters.
    """

    #: One line, one finite verb, and nothing in the grid.
    PROSE = "The compact model replaced the previous range in March."

    def test_the_prose_is_read_by_the_model_as_well(self) -> None:
        """Both readers, so neither half of the passage is thrown away."""
        run = Extraction(table_passage(prose=self.PROSE), facts=ATOMIC)
        run.next()

        assert run.model.calls == 1, "the prose beside the grid is worth a call"
        methods = {one.extraction_method for one in run.queue.stored[1]}
        assert methods == {"deterministic", "llm"}

    def test_a_table_with_no_prose_still_needs_no_model(self) -> None:
        """The rendered rows and the `|---|` separator carry no finite verb."""
        run = Extraction(table_passage(), facts=ATOMIC)
        run.next()

        assert run.model.calls == 0

    def test_the_cell_facts_are_kept_beside_the_prose(self) -> None:
        """The grid is still read deterministically, which is what it is for."""
        run = Extraction(table_passage(prose=self.PROSE), facts=ATOMIC)
        run.next()

        composed = [
            one
            for one in run.queue.stored[1]
            if one.extraction_method == "deterministic"
        ]
        assert composed, "the grid is still read from its cells"


class TestTheDedupGate:
    """A statement the corpus already holds is stored refused, not dropped."""

    TWICE: ClassVar[dict] = {
        "facts": [
            {"sentences": [0], "statement": "The device weighs 4 kg."},
            {"sentences": [0], "statement": "The device weighs 4 kg."},
        ]
    }

    def test_one_passage_saying_a_thing_twice_keeps_it_once(self) -> None:
        """The index holds nothing yet, so only the run's own memory catches it."""
        run = Extraction(passage(), facts=self.TWICE, duplicate_cosine=0.9)
        run.next()

        stored = run.queue.stored[1]
        assert [one.validated for one in stored] == [True, False]
        assert stored[1].rejection_code == Rejection.DUPLICATE

    def test_a_refused_duplicate_is_kept_with_its_reason(self) -> None:
        """Drop-rate evidence, like every other refusal."""
        run = Extraction(passage(), facts=self.TWICE, duplicate_cosine=0.9)
        run.next()

        refused = run.queue.stored[1][1]
        assert "already says" in (refused.validation_error or "")
        assert refused.embedding is not None, "the vector it was judged on is kept"

    def test_a_statement_already_in_the_corpus_is_refused(self) -> None:
        """What the HNSW probe answers is what the gate reads."""
        run = Extraction(passage(), facts=ATOMIC, duplicate_cosine=0.9)
        run.queue.twin = Twin(fact_id=7, statement="Already said.", similarity=0.99)
        run.next()

        stored = run.queue.stored[1]
        assert stored[0].rejection_code == Rejection.DUPLICATE
        assert "fact 7" in (stored[0].validation_error or "")

    def test_a_twin_below_the_threshold_is_not_a_duplicate(self) -> None:
        """Sharing a subject is not saying the same thing."""
        run = Extraction(passage(), facts=ATOMIC, duplicate_cosine=0.9)
        run.queue.twin = Twin(fact_id=7, statement="Something else.", similarity=0.7)
        run.next()

        assert run.queue.stored[1][0].validated

    def test_the_gate_off_still_writes_the_vectors(self) -> None:
        """A deployment may want the embeddings without the refusals."""
        run = Extraction(passage(), facts=self.TWICE, embed=True)
        run.next()

        stored = run.queue.stored[1]
        assert all(one.validated for one in stored)
        assert all(one.embedding is not None for one in stored)

    def test_no_embedder_writes_no_vector_and_refuses_nothing(self) -> None:
        """The default deployment is unchanged."""
        run = Extraction(passage(), facts=self.TWICE)
        run.next()

        stored = run.queue.stored[1]
        assert all(one.validated for one in stored)
        assert all(one.embedding is None for one in stored)
        assert run.queue.embeddings[1] is None

    def test_the_passage_is_embedded_beside_its_facts(self) -> None:
        """One model load writes both, which is why extraction owns the column."""
        run = Extraction(passage(), facts=ATOMIC, embed=True)
        run.next()

        assert run.queue.embeddings[1] is not None

    def test_a_refused_fact_is_not_what_a_later_one_is_compared_against(self) -> None:
        """Otherwise the first copy kept depends on which check ran first."""
        run = Extraction(passage(), facts=self.TWICE, duplicate_cosine=0.9)
        run.next()

        kept = [one for one in run.queue.stored[1] if one.validated]
        assert len(kept) == 1, "the second is refused against the first, not the third"
