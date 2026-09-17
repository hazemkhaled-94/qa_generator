"""How a review sample is drawn, and what a record carries.

The sampling is the part worth pinning. A review answers "is the checker
right", and a sample of only what the checker accepted cannot answer it -
nor can a sample in id order, which for this corpus is a sample of
whichever document was extracted first.
"""

from __future__ import annotations

import pytest

pytest.importorskip("argilla", reason="the observability group is not installed")

from review import datasets
from review.records import FactRow, QuestionRow, _even


def test_the_sample_is_split_across_the_groups() -> None:
    """Two hundred over ten verdicts is twenty of each."""
    assert _even(200, 10) == 20


def test_a_rare_group_still_gets_one() -> None:
    """A rare group still gets one record.

    A rejection code that fired twice in a whole corpus is the
    interesting one, and a proportional sample would never show it.
    """
    assert _even(200, 1000) == 1


def test_no_groups_does_not_divide_by_zero() -> None:
    """An empty corpus has no verdicts to spread a sample over.

    What the answer is does not matter - the loop that asks has nothing to
    iterate - but raising here would make an empty database an error
    instead of an empty push.
    """
    assert _even(200, 0) >= 1


def fact(**overrides) -> FactRow:
    """A fact row with the fields a record needs."""
    return FactRow(
        **{
            "id": 41,
            "statement": "The fee is 120 EUR.",
            "evidence": "The annual fee is 120 EUR for standard members.",
            "kind": "atomic",
            "validated": True,
            "rejection_code": None,
            "validation_error": None,
            **overrides,
        }
    )


def test_a_fact_record_carries_its_id_for_the_pull() -> None:
    """A fact record carries its id twice, and needs to.

    The pull reads the metadata; Argilla deduplicates on the record id.
    """
    record = datasets.fact_record(fact())

    assert record.id == "41"
    assert record.metadata["fact_id"] == 41


def test_an_accepted_fact_says_so_rather_than_saying_nothing() -> None:
    """An accepted fact says so rather than saying nothing.

    `rejection_code` is NULL on one, and a metadata value of None would
    render as an empty filter nobody can select.
    """
    assert datasets.fact_record(fact()).metadata["checker"] == "accepted"


def test_a_refused_fact_carries_the_code_the_checker_used() -> None:
    """A refused fact carries the code the checker used.

    Anchoring is the risk and it is the lesser one: a reviewer told
    nothing is looking at a wall of statements.
    """
    record = datasets.fact_record(fact(validated=False, rejection_code="copied"))

    assert record.metadata["checker"] == "copied"


def question(**overrides) -> QuestionRow:
    """A question row with the fields a record needs."""
    return QuestionRow(
        **{
            "id": 7,
            "question_text": "How long does a standard request take?",
            "target_answer": "48 hours.",
            "answerable": True,
            "difficulty": "easy",
            "question_type": "factoid",
            "language": "en",
            "status": "draft",
            "rejected_reason": None,
            "facts": ["Standard requests are answered within 48 hours."],
            **overrides,
        }
    )


def test_an_unanswerable_question_shows_that_it_is_deliberate() -> None:
    """An unanswerable question shows that it is deliberate.

    Some are written on purpose, to check a chatbot says it does not
    know. Rendered as an empty answer it reads as a bug instead.
    """
    record = datasets.question_record(question(answerable=False, target_answer=None))

    assert "unanswerable" in record.fields["answer"]
    assert record.metadata["answerable"] == "false"


def test_a_question_carries_the_facts_it_was_written_from() -> None:
    """Without them a reviewer cannot judge whether the answer follows."""
    record = datasets.question_record(question())

    assert "48 hours" in record.fields["facts"]


def test_a_questions_gate_falls_back_to_its_status() -> None:
    """A question's gate falls back to its status.

    One nobody refused has no gate to name, and the status is what there
    is to say about it.
    """
    assert datasets.question_record(question()).metadata["gate"] == "draft"
    assert (
        datasets.question_record(question(rejected_reason="compound")).metadata["gate"]
        == "compound"
    )


def test_every_dataset_name_has_a_schema_to_create_it_with() -> None:
    """A name with no builder is a KeyError on the first push.

    The schemas themselves are not built here. `rg.Settings(...)`
    constructs a client and calls the server while being constructed, so
    building one is an integration concern; what this catches is the
    cheap half, which is a name added to NAMES and nowhere else.
    """
    assert set(datasets.SETTINGS) == set(datasets.NAMES)
