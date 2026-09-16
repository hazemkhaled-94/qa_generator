"""What must hold of a fact for any passage at all.

The example-based tests say what happens to the text this corpus holds.
These say what must hold for a document nobody has seen yet.
"""

from __future__ import annotations

import pytest
from drivers import DIGEST_SHARE, Checker, group, passage
from hypothesis import given, settings
from hypothesis import strategies as st

from database.qa_generator import FactKind, Rejection
from extraction.models import BULLET, MIN_POINTS, CandidateFact
from extraction.service import grouped
from extraction.validation import FactChecker

#: Passages of one topic, as (document, passage id).
TOPIC = st.lists(
    st.tuples(st.sampled_from("abcd"), st.integers(min_value=1, max_value=200)),
    min_size=0,
    max_size=24,
    unique_by=lambda one: one[1],
)


def _passages(rows):
    """Turns the generated pairs into the passages a group is formed from."""
    from dataclasses import replace

    template = passage()
    return [
        replace(template, id=passage_id, doc_sha256=document)
        for document, passage_id in rows
    ]


@given(rows=TOPIC, wanted=st.integers(1, 8), size=st.integers(2, 4))
@settings(max_examples=200, deadline=None)
def test_a_group_holds_exactly_the_passages_asked_for(rows, wanted, size) -> None:
    """A short tail is dropped rather than offered as a smaller group."""
    for one in grouped(_passages(rows), wanted, size):
        assert len(one) == size


@given(rows=TOPIC, wanted=st.integers(1, 8), size=st.integers(2, 4))
@settings(max_examples=200, deadline=None)
def test_a_passage_is_never_offered_twice_in_one_group(rows, wanted, size) -> None:
    """The same passage twice bridges nothing."""
    for one in grouped(_passages(rows), wanted, size):
        assert len({passage.id for passage in one}) == size


@given(rows=TOPIC, wanted=st.integers(1, 8), size=st.integers(2, 4))
@settings(max_examples=200, deadline=None)
def test_no_more_groups_than_were_asked_for(rows, wanted, size) -> None:
    """The count multiplied by the topics is what a bridge run costs."""
    assert len(grouped(_passages(rows), wanted, size)) <= wanted


@given(rows=TOPIC, wanted=st.integers(1, 8), size=st.integers(2, 4))
@settings(max_examples=200, deadline=None)
def test_a_topic_smaller_than_one_group_offers_none(rows, wanted, size) -> None:
    """There is no pair to put together."""
    passages = _passages(rows)
    if len(passages) < size:
        assert grouped(passages, wanted, size) == []


@given(
    statement=st.text(max_size=120),
    cited=st.lists(st.integers(-5, 5), max_size=4),
    kind=st.sampled_from([FactKind.ATOMIC, FactKind.SUMMARY, FactKind.OUTLINE]),
)
@settings(max_examples=100, deadline=None)
@pytest.mark.nlp
def test_a_verdict_and_its_code_always_agree(statement, cited, kind) -> None:
    """The column's CHECK constraint refuses a row where they do not."""
    checked = FactChecker(DIGEST_SHARE).check(
        passage(), CandidateFact(statement, tuple(cited), kind=kind), "llm"
    )

    assert checked.validated == (checked.rejection_code is None)
    assert (checked.validation_error is None) == (checked.rejection_code is None)


@given(
    statement=st.text(max_size=120),
    cited=st.lists(st.integers(0, 1), min_size=1, max_size=2, unique=True),
)
@settings(max_examples=100, deadline=None)
@pytest.mark.nlp
def test_an_evidence_span_always_resolves_in_its_passage(statement, cited) -> None:
    """The invariant every reader of the facts table relies on."""
    under = passage()
    checked = FactChecker(DIGEST_SHARE).check(
        under, CandidateFact(statement, tuple(cited)), "llm"
    )

    assert checked.evidence_end >= checked.evidence_start
    assert under.text[checked.evidence_start : checked.evidence_end] == (
        checked.evidence_text
    )


@given(
    statement=st.text(max_size=120),
    rests_on=st.lists(st.integers(-3, 3), max_size=4),
)
@settings(max_examples=100, deadline=None)
@pytest.mark.nlp
def test_a_bridge_records_only_passages_it_was_offered(statement, rests_on) -> None:
    """A position outside the group names nothing, so it rests on nothing."""
    offered = group("Requests are answered within 48 hours.", "Urgent: 4 hours.")
    checked = FactChecker(DIGEST_SHARE).check_bridge(
        offered,
        CandidateFact(statement, (), kind=FactKind.BRIDGE, passages=tuple(rests_on)),
    )
    held = {one.id for one in offered}

    assert set(checked.passage_ids) <= held
    assert len(checked.passage_ids) == len(set(checked.passage_ids))
    # Never validated, rather than always `not_bridging`: an empty statement
    # resting on one passage is refused by `asserts_nothing` first, and which
    # check fires first is the order, not the invariant.
    if len(checked.passage_ids) < 2:
        assert not checked.validated


@given(points=st.lists(st.text(max_size=20), max_size=6))
@settings(max_examples=100, deadline=None)
@pytest.mark.nlp
def test_an_outline_is_refused_on_its_shape_before_anything_else(points) -> None:
    """A bullet is a fragment, so nothing can be read off its grammar."""
    checker = Checker(share=DIGEST_SHARE)
    written = "\n".join(f"{BULLET}{point}" for point in points if point.strip())

    checked = checker.raw_outline(written)
    if written.count(BULLET) < MIN_POINTS:
        assert checked.rejection_code == Rejection.NOT_LISTED
