"""Which facts go together, and which questions get no answer.

Both decisions are deterministic on purpose. A reference dataset whose
contents move between runs of the same corpus is not a reference, which is
the same reason TOPIC_RANDOM_STATE is fixed.
"""

from __future__ import annotations

from factories import source

from database.qa_generator import Difficulty
from question_generation.selection import groups, perturbed


def test_a_group_spans_two_documents_when_the_topic_has_two() -> None:
    """Which is what a cross-document question is, and what it is for."""
    facts = [
        source(1, document="a", passage_id=1),
        source(2, document="a", passage_id=2),
        source(3, document="b", passage_id=3),
        source(4, document="b", passage_id=4),
    ]

    formed = groups(facts, wanted=10, size=2)

    assert [one.difficulty for one in formed] == [Difficulty.CROSS_DOCUMENT] * 2


def test_reading_straight_down_the_list_would_never_cross_a_document() -> None:
    """The failure the interleaving exists to prevent.

    The facts arrive sorted by document, so a group taken in order is one
    document's facts every time.
    """
    facts = [source(n, document="a" if n < 2 else "b", passage_id=n) for n in range(4)]

    straight = [facts[0:2], facts[2:4]]

    assert all(len({one.doc_sha256 for one in pair}) == 1 for pair in straight)
    assert [one.difficulty for one in groups(facts, wanted=4, size=2)] == [
        Difficulty.CROSS_DOCUMENT
    ] * 2


def test_one_document_still_yields_groups() -> None:
    """A corpus of one document has no cross-document question to write."""
    facts = [source(n, document="a", passage_id=n) for n in range(4)]

    formed = groups(facts, wanted=10, size=2)

    assert [one.difficulty for one in formed] == [Difficulty.CROSS_PASSAGE] * 2


def test_two_facts_of_one_passage_are_a_single_passage_question() -> None:
    """Difficulty is the spread of the evidence, not the count of it."""
    formed = groups(
        [source(1, document="a", passage_id=7), source(2, document="a", passage_id=7)],
        wanted=1,
        size=2,
    )

    assert formed[0].difficulty == Difficulty.SINGLE_PASSAGE


def test_a_group_of_one_is_an_ordinary_question() -> None:
    """QUESTIONS_GROUP_SIZE=1 turns cross-document questions off."""
    facts = [source(n, document="a" if n % 2 else "b", passage_id=n) for n in range(4)]

    formed = groups(facts, wanted=10, size=1)

    assert all(len(one.facts) == 1 for one in formed)
    assert {one.difficulty for one in formed} == {Difficulty.SINGLE_PASSAGE}


def test_no_more_groups_than_were_wanted() -> None:
    """QUESTIONS_PER_TOPIC is what a run costs, so it has to be the ceiling."""
    facts = [source(n, document="a", passage_id=n) for n in range(20)]

    assert len(groups(facts, wanted=3, size=2)) == 3


def test_no_facts_is_no_groups_rather_than_an_empty_one() -> None:
    """An empty group has no language and no difficulty to read."""
    assert groups([], wanted=5, size=2) == []


def test_the_same_facts_deal_the_same_groups_whatever_order_they_arrive_in() -> None:
    """The dataset must not depend on what the database returned first."""
    facts = [source(n, document="ab"[n % 2], passage_id=n) for n in range(8)]

    forwards = groups(facts, wanted=4, size=2)
    backwards = groups(list(reversed(facts)), wanted=4, size=2)

    assert [[f.id for f in one.facts] for one in forwards] == [
        [f.id for f in one.facts] for one in backwards
    ]


def test_every_fact_is_used_at_most_once() -> None:
    """A fact in two groups is the same claim tested twice."""
    facts = [source(n, document="abc"[n % 3], passage_id=n) for n in range(9)]

    used = [fact.id for one in groups(facts, wanted=9, size=2) for fact in one.facts]

    assert len(used) == len(set(used))


def test_the_unanswerable_share_is_exact_and_spread_out() -> None:
    """A quarter means one in every four, not a coin weighted a quarter."""
    over = [perturbed(index, 0.25) for index in range(12)]

    assert sum(over) == 3
    assert over == [False, False, False, True] * 3


def test_a_share_of_zero_perturbs_nothing_and_a_share_of_one_perturbs_all() -> None:
    """Both ends without a branch of their own."""
    assert not any(perturbed(index, 0.0) for index in range(10))
    assert all(perturbed(index, 1.0) for index in range(10))
