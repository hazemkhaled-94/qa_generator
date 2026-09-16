"""Which passages of a topic are put in front of the model together.

A bridge is only worth the call if the two passages could plausibly share a
claim, and only worth storing if the reader can check both. Grouping decides
the first of those.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from drivers import passage

from extraction.service import grouped


def topic(*documents: str):
    """Builds one passage per entry, each naming the document it came from.

    Args:
        *documents: The document of each passage, in reading order.

    Returns:
        The passages, numbered from 1.
    """
    return [
        replace(passage(), id=at + 1, doc_sha256=document)
        for at, document in enumerate(documents)
    ]


def documents_of(group) -> list[str | None]:
    """The document each passage of a group came from."""
    return [one.doc_sha256 for one in group]


def test_a_topic_smaller_than_one_group_yields_nothing() -> None:
    """There is no pair to put together."""
    assert grouped(topic("a"), wanted=5, size=2) == []
    assert grouped([], wanted=5, size=2) == []


def test_a_group_of_one_is_not_a_group() -> None:
    """A bridge needs two passages by definition."""
    assert grouped(topic("a", "b", "c"), wanted=5, size=1) == []


def test_nothing_is_asked_for_when_nothing_is_wanted() -> None:
    """A run configured for no groups makes no calls."""
    assert grouped(topic("a", "b"), wanted=0, size=2) == []


def test_two_documents_are_paired_across_rather_than_within() -> None:
    """A cross-document bridge is the one a retriever cannot fake."""
    groups = grouped(topic("a", "a", "b", "b"), wanted=2, size=2)

    assert groups, "two documents of two passages offer two pairs"
    assert all(len(set(documents_of(one))) == 2 for one in groups), [
        documents_of(one) for one in groups
    ]


def test_one_document_still_groups_within_itself() -> None:
    """A topic sitting in one file has no cross-document pair to offer."""
    groups = grouped(topic("a", "a", "a", "a"), wanted=2, size=2)

    assert len(groups) == 2
    assert all(documents_of(one) == ["a", "a"] for one in groups)


def test_no_more_groups_than_were_asked_for() -> None:
    """The count multiplied by the topics is what a run costs."""
    groups = grouped(topic(*("a" * 1), *("abcdefgh")), wanted=3, size=2)
    assert len(groups) <= 3


def test_every_group_is_exactly_the_size_asked_for() -> None:
    """A short tail is dropped rather than offered as a smaller group."""
    groups = grouped(topic("a", "b", "c", "d", "e"), wanted=5, size=2)

    assert groups
    assert all(len(one) == 2 for one in groups), [len(one) for one in groups]


def test_a_passage_is_never_paired_with_itself() -> None:
    """The same passage twice is one passage, and bridges nothing."""
    groups = grouped(topic("a", "b", "c", "d"), wanted=4, size=2)

    for one in groups:
        assert len({passage.id for passage in one}) == len(one), one


def test_the_groups_are_spread_over_the_topic_rather_than_its_start() -> None:
    """Asking about the first four passages of eighty is not asking about it."""
    every = topic(*["a"] * 20)
    groups = grouped(every, wanted=2, size=2)

    reached = [passage.id for one in groups for passage in one]
    assert max(reached) > 8, reached


@pytest.mark.parametrize("size", [2, 3, 4])
def test_a_group_holds_as_many_passages_as_asked(size) -> None:
    """The size is the number the prompt numbers, whatever it is."""
    groups = grouped(topic(*"abcdefghijkl"), wanted=2, size=size)

    assert groups
    assert all(len(one) == size for one in groups)
