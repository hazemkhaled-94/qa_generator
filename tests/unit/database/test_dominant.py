"""Which topic a passage counts towards, when two of them tie.

`DOMINANT` is read by four queries in two services: the facts a topic is the
subject of, the passages a bridge pass groups, what a question listing is
filtered by, and what a question's `topic_scope` is counted over. Every one
of them has to get the same answer for the same passage, or a question is
stored under a subject the page it was drawn on does not list it under.
"""

from __future__ import annotations

from sqlalchemy.dialects import postgresql

from database.qa_generator.passage_topics import DOMINANT


def compiled() -> str:
    """The subquery as PostgreSQL receives it."""
    return str(DOMINANT.original.compile(dialect=postgresql.dialect()))


def test_one_row_per_passage() -> None:
    """Two topics tied at one weight must not yield two rows."""
    assert "DISTINCT ON (passage_topics.passage_id)" in compiled()


def test_the_tie_is_broken_by_something_stable() -> None:
    """The weight alone leaves the choice to the query plan.

    Two topics at the same weight is uncommon and not impossible - the
    factorisation writes floats and the membership floor keeps whatever
    clears it - and with no tiebreaker the row PostgreSQL happens to reach
    first is the one that wins. Every other ordering in this pipeline names
    its tiebreaker for exactly this reason.
    """
    ordering = compiled().split("ORDER BY", 1)[1]

    assert ordering.strip() == (
        "passage_topics.passage_id, passage_topics.weight DESC, passage_topics.topic_id"
    )
