"""The trigger that deletes a question once its last fact is gone.

A foreign key cascades parent to child, never the reverse, so a question
outliving every fact it was drawn from would be a question nothing can be
traced back to.
"""

from __future__ import annotations

import pytest
from seed import digest, document, fact, link, passage, question
from sqlalchemy import text
from sqlalchemy.orm import Session

pytestmark = pytest.mark.integration


@pytest.fixture
def written(engine, database):
    """A question drawn from however many facts a test asks for."""

    def write(facts: int = 1):
        with Session(engine) as session:
            session.add(document())
            session.flush()
            held = passage(digest())
            session.add(held)
            session.flush()
            drawn = [fact(held.id, statement=f"Claim {n}.") for n in range(facts)]
            session.add_all(drawn)
            asked = question()
            session.add(asked)
            session.flush()
            session.add_all(link(asked.id, one.id) for one in drawn)
            session.commit()
            return asked.id, [one.id for one in drawn]

    return write


def counts(engine) -> tuple[int, int]:
    """How many questions and links are left."""
    with engine.connect() as connection:
        return (
            connection.execute(text("SELECT count(*) FROM questions")).scalar_one(),
            connection.execute(
                text("SELECT count(*) FROM question_facts")
            ).scalar_one(),
        )


def test_deleting_the_only_fact_deletes_the_question(written, engine) -> None:
    """Nothing is left that cannot be traced to a passage."""
    _, facts = written(1)

    with engine.begin() as connection:
        connection.execute(text("DELETE FROM facts WHERE id = :id"), {"id": facts[0]})

    assert counts(engine) == (0, 0)


def test_deleting_one_of_two_facts_keeps_the_question(written, engine) -> None:
    """A cross-document question still has a source."""
    _, facts = written(2)

    with engine.begin() as connection:
        connection.execute(text("DELETE FROM facts WHERE id = :id"), {"id": facts[0]})

    assert counts(engine) == (1, 1)


def test_deleting_the_last_of_two_facts_deletes_the_question(written, engine) -> None:
    """The trigger fires per row, and the guard decides."""
    _, facts = written(2)

    with engine.begin() as connection:
        connection.execute(
            text("DELETE FROM facts WHERE id = ANY(:ids)"), {"ids": facts}
        )

    assert counts(engine) == (0, 0)


def test_deleting_the_passage_takes_the_question_with_it(written, engine) -> None:
    """Through the facts: passage to fact cascades, fact to question triggers."""
    written(2)

    with engine.begin() as connection:
        connection.execute(text("DELETE FROM passages"))

    assert counts(engine) == (0, 0)


def test_deleting_the_document_takes_the_whole_chain(written, engine) -> None:
    """One delete at the top, and nothing derived from it survives."""
    written(3)

    with engine.begin() as connection:
        connection.execute(text("DELETE FROM documents"))

    assert counts(engine) == (0, 0)
    with engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM facts")).scalar() == 0


def test_a_question_with_no_facts_at_all_is_left_alone(engine, database) -> None:
    """The trigger fires on a deleted link, not on a question that had none."""
    with Session(engine) as session:
        session.add(question())
        session.commit()

    assert counts(engine) == (1, 0)
