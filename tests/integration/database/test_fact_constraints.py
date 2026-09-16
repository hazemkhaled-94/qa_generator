"""What the database refuses to hold about a fact's kind and its passages.

The CHECK constraints the four kinds added, tested where they run rather
than against the model that declares them: a hand-run UPDATE reaches the
table and never reaches the model.
"""

from __future__ import annotations

import pytest
from seed import digest, document, fact, passage
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session

from database.qa_generator import FactKind, FactPassage, Rejection

pytestmark = pytest.mark.integration


@pytest.fixture
def session(engine, database):
    """A session on the empty database, rolled back if a test leaves work."""
    with Session(engine) as opened:
        yield opened
        opened.rollback()


def refuses(session, row, constraint: str) -> None:
    """Asserts the database rejects a row, naming the constraint it broke."""
    session.add(row)
    with pytest.raises((IntegrityError, DBAPIError)) as raised:
        session.flush()
    assert constraint in str(raised.value), str(raised.value)[:400]
    session.rollback()


@pytest.fixture
def passages(session) -> list[int]:
    """Two passages of two documents, written and flushed."""
    written = []
    for seed in ("a", "b"):
        session.add(document(digest(seed)))
        row = passage(digest(seed))
        session.add(row)
        session.flush()
        written.append(row.id)
    return written


@pytest.mark.parametrize("kind", list(FactKind))
def test_every_declared_kind_is_accepted(session, passages, kind) -> None:
    """The vocabulary in the column and the one in the code are one list."""
    session.add(fact(passages[0], kind=kind))
    session.flush()


@pytest.mark.parametrize("kind", ["paragraph", "", "ATOMIC", "digest"])
def test_a_kind_nothing_writes_is_refused(session, passages, kind) -> None:
    """The kind decides which checks a re-judgement applies."""
    refuses(session, fact(passages[0], kind=kind), "facts_kind_valid")


def test_a_fact_arrives_atomic_when_nothing_says_otherwise(session, passages) -> None:
    """Which is what the server default backfilled the existing corpus with."""
    session.execute(
        text(
            "INSERT INTO facts (statement, evidence_text, extraction_method, "
            "validated, units_statement, units_added, unresolved_references) "
            "VALUES ('A claim.', 'A claim.', 'llm', true, '{}', '{}', '{}')"
        )
    )
    session.flush()

    assert session.execute(text("SELECT kind FROM facts")).scalar() == FactKind.ATOMIC


@pytest.mark.parametrize(
    "code",
    [
        Rejection.ASSERTS_NOTHING,
        Rejection.NOT_CONDENSED,
        Rejection.NOT_LISTED,
        Rejection.NOT_BRIDGING,
    ],
)
def test_every_new_rejection_code_is_accepted(session, passages, code) -> None:
    """A code the checks can emit and the column refuses is a failed write."""
    session.add(fact(passages[0], validated=False, rejection_code=code))
    session.flush()


def test_a_bridge_records_the_passages_it_rests_on(session, passages) -> None:
    """One row per passage, ordered by the position it was shown in."""
    anchor, other = passages
    session.add(fact(anchor, other, kind=FactKind.BRIDGE))
    session.flush()

    held = session.execute(
        text("SELECT passage_id FROM fact_passages ORDER BY position")
    ).scalars()
    assert list(held) == [anchor, other]


@pytest.mark.parametrize("kind", [FactKind.ATOMIC, FactKind.SUMMARY])
def test_every_other_kind_records_the_one_passage_it_rests_on(
    session, passages, kind
) -> None:
    """The link table is the only route to a passage, for every kind."""
    session.add(fact(passages[0], kind=kind))
    session.flush()

    held = session.execute(text("SELECT passage_id, position FROM fact_passages")).all()
    assert held == [(passages[0], 0)]


def test_one_passage_cannot_be_named_twice_by_one_fact(session, passages) -> None:
    """The same passage twice is one passage, and bridges nothing."""
    anchor = passages[0]
    row = fact(anchor, kind=FactKind.BRIDGE)
    session.add(row)
    session.flush()

    refuses(
        session,
        FactPassage(fact_id=row.id, passage_id=anchor, position=1),
        "fact_passages_pkey",
    )


def test_a_link_to_a_passage_that_does_not_exist_is_refused(session, passages) -> None:
    """A bridge cannot rest on something nobody can open."""
    row = fact(passages[0], kind=FactKind.BRIDGE)
    session.add(row)
    session.flush()

    refuses(
        session,
        FactPassage(fact_id=row.id, passage_id=999_999, position=1),
        "fact_passages_passage_id_fkey",
    )


def test_deleting_a_document_takes_the_bridges_that_rested_on_it(
    session, passages
) -> None:
    """One delete, and nothing derived from it survives."""
    anchor, other = passages
    session.add(fact(anchor, other, kind=FactKind.BRIDGE))
    session.flush()

    session.execute(text("DELETE FROM documents"))
    session.flush()

    assert session.execute(text("SELECT count(*) FROM facts")).scalar() == 0
    assert session.execute(text("SELECT count(*) FROM fact_passages")).scalar() == 0


def test_the_orphan_trigger_is_installed(engine, database) -> None:
    """Declared beside the table, executed by the migration that made it."""
    with engine.connect() as connection:
        triggers = set(
            connection.execute(text("SELECT tgname FROM pg_trigger")).scalars()
        )
    assert "fact_passages_delete_orphan_fact" in triggers, sorted(triggers)


def test_losing_any_passage_a_fact_rests_on_takes_the_fact(session, passages) -> None:
    """A claim outliving half its evidence would be a claim nobody can check."""
    anchor, other = passages
    session.add(fact(anchor, other, kind=FactKind.BRIDGE))
    session.add(fact(anchor))
    session.flush()

    # The second passage, not the one either fact opens on.
    session.execute(text("DELETE FROM passages WHERE id = :id"), {"id": other})
    session.flush()

    assert session.execute(text("SELECT count(*) FROM facts")).scalar() == 1
    assert session.execute(text("SELECT kind FROM facts")).scalar() == FactKind.ATOMIC
