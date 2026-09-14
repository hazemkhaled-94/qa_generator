"""The indexes and column types the queries are written against.

An index that is declared but never used is a query doing a sequential scan
over the deliverable, which is slow in a way nothing reports.
"""

from __future__ import annotations

import pytest
from seed import digest, document, fact, passage, question
from sqlalchemy import text
from sqlalchemy.orm import Session

pytestmark = pytest.mark.integration


def indexes(engine, table: str) -> dict[str, str]:
    """The indexes on one table, by name."""
    with engine.connect() as connection:
        return dict(
            connection.execute(
                text("SELECT indexname, indexdef FROM pg_indexes WHERE tablename = :t"),
                {"t": table},
            ).all()
        )


def test_the_question_embedding_is_indexed_for_cosine_search(engine) -> None:
    """The dedup and unanswerability gates search every accepted question."""
    built = indexes(engine, "questions")

    assert "ix_questions_embedding_hnsw" in built, sorted(built)
    definition = built["ix_questions_embedding_hnsw"]
    assert "hnsw" in definition.lower(), definition
    assert "vector_cosine_ops" in definition, definition


def test_the_statement_is_indexed_for_trigram_search(engine) -> None:
    """What the facts page's search box runs against."""
    built = indexes(engine, "facts")

    assert "ix_facts_statement_trgm" in built, sorted(built)
    assert "gin" in built["ix_facts_statement_trgm"].lower()


def test_an_embedding_round_trips_at_the_declared_width(engine, database) -> None:
    """1024, because that is the width of EMBEDDING_MODEL."""
    written = [0.1] * 1024
    with Session(engine) as session:
        asked = question(embedding=written)
        session.add(asked)
        session.commit()
        read = session.get(type(asked), asked.id)
        assert read is not None
        assert len(read.embedding) == 1024
        assert pytest.approx(list(read.embedding), rel=1e-6) == written


def test_an_embedding_of_the_wrong_width_is_refused(engine, database) -> None:
    """A model swapped without a migration would store unsearchable rows."""
    from sqlalchemy.exc import DBAPIError

    with Session(engine) as session:
        session.add(question(embedding=[0.1] * 768))
        with pytest.raises(DBAPIError):
            session.commit()


def test_cosine_distance_orders_by_similarity(engine, database) -> None:
    """What the dedup gate reads, computed by the database."""
    with Session(engine) as session:
        session.add_all(
            [
                question(question_text="near", embedding=[1.0] + [0.0] * 1023),
                question(question_text="far", embedding=[0.0] * 1023 + [1.0]),
            ]
        )
        session.commit()

    with engine.connect() as connection:
        nearest = connection.execute(
            text(
                "SELECT question_text FROM questions "
                "ORDER BY embedding <=> CAST(:probe AS vector) LIMIT 1"
            ),
            {"probe": str([1.0] + [0.0] * 1023)},
        ).scalar_one()

    assert nearest == "near"


def test_a_search_finds_a_statement_by_one_of_its_words(engine, database) -> None:
    """The condition `matching` builds, run against real rows."""
    from database.qa_generator import Fact
    from database.qa_generator.repository import matching

    with Session(engine) as session:
        session.add(document())
        session.flush()
        held = passage(digest())
        session.add(held)
        session.flush()
        session.add_all(
            [
                fact(held.id, statement="The device weighs 4 kg."),
                fact(held.id, statement="The report arrives in March."),
            ]
        )
        session.commit()

        found = session.scalars(
            session.query(Fact.statement)
            .filter(matching("weighs", Fact.statement))
            .statement
        ).all()

    assert found == ["The device weighs 4 kg."]


@pytest.mark.parametrize("typed", ["%", "_", "100%", "a_b"])
def test_a_wildcard_someone_typed_matches_nothing(engine, database, typed) -> None:
    """Unescaped, `%` would return the whole table."""
    from database.qa_generator import Fact
    from database.qa_generator.repository import matching

    with Session(engine) as session:
        session.add(document())
        session.flush()
        held = passage(digest())
        session.add(held)
        session.flush()
        session.add(fact(held.id, statement="The device weighs 4 kg."))
        session.commit()

        found = session.scalars(
            session.query(Fact.statement)
            .filter(matching(typed, Fact.statement))
            .statement
        ).all()

    assert found == []
