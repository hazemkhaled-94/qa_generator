"""Drawing a balanced release, against the table that stores it.

`choose` is covered over lists in tests/unit/questions/test_balance.py. What
is covered here is the round trip: which questions the catalogue offers, what
the draw writes, and that a second draw replaces the first rather than adding
to it.
"""

from __future__ import annotations

import pytest
from seed import digest, document, fact, fitted, membership, passage, question
from sqlalchemy import text
from sqlalchemy.orm import Session

from database.qa_generator import QuestionFact
from question_generation.catalog import QuestionCatalog

pytestmark = pytest.mark.integration


BANDS = {"easy": 1, "medium": 1, "hard": 1}
TYPES = {"factoid": 1, "reason": 1}


@pytest.fixture
def written(engine, database):
    """Writes accepted questions of the bands and kinds a test asks for."""

    def write(rows: list[tuple[str, str, bool, str]]) -> list[int]:
        with Session(engine) as session:
            sha = digest("a")
            session.add(document(sha))
            held = fitted(0)
            session.add(held)
            session.flush()
            at = passage(sha, ordinal=1, text="The device weighs 4 kg.", language="en")
            session.add(at)
            session.flush()
            session.add(membership(at.id, held.id))
            drawn = fact(at.id, statement="The device weighs 4 kg.")
            session.add(drawn)
            session.flush()

            ids = []
            for band, kind, answerable, status in rows:
                asked = question(
                    question_text=f"A {band} {kind} question?",
                    target_answer="4 kg" if answerable else None,
                    answerable=answerable,
                    difficulty=band,
                    question_type=kind,
                    status=status,
                )
                session.add(asked)
                session.flush()
                session.add(QuestionFact(question_id=asked.id, fact_id=drawn.id))
                ids.append(asked.id)
            session.commit()
            return ids

    return write


def released(engine) -> int:
    """How many rows carry a release id."""
    with engine.connect() as connection:
        return connection.execute(
            text("SELECT count(*) FROM questions WHERE release_id IS NOT NULL")
        ).scalar_one()


def test_only_accepted_questions_are_offered(written) -> None:
    """A rejected question is drop-rate evidence, not benchmark material."""
    written(
        [
            ("easy", "factoid", True, "accepted"),
            ("easy", "factoid", True, "rejected"),
            ("easy", "factoid", True, "draft"),
        ]
    )

    assert len(QuestionCatalog().releasable()) == 1


def test_a_question_with_no_band_or_kind_is_not_offered(written, engine) -> None:
    """It would be chosen into a quota it cannot be counted against.

    The report would then disagree with the set it describes.
    """
    written([("easy", "factoid", True, "accepted")])
    with engine.begin() as connection:
        connection.execute(text("UPDATE questions SET difficulty = NULL"))

    assert QuestionCatalog().releasable() == []


def test_the_draw_writes_one_id_on_what_it_chose(written, engine) -> None:
    """A release is a column on the rows already there."""
    ids = written([("easy", "factoid", True, "accepted")] * 3)
    catalog = QuestionCatalog()

    drawn, count = catalog.release(ids[:2])

    assert count == 2
    assert released(engine) == 2
    with engine.connect() as connection:
        held = (
            connection.execute(
                text(
                    "SELECT DISTINCT release_id FROM questions "
                    "WHERE release_id IS NOT NULL"
                )
            )
            .scalars()
            .all()
        )
    assert held == [drawn]


def test_a_second_draw_replaces_the_first(written, engine) -> None:
    """One release at a time, because the column holds one id.

    The question a report answers is what would ship today, and a previous
    draw is not history worth keeping here: every question it held is still
    in the table.
    """
    ids = written([("easy", "factoid", True, "accepted")] * 4)
    catalog = QuestionCatalog()

    first, _ = catalog.release(ids[:3])
    second, count = catalog.release(ids[3:])

    assert first != second
    assert count == 1
    assert released(engine) == 1


def test_drawing_nothing_clears_the_last_release(written, engine) -> None:
    """Rather than leaving a stale set that no settings produced."""
    ids = written([("easy", "factoid", True, "accepted")] * 2)
    catalog = QuestionCatalog()
    catalog.release(ids)

    catalog.release([])

    assert released(engine) == 0


def test_the_balance_command_holds_every_share(written, engine) -> None:
    """The whole round trip, over a pool that can fill the quota exactly."""
    from question_generation.service import balance

    rows = []
    for band in ("easy", "medium", "hard"):
        for kind in ("factoid", "reason"):
            rows += [(band, kind, True, "accepted")] * 6
    rows += [("easy", "factoid", False, "accepted")] * 6
    written(rows)

    count = balance(QuestionCatalog(), _settings())

    with engine.connect() as connection:
        got = dict(
            connection.execute(
                text(
                    "SELECT difficulty, count(*) FROM questions "
                    "WHERE release_id IS NOT NULL GROUP BY 1"
                )
            ).all()
        )
        unanswerable = connection.execute(
            text(
                "SELECT count(*) FROM questions "
                "WHERE release_id IS NOT NULL AND NOT answerable"
            )
        ).scalar_one()

    assert count == released(engine)
    # Within one of each other: a size that does not divide by three has to
    # give one band the remainder, and largest-remainder is what decides
    # which.
    assert max(got.values()) - min(got.values()) <= 1
    assert set(got) == set(BANDS)
    assert unanswerable <= count * 0.1 + 1


def _settings(**overrides):
    """Release settings aiming at a third per band and a tenth unanswerable."""
    from question_generation.config import Settings

    return Settings(
        **{
            "per_topic": 4,
            "sample_size": 4,
            "samples_per_passage": 1,
            "fact_kinds": ("atomic",),
            "type_mix": TYPES,
            "difficulty_mix": {"easy": 1},
            "followup_types": ("condition",),
            "explanation_chars": (150, 900),
            "boilerplate_cosine": 0.0,
            "unanswerable_share": 0.1,
            "followup_share": 0.5,
            "max_followups": 2,
            "retries": 0,
            "answer_chars": {
                "value": (1, 80),
                "list": (3, 300),
                "explanation": (20, 600),
            },
            "answer_coverage": 0.0,
            "party_density": 0.0,
            "meets_floor": 0.0,
            "answer_overlap": 0.6,
            "off_topic_overlap": 0.3,
            "elsewhere_passages": 0,
            "long_answer_chars": 60,
            "duplicate_cosine": 0.93,
            "duplicate_floor": 0.75,
            "release_size": 0,
            "release_unanswerable": 0.1,
            "release_difficulty": BANDS,
            "embedding_model": "stub",
            "model": None,
            "max_tokens": 512,
            "verifier_model": "ollama/verifier",
            **overrides,
        }
    )
