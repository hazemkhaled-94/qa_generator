"""What the database refuses to hold.

Every CHECK here is a claim the pipeline relies on being true of every row,
including rows written by a hand-run UPDATE. They are tested against the
database rather than against the model because that is where they run.
"""

from __future__ import annotations

import pytest
from seed import digest, document, event, fact, passage, question, topic
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session

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


def modelled(session, language: str = "de", index: int = 0):
    """A fitted topic, which needs its index, its language and its terms."""
    return stored(
        session,
        topic(
            language=language,
            topic_index=index,
            status="modelled",
            top_terms=["lieferung", "versand"],
        ),
    )


def stored(session, *rows):
    """Writes rows the database accepts, and hands back the last."""
    session.add_all(rows)
    session.flush()
    return rows[-1]


@pytest.mark.parametrize("language", ["DE", "1a", "", "e "])
def test_a_two_character_language_that_is_not_a_code_is_refused(
    session, language
) -> None:
    """Two lower-case letters, because every reader splits on that."""
    refuses(session, document(language=language), "documents_language_is_iso_639_1")


@pytest.mark.parametrize("language", ["eng", "de-DE"])
def test_a_language_longer_than_a_code_is_refused(session, language) -> None:
    """CHAR(2) refuses it before the CHECK is reached."""
    session.add(document(language=language))
    with pytest.raises(DBAPIError):
        session.flush()
    session.rollback()


@pytest.mark.parametrize("language", ["de", "en"])
def test_a_two_letter_language_is_accepted(session, language) -> None:
    """The form the pipeline writes."""
    assert stored(session, document(digest(language[0]), language=language))


@pytest.mark.parametrize("status", ["chunked", "extracted", "nonsense", ""])
def test_a_parse_status_no_stage_owns_is_refused(session, status) -> None:
    """A stage's column holds the shared states and its own done value."""
    refuses(session, document(parse_status=status), "documents_parse_status_valid")


@pytest.mark.parametrize(
    "status", ["new", "pending", "in_progress", "failed", "parsed"]
)
def test_every_status_parsing_owns_is_accepted(session, status) -> None:
    """Four shared, plus `parsed`."""
    assert stored(session, document(digest(status[0]), parse_status=status))


@pytest.mark.parametrize("confidence", [-0.1, 1.1, 2.0])
def test_a_parse_confidence_outside_zero_to_one_is_refused(session, confidence) -> None:
    """It is a probability."""
    refuses(
        session,
        document(parse_confidence=confidence),
        "documents_parse_confidence_is_a_probability",
    )


@pytest.mark.parametrize("ordinal", [0, -1])
def test_a_passage_ordinal_below_one_is_refused(session, ordinal) -> None:
    """Ordinals run from one, and a reader counts on the first being 1."""
    stored(session, document())
    refuses(session, passage(digest(), ordinal=ordinal), "passages_ordinal_positive")


def test_two_passages_cannot_share_an_ordinal_in_one_document(session) -> None:
    """The pair is the passage's address within its document."""
    stored(session, document(), passage(digest(), ordinal=1))
    session.add(passage(digest(), ordinal=1))
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()


def test_the_same_ordinal_in_another_document_is_fine(session) -> None:
    """Every document numbers its own passages from one."""
    stored(
        session,
        document(digest("a")),
        document(digest("b")),
        passage(digest("a"), ordinal=1),
        passage(digest("b"), ordinal=1),
    )


def test_a_fact_whose_evidence_ends_before_it_starts_is_refused(session) -> None:
    """The span is a slice of the passage text."""
    stored(session, document())
    row = stored(session, passage(digest()))
    refuses(
        session,
        fact(row.id, evidence_start=50, evidence_end=10),
        "facts_evidence_span_ordered",
    )


def test_an_empty_evidence_span_is_accepted(session) -> None:
    """A citation naming no sentence of the passage carries no span."""
    stored(session, document())
    row = stored(session, passage(digest()))
    assert stored(session, fact(row.id, evidence_start=0, evidence_end=0))


@pytest.mark.parametrize("method", ["guessed", "", "LLM"])
def test_an_extraction_method_nothing_produces_is_refused(session, method) -> None:
    """The method is a column a report groups on."""
    stored(session, document())
    row = stored(session, passage(digest()))
    refuses(
        session, fact(row.id, extraction_method=method), "facts_extraction_method_valid"
    )


def test_a_validated_fact_carrying_a_rejection_is_refused(session) -> None:
    """The verdict and the code are one fact stated twice."""
    from database.qa_generator import Rejection

    stored(session, document())
    row = stored(session, passage(digest()))
    refuses(
        session,
        fact(row.id, validated=True, rejection_code=Rejection.COPIED),
        "facts_verdict_agrees",
    )


def test_an_invalid_fact_carrying_no_rejection_is_refused(session) -> None:
    """A refusal a report cannot group on is a refusal nobody can read."""
    stored(session, document())
    row = stored(session, passage(digest()))
    refuses(
        session,
        fact(row.id, validated=False, rejection_code=None),
        "facts_verdict_agrees",
    )


def test_a_rejection_code_nothing_raises_is_refused(session) -> None:
    """Every code is one of the checks."""
    stored(session, document())
    row = stored(session, passage(digest()))
    refuses(
        session,
        fact(row.id, validated=False, rejection_code="made_up"),
        "facts_rejection_code_valid",
    )


@pytest.mark.parametrize("weight", [0, -0.5, 1.0001, 2])
def test_a_membership_weight_outside_its_interval_is_refused(session, weight) -> None:
    """Gensim returns float32, whose last member can round above one."""
    from database.qa_generator import PassageTopic

    stored(session, document())
    row = stored(session, passage(digest()))
    fitted = stored(session, modelled(session))
    refuses(
        session,
        PassageTopic(passage_id=row.id, topic_id=fitted.id, weight=weight),
        "passage_topics_weight_is_a_probability",
    )


@pytest.mark.parametrize("weight", [0.0001, 0.5, 1.0])
def test_a_weight_inside_its_interval_is_accepted(session, weight) -> None:
    """Exactly one is a passage belonging wholly to one topic."""
    from database.qa_generator import PassageTopic

    stored(session, document())
    row = stored(session, passage(digest()))
    fitted = stored(session, modelled(session))
    assert stored(
        session, PassageTopic(passage_id=row.id, topic_id=fitted.id, weight=weight)
    )


def test_two_topics_cannot_share_an_index_in_one_language(session) -> None:
    """The pair is what a membership points at."""
    modelled(session)
    session.add(topic(language="de", topic_index=0, status="modelled", top_terms=["a"]))
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()


def test_the_same_index_in_another_language_is_fine(session) -> None:
    """One model is fitted per language."""
    stored(
        session,
        topic(language="de", topic_index=0, status="modelled", top_terms=["a"]),
        topic(language="en", topic_index=0, status="modelled", top_terms=["a"]),
    )


def test_a_modelled_topic_with_no_index_is_refused(session) -> None:
    """`topics` is both the queue and the result; the index separates them."""
    refuses(
        session,
        topic(language="de", topic_index=None, status="modelled", top_terms=["a"]),
        "topics_modelled_is_a_topic",
    )


def test_a_modelled_topic_with_no_terms_is_refused(session) -> None:
    """A topic nobody can read is not a topic."""
    refuses(
        session,
        topic(language="de", topic_index=1, status="modelled", top_terms=[]),
        "topics_modelled_is_a_topic",
    )


def test_a_request_row_needs_neither_index_nor_language(session) -> None:
    """Asking for a fit is what creates the row."""
    assert stored(session, topic(status="pending"))


def test_an_outcome_nothing_produces_is_refused(session) -> None:
    """Every attempt is recorded under one of the four outcomes."""
    refuses(session, event(outcome="exploded"), "ingest_events_outcome_valid")


@pytest.mark.parametrize(
    "outcome", ["stored", "duplicate_bytes", "too_large", "unsupported_type"]
)
def test_every_outcome_the_service_produces_is_accepted(session, outcome) -> None:
    """The four an upload can end in."""
    assert stored(session, event(outcome=outcome))


def test_an_unanswerable_question_carrying_an_answer_is_refused(session) -> None:
    """It is scored on behaviour, not on content."""
    refuses(
        session,
        question(answerable=False, target_answer="4 kg"),
        "questions_unanswerable_has_no_target",
    )


def test_an_unanswerable_question_with_no_answer_is_accepted(session) -> None:
    """The unanswerable ones are what the dataset tests a chatbot with."""
    assert stored(session, question(answerable=False, target_answer=None))


def test_a_passage_referencing_no_document_is_refused(session) -> None:
    """A passage belongs to a document that exists."""
    session.add(passage(digest("z")))
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()


def test_deleting_a_document_takes_its_passages_and_facts(session) -> None:
    """One delete, and nothing derived from it survives."""
    stored(session, document())
    row = stored(session, passage(digest()))
    stored(session, fact(row.id))

    session.execute(text("DELETE FROM documents"))
    session.flush()

    assert session.execute(text("SELECT count(*) FROM passages")).scalar() == 0
    assert session.execute(text("SELECT count(*) FROM facts")).scalar() == 0
