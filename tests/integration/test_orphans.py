"""What survives deleting every document.

A corpus is deleted one document at a time, and everything derived from a
document is meant to go with it. `topics` is the exception nothing announces:
it has no foreign key to `documents` because a fit is over the corpus rather
than over a file, so deleting every document leaves the topics behind - with
their labels, their coverage flags and their question-generation queue state,
all describing passages that no longer exist.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

pytestmark = pytest.mark.integration


@pytest.fixture
def corpus(engine, database):
    """A document with a passage, a fact, a question and a topic over it."""
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO documents (sha256, mime_type) "
                "VALUES (repeat('a', 64), 'application/pdf')"
            )
        )
        passage = connection.execute(
            text(
                "INSERT INTO passages (doc_sha256, ordinal, text, language, "
                "doc_item_refs, sentences, bbox, table_cells) "
                "VALUES (repeat('a', 64), 1, 'A passage.', 'en', '{}', "
                "'[]'::jsonb, '[]'::jsonb, '[]'::jsonb) RETURNING id"
            )
        ).scalar_one()
        topic = connection.execute(
            text(
                "INSERT INTO topics (topic_index, language, label, top_terms, "
                "include_in_coverage, question_status) "
                "VALUES (0, 'en', 'Support', '{support}', true, 'generated') "
                "RETURNING id"
            )
        ).scalar_one()
        connection.execute(
            text(
                "INSERT INTO passage_topics (passage_id, topic_id, weight) "
                "VALUES (:passage, :topic, 0.9)"
            ),
            {"passage": passage, "topic": topic},
        )
        fact = connection.execute(
            text(
                "INSERT INTO facts (passage_id, statement, evidence_text, "
                "evidence_sentence_ids, evidence_start, evidence_end, "
                "extraction_method, validated, units_statement, units_added, "
                "unresolved_references) VALUES (:passage, 'A claim.', "
                "'A passage.', '{0}', 0, 10, 'llm', true, '{}', '{}', '{}') "
                "RETURNING id"
            ),
            {"passage": passage},
        ).scalar_one()
        question = connection.execute(
            text(
                "INSERT INTO questions (question_text, target_answer, "
                "answerable, language, status) VALUES ('How?', '4 kg', true, "
                "'en', 'accepted') RETURNING id"
            )
        ).scalar_one()
        connection.execute(
            text(
                "INSERT INTO question_facts (question_id, fact_id) "
                "VALUES (:question, :fact)"
            ),
            {"question": question, "fact": fact},
        )
    return {"passage": passage, "topic": topic, "fact": fact, "question": question}


def counts(engine) -> dict[str, int]:
    """How many rows each table holds."""
    tables = (
        "documents",
        "passages",
        "passage_topics",
        "facts",
        "questions",
        "question_facts",
        "topics",
    )
    with engine.connect() as connection:
        return {
            table: connection.execute(
                text(f"SELECT count(*) FROM {table}")
            ).scalar_one()
            for table in tables
        }


def test_deleting_a_document_takes_everything_derived_from_it(corpus, engine) -> None:
    """The passage, its memberships, its facts and the questions on them."""
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM documents WHERE sha256 = repeat('a', 64)"))

    held = counts(engine)

    assert held["passages"] == 0
    assert held["passage_topics"] == 0, "a membership outlived its passage"
    assert held["facts"] == 0
    assert held["question_facts"] == 0
    assert held["questions"] == 0, "a question outlived every fact it rested on"


def test_deleting_every_document_leaves_the_topics_behind(corpus, engine) -> None:
    """The orphan a wipe has to know about.

    A topic has no foreign key to a document because a fit is over the corpus,
    so an empty corpus still has its topics: their labels, their coverage
    flags and their `question_status`. A fit writes them again from scratch,
    which is why this has never broken anything - but `make topics-delete` is
    a second command, and somebody deleting every document has not finished.
    """
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM documents WHERE sha256 = repeat('a', 64)"))

    held = counts(engine)

    assert held["documents"] == 0
    assert held["topics"] == 1, "this is the orphan; if it is 0 the note is stale"


def test_wiping_the_corpus_leaves_no_document_and_no_upload_record(
    corpus, engine, buckets
) -> None:
    """What `make wipe` runs first, and what it has to leave behind.

    The upload history survives a single deletion on purpose - it records
    what was attempted rather than what is held - and that is wrong exactly
    once, when the corpus is being emptied.
    """
    from ingestion.factory import build_removal

    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO ingest_events (sha256, outcome, "
                "submitted_filename, size_bytes) "
                "VALUES (repeat('a', 64), 'stored', 'a.pdf', 1)"
            )
        )

    removed = build_removal().delete_every()

    held = counts(engine)
    assert len(removed) == 1
    assert held["documents"] == 0
    assert held["questions"] == 0
    with engine.connect() as connection:
        assert (
            connection.execute(text("SELECT count(*) FROM ingest_events")).scalar_one()
            == 0
        ), "the corpus is empty and the status panel still reports uploads"


def test_wiping_an_empty_corpus_is_not_an_error(database, engine) -> None:
    """Running it twice is how somebody checks it worked."""
    from ingestion.factory import build_removal

    assert build_removal().delete_every() == []


def test_deleting_the_topics_from_the_command_line_takes_the_figures(
    corpus, engine, buckets
) -> None:
    """The route did this and the command did not.

    `delete_all` reports which languages now have an orphaned figure; the
    route removed each one and the command read the list and dropped it. A
    real wipe left two pyLDAvis pages in the export bucket, each describing
    topics that no longer existed, and nothing said so.
    """
    from blob_store.seaweedfs import ExportBucket
    from topic_modelling.run import main

    bucket = ExportBucket()
    key = bucket.topic_visualisation_key("en")
    bucket.put(
        key,
        b"<html>the en figure</html>",
        content_type=bucket.TOPIC_VISUALISATION_TYPE,
    )

    assert main(["--delete"]) == 0

    assert bucket.find(key) is None, "the figure outlived the topics it drew"
    assert counts(engine)["topics"] == 0


def test_the_api_is_wired_to_the_container_and_not_to_a_developers_database(
    client, engine
) -> None:
    """The isolation failure that made three topic tests flake.

    Every repository captures `sessions()` as it is constructed, and
    `api.dependencies` constructs them all at import. Anything that ran
    earlier and populated that cache - a unit test exercising a stage's
    command line does, against the ambient DATABASE_URL - bound every route
    to the developer's own database for the rest of the process. The
    integration tests then set up a container nothing touched, and read and
    wrote where nobody meant to.

    What the fixture now does is clear the cache before that import. What
    this asserts is the consequence: a row written to the container is a row
    the API can see.
    """
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO documents (sha256, mime_type) "
                "VALUES (repeat('b', 64), 'application/pdf')"
            )
        )

    answered = client.get("/documents")

    assert answered.status_code == 200
    assert [one["sha256"] for one in answered.json()["documents"]] == ["b" * 64], (
        "the API answered from a different database than the fixture wrote to"
    )
