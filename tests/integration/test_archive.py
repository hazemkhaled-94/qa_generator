"""What a deletion leaves behind, and the second deletion that takes it.

The trigger is the whole mechanism, so the tests here delete through the
paths a person actually uses - the removal service, a cascade, a raw
statement - and read `archived_rows` afterwards. A test that inserted into
it directly would pass against no trigger at all.
"""

from __future__ import annotations

import pytest
from seed import digest, document, fact, link, membership, passage, question
from sqlalchemy import text
from sqlalchemy.orm import Session

from archive.store import Archive
from database.qa_generator import Base
from ingestion.factory import build_removal

pytestmark = pytest.mark.integration

PDF = b"%PDF-1.7\nnot really a document\n"


@pytest.fixture
def archive(database, buckets) -> Archive:
    """The archive, over the container's database and object store."""
    return Archive()


@pytest.fixture
def held(engine, buckets) -> str:
    """A document with a file, a converted form and everything under it."""
    from blob_store.seaweedfs import DocumentsBucket, ParsedBucket

    sha = digest()
    source, parsed = DocumentsBucket(), ParsedBucket()
    source.put(
        source.key_for(sha, "application/pdf"), PDF, content_type="application/pdf"
    )
    parsed.put(parsed.key_for(sha), b"{}", content_type="application/json")

    with Session(engine) as session:
        session.add(document(sha, chunk_status="chunked"))
        session.flush()
        first = passage(sha, ordinal=1)
        session.add(first)
        session.flush()
        drawn = fact(first.id)
        session.add(drawn)
        session.flush()
        asked = question()
        session.add(asked)
        session.flush()
        session.add(link(asked.id, drawn.id))
        session.commit()
    return sha


def archived(engine) -> dict[str, int]:
    """How many rows of each table the archive holds."""
    with engine.connect() as connection:
        return dict(
            connection.execute(
                text(
                    "SELECT table_name, count(*) FROM archived_rows GROUP BY table_name"
                )
            ).all()
        )


def test_deleting_a_document_archives_everything_it_cascaded_to(
    archive, held, engine
) -> None:
    """One command, six tables, and a copy of each row it took."""
    build_removal().delete(held)

    assert archived(engine) == {
        "documents": 1,
        "passages": 1,
        "facts": 1,
        "fact_passages": 1,
        "questions": 1,
        "question_facts": 1,
    }


def test_the_archived_row_is_the_whole_row(archive, held, engine) -> None:
    """Every column by name, so a person can read what was deleted."""
    build_removal().delete(held)

    with engine.connect() as connection:
        payload = connection.execute(
            text("SELECT payload FROM archived_rows WHERE table_name = 'questions'")
        ).scalar_one()

    assert payload["question_text"] == "What does the device weigh?"
    assert payload["language"] == "en"
    assert payload["id"] > 0, "the key is kept, so the row can be put back by hand"
    assert "embedding" not in payload, "a vector is ten times the rest of the row"


def test_the_objects_move_to_the_archive_bucket(archive, held) -> None:
    """The uploaded file is readable again from where the deletion put it."""
    from blob_store.seaweedfs import ArchiveBucket, DocumentsBucket

    build_removal().delete(held)

    bucket, source = ArchiveBucket(), DocumentsBucket()
    uploaded = source.key_for(held, "application/pdf")

    assert bucket.find(bucket.key_for(source.name, uploaded)) == PDF
    assert source.find(uploaded) is None, "moved, not copied"


def test_a_statement_nothing_in_python_ran_is_archived_too(
    archive, held, engine
) -> None:
    """The reason this is a trigger: psql and Adminer delete rows as well."""
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM questions"))

    assert archived(engine) == {"questions": 1, "question_facts": 1}


def test_every_table_archives_its_deletions(archive, engine) -> None:
    """A table added without a trigger is a table that deletes for good."""
    with engine.connect() as connection:
        attached = {
            row[0]
            for row in connection.execute(
                text(
                    "SELECT c.relname FROM pg_trigger t JOIN pg_class c "
                    "ON c.oid = t.tgrelid WHERE NOT t.tgisinternal "
                    "AND t.tgname LIKE '%_archive_deleted_row'"
                )
            ).all()
        }

    assert attached == set(Base.metadata.tables) - {"archived_rows"}


def test_the_archive_is_not_archived(archive, held, engine) -> None:
    """Otherwise purging it would write one row for every row it took."""
    build_removal().delete(held)

    archive.purge(table="questions")

    assert "archived_rows" not in archived(engine)


def test_purging_one_table_leaves_the_others(archive, held, engine) -> None:
    """A table is named to purge the questions and keep the document."""
    build_removal().delete(held)

    purged = archive.purge(table="questions")

    assert purged.rows == 1
    assert purged.objects == 0, "an object belongs to no table"
    assert "questions" not in archived(engine)
    assert archived(engine)["documents"] == 1


def test_purging_by_age_keeps_what_is_newer(archive, held) -> None:
    """The row is seconds old, so a day's cutoff must not reach it."""
    build_removal().delete(held)

    assert archive.purge(older_than_days=1).rows == 0
    assert archive.purge(older_than_days=0).rows == 6, (
        "measured by the clock that wrote the row, not the one asking"
    )


def test_purging_everything_takes_the_objects_with_it(archive, held) -> None:
    """Naming no table means the whole archive, bucket included."""
    build_removal().delete(held)

    purged = archive.purge()

    assert purged.rows == 6
    assert purged.objects == 2, "the uploaded file and the converted form"
    assert archive.objects() == (0, 0)


def test_the_status_report_says_what_is_held(archive, held, engine) -> None:
    """What `make archive` prints, heaviest table first."""
    build_removal().delete(held)
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM documents"))

    held_rows = archive.rows()

    assert {one.table_name for one in held_rows} >= {"documents", "questions"}
    assert all(one.bytes > 0 for one in held_rows)
    assert all(one.oldest <= one.newest for one in held_rows)
    assert archive.objects() == (2, len(PDF) + len(b"{}"))


def test_a_topic_membership_is_archived_with_its_passage(archive, held, engine) -> None:
    """`topics` has no foreign key to a document; the membership does."""
    from seed import fitted

    with Session(engine) as session:
        subject = fitted()
        session.add(subject)
        session.flush()
        first = session.execute(text("SELECT id FROM passages LIMIT 1")).scalar_one()
        session.add(membership(first, subject.id))
        session.commit()

    build_removal().delete(held)

    assert archived(engine)["passage_topics"] == 1
    assert "topics" not in archived(engine), "a fit is over the corpus, not a file"
