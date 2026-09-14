"""The queue every stage claims from, against the database that runs it.

The unit tests prove the SQL a narrowing compiles to. These prove what it
does to rows: which ones move, which ones are left alone, and that two
workers never take the same one.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from datetime import timedelta

import pytest
from seed import digest, document, passage
from sqlalchemy import text
from sqlalchemy.orm import Session

from database.qa_generator import Status
from extraction.repository import PassageQueue
from preprocessing.parsing.repository import ParseQueue
from stages.queue import ABANDONED

pytestmark = pytest.mark.integration


@pytest.fixture
def documents(engine, database):
    """Writes documents in named states, and hands back a reader."""

    def write(**by_status: int) -> None:
        with Session(engine) as session:
            letters = iter("abcdefghijklmnopqrstuvwxyz")
            for status, count in by_status.items():
                for _ in range(count):
                    session.add(document(digest(next(letters)), parse_status=status))
            session.commit()

    return write


def statuses(engine) -> dict[str, int]:
    """Counts documents by parse status."""
    with engine.connect() as connection:
        return dict(
            connection.execute(
                text("SELECT parse_status, count(*) FROM documents GROUP BY 1")
            ).all()
        )


def test_a_claim_takes_one_pending_row_and_marks_it_in_progress(
    documents, engine
) -> None:
    """One row, claimed and timestamped."""
    documents(pending=2)

    claimed = ParseQueue().claim()

    assert claimed is not None
    assert statuses(engine) == {Status.IN_PROGRESS: 1, Status.PENDING: 1}


def test_two_claims_never_take_the_same_row(documents) -> None:
    """Every claim narrows to a row nothing else holds."""
    documents(pending=3)
    queue = ParseQueue()

    taken = [queue.claim(), queue.claim(), queue.claim()]

    assert all(one is not None for one in taken)
    assert len({one.sha256 for one in taken}) == 3


def test_claiming_an_empty_queue_answers_nothing(documents) -> None:
    """A drain stops when the queue does."""
    documents(new=1, parsed=1, failed=1)

    assert ParseQueue().claim() is None


def test_concurrent_claims_never_collide(documents) -> None:
    """FOR UPDATE SKIP LOCKED, exercised by two workers at once."""
    documents(pending=6)
    queue = ParseQueue()

    with ThreadPoolExecutor(max_workers=6) as pool:
        taken = [one for one in pool.map(lambda _: queue.claim(), range(6)) if one]

    assert len(taken) == 6
    assert len({one.sha256 for one in taken}) == 6, "a row was claimed twice"


def test_only_pending_rows_are_claimable(documents, engine) -> None:
    """A row arrives `new` and no worker looks at it until asked."""
    documents(new=3)

    assert ParseQueue().claim() is None
    assert statuses(engine) == {Status.NEW: 3}


def test_start_queues_what_was_never_asked_for(documents, engine) -> None:
    """`new` becomes `pending`, and nothing else moves."""
    documents(new=2, parsed=1, failed=1)

    assert ParseQueue().start() == 2
    assert statuses(engine) == {
        Status.PENDING: 2,
        Status.PARSED: 1,
        Status.FAILED: 1,
    }


def test_stop_takes_back_only_what_has_not_begun(documents, engine) -> None:
    """A row a worker holds finishes; a queued one is withdrawn."""
    documents(pending=2, in_progress=1)

    assert ParseQueue().stop() == 2
    assert statuses(engine) == {Status.NEW: 2, Status.IN_PROGRESS: 1}


def test_retry_returns_failures_and_clears_the_error(documents, engine) -> None:
    """The error goes with the status, so a stale one cannot be read."""
    with Session(engine) as session:
        session.add(document(digest(), parse_status=Status.FAILED, parse_error="boom"))
        session.commit()

    assert ParseQueue().retry() == 1
    with engine.connect() as connection:
        row = connection.execute(
            text("SELECT parse_status, parse_error FROM documents")
        ).one()
    assert row.parse_status == Status.PENDING
    assert row.parse_error is None


def test_reset_queues_finished_rows_too_but_not_a_held_one(documents, engine) -> None:
    """Skips whatever a worker holds right now."""
    documents(parsed=2, failed=1, in_progress=1)

    assert ParseQueue().reset() == 3
    assert statuses(engine) == {Status.PENDING: 3, Status.IN_PROGRESS: 1}


def test_a_claim_that_outlived_its_lease_is_failed(engine, database) -> None:
    """A worker killed mid-item leaves its row claimed."""
    with Session(engine) as session:
        session.add(document(digest(), parse_status=Status.IN_PROGRESS))
        session.commit()
    with engine.begin() as connection:
        connection.execute(
            text("UPDATE documents SET parse_claimed_at = now() - interval '3 hours'")
        )

    assert ParseQueue().abandon() == 1
    with engine.connect() as connection:
        row = connection.execute(
            text("SELECT parse_status, parse_error, parse_claimed_at FROM documents")
        ).one()
    assert row.parse_status == Status.FAILED
    assert row.parse_error == ABANDONED
    assert row.parse_claimed_at is None


def test_a_live_claim_is_left_alone(engine, database) -> None:
    """The lease is what tells a live worker from a dead one."""
    with Session(engine) as session:
        session.add(document(digest(), parse_status=Status.IN_PROGRESS))
        session.commit()
    with engine.begin() as connection:
        connection.execute(text("UPDATE documents SET parse_claimed_at = now()"))

    assert ParseQueue().abandon() == 0


def test_a_claim_with_no_timestamp_counts_as_abandoned(engine, database) -> None:
    """Such a row matches no other operation and would sit there forever."""
    with Session(engine) as session:
        session.add(document(digest(), parse_status=Status.IN_PROGRESS))
        session.commit()

    assert ParseQueue().abandon() == 1


def test_a_shorter_lease_sweeps_sooner(engine, database) -> None:
    """A worker that knows what one row costs may say so."""
    with Session(engine) as session:
        session.add(document(digest(), parse_status=Status.IN_PROGRESS))
        session.commit()
    with engine.begin() as connection:
        connection.execute(
            text("UPDATE documents SET parse_claimed_at = now() - interval '5 minutes'")
        )

    assert ParseQueue().abandon() == 0
    assert ParseQueue(lease=timedelta(minutes=1)).abandon() == 1


def test_the_queue_reports_its_depth_and_whether_anyone_is_on_it(
    documents, engine
) -> None:
    """One query for both: a page polls this every few seconds."""
    documents(pending=2, parsed=1)
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE documents SET parse_status = 'in_progress', "
                "parse_claimed_at = now() WHERE parse_status = 'parsed'"
            )
        )

    state = ParseQueue().queue_state()

    assert state.working is True
    assert state.rows == {Status.PENDING: 2, Status.IN_PROGRESS: 1}


def test_a_stale_claim_does_not_count_as_a_worker(engine, database) -> None:
    """`working` means a worker is on it now, not that a row says so."""
    with Session(engine) as session:
        session.add(document(digest(), parse_status=Status.IN_PROGRESS))
        session.commit()
    with engine.begin() as connection:
        connection.execute(
            text("UPDATE documents SET parse_claimed_at = now() - interval '9 hours'")
        )

    assert ParseQueue().queue_state().working is False


def test_a_narrowed_stop_leaves_the_rest_of_the_queue_alone(documents, engine) -> None:
    """The failure this catches answers with a count that looks like success."""
    documents(pending=3)
    queue = ParseQueue()

    moved = queue.stop(queue.narrowed("document", digest("a")))

    assert moved == 1
    assert statuses(engine) == {Status.NEW: 1, Status.PENDING: 2}


def test_a_narrowed_start_reaches_only_its_document(engine, database) -> None:
    """One document's passages, not the corpus's."""
    with Session(engine) as session:
        session.add_all([document(digest("a")), document(digest("b"))])
        session.add_all(
            [
                passage(digest("a"), ordinal=1),
                passage(digest("a"), ordinal=2),
                passage(digest("b"), ordinal=1),
            ]
        )
        session.commit()
    queue = PassageQueue()

    moved = queue.start(queue.narrowed("document", digest("a")))

    assert moved == 2
    with engine.connect() as connection:
        pending = connection.scalars(
            text("SELECT doc_sha256 FROM passages WHERE extract_status = 'pending'")
        ).all()
    assert set(pending) == {digest("a")}


def test_a_narrowing_to_one_passage_moves_one_row(engine, database) -> None:
    """Extraction queues over passages and answers for one of them."""
    with Session(engine) as session:
        session.add(document(digest("a")))
        session.add_all(
            [passage(digest("a"), ordinal=1), passage(digest("a"), ordinal=2)]
        )
        session.commit()
        first = session.execute(text("SELECT min(id) FROM passages")).scalar()
    queue = PassageQueue()

    assert queue.start(queue.narrowed("passage", str(first))) == 1


def test_failing_a_row_records_the_reason_and_releases_the_claim(
    documents, engine
) -> None:
    """A failure a person can read, and a row nothing still holds."""
    documents(pending=1)
    queue = ParseQueue()
    claimed = queue.claim()
    assert claimed is not None

    queue.fail(claimed.sha256, "the converter gave up")

    with engine.connect() as connection:
        row = connection.execute(
            text("SELECT parse_status, parse_error, parse_claimed_at FROM documents")
        ).one()
    assert row.parse_status == Status.FAILED
    assert row.parse_error == "the converter gave up"
    assert row.parse_claimed_at is None


def test_a_very_long_failure_is_truncated_rather_than_refused(
    documents, engine
) -> None:
    """A traceback must not be what stops a row being marked failed."""
    documents(pending=1)
    queue = ParseQueue()
    claimed = queue.claim()
    assert claimed is not None

    queue.fail(claimed.sha256, "x" * 5000)

    with engine.connect() as connection:
        stored = connection.execute(text("SELECT parse_error FROM documents")).scalar()
    assert stored is not None
    assert len(stored) == 2000


def test_a_row_another_worker_holds_is_skipped_rather_than_waited_for(
    documents, engine
) -> None:
    """SKIP LOCKED, which is the difference between two workers and one.

    Without it the second claim blocks on the first worker's row lock until
    that transaction ends, so a stage scaled to four workers runs at the
    speed of one. Claimed in a thread so a block fails the test rather than
    hanging it.
    """
    documents(pending=2)
    first, second = sorted(digest(letter) for letter in "ab")

    # Not a context manager: a claim that blocks leaves its thread running,
    # and shutting the pool down would wait for exactly what is stuck.
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        with engine.begin() as holding:
            holding.execute(
                text("SELECT sha256 FROM documents WHERE sha256 = :sha FOR UPDATE"),
                {"sha": first},
            )
            claiming = pool.submit(ParseQueue().claim)
            try:
                claimed = claiming.result(timeout=10)
            except FuturesTimeout:
                pytest.fail("the claim blocked on a row another worker holds")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    assert claimed is not None, "the second row was never claimed"
    assert claimed.sha256 == second, "the locked row should have been skipped"
