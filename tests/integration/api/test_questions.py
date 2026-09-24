"""The /questions surface, against the database that answers it.

The contract layer proves the shape of this surface; this proves what it
does. The two halves matter separately: a route can be published, typed and
documented and still count a question twice, filter on the wrong column, or
lose a person's verdict.

The route also carries a stage's queue under the same word, which no other
stage does, so the verbs are exercised here beside the reads.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest
from seed import digest, document, fact, fitted, link, membership, passage, question
from sqlalchemy.orm import Session

pytestmark = pytest.mark.integration


@pytest.fixture
def written(client, engine):
    """A corpus with questions in every state a run can leave one in.

    Two documents, two topics and four questions: one ordinary, one spanning
    both documents, one rejected by a gate and one written to have no answer.
    """
    with Session(engine) as session:
        session.add_all(
            [
                document(digest("a"), language="en", chunk_status="chunked"),
                document(digest("b"), language="en", chunk_status="chunked"),
            ]
        )
        subject = fitted(0, label="Support")
        aside = fitted(1, label="Boilerplate", include_in_coverage=False)
        session.add_all([subject, aside])
        session.flush()

        first = passage(
            digest("a"), ordinal=1, text="Answered in 48 hours.", language="en"
        )
        second = passage(digest("a"), ordinal=2, text="Raised by phone.", language="en")
        third = passage(digest("b"), ordinal=1, text="Reviewed yearly.", language="en")
        session.add_all([first, second, third])
        session.flush()
        session.add_all(membership(at.id, subject.id) for at in (first, second, third))

        facts = [
            fact(first.id, statement="A standard request is answered in 48 hours."),
            fact(second.id, statement="An urgent request may be raised by phone."),
            fact(third.id, statement="The policy is reviewed yearly."),
        ]
        session.add_all(facts)
        session.flush()

        rows = {
            "ordinary": question(
                question_text="Within how many hours is a request answered?",
                target_answer="48 hours",
                status="accepted",
                difficulty="easy",
                passage_scope="single_passage",
                document_scope="single_document",
                topic_scope="single_topic",
                answer_chars=8,
                embedding=[1.0] + [0.0] * 1023,
            ),
            "crossing": question(
                question_text="How is an urgent request raised, and how often reviewed?",
                target_answer="by phone, yearly",
                status="accepted",
                difficulty="hard",
                passage_scope="multi_passage",
                document_scope="cross_document",
                topic_scope="multi_topic",
                answer_chars=16,
            ),
            "refused": question(
                question_text="Within how many hours is a request answered again?",
                target_answer="48 hours",
                status="rejected",
                rejected_reason="duplicate",
                difficulty="easy",
                passage_scope="single_passage",
                document_scope="single_document",
                topic_scope="single_topic",
                answer_chars=8,
            ),
            "unanswered": question(
                question_text="Within how many hours is a request answered on a holiday?",
                target_answer=None,
                answerable=False,
                status="accepted",
                difficulty="easy",
                passage_scope="single_passage",
                document_scope="single_document",
                topic_scope="single_topic",
            ),
        }
        session.add_all(rows.values())
        session.flush()
        session.add_all(
            [
                link(rows["ordinary"].id, facts[0].id),
                # Two citations, in two documents: the row the listing must
                # not return twice.
                link(rows["crossing"].id, facts[1].id),
                link(rows["crossing"].id, facts[2].id),
                link(rows["refused"].id, facts[0].id),
                link(rows["unanswered"].id, facts[1].id),
            ]
        )
        session.commit()
        return client, {
            "ids": {name: row.id for name, row in rows.items()},
            "facts": [one.id for one in facts],
            "topic": subject.id,
            "aside": aside.id,
        }


# ── Listing ────────────────────────────────────────────────────────────────


def test_the_listing_pages_and_totals(written) -> None:
    """What the Questions page draws."""
    client, _ = written

    listed = client.get("/questions").json()

    assert listed["total"] == 4
    assert len(listed["questions"]) == 4


def test_a_question_citing_two_facts_is_returned_once(written) -> None:
    """The listing joins through question_facts to reach its documents.

    Without a grouping that collapses them, a cross-document question is one
    row per citation: the page shows it twice and the total says five.
    """
    client, held = written

    listed = client.get("/questions").json()
    crossing = [
        one for one in listed["questions"] if one["id"] == held["ids"]["crossing"]
    ]

    assert len(crossing) == 1
    assert crossing[0]["facts"] == 2
    assert sorted(crossing[0]["documents"]) == sorted([digest("a"), digest("b")])


def test_rejected_questions_are_listed_too(written) -> None:
    """They are the drop-rate evidence, so the page has to be able to show them."""
    client, _ = written

    refused = client.get("/questions", params={"status": "rejected"}).json()

    assert refused["total"] == 1
    assert refused["questions"][0]["rejected_reason"] == "duplicate"


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"document": digest("b")}, 1),
        ({"status": "accepted"}, 3),
        ({"answerable": True}, 3),
        ({"answerable": False}, 1),
        ({"q": "holiday"}, 1),
        ({"q": "48 hours", "field": "answer"}, 2),
        ({"q": "48 hours", "field": "question"}, 0),
        ({"q": "nothing here matches"}, 0),
    ],
)
def test_each_filter_narrows_the_listing(written, params, expected) -> None:
    """Every filter the page offers, one at a time."""
    client, _ = written

    assert client.get("/questions", params=params).json()["total"] == expected


def test_filtering_by_topic_reaches_through_the_passages(written) -> None:
    """A question has no topic of its own; it joins to one through its facts."""
    client, held = written

    covered = client.get("/questions", params={"topic": held["topic"]}).json()
    aside = client.get("/questions", params={"topic": held["aside"]}).json()

    assert covered["total"] == 4
    assert aside["total"] == 0


def test_two_filters_narrow_together_rather_than_either_way(written) -> None:
    """An OR here would quietly widen every figure on the page."""
    client, _ = written

    both = client.get(
        "/questions", params={"document": digest("a"), "answerable": False}
    ).json()

    assert both["total"] == 1


def test_a_search_is_characters_and_not_a_pattern(written) -> None:
    """What a person types is a substring, wildcards included."""
    client, _ = written

    assert client.get("/questions", params={"q": "%"}).json()["total"] == 0
    assert client.get("/questions", params={"q": "_"}).json()["total"] == 0


def test_paging_returns_each_question_once_across_the_pages(written) -> None:
    """The total is the filtered total, not the size of the page."""
    client, _ = written

    first = client.get("/questions", params={"limit": 3, "offset": 0}).json()
    second = client.get("/questions", params={"limit": 3, "offset": 3}).json()

    assert first["total"] == second["total"] == 4
    assert len(first["questions"]) == 3
    assert len(second["questions"]) == 1
    seen = [one["id"] for one in first["questions"] + second["questions"]]
    assert len(seen) == len(set(seen)) == 4


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 501}, {"offset": -1}])
def test_a_page_size_outside_the_bounds_is_refused(written, params) -> None:
    """The bounds are declared, so they are enforced before a query runs."""
    client, _ = written

    assert client.get("/questions", params=params).status_code == 422


def test_an_unknown_status_is_refused_rather_than_matching_nothing(written) -> None:
    """A typo that answers `no questions` is worse than one that answers 422."""
    client, _ = written

    assert client.get("/questions", params={"status": "acepted"}).status_code == 422


# ── Quality ────────────────────────────────────────────────────────────────


def test_quality_counts_the_same_set_the_listing_pages_through(written) -> None:
    """The figures sit above the table and must describe it."""
    client, _ = written

    for params in ({}, {"document": digest("a")}, {"status": "accepted"}):
        listed = client.get("/questions", params=params).json()
        measured = client.get("/questions/quality", params=params).json()
        assert measured["total"] == listed["total"], params


def test_quality_splits_the_states_and_the_gates(written) -> None:
    """The numbers the page reads as `passed`, `rejected` and `unanswerable`."""
    client, _ = written

    measured = client.get("/questions/quality").json()

    assert measured["accepted"] == 3
    assert measured["unanswerable"] == 1
    assert measured["draft"] == 0
    assert measured["rejected"] == {"duplicate": 1}
    assert measured["difficulty"] == {"easy": 3, "hard": 1}
    assert measured["document_scope"] == {"single_document": 3, "cross_document": 1}
    assert measured["topic_scope"] == {"single_topic": 3, "multi_topic": 1}


def test_the_mean_length_is_not_weighted_by_how_many_facts_a_question_cites(
    written,
) -> None:
    """The join carries a question once per citation.

    A mean taken over that counts the cross-document question twice, so the
    figure drifts with the shape of the evidence rather than the questions.
    """
    client, _ = written

    measured = client.get("/questions/quality").json()

    listed = client.get("/questions").json()["questions"]
    expected = sum(len(one["question_text"]) for one in listed) / len(listed)
    assert measured["mean_question_chars"] == pytest.approx(expected, abs=0.05)


def test_coverage_counts_the_topics_rather_than_the_questions(written) -> None:
    """A topic taken out of coverage is in neither number."""
    client, _ = written

    measured = client.get("/questions/quality").json()

    assert measured["topics_in_coverage"] == 1, "the aside topic is excluded"
    assert measured["topics_covered"] == 0, "nothing has been generated yet"


# ── One question ───────────────────────────────────────────────────────────


def test_one_question_carries_the_facts_it_was_written_from(written) -> None:
    """Which is the only route from a question back to a passage."""
    client, held = written

    found = client.get(f"/questions/{held['ids']['crossing']}").json()

    assert found["question"]["id"] == held["ids"]["crossing"]
    assert len(found["sources"]) == 2
    assert {one["doc_sha256"] for one in found["sources"]} == {
        digest("a"),
        digest("b"),
    }


def test_a_source_says_whether_its_fact_still_holds(written, engine) -> None:
    """Re-judging a fact leaves the question alone, so the page has to say so."""
    client, held = written
    from sqlalchemy import text

    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE facts SET validated = false, rejection_code = 'copied' "
                "WHERE id = :id"
            ),
            {"id": held["facts"][0]},
        )

    found = client.get(f"/questions/{held['ids']['ordinary']}").json()

    assert found["sources"][0]["validated"] is False


def test_an_unknown_question_is_a_named_refusal(written) -> None:
    """A caller branches on the code, not on the message."""
    client, _ = written

    refused = client.get("/questions/999999")

    assert refused.status_code == 404
    assert refused.json()["code"] == "unknown_question"


def test_a_question_id_that_is_not_a_number_is_refused(written) -> None:
    """`/questions/status` is a route; `/questions/banana` is a mistake."""
    client, _ = written

    assert client.get("/questions/banana").status_code == 422


# ── Accepting and rejecting ────────────────────────────────────────────────


def test_accepting_a_rejected_question_keeps_the_gate_that_refused_it(
    written,
) -> None:
    """A person overruling a gate is a disagreement, and both halves are kept.

    This used to clear the reason, on the grounds that a reason for a
    rejection somebody overturned is a stale one. It is not stale, it is
    the other half of the disagreement: cleared, the one number a review
    sample exists to produce - how often the reviewer and the gates agreed
    - could not be computed at all, because an overruled rejection was
    indistinguishable from a question no gate ever stopped.
    """
    client, held = written

    decided = client.patch(
        f"/questions/{held['ids']['refused']}", json={"status": "accepted"}
    ).json()

    assert decided["status"] == "accepted"
    assert decided["rejected_reason"] == "duplicate"
    assert decided["reviewed_verdict"] == "accepted"


def test_a_question_nobody_looked_at_carries_no_verdict(written) -> None:
    """NULL means nobody has looked, which is most questions."""
    client, held = written

    listed = client.get("/questions").json()["questions"]
    untouched = [one for one in listed if one["id"] == held["ids"]["ordinary"]]

    assert untouched[0]["reviewed_verdict"] is None


def test_putting_a_question_back_to_draft_clears_the_verdict(written) -> None:
    """`draft` says "I have not decided", which is not a verdict to record."""
    client, held = written
    where = f"/questions/{held['ids']['ordinary']}"

    client.patch(where, json={"status": "rejected"})
    back = client.patch(where, json={"status": "draft"}).json()

    assert back["status"] == "draft"
    assert back["reviewed_verdict"] is None


def test_rejecting_a_question_keeps_the_row(written) -> None:
    """The share thrown away is the evidence behind the coverage report."""
    client, held = written

    client.patch(f"/questions/{held['ids']['ordinary']}", json={"status": "rejected"})

    assert client.get("/questions").json()["total"] == 4
    assert client.get("/questions", params={"status": "rejected"}).json()["total"] == 2


def test_a_verdict_survives_being_read_back(written) -> None:
    """The route answers with the row it wrote, not with what it was sent."""
    client, held = written

    client.patch(f"/questions/{held['ids']['ordinary']}", json={"status": "rejected"})
    found = client.get(f"/questions/{held['ids']['ordinary']}").json()

    assert found["question"]["status"] == "rejected"


def test_deciding_an_unknown_question_is_a_named_refusal(written) -> None:
    """Not a silent no-op, which would read as success."""
    client, _ = written

    refused = client.patch("/questions/999999", json={"status": "accepted"})

    assert refused.status_code == 404
    assert refused.json()["code"] == "unknown_question"


@pytest.mark.parametrize("body", [{"status": "approved"}, {"status": None}, {}])
def test_a_verdict_that_is_not_one_of_the_three_is_refused(written, body) -> None:
    """The column's CHECK would refuse it anyway, with a worse message."""
    client, held = written

    answered = client.patch(f"/questions/{held['ids']['ordinary']}", json=body)

    assert answered.status_code == 422


# ── Export ─────────────────────────────────────────────────────────────────


def workbook_from(response) -> dict:
    """Reads a downloaded workbook back the way a recipient opens it."""
    book = pd.ExcelFile(io.BytesIO(response.content))
    return {name: book.parse(name) for name in book.sheet_names}


def test_the_export_is_a_workbook_a_spreadsheet_can_open(written) -> None:
    """Served as an .xlsx, named, and offered as a download."""
    client, _ = written

    drawn = client.get("/questions/export")

    assert drawn.status_code == 200
    assert drawn.headers["content-type"] == (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert "questions.xlsx" in drawn.headers["content-disposition"]
    assert list(workbook_from(drawn)) == ["Questions", "Citations", "Summary"]


def test_the_export_has_no_default_scope(written) -> None:
    """A request naming nothing gets everything the listing would.

    Not "the release", not "everything accepted": a reader handed a
    narrower set than they asked for without having asked has been given
    the wrong file under a name that looks right. All four questions, the
    rejected one included.
    """
    client, _ = written

    questions = workbook_from(client.get("/questions/export"))["Questions"]

    assert len(questions) == 4
    assert set(questions["Status"]) == {"accepted", "rejected"}


def test_the_export_takes_the_listings_filter(written) -> None:
    """What downloads is what the page was showing."""
    client, _ = written

    questions = workbook_from(
        client.get("/questions/export?status=accepted&answerable=false")
    )["Questions"]

    assert len(questions) == 1
    assert not questions.iloc[0]["Answerable"]


def test_a_citation_row_is_one_fact_in_one_passage(written) -> None:
    """The grain of the Citations sheet, and the bug it is pinned against.

    `crossing` cites two facts in two documents and the other three cite
    one apiece, so five rows. Built as a WHERE subquery this came back as
    the product of every join instead: SQLAlchemy correlated the four
    tables the id query is joined over to the outer query, emptied the
    subquery's own FROM, and turned 1,229 rows into six million on the
    real corpus - which Excel then refused outright.
    """
    client, held = written

    citations = workbook_from(client.get("/questions/export"))["Citations"]

    assert len(citations) == 5
    assert sorted(citations["Question ID"]) == sorted(
        [
            held["ids"]["ordinary"],
            held["ids"]["crossing"],
            held["ids"]["crossing"],
            held["ids"]["refused"],
            held["ids"]["unanswered"],
        ]
    )


def test_citations_can_be_left_out(written) -> None:
    """The expensive half, and the questions alone run a benchmark."""
    client, _ = written

    read = workbook_from(client.get("/questions/export?citations=false"))

    assert read["Citations"].empty
    assert len(read["Questions"]) == 4


def test_a_filter_matching_nothing_still_exports(written) -> None:
    """The one filter that matches no question is not the one that raises."""
    client, _ = written

    read = workbook_from(client.get("/questions/export?status=draft"))

    assert read["Questions"].empty
    assert "Question" in read["Questions"].columns


# ── The queue under the same word ──────────────────────────────────────────


def test_the_queue_answers_for_the_topics_and_not_the_questions(written) -> None:
    """`/questions/status` is the stage; `/questions` is what it produced.

    Both are served by one router, which is the arrangement that makes
    `/questions/status` worth a test of its own: declared the other way
    round it would be read as a question with the id `status`.
    """
    client, _ = written

    state = client.get("/questions/status").json()

    assert state["stage"] == "questions"
    assert state["rows"] == {"new": 2}, "two topics, and no fit request among them"


def test_the_queue_verbs_move_topics(written) -> None:
    """The same five verbs as every other stage."""
    client, _ = written

    started = client.post("/questions/start").json()
    assert started["rows"] == 2
    assert client.get("/questions/status").json()["rows"] == {"pending": 2}

    client.post("/questions/stop")
    assert client.get("/questions/status").json()["rows"] == {"new": 2}


def test_a_verb_narrows_to_one_topic(written) -> None:
    """A topic is the smallest subject there is."""
    client, held = written

    started = client.post(f"/questions/topic/{held['topic']}/start").json()

    assert started["rows"] == 1
    assert client.get("/questions/status").json()["rows"] == {"pending": 1, "new": 1}


def test_a_scope_this_stage_does_not_take_is_a_named_refusal(written) -> None:
    """Question generation queues over topics and over nothing else."""
    client, _ = written

    refused = client.get(f"/questions/document/{digest('a')}/status")

    assert refused.status_code == 404
    assert refused.json()["code"] == "unknown_scope"


def test_a_value_the_column_cannot_hold_is_a_named_refusal(written) -> None:
    """A topic id is a number, and `banana` is not one."""
    client, _ = written

    refused = client.get("/questions/topic/banana/status")

    assert refused.status_code == 400
    assert refused.json()["code"] == "invalid_value"
