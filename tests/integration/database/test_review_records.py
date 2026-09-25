"""The review repository against the database that runs it.

Every other repository in this project has a file here. This one did not,
and it is the one whose whole job is a query nothing else writes: a sample
stratified over the verdicts, and an agreement rate computed off two
different columns in two different tables.

`tests/unit/review/test_records.py` covers `_even` and what a record carries
once a row has been read. None of that touches SQL, which is why the module
sat at 42% while the repositories tested here sit near complete.

What is worth pinning is the sampling. A review answers "is the checker
right", and every way of getting that wrong is a query:

- a sample of only what the checker accepted cannot answer it,
- a sample in id order is a sample of whichever document was extracted
  first,
- a sample that can return a row somebody has already judged asks the same
  question twice.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from seed import digest, document, fact, link, passage, question
from sqlalchemy import select
from sqlalchemy.orm import Session

from database.qa_generator import Fact, Question, QuestionStatus, ReviewVerdict
from review.records import ReviewRepository

pytestmark = pytest.mark.integration


@pytest.fixture
def corpus(engine, database):
    """One passage, and facts spread over the verdicts a review samples.

    Four codes and two accepted, because a stratified draw over six rows in
    five groups is what tells a stratified draw from `LIMIT n`.
    """
    with Session(engine) as session:
        sha = digest("a")
        session.add(document(sha))
        session.flush()
        held = passage(sha)
        session.add(held)
        session.flush()

        written = {}
        for statement, code in (
            ("A reply is due in five days.", None),
            ("The fee is 120 EUR.", None),
            ("The device weighs 4 kg.", "copied"),
            ("It arrives in March.", "unresolved_reference"),
            ("The plant produces two lines.", "unsupported_addition"),
            ("Everything is fine.", "asserts_nothing"),
        ):
            row = fact(
                held.id,
                statement=statement,
                validated=code is None,
                rejection_code=code,
            )
            session.add(row)
            session.flush()
            written[statement] = row.id
        session.commit()
        return {"passage": held.id, "facts": written}


@pytest.fixture
def asked(corpus, engine):
    """Questions spread over difficulty and verdict, each citing a fact."""
    with Session(engine) as session:
        one = next(iter(corpus["facts"].values()))
        written = []
        for text, difficulty, reason in (
            ("How long is allowed for a reply?", "easy", None),
            ("What does the device weigh?", "easy", "compound"),
            ("Why is the fee 120 EUR?", "hard", None),
            ("Which plant ships in March?", "hard", "restates_question"),
        ):
            row = question(
                question_text=text,
                difficulty=difficulty,
                rejected_reason=reason,
                status=(
                    QuestionStatus.ACCEPTED if reason is None else QuestionStatus.DRAFT
                ),
                target_answer="Five days.",
                answer_explanation="Because the policy says so.",
            )
            session.add(row)
            session.flush()
            session.add(link(row.id, one))
            written.append(row.id)
        session.commit()
        return written


# ── The fact sample ───────────────────────────────────────────────────────


def test_the_sample_reaches_every_verdict_and_not_just_the_first_rows(
    corpus,
) -> None:
    """A draw of one per group, which `LIMIT 5` over six rows would not give.

    In id order the first five rows are both accepted facts and three of
    the four codes, so `asserts_nothing` - the rarest thing in the corpus and
    the one a reviewer most wants to see - would never be drawn.
    """
    drawn = ReviewRepository().facts(sample=5)

    assert {one.rejection_code for one in drawn} == {
        None,
        "copied",
        "unresolved_reference",
        "unsupported_addition",
        "asserts_nothing",
    }


def test_a_sample_smaller_than_the_number_of_groups_still_reaches_them_all(
    corpus,
) -> None:
    """`_even` floors at one, and this is that floor against real groups."""
    drawn = ReviewRepository().facts(sample=1)

    assert len(drawn) == 5, [one.rejection_code for one in drawn]


def test_asking_for_everything_takes_every_unreviewed_fact(corpus) -> None:
    """Which is what `--all` is: the corpus rather than a draw from it."""
    drawn = ReviewRepository().facts(sample=None)

    assert len(drawn) == 6


def test_a_fact_already_judged_is_never_drawn_again(corpus, engine) -> None:
    """A second push would otherwise ask for a verdict already given."""
    judged = corpus["facts"]["Everything is fine."]
    ReviewRepository().review_facts([(judged, "rejected")])

    drawn = ReviewRepository().facts(sample=None)

    assert judged not in {one.id for one in drawn}
    assert len(drawn) == 5


def test_a_drawn_fact_carries_what_a_reviewer_is_shown(corpus) -> None:
    """The statement, the sentence behind it, and why the checker refused."""
    (refused,) = [
        one
        for one in ReviewRepository().facts(sample=None)
        if one.rejection_code == "copied"
    ]

    assert refused.statement == "The device weighs 4 kg."
    assert refused.evidence == "The device weighs 4 kg."
    assert refused.validated is False
    assert refused.kind == "atomic"


def test_an_empty_corpus_is_an_empty_push_rather_than_an_error(database) -> None:
    """Nothing to group by, so nothing to divide a sample over."""
    assert ReviewRepository().facts(sample=200) == []


# ── The question sample ───────────────────────────────────────────────────


def test_questions_are_spread_over_the_band_and_the_verdict_together(
    asked,
) -> None:
    """Both, because a reviewer is judging both at once.

    Four groups over the pair, so a draw of four is one of each: whether
    the gate was right, and whether a band called hard is actually hard.
    """
    drawn = ReviewRepository().questions(sample=4)

    assert {(one.difficulty, one.rejected_reason) for one in drawn} == {
        ("easy", None),
        ("easy", "compound"),
        ("hard", None),
        ("hard", "restates_question"),
    }


def test_naming_the_rows_replaces_the_sampling_entirely(asked) -> None:
    """How a queue somebody else built gets reviewed.

    `evaluation.second_opinion` finds disagreements spread thinly over
    every band and every verdict, which is the shape a proportional draw
    misses.
    """
    wanted = [asked[3], asked[1]]

    drawn = ReviewRepository().questions(sample=1, ids=wanted)

    assert [one.id for one in drawn] == sorted(wanted), "in id order, not as given"


def test_a_drawn_question_carries_the_facts_and_the_sentences_behind_them(
    asked,
) -> None:
    """A reviewer asked whether the answer follows needs both."""
    (drawn,) = ReviewRepository().questions(sample=1, ids=[asked[0]])

    assert drawn.facts == ["A reply is due in five days."]
    assert drawn.evidence == ["A reply is due in five days."]
    assert drawn.answer_explanation == "Because the policy says so."


def test_no_questions_at_all_is_an_empty_push(database) -> None:
    """The same empty corpus the facts side has to survive."""
    assert ReviewRepository().questions(sample=200) == []


# ── Writing a verdict back ────────────────────────────────────────────────


def test_a_verdict_is_written_against_the_row_it_names(corpus, engine) -> None:
    """With the time it was given, which is what a second push reads."""
    judged = corpus["facts"]["The fee is 120 EUR."]

    assert ReviewRepository().review_facts([(judged, "accepted")]) == 1

    with Session(engine) as session:
        row = session.get(Fact, judged)
        assert row.reviewed_verdict == ReviewVerdict.ACCEPTED
        assert row.reviewed_at is not None


def test_a_verdict_never_touches_the_checkers_own_column(corpus, engine) -> None:
    """`validated` is the checker's, and revalidate rewrites it in full.

    A person's verdict written there would last until the next
    re-judgement and then vanish, which is the worst of both.
    """
    judged = corpus["facts"]["The device weighs 4 kg."]
    ReviewRepository().review_facts([(judged, "accepted")])

    with Session(engine) as session:
        row = session.get(Fact, judged)
        assert row.validated is False, "the checker still says it is copied"
        assert row.rejection_code == "copied"
        assert row.reviewed_verdict == ReviewVerdict.ACCEPTED


def test_writing_nothing_writes_nothing(corpus, engine) -> None:
    """An empty pull is the ordinary case, not an error."""
    assert ReviewRepository().review_facts([]) == 0

    with Session(engine) as session:
        judged = session.execute(
            select(Fact).where(Fact.reviewed_verdict.is_not(None))
        ).all()
        assert judged == []


def test_a_verdict_on_a_row_that_is_gone_is_counted_as_nothing_written(
    corpus,
) -> None:
    """A dataset outlives the corpus it was pushed from."""
    assert ReviewRepository().review_facts([(999_999, "accepted")]) == 0


# ── What the review found ─────────────────────────────────────────────────


def test_nothing_reviewed_reports_no_agreement_rather_than_none_agreeing(
    corpus,
) -> None:
    """0% would read as "the reviewer never agreed" instead of "nobody looked"."""
    counted = ReviewRepository().counts()

    assert counted["facts"]["total"] == 6
    assert counted["facts"]["reviewed"] == 0
    assert counted["facts"]["agreement"] is None


def test_a_fact_is_agreed_with_when_the_person_matches_validated(corpus) -> None:
    """Which is the column the checker's own verdict lives in.

    Two agreements and one disagreement: the accepted fact confirmed, the
    copied one confirmed as a refusal, and the second accepted one
    overruled.
    """
    ReviewRepository().review_facts(
        [
            (corpus["facts"]["A reply is due in five days."], "accepted"),
            (corpus["facts"]["The device weighs 4 kg."], "rejected"),
            (corpus["facts"]["The fee is 120 EUR."], "rejected"),
        ]
    )

    counted = ReviewRepository().counts()["facts"]

    assert counted["reviewed"] == 3
    assert counted["verdicts"] == {"accepted": 1, "rejected": 2}
    assert counted["agreed"] == 2
    assert counted["agreement"] == round(2 / 3, 3)


def test_a_question_is_agreed_with_off_the_gate_rather_than_the_status(
    asked, engine
) -> None:
    """A question carries the gates' verdict as `rejected_reason`.

    NULL means no gate stopped it, which is only readable because
    `decide()` stopped clearing the column.
    """
    # Written here rather than through the repository, which only writes
    # facts: a question's verdict comes back through the Argilla pull. Both
    # columns, because `questions_reviewed_together` insists on the pair -
    # a verdict with no time is a row nothing can order by.
    with Session(engine) as session:
        for row_id in (asked[0], asked[1]):
            held = session.get(Question, row_id)
            held.reviewed_verdict = ReviewVerdict.ACCEPTED
            held.reviewed_at = datetime.now(UTC)
        session.commit()

    counted = ReviewRepository().counts()["questions"]

    assert counted["total"] == 4
    assert counted["reviewed"] == 2
    # The first was let through and the person agreed; the second was
    # refused by `compound` and the person overruled it.
    assert counted["agreed"] == 1
    assert counted["agreement"] == 0.5


def test_both_datasets_are_counted_even_when_only_one_was_reviewed(
    corpus, asked
) -> None:
    """A report naming one dataset cannot say a review is half done."""
    ReviewRepository().review_facts(
        [(corpus["facts"]["A reply is due in five days."], "accepted")]
    )

    counted = ReviewRepository().counts()

    assert set(counted) == {"facts", "questions"}
    assert counted["facts"]["reviewed"] == 1
    assert counted["questions"]["reviewed"] == 0
    assert counted["questions"]["agreement"] is None
