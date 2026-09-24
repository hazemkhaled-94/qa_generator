"""Reading the rows a person should look at, and writing back what they said.

Three things in this pipeline are decided by a model and have somewhere for
a person to disagree: a fact's verdict, a topic's name, and whether a
question is any good. Those are the three datasets, and this is the half of
them that touches the database.

Stratified samples, not the first N. A review is a sample - nobody reads
twenty thousand facts - and a sample taken in id order is a sample of
whatever was extracted first, which for this corpus is one document. Drawn
evenly across the verdicts instead, because the question a review answers
is "is the checker right", and a sample of only what it accepted cannot
answer it. That the bands and the rejection codes exist to be sampled on
is what facts.rejection_code and questions.difficulty were for.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import case, func, select, update

from database.qa_generator import Fact, Question, ReviewVerdict
from database.qa_generator.repository import Repository


@dataclass(frozen=True)
class FactRow:
    """One fact as a reviewer sees it."""

    id: int
    statement: str
    evidence: str
    kind: str
    validated: bool
    rejection_code: str | None
    validation_error: str | None


@dataclass(frozen=True)
class QuestionRow:
    """One question as a reviewer sees it."""

    id: int
    question_text: str
    target_answer: str | None
    answerable: bool
    difficulty: str | None
    question_type: str | None
    language: str
    status: str
    rejected_reason: str | None
    facts: list[str]


def _even(total: int, groups: int) -> int:
    """How many to take from each group so the whole sample is `total`.

    At least one: a group with a share below one is a rejection code that
    fired twice in a corpus, and those are the interesting ones.
    """
    return max(1, total // max(groups, 1))


class ReviewRepository(Repository):
    """The rows a review is drawn from, and the verdicts it produces."""

    def facts(self, sample: int | None) -> list[FactRow]:
        """Draws a sample of facts, spread over the verdicts.

        Everything the checker rejected is grouped by its code, and what it
        accepted is one more group beside them, so a reviewer sees both the
        drops and the keeps in one sitting.

        `sample` of None takes every unreviewed fact instead, which is what
        `--all` asks for: the dataset is then the corpus rather than a draw
        from it, and a reviewer filters in Argilla rather than trusting the
        draw to have included what they were looking for.
        """
        with self._session() as session:
            groups = [
                code
                for (code,) in session.execute(
                    select(Fact.rejection_code).group_by(Fact.rejection_code)
                )
            ]
            each = None if sample is None else _even(sample, len(groups))
            rows: list[FactRow] = []
            for code in groups:
                found = session.execute(
                    select(
                        Fact.id,
                        Fact.statement,
                        Fact.evidence_text,
                        Fact.kind,
                        Fact.validated,
                        Fact.rejection_code,
                        Fact.validation_error,
                    )
                    .where(
                        Fact.rejection_code.is_(code)
                        if code is None
                        else Fact.rejection_code == code,
                        # Never the same row twice: a second push would
                        # otherwise ask for a verdict already given.
                        Fact.reviewed_verdict.is_(None),
                    )
                    # Randomly, so two pushes are two samples rather than
                    # the same rows in the same order.
                    .order_by(Fact.id if each is None else func.random())
                    .limit(each)
                ).all()
                rows += [FactRow(*one) for one in found]
        return rows

    def questions(
        self, sample: int | None, ids: Sequence[int] | None = None
    ) -> list[QuestionRow]:
        """Draws a sample of questions, spread over difficulty and verdict.

        Both, because they are the two things a reviewer is judging at
        once: whether the gate was right, and whether a question the band
        calls hard is actually hard.

        `ids` replaces the sampling with exactly those rows, in id order.
        That is how a queue somebody else built gets reviewed - the
        disagreements `evaluation.second_opinion` found, where the gates
        kept a question and an independent judge says the passages do not
        support its answer. A stratified sample cannot find those: they are
        spread over every band and every verdict and are rare in all of
        them, which is exactly the shape a proportional draw misses.
        """
        if ids:
            with self._session() as session:
                found = session.execute(
                    select(Question).where(Question.id.in_(ids)).order_by(Question.id)
                ).scalars()
                return [self._question(one) for one in found]
        with self._session() as session:
            groups = session.execute(
                select(Question.difficulty, Question.rejected_reason).group_by(
                    Question.difficulty, Question.rejected_reason
                )
            ).all()
            each = None if sample is None else _even(sample, len(groups))
            rows: list[QuestionRow] = []
            for difficulty, reason in groups:
                found = session.execute(
                    select(Question)
                    .where(
                        Question.difficulty.is_(difficulty)
                        if difficulty is None
                        else Question.difficulty == difficulty,
                        Question.rejected_reason.is_(reason)
                        if reason is None
                        else Question.rejected_reason == reason,
                    )
                    .order_by(Question.id if each is None else func.random())
                    .limit(each)
                ).scalars()
                rows += [self._question(one) for one in found]
        return rows

    @staticmethod
    def _question(one) -> QuestionRow:
        """One stored question, as a reviewer is shown it."""
        return QuestionRow(
            id=one.id,
            question_text=one.question_text,
            target_answer=one.target_answer,
            answerable=one.answerable,
            difficulty=one.difficulty,
            question_type=one.question_type,
            language=one.language,
            status=one.status,
            rejected_reason=one.rejected_reason,
            facts=[link.fact.statement for link in one.fact_links],
        )

    def review_facts(self, verdicts: list[tuple[int, str]]) -> int:
        """Records what a person decided about some facts.

        Writes `reviewed_verdict` and never `validated`: that one is the
        checker's and extract-revalidate rewrites it in full, so a verdict
        written there would last until the next re-judgement.

        Args:
            verdicts: Each fact's id beside `accepted` or `rejected`.

        Returns:
            How many rows were written.
        """
        if not verdicts:
            return 0
        now = datetime.now(UTC)
        written = 0
        with self._session.begin() as session:
            for fact_id, verdict in verdicts:
                written += session.execute(
                    update(Fact)
                    .where(Fact.id == fact_id)
                    .values(
                        reviewed_verdict=ReviewVerdict(verdict),
                        reviewed_at=now,
                    )
                ).rowcount
        return written

    def counts(self) -> dict[str, Any]:
        """How much of the corpus has been looked at, and who agreed.

        Both datasets a verdict can land in, and for each the number the
        sample exists to produce: how often the person and the model
        reached the same verdict. A count of rows reviewed says a review
        happened; the agreement says what it found.

        The model's verdict is read off a different column in each. A fact
        carries the checker's as `validated`. A question carries the
        gates' as `rejected_reason` - NULL means no gate stopped it - which
        is only readable because `decide()` stopped clearing it.
        """
        gated = case((Question.rejected_reason.is_(None), "accepted"), else_="rejected")
        checked = case((Fact.validated, "accepted"), else_="rejected")
        with self._session() as session:
            return {
                "facts": self._looked(session, Fact, Fact.reviewed_verdict, checked),
                "questions": self._looked(
                    session, Question, Question.reviewed_verdict, gated
                ),
            }

    @staticmethod
    def _looked(session, table, verdict, model) -> dict[str, Any]:
        """One dataset: how big it is, what was said, and how often it matched."""
        total = session.execute(select(func.count()).select_from(table)).scalar_one()
        rows = session.execute(
            select(verdict, model.label("model"), func.count())
            .where(verdict.is_not(None))
            .group_by(verdict, model)
        ).all()
        said: dict[str, int] = {}
        agreed = 0
        for person, machine, count in rows:
            said[person] = said.get(person, 0) + count
            if person == machine:
                agreed += count
        reviewed = sum(said.values())
        return {
            "total": total,
            "reviewed": reviewed,
            "verdicts": said,
            "agreed": agreed,
            # None rather than 0.0 when nothing has been reviewed: a rate
            # over no rows is not a rate, and printing 0% would read as
            # "the reviewer never agreed" instead of "nobody has looked".
            "agreement": round(agreed / reviewed, 3) if reviewed else None,
        }
