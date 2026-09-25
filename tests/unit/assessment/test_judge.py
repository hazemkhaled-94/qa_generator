"""Asking the judge, and what happens when it will not answer.

The distinction this file exists for is abstention against refusal. A model
that could not be reached has found nothing wrong with the artefact, and a
phase that recorded that as disapproval would fill a review queue with rows
whose only fault was that the runtime was restarting.
"""

from __future__ import annotations

import pytest

from assessment.judge import Judge, JudgeUnavailable
from assessment.models import ArtifactToJudge
from assessment.templates import TEMPLATES
from database.qa_generator import ArtifactKind, JudgeMetric
from llm.client import ModelUnavailable

#: Which of each metric's two labels approves, by the shape's name.
_GOOD = {
    one.shape().__name__: one.good for group in TEMPLATES.values() for one in group
}


def _approving(shape) -> str:
    """The label that approves, for whichever metric this shape belongs to."""
    return _GOOD[shape.__name__]


class _Answering:
    """A client that answers every call with a named label."""

    model = "test/judge"

    def __init__(self, *labels: str, fails: int = 0) -> None:
        """Answers with each label in turn, after `fails` refusals."""
        self._labels = list(labels)
        self._fails = fails
        self.asked: list[tuple[str, str]] = []

    def answer(self, *, system, user, shape, prompt_version=None):
        """Answers one judgement, or refuses."""
        self.asked.append((shape.__name__, user))
        if self._fails:
            self._fails -= 1
            raise ModelUnavailable("the runtime is restarting")
        # Left to itself, each metric's own approving label. A fixed
        # default would fail the shape's Literal the moment a test asked
        # about a metric whose rails are not the hallucination pair.
        label = self._labels.pop(0) if self._labels else _approving(shape)
        return shape(label=label, explanation="  because   the evidence  says so ")


def _fact() -> ArtifactToJudge:
    """One fact for the judge to read."""
    return ArtifactToJudge(
        assessment_id=1,
        kind=ArtifactKind.FACT,
        artifact_id=7,
        fields={"statement": "A reply is due in five days.", "evidence": "…"},
    )


def test_every_metric_of_a_kind_is_asked() -> None:
    """One call per judgement, never one call asking all of them.

    `question_generation/phrasing.py` exists because three judgements in
    one call were answered at chance in German while the harder task in the
    same call was answered well in both. Asking separately is the fix, and
    it is the fix here for the same reason.
    """
    client = _Answering("factual", "relevant", "non-toxic")

    judged = Judge(client).judge(_fact())

    assert [one.metric for one in judged] == [
        JudgeMetric.HALLUCINATION,
        JudgeMetric.RELEVANCE,
        JudgeMetric.TOXICITY,
    ]
    assert len(client.asked) == 3
    assert {shape for shape, _ in client.asked} == {
        "_Hallucination",
        "_Relevance",
        "_Toxicity",
    }


def test_the_artefact_reaches_the_prompt() -> None:
    """The statement and its evidence, substituted into the template."""
    client = _Answering()

    Judge(client).judge(_fact())

    assert "A reply is due in five days." in client.asked[0][1]
    assert "{{statement}}" not in client.asked[0][1]


def test_a_label_carries_the_score_and_the_side_phoenix_gives_it() -> None:
    """`factual` approves and scores 0.0, which is the confusable pair."""
    judged = Judge(_Answering("factual", "relevant", "non-toxic")).judge(_fact())
    by_metric = {one.metric: one for one in judged}

    assert by_metric[JudgeMetric.HALLUCINATION].approved is True
    assert by_metric[JudgeMetric.HALLUCINATION].score == 0.0
    assert by_metric[JudgeMetric.RELEVANCE].approved is True
    assert by_metric[JudgeMetric.RELEVANCE].score == 1.0


def test_a_hallucinated_label_does_not_approve() -> None:
    """And scores 1.0, because the metric is minimised."""
    judged = Judge(_Answering("hallucinated", "relevant", "non-toxic")).judge(_fact())

    assert judged[0].approved is False
    assert judged[0].score == 1.0


def test_an_explanation_is_kept_and_its_whitespace_is_not() -> None:
    """A reviewer reads this. It must not arrive with the model's spacing."""
    judged = Judge(_Answering()).judge(_fact())

    assert judged[0].explanation == "because the evidence says so"


def test_one_metric_failing_still_records_the_others() -> None:
    """An abstention is a judgement nobody made, not a refusal.

    Losing one opinion must not fail a row the rest of the metrics judged,
    so what comes back is short rather than empty.
    """
    judged = Judge(_Answering("relevant", "non-toxic", fails=1)).judge(_fact())

    assert [one.metric for one in judged] == [
        JudgeMetric.RELEVANCE,
        JudgeMetric.TOXICITY,
    ]


def test_no_metric_answering_is_a_failed_row_and_not_a_refusal() -> None:
    """The model being down is a row to retry, not an artefact to reject.

    This is the whole distinction the phase rests on. Recorded as
    disapproval, a Phoenix that was restarting would read as a corpus the
    judge rejected.
    """
    client = _Answering(fails=len(TEMPLATES[ArtifactKind.FACT]))

    with pytest.raises(JudgeUnavailable, match="answered none"):
        Judge(client).judge(_fact())


def test_a_question_is_asked_every_metric_its_kind_declares() -> None:
    """Six of them, and three have no counterpart anywhere upstream.

    `qa_correctness` asks whether the answer answers the question,
    `refusal` whether it is an answer at all, and `conciseness` whether it
    is only the answer. No gate asks any of those.
    """
    client = _Answering(
        "factual", "correct", "relevant", "answered", "concise", "non-toxic"
    )
    question = ArtifactToJudge(
        assessment_id=2,
        kind=ArtifactKind.QUESTION,
        artifact_id=9,
        fields={
            "question": "How long is allowed for a reply?",
            "answer": "Five days.",
            "facts": "- A reply is due in five days.",
        },
    )

    judged = Judge(client).judge(question)

    assert [one.metric for one in judged] == [
        JudgeMetric.HALLUCINATION,
        JudgeMetric.QA_CORRECTNESS,
        JudgeMetric.RELEVANCE,
        JudgeMetric.REFUSAL,
        JudgeMetric.CONCISENESS,
        JudgeMetric.TOXICITY,
    ]
    assert all(one.approved for one in judged)


def test_an_answer_that_declines_to_answer_is_caught_by_nothing_else() -> None:
    """The judgement this phase added that no gate makes.

    A question marked answerable whose stored answer is "the document does
    not specify" is well formed, the right length, cites its facts and
    leaks no source. Every gate passes it, and a benchmark built from it
    scores a chatbot against a non-answer.
    """
    client = _Answering(
        "factual", "correct", "relevant", "refused", "concise", "non-toxic"
    )
    question = ArtifactToJudge(
        assessment_id=3,
        kind=ArtifactKind.QUESTION,
        artifact_id=11,
        fields={
            "question": "How long is allowed for a reply?",
            "answer": "The document does not specify a period.",
            "facts": "- A reply is due in five days.",
        },
    )

    judged = {one.metric: one for one in Judge(client).judge(question)}

    assert judged[JudgeMetric.REFUSAL].label == "refused"
    assert judged[JudgeMetric.REFUSAL].approved is False
    assert judged[JudgeMetric.REFUSAL].score == 1.0
