"""What one judged artefact leaves behind, in the database and in Phoenix.

Two halves, and the split is the one `telemetry/evaluations.py` already
draws for a gate verdict. The ROW is what the UI, the API and the export
read. The ANNOTATIONS are what put a label, a score and an explanation in
Phoenix's Evaluations view, and `annotator_kind` is what separates them
from the gates' - CODE for a rule, LLM for everything this phase writes.

The third thing checked here is what the phase does NOT do. It writes no
column any stage reads, and a judge that could not be reached fails a row
rather than refusing an artefact.
"""

from __future__ import annotations

import pytest

from assessment.judge import JudgeUnavailable
from assessment.models import ArtifactToJudge, Judgement
from assessment.service import SUMMARY, AssessmentService
from database.qa_generator import ArtifactKind, JudgeMetric


class _Queue:
    """A queue holding one artefact, recording what was written back."""

    done = "assessed"

    def __init__(self, *artifacts: ArtifactToJudge) -> None:
        """Holds the artefacts a drain will claim, in order."""
        self._artifacts = list(artifacts)
        self.recorded: list = []
        self.failed: list[tuple[int, str]] = []

    def claim(self):
        """Hands over the next artefact, or nothing."""
        return self._artifacts.pop(0) if self._artifacts else None

    def record(self, assessed) -> None:
        """Records a verdict."""
        self.recorded.append(assessed)

    def fail(self, key, error) -> None:
        """Records a row the judge could not be asked about."""
        self.failed.append((key, error))

    def abandon(self) -> int:
        """No claim has outlived its lease."""
        return 0

    def touch(self) -> bool:
        """Refreshes the claim this queue holds."""
        return True


class _Judge:
    """A judge answering with whatever it was built with."""

    model = "test/judge"

    def __init__(self, *judgements: Judgement, unavailable: bool = False) -> None:
        """Answers with these, or refuses."""
        self._judgements = judgements
        self._unavailable = unavailable

    def judge(self, artifact):
        """Answers, or says the model could not be reached."""
        if self._unavailable:
            raise JudgeUnavailable("nothing answered")
        return self._judgements


class _Evaluations:
    """A Phoenix that records what it was posted."""

    def __init__(self) -> None:
        """Holds nothing yet."""
        self.verdicts: list = []
        self.flushes = 0

    def record(self, *verdicts) -> None:
        """Holds some verdicts."""
        self.verdicts.extend(verdicts)

    def flush(self) -> int:
        """Counts a flush."""
        self.flushes += 1
        return 0


def _fact() -> ArtifactToJudge:
    """One fact to judge."""
    return ArtifactToJudge(
        assessment_id=11,
        kind=ArtifactKind.FACT,
        artifact_id=7,
        fields={"statement": "…", "evidence": "…"},
    )


def _judgement(metric: str, label: str, score: float, approved: bool) -> Judgement:
    """One judgement as the judge returns it."""
    return Judgement(
        metric=metric,
        label=label,
        score=score,
        approved=approved,
        explanation=f"{label} because so",
    )


APPROVING = (
    _judgement(JudgeMetric.HALLUCINATION, "factual", 0.0, True),
    _judgement(JudgeMetric.RELEVANCE, "relevant", 1.0, True),
)
REFUSING = (
    _judgement(JudgeMetric.HALLUCINATION, "hallucinated", 1.0, False),
    _judgement(JudgeMetric.RELEVANCE, "relevant", 1.0, True),
)


def _service(queue, judge, evaluations=None) -> AssessmentService:
    """The service, wired to fakes."""
    return AssessmentService(queue, judge, evaluations or _Evaluations())


def test_a_verdict_is_written_with_the_judge_that_reached_it() -> None:
    """The row carries who judged and under which templates."""
    queue = _Queue(_fact())

    _service(queue, _Judge(*APPROVING)).process_next()

    written = queue.recorded[0]
    assert written.assessment_id == 11
    assert written.judge_model == "test/judge"
    assert written.prompt_version
    assert written.approved is True


def test_one_metric_refusing_refuses_the_artefact() -> None:
    """All of them, not a majority.

    What this produces is a queue for a person, and an answer that is
    well-sourced but answers the wrong question is worth somebody's time
    even though the other metrics passed.
    """
    queue = _Queue(_fact())

    _service(queue, _Judge(*REFUSING)).process_next()

    written = queue.recorded[0]
    assert written.approved is False
    assert written.refused == (JudgeMetric.HALLUCINATION,)


def test_a_judge_that_cannot_be_reached_fails_the_row() -> None:
    """And records no verdict at all.

    The distinction the phase rests on: a model that is down has found
    nothing wrong with the artefact. Recorded as a refusal, a restarting
    runtime would read as a corpus the judge rejected.
    """
    queue = _Queue(_fact())

    _service(queue, _Judge(unavailable=True)).process_next()

    assert not queue.recorded
    assert queue.failed == [(11, "nothing answered")]


def test_every_metric_becomes_an_annotation_named_the_way_phoenix_names_it(
    monkeypatch,
) -> None:
    """One per judgement, plus a summary, and all of them LLM.

    The names are the point. A `hallucination` annotation from this
    pipeline has to mean what a `hallucination` annotation from anything
    else does, or a Phoenix chart over two projects is adding two
    different measurements together.
    """
    queue, evaluations = _Queue(_fact()), _Evaluations()
    monkeypatch.setattr("assessment.service.current_ids", lambda: ("a" * 32, "b" * 16))

    _service(queue, _Judge(*APPROVING), evaluations).process_next()

    by_name = {one.name: one for one in evaluations.verdicts}
    assert set(by_name) == {SUMMARY, JudgeMetric.HALLUCINATION, JudgeMetric.RELEVANCE}
    assert all(one.by_model for one in evaluations.verdicts), "must post as LLM"
    assert all(one.span_id == "b" * 16 for one in evaluations.verdicts)


def test_an_annotation_carries_the_label_score_and_reason(monkeypatch) -> None:
    """A verdict with no explanation is one a reviewer cannot act on."""
    queue, evaluations = _Queue(_fact()), _Evaluations()
    monkeypatch.setattr("assessment.service.current_ids", lambda: ("a" * 32, "b" * 16))

    _service(queue, _Judge(*REFUSING), evaluations).process_next()

    posted = {one.name: one for one in evaluations.verdicts}
    hallucination = posted[JudgeMetric.HALLUCINATION]
    assert hallucination.label == "hallucinated"
    assert hallucination.score == 1.0
    assert "because so" in hallucination.explanation
    # The direction rides along, because this metric's mean reads the
    # other way up from the rest.
    assert hallucination.metadata["direction"] == "minimize"


def test_the_summary_scores_one_for_approved(monkeypatch) -> None:
    """So a project's mean under that name IS the approval rate.

    Unlike the per-metric scores, which take phoenix-evals' direction.
    """
    queue, evaluations = _Queue(_fact()), _Evaluations()
    monkeypatch.setattr("assessment.service.current_ids", lambda: ("a" * 32, "b" * 16))

    _service(queue, _Judge(*APPROVING), evaluations).process_next()

    summary = next(one for one in evaluations.verdicts if one.name == SUMMARY)
    assert summary.label == "approved"
    assert summary.score == 1.0
    assert summary.metadata["direction"] == "maximize"


def test_nothing_is_posted_without_a_span(monkeypatch) -> None:
    """A verdict with no span to hang off is dropped rather than invented.

    Which is every process that configured telemetry without an exporter.
    """
    queue, evaluations = _Queue(_fact()), _Evaluations()
    monkeypatch.setattr("assessment.service.current_ids", lambda: ("", ""))

    _service(queue, _Judge(*APPROVING), evaluations).process_next()

    assert queue.recorded, "the row is still written"
    assert not evaluations.verdicts


def test_an_empty_queue_flushes_what_is_held() -> None:
    """A corpus smaller than the batch would otherwise never reach Phoenix.

    The last batch of every run is the one this loses without a flush, and
    on a corpus of eighty artefacts that is all of them.
    """
    queue, evaluations = _Queue(_fact()), _Evaluations()

    drained = _service(queue, _Judge(*APPROVING), evaluations).drain()

    assert drained == 1
    assert evaluations.flushes >= 1


def test_a_drain_works_every_queued_artefact() -> None:
    """And stops when the queue is empty rather than spinning."""
    queue = _Queue(_fact(), _fact(), _fact())

    assert _service(queue, _Judge(*APPROVING)).drain() == 3
    assert len(queue.recorded) == 3


@pytest.mark.parametrize("judgements", [APPROVING, REFUSING])
def test_the_phase_writes_no_column_a_stage_reads(judgements) -> None:
    """The guarantee the whole design rests on.

    `facts.validated` and `questions.status` decide what is kept. This
    phase touches neither: the only thing it hands its queue is an
    `Assessed`, and the only row that changes is the assessment's own.
    """
    queue = _Queue(_fact())

    _service(queue, _Judge(*judgements)).process_next()

    written = queue.recorded[0]
    assert set(vars(written)) == {
        "assessment_id",
        "judgements",
        "judge_model",
        "prompt_version",
        "trace_id",
        "span_id",
    }


def test_the_refused_metrics_reach_the_span_as_plain_strings() -> None:
    """OpenTelemetry drops a sequence attribute holding a StrEnum.

    It type-checks each item with `type()` rather than `isinstance`, so a
    `JudgeMetric` - which IS a `str` - is refused, and the whole attribute
    is dropped with a warning and no error. The run looks fine and the one
    span attribute naming what the judge refused is simply absent.
    """
    recorded: dict = {}

    class _Span:
        """A span that records what was set on it."""

        def set_attribute(self, key, value):
            """Records one attribute."""
            recorded[key] = value

        def __enter__(self):
            """Enters the span."""
            return self

        def __exit__(self, *_):
            """Leaves it."""
            return False

    queue = _Queue(_fact())
    service = _service(queue, _Judge(*REFUSING))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "assessment.service.working",
            lambda *a, **k: _Span(),
        )
        service.process_next()

    assert recorded["assessment.refused"] == ["hallucination"]
    assert all(type(one) is str for one in recorded["assessment.refused"])
    assert type(recorded["assessment.hallucination"]) is str
