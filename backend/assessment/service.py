"""The drain loop of the assessment phase.

One artefact at a time: claim it, ask the judge each metric its kind
declares, write the verdict, and post the same verdict to Phoenix as
annotations.

**A verdict is written three times, and the split is the one
`telemetry/evaluations.py` already draws for a gate.** The row in Postgres
is what the UI, the API and the export read. The span attributes are what a
trace is filtered by. The annotations are what put a label, a score and an
explanation in Phoenix's Evaluations view, where they sort and chart
beside the gate verdicts - with `annotator_kind` telling the two apart,
CODE for a rule and **LLM for everything here**. That distinction is the
one this repository cares about most, and this phase is the first thing in
it that produces LLM annotations in bulk.

**Nothing here gates anything.** `facts.validated` and `questions.status`
are what decide whether an artefact is kept, and this phase does not write
either. It writes a column beside them, so the two can be compared and the
disagreements can be put in front of a person. That is the position
`evaluation/README.md` argues for and the measurement behind it has not
changed: an LLM judge was at chance on the German half of this corpus, and
a phase that could reject rows would have thrown away good German
questions on the strength of it.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from assessment.judge import Judge, JudgeUnavailable
from assessment.models import ArtifactToJudge, Assessed
from assessment.repository import AssessmentQueue
from assessment.templates import PROMPT_VERSION, direction_of, metrics_of
from stages import StageService
from telemetry import tracer, working
from telemetry.evaluations import Evaluations, Verdict, current_ids

log = logging.getLogger(__name__)
span = tracer(__name__)

#: What the summary annotation is called in Phoenix. One name per artefact
#: beside the per-metric ones, so a project's mean under `assessment` is
#: the share of artefacts the judge backed whole.
SUMMARY = "assessment"


class AssessmentService(StageService):
    """Judges one artefact at a time and records what was said."""

    name: ClassVar[str] = "assessment"
    unit: ClassVar[str] = "artifact"

    def __init__(
        self,
        repository: AssessmentQueue,
        judge: Judge,
        evaluations: Evaluations | None = None,
    ) -> None:
        """Initialises the service with its queue, its judge and Phoenix."""
        super().__init__(repository)
        self._repository = repository
        self._judge = judge
        # Built here rather than per row: it batches, and a Phoenix that is
        # down should warn once for the run rather than once per artefact.
        self._evaluations = evaluations if evaluations is not None else Evaluations()

    def process_next(self) -> int | None:
        """Judges one artefact and records the verdict.

        Returns:
            The assessment's id, or None when the queue is empty.
        """
        artifact = self._repository.claim()
        if artifact is None:
            # The flush is here as well as at the end of the drain: a
            # worker that empties its queue and then sleeps would
            # otherwise hold the last batch until the next artefact
            # arrived, which on an idle corpus is never.
            self._evaluations.flush()
            return None

        with working(
            span,
            "assess",
            {
                "stage": self.name,
                "assessment.id": artifact.assessment_id,
                "artifact.kind": artifact.kind,
                "artifact.id": artifact.artifact_id,
            },
        ) as current:
            try:
                judgements = self._judge.judge(artifact)
            except JudgeUnavailable as exc:
                # Failed rather than recorded as disapproval. A judge that
                # could not be reached has found nothing wrong with the
                # artefact, and `retry` is what moves this row.
                self._fail(artifact.assessment_id, str(exc), current)
                return artifact.assessment_id

            trace_id, span_id = current_ids()
            assessed = Assessed(
                assessment_id=artifact.assessment_id,
                judgements=judgements,
                judge_model=self._judge.model,
                prompt_version=PROMPT_VERSION,
                # What the kind was DUE, which is not what came back: the
                # judge abstains per metric when the model will not answer,
                # and the row would otherwise read `assessed` whether it got
                # six judgements or two.
                metrics_due=len(metrics_of(artifact.kind)),
                trace_id=trace_id,
                span_id=span_id,
            )
            self._repository.record(assessed)
            self._done(current)

            current.set_attribute("assessment.approved", assessed.approved)
            # `str` per item, not the JudgeMetric members themselves.
            # OpenTelemetry type-checks a sequence attribute by `type()`
            # rather than by isinstance, so a StrEnum - which IS a str -
            # is refused and the whole attribute is DROPPED with a warning
            # and no error. The run looks fine and the one attribute
            # naming what the judge refused is missing from every span.
            current.set_attribute(
                "assessment.refused", [str(one) for one in assessed.refused]
            )
            for one in judgements:
                # One attribute per metric, so a trace can be filtered on
                # `assessment.hallucination = hallucinated` without anybody
                # joining to Postgres.
                current.set_attribute(f"assessment.{one.metric}", str(one.label))

            if assessed.abstained:
                # The judge already logged each one. This says what it cost
                # the ROW, which is the thing that outlives the run: a
                # verdict standing on fewer metrics than its version asked
                # for, and `assess-retry` will not find it because the row
                # did not fail.
                log.warning(
                    "%s %d is judged on %d of %d metric(s): the rest abstained, "
                    "so this verdict is thinner than version %s asks for. "
                    "`make assess-rerun` is what asks them again.",
                    artifact.kind,
                    artifact.artifact_id,
                    len(judgements),
                    assessed.metrics_due,
                    PROMPT_VERSION,
                )
            self._record(artifact, assessed, span_id)
            log.info(
                "%s %d: the judge %s%s",
                artifact.kind,
                artifact.artifact_id,
                "approved it" if assessed.approved else "did not approve it",
                f" ({', '.join(assessed.refused)})" if assessed.refused else "",
                extra={
                    "artifact.kind": artifact.kind,
                    "artifact.id": artifact.artifact_id,
                    "assessment.approved": assessed.approved,
                    "assessment.metrics": len(judgements),
                },
            )
        return artifact.assessment_id

    def _record(
        self, artifact: ArtifactToJudge, assessed: Assessed, span_id: str
    ) -> None:
        """Posts this artefact's verdict to Phoenix as annotations.

        One per metric, named exactly as phoenix-evals names it, plus a
        summary. The per-metric names are what makes this comparable with
        anything else that posts a `hallucination`: a Phoenix chart over
        two projects does not care which pipeline wrote them, only that
        the name and the score mean the same thing.

        Best-effort throughout, like the gate verdicts: a Phoenix that is
        down costs the annotation and not the run, and the verdict is in
        Postgres either way.
        """
        if not span_id:
            return
        about = {
            "artifact.kind": artifact.kind,
            "artifact.id": artifact.artifact_id,
            "prompt_version": assessed.prompt_version,
            "judge_model": assessed.judge_model,
        }
        self._evaluations.record(
            Verdict(
                span_id=span_id,
                name=SUMMARY,
                label="approved" if assessed.approved else "refused",
                # 1 for approved, so a project's mean under this name IS
                # the share of artefacts the judge backed whole.
                score=1.0 if assessed.approved else 0.0,
                explanation=(
                    "every metric approved"
                    if assessed.approved
                    else f"not approved by: {', '.join(assessed.refused)}"
                ),
                by_model=True,
                metadata=about | {"direction": "maximize"},
            ),
            *(
                Verdict(
                    span_id=span_id,
                    name=one.metric,
                    label=one.label,
                    score=one.score,
                    explanation=one.explanation,
                    by_model=True,
                    metadata=about | {"direction": direction_of(one.metric)},
                )
                for one in assessed.judgements
            ),
        )

    def drain(self, stopping=None) -> int:
        """Works the queue, then posts whatever annotations are still held.

        The flush is the reason this is overridden. A corpus of eighty
        artefacts never reaches the batch size, and without it the last
        batch of every run is the one missing from Phoenix.
        """
        try:
            return super().drain(stopping) if stopping else super().drain()
        finally:
            self._evaluations.flush()
