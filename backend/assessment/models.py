"""The shapes this stage moves between its queue, its judge and its readers.

Dataclasses rather than the ORM rows, for the reason every other stage
here keeps a pair: the service reads an artefact once, hands it to a model
that may take minutes, and writes a verdict back. A detached ORM row held
across that is a row whose session has gone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class ArtifactToJudge:
    """One artefact, as the judge is given it.

    Attributes:
        assessment_id: The queue row, which is what a verdict is written
            against.
        kind: fact, topic or question.
        artifact_id: The row the assessment is about, for the log line and
            for whatever reads the verdict back.
        fields: The text the templates substitute, by the name they use -
            `statement` and `evidence` for a fact, `terms` and `label` for a
            topic, `question`, `answer` and `facts` for a question.
        language: What the artefact is written in, recorded on the span. Not
            sent to the model: every template says the material may be in
            any language, and naming it invites a judgement about the
            language rather than about the artefact.
    """

    assessment_id: int
    kind: str
    artifact_id: int
    fields: dict[str, str]
    language: str | None = None


@dataclass(frozen=True)
class Judgement:
    """What the judge said about one metric.

    Attributes:
        metric: The phoenix-evals name, which is what Phoenix files it as.
        label: One of that metric's own labels.
        score: What phoenix-evals scores that label.
        approved: Whether this label is the good side of this metric.
        explanation: Why, in the model's words.
    """

    metric: str
    label: str
    score: float
    approved: bool
    explanation: str


@dataclass(frozen=True)
class Assessed:
    """Every judgement about one artefact, ready to be written.

    Attributes:
        assessment_id: The queue row this belongs to.
        judgements: One per metric the kind declares, in template order.
        judge_model: Which model answered.
        prompt_version: Which template version it was asked under.
        metrics_due: How many metrics the kind declares under that version.
            Recorded because `judgements` is short of it wherever a metric
            ABSTAINED - the model would not answer - and without the count
            a thin verdict and a whole one are the same row. The count is
            in the source under a version the row outlives, which is the
            argument the `prompts` table is built on.
        trace_id: The trace it was judged in, for the row to link to.
        span_id: The span its annotations hang off.
    """

    assessment_id: int
    judgements: tuple[Judgement, ...]
    judge_model: str
    prompt_version: str
    metrics_due: int = 0
    trace_id: str = ""
    span_id: str = ""

    @property
    def abstained(self) -> int:
        """How many metrics were due and did not answer."""
        return max(self.metrics_due - len(self.judgements), 0)

    @property
    def approved(self) -> bool:
        """Whether every metric landed on its good side.

        All of them, not a majority. What this produces is a queue for a
        person, and a question whose answer is well-sourced but answers the
        wrong question is worth somebody's time even though two of its
        three metrics passed.
        """
        return all(one.approved for one in self.judgements)

    @property
    def refused(self) -> tuple[str, ...]:
        """Which metrics did not approve, for the log line and the record."""
        return tuple(one.metric for one in self.judgements if not one.approved)


@dataclass(frozen=True)
class StoredMetric:
    """One metric of one assessment, as a reader gets it back."""

    metric: str
    label: str
    score: float
    approved: bool
    explanation: str


@dataclass(frozen=True)
class StoredAssessment:
    """One assessment as the API serves it and the page draws it.

    Carries the artefact's own text, because every reader of this wants the
    verdict beside the thing it is about and neither the page nor the
    workbook should have to make a second request per row to get it.
    """

    id: int
    kind: str
    artifact_id: int
    status: str
    approved: bool | None
    judge_model: str | None
    prompt_version: str | None
    run_id: str | None
    trace_id: str | None
    span_id: str | None
    assessed_at: datetime | None
    error: str | None
    #: What the pipeline's own checker or gates decided, so the two verdicts
    #: are read together. `accepted` where nothing refused it.
    verdict: str | None = None
    #: The artefact in one line, for a table cell.
    summary: str = ""
    metrics: list[StoredMetric] = field(default_factory=list)

    @property
    def refused(self) -> list[str]:
        """Which metrics did not approve."""
        return [one.metric for one in self.metrics if not one.approved]

    @property
    def disagrees(self) -> bool:
        """Whether the judge and the pipeline point opposite ways.

        One direction only, as `evaluation/second_opinion.py` argues: an
        artefact the pipeline KEPT and the judge refused is either a check
        that let something through or a judge that is wrong, and only a
        person settles which. The other direction is usually neither,
        because most rejections are for something the judge is not asked
        about at all.
        """
        return self.approved is False and self.verdict == "accepted"


@dataclass(frozen=True)
class MetricQuality:
    """How one metric did over whatever was asked for."""

    metric: str
    judged: int
    approved: int
    #: `maximize` or `minimize`, as phoenix-evals declares it, so a reader
    #: knows which way up to read the mean of `score`.
    direction: str

    @property
    def share(self) -> float:
        """What fraction this metric approved."""
        return self.approved / self.judged if self.judged else 0.0


@dataclass(frozen=True)
class AssessmentQuality:
    """What the judge made of the corpus, as counts.

    Served by the API, drawn by the page, and written onto the workbook's
    own sheet - one shape for all three, so the figures in an export are
    the figures that were on screen.
    """

    total: int
    judged: int
    approved: int
    refused: int
    #: Kept by the pipeline and refused by the judge. The queue worth a
    #: person's time, and the only number here anybody should act on.
    disagreements: int
    by_kind: dict[str, int]
    approved_by_kind: dict[str, int]
    metrics: list[MetricQuality]
    judge_models: list[str]
    #: Enrolled but not yet judged, which is what says a phase is part-way
    #: through rather than that the judge refused everything.
    outstanding: int = 0
