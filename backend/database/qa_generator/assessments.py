"""The assessments table, and the metrics one assessment holds.

One assessment per artefact the judge was asked about, and one metric row
per judgement it made. Two tables rather than one because the metrics a
kind carries differ by kind - a fact is asked two questions and a question
three - and a column per metric would be a column that is NULL for two
kinds out of three.

Also the queue. Nothing upstream creates these rows: a fact arrives from
extraction with nowhere to record a judgement, so `start` is what enrols
it. That is topic modelling's arrangement rather than extraction's, and
for the same reason - asking is what creates the work.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.qa_generator.base import Base
from database.qa_generator.outcomes import ArtifactKind, JudgeMetric, one_of
from database.qa_generator.status import Status, check, queued


class Assessment(Base):
    """What an independent judge made of one artefact.

    Reaches its artefact through exactly one of the three foreign keys, each
    of which cascades: an assessment of a fact that has been deleted is an
    opinion about nothing, and a polymorphic (kind, id) pair would have left
    it behind for a trigger to sweep.
    """

    __tablename__ = "assessments"
    __table_args__ = (
        CheckConstraint(
            check("assess_status", Status.ASSESSED), name="assessments_status_valid"
        ),
        CheckConstraint(
            one_of("artifact_kind", ArtifactKind), name="assessments_kind_valid"
        ),
        # Exactly one artefact. Without it a row could point at a fact and a
        # question at once, and the metric rows beneath it would be about
        # two different things.
        CheckConstraint(
            "(fact_id IS NOT NULL)::int + (topic_id IS NOT NULL)::int + "
            "(question_id IS NOT NULL)::int = 1",
            name="assessments_one_artifact",
        ),
        # And the stored kind says which one. Derivable, and stored anyway:
        # the queue narrows on it and the report groups on it, and a CASE
        # over three columns in every such query is the version of this that
        # cannot be indexed.
        CheckConstraint(
            "(artifact_kind = 'fact') = (fact_id IS NOT NULL) AND "
            "(artifact_kind = 'topic') = (topic_id IS NOT NULL) AND "
            "(artifact_kind = 'question') = (question_id IS NOT NULL)",
            name="assessments_kind_agrees",
        ),
        # One assessment per artefact. `start` enrols whatever has no row
        # yet, and this is what makes running it twice cost nothing.
        UniqueConstraint("fact_id", name="assessments_one_per_fact"),
        UniqueConstraint("topic_id", name="assessments_one_per_topic"),
        UniqueConstraint("question_id", name="assessments_one_per_question"),
        # Partial, as on every other stage: the queue is a shrinking
        # fraction of the table.
        Index(
            "ix_assessments_queue",
            "assess_status",
            postgresql_where=text(queued("assess_status")),
        ),
        # The report groups by kind and by whether the judge approved.
        Index("ix_assessments_kind", "artifact_kind"),
        Index(
            "ix_assessments_approved",
            "approved",
            postgresql_where=text("approved IS NOT NULL"),
        ),
        # A verdict is only half written if one of these is missing.
        CheckConstraint(
            "(approved IS NULL) = (assessed_at IS NULL)",
            name="assessments_verdict_together",
        ),
        {
            "comment": "What an LLM judge made of one fact, topic or question, "
            "and the queue that asked it. Never a gate: the checker and the "
            "gates decide what is kept, and this is recorded beside their "
            "verdict so the two can be compared. One metric row per judgement "
            "is in assessment_metrics."
        },
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    artifact_kind: Mapped[str] = mapped_column(
        Text,
        comment="fact | topic | question. Which of the three foreign keys "
        "below is set, stored so the queue can narrow on it and the report "
        "can group on it without a CASE over three columns.",
    )
    fact_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("facts.id", ondelete="CASCADE"),
        comment="The fact this is about, NULL unless artifact_kind is `fact`. "
        "Cascades: a judgement about a deleted fact is an opinion about "
        "nothing.",
    )
    topic_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("topics.id", ondelete="CASCADE"),
        comment="The topic this is about, NULL unless artifact_kind is "
        "`topic`. A refit replaces every topic, so this cascades on every "
        "`make topics-discover`.",
    )
    question_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("questions.id", ondelete="CASCADE"),
        comment="The question this is about, NULL unless artifact_kind is `question`.",
    )
    assess_status: Mapped[str] = mapped_column(
        Text,
        server_default=Status.NEW,
        comment="This stage's own queue column. A row is enrolled `new` by "
        "`start`, which is also what makes it claimable - nothing upstream "
        "creates one.",
    )
    assess_error: Mapped[str | None] = mapped_column(
        Text, comment="Why the judge could not be asked, NULL when it could."
    )
    assess_claimed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), comment="When a worker took this row."
    )
    approved: Mapped[bool | None] = mapped_column(
        Boolean,
        comment="Whether every metric landed on its good side. NULL until "
        "the judge has answered, which is most rows until the phase runs. "
        "Never read by a gate: `facts.validated` and `questions.status` are "
        "what decide whether an artefact is kept, and this is the second "
        "opinion recorded beside them.",
    )
    judge_model: Mapped[str | None] = mapped_column(
        Text,
        comment="The model that judged this, as ASSESSMENT_JUDGE_MODEL named "
        "it. Recorded per row because a judge's answers move between models "
        "the way a writer's do.",
    )
    prompt_version: Mapped[str | None] = mapped_column(
        Text,
        comment="Version of the judging templates, as "
        "assessment.templates.PROMPT_VERSION declares it. Two template "
        "versions are two judgements, the same way two extraction prompts "
        "are two datasets.",
    )
    settings_version: Mapped[str | None] = mapped_column(
        Text,
        index=True,
        comment="The configuration this judgement was made under, as the "
        "digest settings.store computes, or 'environment' when nothing was "
        "overridden.",
    )
    run_id: Mapped[str | None] = mapped_column(
        Text,
        index=True,
        comment="Which run judged this row, as `settings.runs.run_id` named "
        "it. Two runs of the judge over one corpus compare on this.",
    )
    trace_id: Mapped[str | None] = mapped_column(
        Text,
        comment="The OpenTelemetry trace this judgement was made in, as 32 "
        "hex characters. Phoenix resolves a trace from this alone.",
    )
    span_id: Mapped[str | None] = mapped_column(
        Text,
        comment="The span the judge ran in, as 16 hex characters, which is "
        "the span this row's annotations hang off in Phoenix's Evaluations "
        "view.",
    )
    assessed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        comment="When the verdict was reached. NULL exactly when `approved` is.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="When the row was enrolled, which is when `start` first saw "
        "the artefact.",
    )

    metrics: Mapped[list[AssessmentMetric]] = relationship(
        back_populates="assessment", cascade="all, delete-orphan", passive_deletes=True
    )


class AssessmentMetric(Base):
    """One judgement one judge made about one artefact.

    The name, the labels and the scores are arize-phoenix-evals', so what
    Phoenix's Evaluations view shows under `hallucination` here means what
    it means in any other project that posts one. The templates are this
    repository's own - see `assessment/templates.py` for why - but a metric
    a reader already knows must not be renamed.
    """

    __tablename__ = "assessment_metrics"
    __table_args__ = (
        # One answer per metric per artefact. A re-run overwrites rather than
        # appending, so a row cannot hold two verdicts from two runs.
        UniqueConstraint(
            "assessment_id", "metric", name="assessment_metrics_one_per_metric"
        ),
        CheckConstraint(
            one_of("metric", JudgeMetric), name="assessment_metrics_metric_valid"
        ),
        CheckConstraint(
            "score >= 0 AND score <= 1", name="assessment_metrics_score_in_range"
        ),
        # The report counts each metric's approvals over the whole corpus.
        Index("ix_assessment_metrics_metric", "metric"),
        {
            "comment": "One metric of one assessment: what the judge called "
            "it, the score that label carries, and why. The names, the labels "
            "and the scores are arize-phoenix-evals', so a reader who knows "
            "`hallucination` there knows what this column holds."
        },
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    assessment_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("assessments.id", ondelete="CASCADE"),
        index=True,
        comment="The assessment this metric belongs to.",
    )
    metric: Mapped[str] = mapped_column(
        Text,
        comment="hallucination | relevance | qa_correctness | summarization. "
        "Spelled as arize-phoenix-evals spells it.",
    )
    label: Mapped[str] = mapped_column(
        Text,
        comment="What the judge answered, as one of that metric's own labels "
        "- `factual` or `hallucinated`, `relevant` or `unrelated`, and so on. "
        "The word rather than the number, because that is what Phoenix's "
        "Evaluations view groups by.",
    )
    score: Mapped[float] = mapped_column(
        Float,
        comment="The score that label carries, as phoenix-evals scores it. "
        "Note the direction is the metric's: `hallucination` scores 1.0 for "
        "`hallucinated` because its optimisation direction is minimise, and "
        "every other metric here scores 1.0 for its good label.",
    )
    approved: Mapped[bool] = mapped_column(
        Boolean,
        comment="Whether this label is the good side of this metric. Stored "
        "rather than derived from the score, because the score's direction "
        "differs per metric and a reader summing scores across metrics would "
        "be adding a hallucination rate to a relevance rate.",
    )
    explanation: Mapped[str] = mapped_column(
        Text,
        server_default="",
        comment="Why the judge said so, in its own words. Asked for in the "
        "same call as the label: an unexplained verdict is one a reviewer "
        "cannot act on, and the queue this feeds is for people.",
    )

    assessment: Mapped[Assessment] = relationship(back_populates="metrics")
