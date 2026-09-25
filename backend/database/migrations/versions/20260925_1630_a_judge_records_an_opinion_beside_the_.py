"""A judge records an opinion beside the checker's, and never over it.

The pipeline decides what it keeps twice: `facts.validated` is the
checker's verdict and `questions.status` is the gates'. Both are the
pipeline's own reading of its own work, and `evaluation/README.md` has
always said an LLM judge must not be a third - the German half of this
corpus is where one was measured at chance, and a judge that could reject
a row would have thrown away good German questions on that basis.

What was missing was anywhere to put the opinion. `make second-opinion`
asked a judge about one run's answers and printed the disagreements to a
terminal, so the verdict existed for as long as the scrollback did: no
column, no route, no page, no workbook, nothing in Argilla, and in Phoenix
only whatever the host command's own process happened to export.

So: two tables. `assessments` holds one row per artefact the judge was
asked about - a fact, a topic or a question - carrying the verdict and the
queue that asked for it. `assessment_metrics` holds one row per judgement
behind that verdict, named, labelled and scored the way
arize-phoenix-evals names, labels and scores its own, so a `hallucination`
here means in Phoenix what a `hallucination` anywhere else means.

Three foreign keys rather than a polymorphic (kind, id) pair, each
cascading. An assessment of a deleted fact is an opinion about nothing,
and this corpus deletes: removing a document cascades to its passages, its
facts and the questions resting on them, and a refit replaces every topic
there is. A polymorphic pair would have left orphans for a third trigger
to sweep.

`artifact_kind` is stored beside them anyway, because the queue narrows on
it and the report groups on it, and a CASE over three columns in every
such query is the version of this that cannot be indexed. Two CHECK
constraints keep it honest: exactly one key is set, and the kind says
which.

Revision ID: 8b3d5e1f0c74
Revises: 2f6c04b98ae3
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from database.qa_generator.archived_rows import archive_trigger, drop_archive_trigger

revision: str = "8b3d5e1f0c74"
down_revision: str | None = "2f6c04b98ae3"
branch_labels: str | None = None
depends_on: str | None = None

#: The states this stage may put a row in: the four every stage shares, and
#: its own done value.
_STATUSES = "'new', 'pending', 'in_progress', 'failed', 'assessed'"

#: The three artefacts a judge is asked about. A passage and a parsed
#: document are absent: judging a conversion needs the source page rendered
#: beside it, and nothing here keeps one.
_KINDS = "'fact', 'topic', 'question'"

#: The metric names, which are arize-phoenix-evals'.
_METRICS = "'hallucination', 'relevance', 'qa_correctness', 'summarization'"


def upgrade() -> None:
    """Applies the change."""
    op.create_table(
        "assessments",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column(
            "artifact_kind",
            sa.Text(),
            nullable=False,
            comment="fact | topic | question. Which of the three foreign keys below is set, stored so the queue can narrow on it and the report can group on it without a CASE over three columns.",
        ),
        sa.Column(
            "fact_id",
            sa.BigInteger(),
            nullable=True,
            comment="The fact this is about, NULL unless artifact_kind is `fact`. Cascades: a judgement about a deleted fact is an opinion about nothing.",
        ),
        sa.Column(
            "topic_id",
            sa.BigInteger(),
            nullable=True,
            comment="The topic this is about, NULL unless artifact_kind is `topic`. A refit replaces every topic, so this cascades on every `make topics-discover`.",
        ),
        sa.Column(
            "question_id",
            sa.BigInteger(),
            nullable=True,
            comment="The question this is about, NULL unless artifact_kind is `question`.",
        ),
        sa.Column(
            "assess_status",
            sa.Text(),
            server_default="new",
            nullable=False,
            comment="This stage's own queue column. A row is enrolled `new` by `start`, which is also what makes it claimable - nothing upstream creates one.",
        ),
        sa.Column(
            "assess_error",
            sa.Text(),
            nullable=True,
            comment="Why the judge could not be asked, NULL when it could.",
        ),
        sa.Column(
            "assess_claimed_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="When a worker took this row.",
        ),
        sa.Column(
            "approved",
            sa.Boolean(),
            nullable=True,
            comment="Whether every metric landed on its good side. NULL until the judge has answered, which is most rows until the phase runs. Never read by a gate: `facts.validated` and `questions.status` are what decide whether an artefact is kept, and this is the second opinion recorded beside them.",
        ),
        sa.Column(
            "judge_model",
            sa.Text(),
            nullable=True,
            comment="The model that judged this, as ASSESSMENT_JUDGE_MODEL named it. Recorded per row because a judge's answers move between models the way a writer's do.",
        ),
        sa.Column(
            "prompt_version",
            sa.Text(),
            nullable=True,
            comment="Version of the judging templates, as assessment.templates.PROMPT_VERSION declares it. Two template versions are two judgements, the same way two extraction prompts are two datasets.",
        ),
        sa.Column(
            "settings_version",
            sa.Text(),
            nullable=True,
            comment="The configuration this judgement was made under, as the digest settings.store computes, or 'environment' when nothing was overridden.",
        ),
        sa.Column(
            "run_id",
            sa.Text(),
            nullable=True,
            comment="Which run judged this row, as `settings.runs.run_id` named it. Two runs of the judge over one corpus compare on this.",
        ),
        sa.Column(
            "trace_id",
            sa.Text(),
            nullable=True,
            comment="The OpenTelemetry trace this judgement was made in, as 32 hex characters. Phoenix resolves a trace from this alone.",
        ),
        sa.Column(
            "span_id",
            sa.Text(),
            nullable=True,
            comment="The span the judge ran in, as 16 hex characters, which is the span this row's annotations hang off in Phoenix's Evaluations view.",
        ),
        sa.Column(
            "assessed_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="When the verdict was reached. NULL exactly when `approved` is.",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
            comment="When the row was enrolled, which is when `start` first saw the artefact.",
        ),
        sa.CheckConstraint(
            f"assess_status IN ({_STATUSES})", name="assessments_status_valid"
        ),
        sa.CheckConstraint(
            f"artifact_kind IN ({_KINDS})", name="assessments_kind_valid"
        ),
        sa.CheckConstraint(
            "(fact_id IS NOT NULL)::int + (topic_id IS NOT NULL)::int + "
            "(question_id IS NOT NULL)::int = 1",
            name="assessments_one_artifact",
        ),
        sa.CheckConstraint(
            "(artifact_kind = 'fact') = (fact_id IS NOT NULL) AND "
            "(artifact_kind = 'topic') = (topic_id IS NOT NULL) AND "
            "(artifact_kind = 'question') = (question_id IS NOT NULL)",
            name="assessments_kind_agrees",
        ),
        sa.CheckConstraint(
            "(approved IS NULL) = (assessed_at IS NULL)",
            name="assessments_verdict_together",
        ),
        sa.ForeignKeyConstraint(["fact_id"], ["facts.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["topic_id"], ["topics.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["question_id"], ["questions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("fact_id", name="assessments_one_per_fact"),
        sa.UniqueConstraint("topic_id", name="assessments_one_per_topic"),
        sa.UniqueConstraint("question_id", name="assessments_one_per_question"),
        comment="What an LLM judge made of one fact, topic or question, and the queue that asked it. Never a gate: the checker and the gates decide what is kept, and this is recorded beside their verdict so the two can be compared. One metric row per judgement is in assessment_metrics.",
    )
    op.create_index(
        "ix_assessments_queue",
        "assessments",
        ["assess_status"],
        postgresql_where=sa.text("assess_status IN ('pending', 'in_progress')"),
    )
    op.create_index("ix_assessments_kind", "assessments", ["artifact_kind"])
    op.create_index(
        "ix_assessments_approved",
        "assessments",
        ["approved"],
        postgresql_where=sa.text("approved IS NOT NULL"),
    )
    op.create_index(
        op.f("ix_assessments_run_id"), "assessments", ["run_id"], unique=False
    )
    op.create_index(
        op.f("ix_assessments_settings_version"),
        "assessments",
        ["settings_version"],
        unique=False,
    )

    op.create_table(
        "assessment_metrics",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column(
            "assessment_id",
            sa.BigInteger(),
            nullable=False,
            comment="The assessment this metric belongs to.",
        ),
        sa.Column(
            "metric",
            sa.Text(),
            nullable=False,
            comment="hallucination | relevance | qa_correctness | summarization. Spelled as arize-phoenix-evals spells it.",
        ),
        sa.Column(
            "label",
            sa.Text(),
            nullable=False,
            comment="What the judge answered, as one of that metric's own labels - `factual` or `hallucinated`, `relevant` or `unrelated`, and so on. The word rather than the number, because that is what Phoenix's Evaluations view groups by.",
        ),
        sa.Column(
            "score",
            sa.Float(),
            nullable=False,
            comment="The score that label carries, as phoenix-evals scores it. Note the direction is the metric's: `hallucination` scores 1.0 for `hallucinated` because its optimisation direction is minimise, and every other metric here scores 1.0 for its good label.",
        ),
        sa.Column(
            "approved",
            sa.Boolean(),
            nullable=False,
            comment="Whether this label is the good side of this metric. Stored rather than derived from the score, because the score's direction differs per metric and a reader summing scores across metrics would be adding a hallucination rate to a relevance rate.",
        ),
        sa.Column(
            "explanation",
            sa.Text(),
            server_default="",
            nullable=False,
            comment="Why the judge said so, in its own words. Asked for in the same call as the label: an unexplained verdict is one a reviewer cannot act on, and the queue this feeds is for people.",
        ),
        sa.CheckConstraint(
            f"metric IN ({_METRICS})", name="assessment_metrics_metric_valid"
        ),
        sa.CheckConstraint(
            "score >= 0 AND score <= 1", name="assessment_metrics_score_in_range"
        ),
        sa.ForeignKeyConstraint(
            ["assessment_id"], ["assessments.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "assessment_id", "metric", name="assessment_metrics_one_per_metric"
        ),
        comment="One metric of one assessment: what the judge called it, the score that label carries, and why. The names, the labels and the scores are arize-phoenix-evals', so a reader who knows `hallucination` there knows what this column holds.",
    )
    op.create_index(
        op.f("ix_assessment_metrics_assessment_id"),
        "assessment_metrics",
        ["assessment_id"],
        unique=False,
    )
    op.create_index(
        "ix_assessment_metrics_metric", "assessment_metrics", ["metric"]
    )

    # Every table in this schema archives its deletions, and these two are
    # the tables most likely to be deleted from without anybody typing a
    # deletion: all three foreign keys cascade, so removing a document
    # takes the judgements of its facts and of the questions resting on
    # them, and `topics-discover` takes every topic's. What the judge said
    # about a corpus is exactly the sort of thing to want back afterwards.
    for table in ("assessments", "assessment_metrics"):
        op.execute(archive_trigger(table))


def downgrade() -> None:
    """Takes it back out."""
    for table in ("assessments", "assessment_metrics"):
        op.execute(drop_archive_trigger(table))
    op.drop_index("ix_assessment_metrics_metric", table_name="assessment_metrics")
    op.drop_index(
        op.f("ix_assessment_metrics_assessment_id"), table_name="assessment_metrics"
    )
    op.drop_table("assessment_metrics")
    op.drop_index(op.f("ix_assessments_settings_version"), table_name="assessments")
    op.drop_index(op.f("ix_assessments_run_id"), table_name="assessments")
    op.drop_index("ix_assessments_approved", table_name="assessments")
    op.drop_index("ix_assessments_kind", table_name="assessments")
    op.drop_index("ix_assessments_queue", table_name="assessments")
    op.drop_table("assessments")
