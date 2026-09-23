"""A question links to the trace that produced it.

A stored question and the span that judged it had nothing in common. The
span carries no question id - the question has none yet when the gates
run, because it is written afterwards - and the row carried no span. So a
bad question in the Questions page could not be opened as the calls that
produced it, and a gate verdict in Phoenix could not be opened as the row
it judged. Neither direction existed.

Two columns, because they answer different questions. The trace is the
whole topic - the writer call, the gates, the verifier. The span is the
gate decision itself, and it is the one the verdict annotations hang off.

Phoenix resolves either from the hex alone, through `getSpanByOtelId` and
`getTraceByOtelId`, which is what its `/redirects/spans/` and
`/redirects/traces/` routes are built on. So a link needs nothing but
these and the address Phoenix is served at - no project id, and nothing
here that breaks when Phoenix renumbers its own.

NULL for every question written before this, and nothing is backfilled:
there is no key to backfill on, which is the whole reason for the
columns.

Revision ID: 175d0f5da371
Revises: 17536718f959
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "175d0f5da371"
down_revision: str | None = "17536718f959"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Applies the change."""
    op.add_column(
        "questions",
        sa.Column(
            "trace_id",
            sa.Text(),
            nullable=True,
            comment="The OpenTelemetry trace this question was written and judged in, as 32 hex characters. What gets from a row to the calls that produced it: Phoenix resolves a trace from this alone, so <phoenix>/redirects/traces/<trace_id> opens it without anything knowing Phoenix's internal ids. NULL for a question written before this column, and for one written by a process exporting no spans.",
        ),
    )
    op.add_column(
        "questions",
        sa.Column(
            "span_id",
            sa.Text(),
            nullable=True,
            comment="The span the gates ran in, as 16 hex characters, which is the span this question's verdict annotations hang off. Stored beside the trace because it is the more useful of the two: /redirects/spans/ opens the gate decision itself rather than the whole topic. The span cannot carry the question id instead - the question has none yet when the gates run.",
        ),
    )
    op.create_index(
        op.f("ix_questions_span_id"), "questions", ["span_id"], unique=False
    )
    op.create_index(
        op.f("ix_questions_trace_id"), "questions", ["trace_id"], unique=False
    )


def downgrade() -> None:
    """Takes it back out."""
    op.drop_index(op.f("ix_questions_trace_id"), table_name="questions")
    op.drop_index(op.f("ix_questions_span_id"), table_name="questions")
    op.drop_column("questions", "span_id")
    op.drop_column("questions", "trace_id")
