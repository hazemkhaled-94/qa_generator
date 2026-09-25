"""Every artefact links to the trace that produced it.

`questions` and `assessments` carried `trace_id` and `span_id`; a document,
a passage, a fact and a topic carried neither. Adds them, and adds the run
to the two tables that named none.

Parsing's columns on `documents` are prefixed like `parse_status` and
`parse_error` beside them, because chunking owns the other half of that
row. Chunking's provenance is on `passages`, which is what chunking
produces.

NULL for every row written before this. Nothing is backfilled: there is no
key to backfill on.

Revision ID: 4d7e2b9a15c3
Revises: 8b3d5e1f0c74
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "4d7e2b9a15c3"
down_revision: str | None = "8b3d5e1f0c74"
branch_labels: str | None = None
depends_on: str | None = None

#: Table, column, and what the column holds.
_COLUMNS = (
    (
        "documents",
        "parse_run_id",
        (
            "Which run parsed this document, as `settings.runs.run_id` named it. "
            "NULL for a document parsed before the run was recorded."
        ),
    ),
    (
        "documents",
        "parse_trace_id",
        (
            "The OpenTelemetry trace this document was parsed in, as 32 hex "
            "characters. <phoenix>/redirects/traces/<trace_id> opens it."
        ),
    ),
    (
        "documents",
        "parse_span_id",
        (
            "The span the parse ran in, as 16 hex characters. "
            "<phoenix>/redirects/spans/<span_id> opens it."
        ),
    ),
    (
        "passages",
        "run_id",
        (
            "Which run chunked this passage out of its document, as "
            "`settings.runs.run_id` named it."
        ),
    ),
    (
        "passages",
        "trace_id",
        ("The OpenTelemetry trace this passage was chunked in, as 32 hex characters."),
    ),
    (
        "passages",
        "span_id",
        (
            "The span the chunking ran in, as 16 hex characters. One span per "
            "document, so every passage of one document shares it."
        ),
    ),
    (
        "facts",
        "trace_id",
        (
            "The OpenTelemetry trace this fact was extracted and checked in, as "
            "32 hex characters."
        ),
    ),
    (
        "facts",
        "span_id",
        (
            "The span the extraction ran in, as 16 hex characters. One span per "
            "passage, so every fact from one passage shares it."
        ),
    ),
    (
        "topics",
        "trace_id",
        (
            "The OpenTelemetry trace this topic was fitted and labelled in, as "
            "32 hex characters."
        ),
    ),
    (
        "topics",
        "span_id",
        (
            "The span the fit ran in, as 16 hex characters. One span per "
            "language, so every topic of one language shares it."
        ),
    ),
)


def upgrade() -> None:
    """Applies the change."""
    for table, column, comment in _COLUMNS:
        op.add_column(
            table, sa.Column(column, sa.Text(), nullable=True, comment=comment)
        )
        op.create_index(op.f(f"ix_{table}_{column}"), table, [column], unique=False)


def downgrade() -> None:
    """Takes it back out."""
    for table, column, _ in reversed(_COLUMNS):
        op.drop_index(op.f(f"ix_{table}_{column}"), table_name=table)
        op.drop_column(table, column)
