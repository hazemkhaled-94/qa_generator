"""Refresh the column comments the `new` status changed.

The comments on the three status columns and on facts.evidence_end still
described the schema as it was before rows arrived `new` and before the
evidence span was documented as half-open. No data changes.

Revision: 5ada5ee19fce
Parent:   e68b49264a95
Created:  2026-09-12 11:40:14.253040
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "5ada5ee19fce"
down_revision: str | None = "e68b49264a95"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Applies the change."""
    op.alter_column(
        "documents",
        "parse_status",
        existing_type=sa.TEXT(),
        comment="new | pending | in_progress | parsed | failed. The queue the parsing service selects on; in_progress is how a worker claims a document so a second worker skips it. A worker that dies leaves the row claimed, and the next run fails it with a reason and retry returns it to pending.",
        existing_comment="pending | in_progress | parsed | failed. The queue the parsing service selects on; in_progress is how a worker claims a document so a second worker skips it. A worker that dies leaves the row claimed, and the next run fails it with a reason and retry returns it to pending.",
        existing_nullable=False,
        existing_server_default=sa.text("'new'::text"),
    )
    op.alter_column(
        "documents",
        "chunk_status",
        existing_type=sa.TEXT(),
        comment="new | pending | in_progress | chunked | failed. The queue the chunking service selects on, and the only column it reads to find work: it does not consult parse_status, because a stage knows of no other stage. Starting chunking on a document that was never parsed fails that document, which is the orchestrator's mistake to avoid. Mirrors parse_status in every other respect, including how a dead worker's claim becomes a failure the next run can retry. Extraction has no column here: it queues over passages, because a document is a unit of work that divides very unevenly.",
        existing_comment="pending | in_progress | chunked | failed. The queue the chunking service selects on, together with parse_status = 'parsed': a document cannot be split until it has been read. Mirrors parse_status in every respect, including how a dead worker's claim becomes a failure the next run can retry. Extraction has no column here: it queues over passages, because a document is a unit of work that divides very unevenly.",
        existing_nullable=False,
        existing_server_default=sa.text("'new'::text"),
    )
    op.alter_column(
        "documents",
        "dropped_short",
        existing_type=sa.INTEGER(),
        comment='Chunks the last run discarded for falling under CHUNKING_MIN_CHARS. Stored rather than logged, because it is content that reached no passage and no fact: "which documents lost content to the size bounds?" is otherwise unanswerable from the database.',
        existing_nullable=True,
    )
    op.alter_column(
        "documents",
        "dropped_long",
        existing_type=sa.INTEGER(),
        comment="Chunks the last run discarded for exceeding CHUNKING_MAX_CHARS, typically one large unsplittable table. Counted for the same reason as dropped_short.",
        existing_nullable=True,
    )
    op.alter_column(
        "facts",
        "evidence_end",
        existing_type=sa.INTEGER(),
        comment="Offset one past the evidence's last character, relative to passages.text, so the span is text[evidence_start:evidence_end] and its length is the difference. Present so that evidence appearing twice in one passage is still unambiguous.",
        existing_comment="Offset of the evidence's last character, relative to passages.text. Present so that evidence appearing twice in one passage is still unambiguous.",
        existing_nullable=False,
    )
    op.alter_column(
        "passages",
        "extract_status",
        existing_type=sa.TEXT(),
        comment="new | pending | in_progress | extracted | failed. The extraction queue runs over passages rather than documents: one document holds four hundred passages and another thirty, so claiming a document hands one worker hours of work and its neighbour minutes. A passage is one model call, which makes it the unit that divides evenly.",
        existing_comment="pending | in_progress | extracted | failed. The extraction queue runs over passages rather than documents: one document holds four hundred passages and another thirty, so claiming a document hands one worker hours of work and its neighbour minutes. A passage is one model call, which makes it the unit that divides evenly.",
        existing_nullable=False,
        existing_server_default=sa.text("'new'::text"),
    )
    op.alter_column(
        "passages",
        "table_cells",
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        comment="The cell grid behind a table passage. A list of grids, since one passage can hold more than one table, each {caption, num_rows, num_cols, cells:[{row, col, row_span, col_span, column_header, row_header, text, line}]}. NULL unless block_type is table. `line` is the rendered Markdown row of passages.text the cell sits in, and the evidence any fact drawn from that cell quotes; it is NULL on a header cell, which yields no fact. Only the rows this passage renders are stored: a split table points every piece at the whole item, and a row this piece does not show would describe a value with a row the reader cannot see.",
        existing_comment='The cell grid behind a table passage, and the only column in this schema that is not flat. It is here because text alone cannot carry a table: serialising one to a string loses which cells are headers, which row and column a value sits at, and which cells span several. Without that, the deterministic cell reader has no input and every fee table would have to go to the model. A list of grids, since one passage can hold more than one table, each {caption, num_rows, num_cols, cells:[{row, col, row_span, col_span, column_header, row_header, text}]}. The two header flags stay apart because a corner cell heads its column and not its row. The caption is repeated here although it is already in the text, because it is what scopes the cells - a fee is only meaningful once "Fees effective 2024" is attached to it. NULL unless block_type is table.',
        existing_nullable=True,
    )


def downgrade() -> None:
    """Takes it back out."""
    op.alter_column(
        "passages",
        "table_cells",
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        comment='The cell grid behind a table passage, and the only column in this schema that is not flat. It is here because text alone cannot carry a table: serialising one to a string loses which cells are headers, which row and column a value sits at, and which cells span several. Without that, the deterministic cell reader has no input and every fee table would have to go to the model. A list of grids, since one passage can hold more than one table, each {caption, num_rows, num_cols, cells:[{row, col, row_span, col_span, column_header, row_header, text}]}. The two header flags stay apart because a corner cell heads its column and not its row. The caption is repeated here although it is already in the text, because it is what scopes the cells - a fee is only meaningful once "Fees effective 2024" is attached to it. NULL unless block_type is table.',
        existing_comment="The cell grid behind a table passage. A list of grids, since one passage can hold more than one table, each {caption, num_rows, num_cols, cells:[{row, col, row_span, col_span, column_header, row_header, text, line}]}. NULL unless block_type is table. `line` is the rendered Markdown row of passages.text the cell sits in, and the evidence any fact drawn from that cell quotes; it is NULL on a header cell, which yields no fact. Only the rows this passage renders are stored: a split table points every piece at the whole item, and a row this piece does not show would describe a value with a row the reader cannot see.",
        existing_nullable=True,
    )
    op.alter_column(
        "passages",
        "extract_status",
        existing_type=sa.TEXT(),
        comment="pending | in_progress | extracted | failed. The extraction queue runs over passages rather than documents: one document holds four hundred passages and another thirty, so claiming a document hands one worker hours of work and its neighbour minutes. A passage is one model call, which makes it the unit that divides evenly.",
        existing_comment="new | pending | in_progress | extracted | failed. The extraction queue runs over passages rather than documents: one document holds four hundred passages and another thirty, so claiming a document hands one worker hours of work and its neighbour minutes. A passage is one model call, which makes it the unit that divides evenly.",
        existing_nullable=False,
        existing_server_default=sa.text("'new'::text"),
    )
    op.alter_column(
        "facts",
        "evidence_end",
        existing_type=sa.INTEGER(),
        comment="Offset of the evidence's last character, relative to passages.text. Present so that evidence appearing twice in one passage is still unambiguous.",
        existing_comment="Offset one past the evidence's last character, relative to passages.text, so the span is text[evidence_start:evidence_end] and its length is the difference. Present so that evidence appearing twice in one passage is still unambiguous.",
        existing_nullable=False,
    )
    op.alter_column(
        "documents",
        "dropped_long",
        existing_type=sa.INTEGER(),
        comment=None,
        existing_comment="Chunks the last run discarded for exceeding CHUNKING_MAX_CHARS, typically one large unsplittable table. Counted for the same reason as dropped_short.",
        existing_nullable=True,
    )
    op.alter_column(
        "documents",
        "dropped_short",
        existing_type=sa.INTEGER(),
        comment=None,
        existing_comment='Chunks the last run discarded for falling under CHUNKING_MIN_CHARS. Stored rather than logged, because it is content that reached no passage and no fact: "which documents lost content to the size bounds?" is otherwise unanswerable from the database.',
        existing_nullable=True,
    )
    op.alter_column(
        "documents",
        "chunk_status",
        existing_type=sa.TEXT(),
        comment="pending | in_progress | chunked | failed. The queue the chunking service selects on, together with parse_status = 'parsed': a document cannot be split until it has been read. Mirrors parse_status in every respect, including how a dead worker's claim becomes a failure the next run can retry. Extraction has no column here: it queues over passages, because a document is a unit of work that divides very unevenly.",
        existing_comment="new | pending | in_progress | chunked | failed. The queue the chunking service selects on, and the only column it reads to find work: it does not consult parse_status, because a stage knows of no other stage. Starting chunking on a document that was never parsed fails that document, which is the orchestrator's mistake to avoid. Mirrors parse_status in every other respect, including how a dead worker's claim becomes a failure the next run can retry. Extraction has no column here: it queues over passages, because a document is a unit of work that divides very unevenly.",
        existing_nullable=False,
        existing_server_default=sa.text("'new'::text"),
    )
    op.alter_column(
        "documents",
        "parse_status",
        existing_type=sa.TEXT(),
        comment="pending | in_progress | parsed | failed. The queue the parsing service selects on; in_progress is how a worker claims a document so a second worker skips it. A worker that dies leaves the row claimed, and the next run fails it with a reason and retry returns it to pending.",
        existing_comment="new | pending | in_progress | parsed | failed. The queue the parsing service selects on; in_progress is how a worker claims a document so a second worker skips it. A worker that dies leaves the row claimed, and the next run fails it with a reason and retry returns it to pending.",
        existing_nullable=False,
        existing_server_default=sa.text("'new'::text"),
    )
