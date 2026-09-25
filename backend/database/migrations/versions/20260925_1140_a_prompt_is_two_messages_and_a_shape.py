"""A prompt is two messages and a shape.

Every call this pipeline makes sends a SYSTEM message and a USER message,
and asks for the answer in a declared Pydantic shape. The row recorded the
first of those three. So `prompts` said what a version instructed and not
what it was given to work on, and a prompt published to Phoenix arrived as
one system message with no input under it and no response format beside it.

`user_text` holds the second message as its TEMPLATE, with `{{name}}` where
the call substitutes a passage, a fact or a question. The template and not
one rendered instance: what the model was given is the row's own columns
put into this, and a record of one call's passages would be a record of one
call.

`response_schema` holds the JSON schema of the shape. NULL on the rows
already here, which is honest - no call ever asked for no shape, so NULL
means "recorded before this column existed" and never "asked for nothing".

Both are backfilled by the stages themselves at the next start-up, for the
versions the source still declares. A row under a version the code has
moved past keeps its empty template, because nothing can now say what that
version's user message was.

Revision ID: 2f6c04b98ae3
Revises: 3c81a9e4d7b2
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "2f6c04b98ae3"
down_revision: str | None = "3c81a9e4d7b2"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Applies the change."""
    op.add_column(
        "prompts",
        sa.Column(
            "user_text",
            sa.Text(),
            server_default="",
            nullable=False,
            comment="The USER message as its template, with `{{name}}` where the call fills a passage, a fact or a question in. Every call sends both halves; this is the one the row used not to carry, so Phoenix showed a system message and nothing under it. Empty on a row recorded before the column existed.",
        ),
    )
    op.add_column(
        "prompts",
        sa.Column(
            "response_schema",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="The JSON schema the answer had to come back in, from the Pydantic shape the call asks for. NULL on a row recorded before the column existed, which is not the same as a call that asked for no shape - there is no such call.",
        ),
    )
    op.alter_column(
        "prompts",
        "text",
        existing_type=sa.Text(),
        existing_nullable=False,
        comment="The SYSTEM prompt as composed, which is what the model was given as its instructions.",
        existing_comment="The prompt as composed, which is what the model was given.",
    )


def downgrade() -> None:
    """Takes it back out."""
    op.alter_column(
        "prompts",
        "text",
        existing_type=sa.Text(),
        existing_nullable=False,
        comment="The prompt as composed, which is what the model was given.",
        existing_comment="The SYSTEM prompt as composed, which is what the model was given as its instructions.",
    )
    op.drop_column("prompts", "response_schema")
    op.drop_column("prompts", "user_text")
