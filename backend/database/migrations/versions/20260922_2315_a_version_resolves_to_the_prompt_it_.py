"""A version resolves to the prompt it named.

`facts.prompt_version` and the `questions.prompt_version` added one
revision ago both name a version, and nothing in the deployment resolved
one. The prompt itself was in two places, neither of them durable beside
the row: in a git commit, and on the span of the call that sent it - and a
Phoenix project is one run with a retention of its own, while `questions`
is append-only and never hard-deleted. So a question written under version
5 outlived every record of what version 5 asked for, and an exported
dataset could not say what wrote it without the repository beside it.

The text is stored COMPOSED. A question type's system prompt is the shared
rules plus that type's directive and worked examples, assembled at call
time; the pieces are in the source and what the model was given is here.

Written by the stages at start-up, each its own and no other's. A row is a
record, not a place to edit: nothing reads it back into a call, and the
next start-up writes the source's text over it.

Revision ID: 17536718f959
Revises: 91f74559da2d
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from database.qa_generator.archived_rows import (
    archive_trigger,
    drop_archive_trigger,
)

revision: str = "17536718f959"
down_revision: str | None = "91f74559da2d"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Applies the change."""
    op.create_table(
        "prompts",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column(
            "service",
            sa.Text(),
            nullable=False,
            comment="Which stage sends it: extraction or questions. The stage writes its own and no other's, because no backend service imports another.",
        ),
        sa.Column(
            "name",
            sa.Text(),
            nullable=False,
            comment="What the prompt is, within its service: a question type, `perturb`, `follow`, one of the phrasing judgements, or one of the verifier's four.",
        ),
        sa.Column(
            "version",
            sa.Text(),
            nullable=False,
            comment="The PROMPT_VERSION its module declared when this text was written. Indexed: resolving a question's prompt_version is the query this table is for.",
        ),
        sa.Column(
            "text",
            sa.Text(),
            nullable=False,
            comment="The prompt as composed, which is what the model was given.",
        ),
        sa.Column(
            "digest",
            sa.Text(),
            nullable=False,
            comment="The first sixteen hex characters of the text's SHA-256, spelled as tests/static/test_prompts_pinned.py spells it, so a row and that file's pin can be compared by eye. What a changed prompt under an unchanged version is noticed by at run time.",
        ),
        sa.Column(
            "first_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
            comment="When this version of this prompt was first recorded. Not updated when the text is rewritten under the same version: what that would record is the last restart, and the warning is where a rewrite is reported.",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
            comment="When the text was last written. Moves only when the text actually changed, which under an unchanged version is drift.",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "service", "version", "name", name="prompts_service_version_name_unique"
        ),
        comment="Every prompt this pipeline sends, as it was composed, keyed by the PROMPT_VERSION that composed it. What facts.prompt_version and questions.prompt_version point at. Written at stage start-up from the source; read-only to everything else, and editing a row changes nothing about what is sent.",
    )
    op.create_index(op.f("ix_prompts_service"), "prompts", ["service"], unique=False)
    op.create_index(op.f("ix_prompts_version"), "prompts", ["version"], unique=False)
    # Every table archives its deletions, and this one has the most to lose
    # by not: nothing in the application deletes a prompt, and the row for a
    # version the source has moved past is the only copy of what that
    # version asked for. Autogenerate sees tables and columns and never a
    # trigger, so it is executed here.
    op.execute(archive_trigger("prompts"))


def downgrade() -> None:
    """Takes it back out."""
    op.execute(drop_archive_trigger("prompts"))
    op.drop_index(op.f("ix_prompts_version"), table_name="prompts")
    op.drop_index(op.f("ix_prompts_service"), table_name="prompts")
    op.drop_table("prompts")
