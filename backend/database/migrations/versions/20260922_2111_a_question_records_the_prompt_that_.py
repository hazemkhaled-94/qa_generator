"""A question records the prompt that wrote it.

`facts.prompt_version` has existed since the initial schema, for the reason
`extraction/extractors/llm.py` states: the prompt decides what counts as a
fact, so two prompts are two datasets. Questions had no equivalent, and
`question_generation.types.PROMPT_VERSION` - eight versions in - was
declared, pinned by a static test, imported once and recorded nowhere. Two
runs, one under prompt 7 and one under prompt 8, were the same rows.

`settings_version` does not cover it. That is a digest of the STORED
OVERRIDES, and editing a prompt in the source moves neither it nor any
setting it names.

NULL for everything written before this, and nothing is backfilled:
inventing a version for rows whose prompt is not known would make two
datasets look like one, which is the thing the column exists to prevent.

Revision ID: 91f74559da2d
Revises: 9c04a7f1e6b2
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "91f74559da2d"
down_revision: str | None = "9c04a7f1e6b2"
branch_labels: str | None = None
depends_on: str | None = None

COMMENT = (
    "Version of the prompt that wrote this question, as "
    "question_generation.types.PROMPT_VERSION declares it. The prompt decides "
    "what a question IS, so two prompts are two datasets - the same reason "
    "facts.prompt_version exists. Not covered by settings_version, which is a "
    "digest of the stored overrides and does not move when a prompt is "
    "edited. NULL for a question written before the version was recorded."
)

#: The sentence dropped from `settings_version`, which used to call itself
#: the only provenance a question carries. It is not, now.
SETTINGS_WAS = (
    "The configuration this question was written under, as the digest "
    "settings.store computes, or 'environment' when nothing was overridden. "
    "The only provenance a question carries: which model wrote it, what mix "
    "the plan aimed for and what bounds the gates held it to are all in the "
    "settings that version names. NULL for a question written before a "
    "setting could be changed without a restart."
)

SETTINGS_IS = (
    "The configuration this question was written under, as the digest "
    "settings.store computes, or 'environment' when nothing was overridden. "
    "Which model wrote it, what mix the plan aimed for and what bounds the "
    "gates held it to are all in the settings that version names. NULL for a "
    "question written before a setting could be changed without a restart."
)


def upgrade() -> None:
    """Applies the change."""
    op.add_column(
        "questions",
        sa.Column("prompt_version", sa.Text(), nullable=True, comment=COMMENT),
    )
    # Indexed for the same reason run_id is: what this column is for is
    # `WHERE prompt_version = '8'` against a table with a run's worth of
    # rows in it.
    op.create_index(
        op.f("ix_questions_prompt_version"),
        "questions",
        ["prompt_version"],
        unique=False,
    )
    op.alter_column(
        "questions",
        "settings_version",
        existing_type=sa.TEXT(),
        comment=SETTINGS_IS,
        existing_comment=SETTINGS_WAS,
        existing_nullable=True,
    )


def downgrade() -> None:
    """Takes it back out."""
    op.alter_column(
        "questions",
        "settings_version",
        existing_type=sa.TEXT(),
        comment=SETTINGS_WAS,
        existing_comment=SETTINGS_IS,
        existing_nullable=True,
    )
    op.drop_index(op.f("ix_questions_prompt_version"), table_name="questions")
    op.drop_column("questions", "prompt_version")
