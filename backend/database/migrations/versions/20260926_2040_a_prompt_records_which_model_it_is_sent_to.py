"""A prompt records which model it is sent to.

A stage publishes each of its prompts to Phoenix against the model that
prompt actually goes to, because a published prompt is replayed in the
playground against the model named on it. One service can send to several:
question generation writes with QUESTIONS_MODEL, judges wording with
QUESTIONS_PHRASING_MODEL and checks with QUESTIONS_VERIFIER_MODEL, and two
of those are usually different providers.

That attribution lived only in `Composed.model`, in the process that
composed it, and `make prompts-publish` reads this table. So a republish
sent every row against LLM_MODEL and overwrote what the stage got right -
Phoenix keeps the version created last - which made the repair for a wiped
Phoenix cost the thing it was repairing.

NULL means the stage's own model, which is what every row already recorded
means: absent, `Settings.overridden` returns the settings unchanged.

No index. It is read one row at a time on the way out of `stored`.

Revision ID: d3f8a1c47b52
Revises: c5a71e83b94f
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "d3f8a1c47b52"
down_revision: str | None = "c5a71e83b94f"
branch_labels: str | None = None
depends_on: str | None = None

#: What the column holds. Word for word what `database/qa_generator/prompts.py`
#: declares: `tests/integration/database/test_schema.py` compares the two.
_COMMENT = (
    "Which model this prompt is sent to, where the stage names one "
    "instead of the shared LLM_MODEL. A name and not a setting: the publisher "
    "resolves the rest through llm.config.Settings.overridden. NULL means the "
    "stage's own model, which is what a row recorded before this column "
    "existed means too."
)


def upgrade() -> None:
    """Applies the change."""
    op.add_column(
        "prompts", sa.Column("model", sa.Text(), nullable=True, comment=_COMMENT)
    )


def downgrade() -> None:
    """Takes it back out."""
    op.drop_column("prompts", "model")
