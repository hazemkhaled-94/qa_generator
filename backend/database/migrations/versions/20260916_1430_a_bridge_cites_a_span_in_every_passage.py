"""A bridge cites a span in every passage it rests on.

A bridge was stored with one citation: `evidence_start = 0`,
`evidence_end = len(anchor.text)` and every sentence of the anchor, because
the prompt returned excerpt numbers and no sentence numbers. So the evidence
on a bridge was truthful about one of its passages and silent about the rest,
and a question written from one could be traced no further than "somewhere in
this passage".

Prompt version 2 numbers the sentences inside each excerpt and asks which of
them the claim rests on, per passage. Those citations land here, one row per
passage, beside the position that was already recorded.

The three columns are nullable rather than filled in, because nothing can
fill them in: a bridge drawn under prompt version 1 recorded no sentence
numbers, and there is no way back to them without asking the model again.
Such a row keeps its position and carries no span, which question generation
reads as "not citable" and skips. `make extract-bridge` replaces them when
somebody chooses to pay for it; until then they sit exactly as they are.

Revision: 8d2c47b1e6f9
Parent:   6b4e91a0c37d
Created:  2026-09-16
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from database.qa_generator.fact_passages import CITATION_COMPLETE

revision: str = "8d2c47b1e6f9"
down_revision: str | None = "6b4e91a0c37d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CHECK = "fact_passages_citation_complete"


def upgrade() -> None:
    """Applies the change."""
    op.add_column(
        "fact_passages",
        sa.Column(
            "sentence_ids",
            sa.ARRAY(sa.Integer()),
            nullable=True,
            comment="Which of this passage's sentences the claim rests on. NULL on "
            "a bridge drawn before prompt version 2, which cited a passage without "
            "saying where in it; such a bridge is not offered to question "
            "generation.",
        ),
    )
    op.add_column(
        "fact_passages",
        sa.Column(
            "evidence_start",
            sa.Integer(),
            nullable=True,
            comment="Offset of the cited span's first character within this "
            "passage's text. NULL alongside sentence_ids.",
        ),
    )
    op.add_column(
        "fact_passages",
        sa.Column(
            "evidence_end",
            sa.Integer(),
            nullable=True,
            comment="Offset one past its last character.",
        ),
    )
    op.create_check_constraint(_CHECK, "fact_passages", CITATION_COMPLETE)
    op.alter_column(
        "fact_passages",
        "passage_id",
        existing_type=sa.BigInteger(),
        comment="A passage the claim rests on. Deleting the passage deletes this "
        "row, and the fact with it when the passage is the anchor.",
    )
    op.execute(
        "COMMENT ON TABLE fact_passages IS 'Which passages a bridge fact rests "
        "on, one row per passage including the anchor in facts.passage_id, and "
        "where in each it rests. Empty for every other kind, which rests on that "
        "anchor alone.'"
    )


def downgrade() -> None:
    """Takes it back out."""
    op.execute(
        "COMMENT ON TABLE fact_passages IS 'Which passages a bridge fact rests "
        "on, one row per passage including the anchor in facts.passage_id. Empty "
        "for every other kind, which rests on that anchor alone.'"
    )
    op.drop_constraint(_CHECK, "fact_passages", type_="check")
    op.drop_column("fact_passages", "evidence_end")
    op.drop_column("fact_passages", "evidence_start")
    op.drop_column("fact_passages", "sentence_ids")
