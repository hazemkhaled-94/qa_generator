"""Four kinds of fact, and the passages a bridge rests on.

`facts.kind` joins the three new readings to the atomic one. A summary and an
outline stand in for one passage, as prose and as bullets; a bridge carries a
claim no single passage states. Every fact that exists now is atomic, which
is what the server default backfills.

`fact_passages` holds the passages a bridge rests on, the anchor in
facts.passage_id included. It is empty for every other kind. A trigger
deletes a bridge that falls below two passages, because re-chunking one side
cascades to this table and not to the fact.

Four rejection codes join the vocabulary. `asserts_nothing` takes the
no-finite-verb half of `not_atomic`, so `not_atomic` now means what its name
says - more than one claim - and a summary can be refused for asserting
nothing without being refused for carrying several claims. `not_condensed`
refuses a summary or an outline no shorter than the passage it stands in
for, `not_listed` refuses an outline of fewer than two points, and
`not_bridging` refuses a bridge resting on one passage.

Rows already carrying `not_atomic` are left alone: which half of the old
check refused them is not recorded, and `extract-revalidate` re-reads them
without calling the model.

Revision: 6b1e93f7c204
Parent:   3f7b21d9e4a5
Created:  2026-09-16
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from database.qa_generator.fact_passages import (
    DELETE_ORPHAN_FUNCTION,
    DELETE_ORPHAN_TRIGGER,
    DROP_DELETE_ORPHAN,
)

revision: str = "6b1e93f7c204"
down_revision: str | None = "3f7b21d9e4a5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_REJECTION = "facts_rejection_code_valid"

_REJECTION_WAS = (
    "rejection_code IS NULL OR rejection_code IN ('evidence_absent', 'copied', "
    "'not_atomic', 'unsupported_addition', 'unresolved_reference')"
)

_REJECTION_NOW = (
    "rejection_code IS NULL OR rejection_code IN ('evidence_absent', 'copied', "
    "'asserts_nothing', 'not_atomic', 'unsupported_addition', "
    "'unresolved_reference', 'not_condensed', 'not_listed', 'not_bridging')"
)


def upgrade() -> None:
    """Applies the change."""
    op.add_column(
        "facts",
        sa.Column(
            "kind",
            sa.Text(),
            nullable=False,
            server_default="atomic",
            comment="atomic | summary | outline | bridge. Decides which checks the "
            "statement is held to and what its evidence is: an atomic fact cites "
            "sentences, a summary and an outline stand in for the whole passage, "
            "and a bridge rests on the passages in fact_passages.",
        ),
    )
    op.create_index("ix_facts_kind", "facts", ["kind"])
    op.create_check_constraint(
        "facts_kind_valid",
        "facts",
        "kind IN ('atomic', 'summary', 'outline', 'bridge')",
    )

    op.drop_constraint(_REJECTION, "facts", type_="check")
    op.create_check_constraint(_REJECTION, "facts", _REJECTION_NOW)

    op.alter_column(
        "facts",
        "statement",
        existing_type=sa.Text(),
        comment="The fact as written. One self-contained sentence for an atomic "
        "fact or a bridge, a short paragraph for a summary, and newline-separated "
        "`- ` bullets for an outline.",
        existing_comment="The fact as a single self-contained sentence.",
    )
    op.alter_column(
        "facts",
        "passage_id",
        existing_type=sa.BigInteger(),
        comment="The passage this fact was drawn from, and the route to both its "
        "document and its topics. On a bridge fact this is the anchor, the first "
        "of the passages listed in fact_passages.",
        existing_comment="The passage this fact was drawn from, and the route to "
        "both its document and its topics.",
    )
    op.alter_column(
        "facts",
        "evidence_sentence_ids",
        existing_type=sa.ARRAY(sa.Integer()),
        comment="Which of passages.sentences the claim was drawn from. What the "
        "extractor chooses; the span below is resolved from it, so a citation is "
        "exact by construction rather than by searching for a quote. Every "
        "sentence of the passage on a summary, an outline or a bridge.",
        existing_comment="Which of passages.sentences the claim was drawn from. "
        "What the extractor chooses; the span below is resolved from it, so a "
        "citation is exact by construction rather than by searching for a quote.",
    )

    op.create_table(
        "fact_passages",
        sa.Column(
            "fact_id", sa.BigInteger(), nullable=False, comment="The bridge fact."
        ),
        sa.Column(
            "passage_id",
            sa.BigInteger(),
            nullable=False,
            comment="A passage the claim rests on. Deleting the passage deletes "
            "this row, and the fact with it when the passage is the anchor.",
        ),
        sa.Column(
            "position",
            sa.Integer(),
            nullable=False,
            comment="Where this passage sat in the excerpt the model was shown, "
            "from 0. Position 0 is the anchor.",
        ),
        sa.ForeignKeyConstraint(["fact_id"], ["facts.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["passage_id"], ["passages.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("fact_id", "passage_id"),
        comment="Which passages a bridge fact rests on, one row per passage "
        "including the anchor in facts.passage_id. Empty for every other kind, "
        "which rests on that anchor alone.",
    )
    op.create_index(
        "ix_fact_passages_passage_id", "fact_passages", ["passage_id"], unique=False
    )
    op.execute(DELETE_ORPHAN_FUNCTION)
    op.execute(DELETE_ORPHAN_TRIGGER)


def downgrade() -> None:
    """Drops the three new kinds and the passages they rested on.

    Every fact that is not atomic goes with them: there is no atomic reading
    of a summary, and a bridge has no single passage to keep it against.
    """
    op.execute(DROP_DELETE_ORPHAN)
    op.drop_index("ix_fact_passages_passage_id", table_name="fact_passages")
    op.drop_table("fact_passages")

    op.execute("DELETE FROM facts WHERE kind <> 'atomic'")
    op.execute(
        "UPDATE facts SET validated = false, rejection_code = 'not_atomic' "
        "WHERE rejection_code IN ('asserts_nothing', 'not_condensed', "
        "'not_listed', 'not_bridging')"
    )
    op.drop_constraint(_REJECTION, "facts", type_="check")
    op.create_check_constraint(_REJECTION, "facts", _REJECTION_WAS)

    op.alter_column(
        "facts",
        "evidence_sentence_ids",
        existing_type=sa.ARRAY(sa.Integer()),
        comment="Which of passages.sentences the claim was drawn from. What the "
        "extractor chooses; the span below is resolved from it, so a citation is "
        "exact by construction rather than by searching for a quote.",
    )
    op.alter_column(
        "facts",
        "passage_id",
        existing_type=sa.BigInteger(),
        comment="The passage this fact was drawn from, and the route to both its "
        "document and its topics.",
    )
    op.alter_column(
        "facts",
        "statement",
        existing_type=sa.Text(),
        comment="The fact as a single self-contained sentence.",
    )
    op.drop_constraint("facts_kind_valid", "facts", type_="check")
    op.drop_index("ix_facts_kind", table_name="facts")
    op.drop_column("facts", "kind")
