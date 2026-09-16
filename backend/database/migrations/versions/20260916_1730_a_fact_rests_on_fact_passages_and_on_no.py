"""A fact rests on fact_passages, and on nothing else.

`facts.passage_id` said one passage was special. For an atomic fact, a
summary and an outline that was true, because there was only one. For a
bridge it was a pick - the first passage the claim cited, where first meant
the position the model happened to be shown it in - and no reader could
answer why that one. The evidence span went with it: one offset pair into
one `passages.text`, which is the wrong shape for a claim resting on two.

`fact_passages` already held a bridge's passages and, since the previous
revision, a span in each. It becomes the only link, one row per passage for
every kind. `facts` keeps `evidence_text` as the denormalised copy the
trigram index searches, now defined as every span joined in position order.

The orphan trigger goes from bridge-specific to universal. It deleted a
bridge left with fewer than two passages; it now deletes any fact any of
whose passages is gone, which is what the link table means for all four
kinds, and which replaces both that rule and the foreign key cascade
`facts.passage_id` used to carry.

Revision: c4a7f2e918bd
Parent:   8d2c47b1e6f9
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

revision: str = "c4a7f2e918bd"
down_revision: str | None = "8d2c47b1e6f9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SPAN = "facts_evidence_span_ordered"

#: The bridge-specific trigger this revision replaces. Dropped by name
#: because the shared declaration now carries the universal one, so a
#: database migrated before this revision is the only place it still exists.
_DROP_BRIDGE_TRIGGER = """
DROP TRIGGER IF EXISTS fact_passages_delete_orphan_bridge ON fact_passages;
DROP FUNCTION IF EXISTS delete_bridge_without_passages();
"""

#: One link row for every fact that has none: every atomic fact, summary and
#: outline, and the refused bridges that resolved no passage and so were
#: written without links at all.
_LINK_THE_REST = """
INSERT INTO fact_passages (fact_id, passage_id, position,
                           sentence_ids, evidence_start, evidence_end)
SELECT f.id, f.passage_id, 0,
       f.evidence_sentence_ids, f.evidence_start, f.evidence_end
  FROM facts f
 WHERE NOT EXISTS (SELECT 1 FROM fact_passages fp WHERE fp.fact_id = f.id)
"""

#: A bridge drawn before prompt version 2 has a link row per passage and a
#: span on none of them. The span on the fact is the first passage's and is
#: correct there; the rest stay NULL, which is what question generation
#: reads as "not citable".
_FILL_ANCHOR_SPAN = """
UPDATE fact_passages fp
   SET sentence_ids = f.evidence_sentence_ids,
       evidence_start = f.evidence_start,
       evidence_end = f.evidence_end
  FROM facts f
 WHERE fp.fact_id = f.id
   AND fp.position = 0
   AND fp.sentence_ids IS NULL
   AND f.evidence_sentence_ids IS NOT NULL
"""

#: A fact refused for evidence_absent cited sentences its passage does not
#: have. It kept them in the column beside an empty span; the link row says
#: what is true instead, which is that it resolves nowhere.
_CLEAR_ABSENT = """
UPDATE fact_passages fp
   SET sentence_ids = NULL, evidence_start = NULL, evidence_end = NULL
  FROM facts f
 WHERE fp.fact_id = f.id AND f.rejection_code = 'evidence_absent'
"""

#: evidence_text on a fact resting on several passages held the first span
#: alone. It becomes every span, joined in position order, which is what the
#: checker now writes and what the search index should be able to find.
_REJOIN_EVIDENCE = r"""
UPDATE facts f
   SET evidence_text = joined.text
  FROM (
    SELECT fp.fact_id,
           string_agg(
               substring(p.text FROM fp.evidence_start + 1
                                FOR fp.evidence_end - fp.evidence_start),
               E'\n' ORDER BY fp.position) AS text
      FROM fact_passages fp
      JOIN passages p ON p.id = fp.passage_id
     WHERE fp.sentence_ids IS NOT NULL
     GROUP BY fp.fact_id
    HAVING count(*) > 1
  ) joined
 WHERE f.id = joined.fact_id
"""

#: The reverse: the first passage's span back onto the fact. Coalesced,
#: because the columns are NOT NULL there and a link row may carry no span.
_RESTORE_FROM_ANCHOR = """
UPDATE facts f
   SET passage_id = fp.passage_id,
       evidence_sentence_ids = coalesce(fp.sentence_ids, '{}'),
       evidence_start = coalesce(fp.evidence_start, 0),
       evidence_end = coalesce(fp.evidence_end, 0)
  FROM fact_passages fp
 WHERE fp.fact_id = f.id AND fp.position = 0
"""

#: The bridge-specific trigger, restored by the downgrade.
_BRIDGE_TRIGGER = """
CREATE OR REPLACE FUNCTION delete_bridge_without_passages() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    DELETE FROM facts f
     WHERE f.id = OLD.fact_id
       AND (SELECT count(*) FROM fact_passages fp WHERE fp.fact_id = f.id) < 2;
    RETURN NULL;
END;
$$;

CREATE TRIGGER fact_passages_delete_orphan_bridge
AFTER DELETE ON fact_passages
FOR EACH ROW EXECUTE FUNCTION delete_bridge_without_passages();
"""

_LINKS_COMMENT = (
    "Which passages a fact rests on and where in each, one row per passage "
    "for every kind. The only route from a fact to a passage."
)

_FACTS_COMMENT = (
    "One statement and the verdicts the checks reached on it. Which passages "
    "it rests on, and where in each, is fact_passages. The unit a question is "
    "generated from and scored against."
)

#: Reached facts.passage_id, and reaches the link table instead.
_QUESTION_FACTS_COMMENT = (
    "Which facts a question was generated from. One row for an ordinary "
    "question, two or more for a cross-document question. The passages a "
    "question needs retrieved are derived from here through fact_passages "
    "rather than stored separately."
)


def upgrade() -> None:
    """Applies the change."""
    op.execute(_LINK_THE_REST)
    op.execute(_FILL_ANCHOR_SPAN)
    op.execute(_CLEAR_ABSENT)
    op.execute(_REJOIN_EVIDENCE)

    op.drop_constraint(_SPAN, "facts", type_="check")
    op.drop_index(op.f("ix_facts_passage_id"), table_name="facts")
    op.drop_column("facts", "evidence_end")
    op.drop_column("facts", "evidence_start")
    op.drop_column("facts", "evidence_sentence_ids")
    op.drop_column("facts", "passage_id")

    op.execute(_DROP_BRIDGE_TRIGGER)
    op.execute(DROP_DELETE_ORPHAN)
    op.execute(DELETE_ORPHAN_FUNCTION)
    op.execute(DELETE_ORPHAN_TRIGGER)

    op.alter_column(
        "facts",
        "kind",
        existing_type=sa.Text(),
        comment="atomic | summary | outline | bridge. Decides which checks the "
        "statement is held to and how many passages it rests on: atomic, "
        "summary and outline rest on one, a bridge on two or more.",
    )
    op.alter_column(
        "facts",
        "evidence_text",
        existing_type=sa.Text(),
        comment="Every span in fact_passages, resolved and joined in position "
        "order. The denormalised copy the search index reads; the spans "
        "themselves are the record.",
    )
    op.alter_column(
        "fact_passages",
        "fact_id",
        existing_type=sa.BigInteger(),
        comment="The fact.",
    )
    op.alter_column(
        "fact_passages",
        "passage_id",
        existing_type=sa.BigInteger(),
        comment="A passage the claim rests on. Deleting the passage deletes "
        "this row and the fact with it.",
    )
    op.alter_column(
        "fact_passages",
        "position",
        existing_type=sa.Integer(),
        comment="Where this passage sat in what the model was shown, from 0. "
        "0 for the one passage of an atomic fact, a summary or an outline.",
    )
    op.alter_column(
        "fact_passages",
        "sentence_ids",
        existing_type=sa.ARRAY(sa.Integer()),
        comment="Which of this passage's sentences the claim rests on. NULL "
        "when the claim named no sentence this passage has: a refused fact, "
        "or a bridge drawn before prompt version 2. Neither is offered to "
        "question generation.",
    )
    op.alter_column(
        "passages",
        "text",
        existing_type=sa.Text(),
        comment="The passage content. The evidence offsets in fact_passages "
        "are relative to this string.",
    )
    op.execute(f"COMMENT ON TABLE fact_passages IS '{_LINKS_COMMENT}'")
    op.execute(f"COMMENT ON TABLE facts IS '{_FACTS_COMMENT}'")
    op.execute(f"COMMENT ON TABLE question_facts IS '{_QUESTION_FACTS_COMMENT}'")


def downgrade() -> None:
    """Puts the anchor and its span back on the fact.

    The trigger is taken out before the link rows are, and put back after:
    either rule would read a row disappearing as its fact disappearing, and
    delete every atomic fact in the table.
    """
    op.execute(DROP_DELETE_ORPHAN)

    op.add_column(
        "facts",
        sa.Column(
            "passage_id",
            sa.BigInteger(),
            nullable=True,
            comment="The passage this fact was drawn from, and the route to "
            "both its document and its topics. On a bridge fact this is the "
            "anchor, the first of the passages listed in fact_passages.",
        ),
    )
    op.add_column(
        "facts",
        sa.Column("evidence_sentence_ids", sa.ARRAY(sa.Integer()), nullable=True),
    )
    op.add_column("facts", sa.Column("evidence_start", sa.Integer(), nullable=True))
    op.add_column("facts", sa.Column("evidence_end", sa.Integer(), nullable=True))
    op.execute(_RESTORE_FROM_ANCHOR)
    op.execute("DELETE FROM facts WHERE passage_id IS NULL")
    op.execute(
        "DELETE FROM fact_passages fp USING facts f "
        "WHERE fp.fact_id = f.id AND f.kind <> 'bridge'"
    )

    for column, kind in (
        ("passage_id", sa.BigInteger()),
        ("evidence_sentence_ids", sa.ARRAY(sa.Integer())),
        ("evidence_start", sa.Integer()),
        ("evidence_end", sa.Integer()),
    ):
        op.alter_column("facts", column, existing_type=kind, nullable=False)
    op.create_foreign_key(
        "facts_passage_id_fkey",
        "facts",
        "passages",
        ["passage_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index(op.f("ix_facts_passage_id"), "facts", ["passage_id"], unique=False)
    op.create_check_constraint(_SPAN, "facts", "evidence_end >= evidence_start")
    op.execute(_BRIDGE_TRIGGER)
