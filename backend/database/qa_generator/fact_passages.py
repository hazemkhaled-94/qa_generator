"""The fact_passages table."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.qa_generator.base import Base

if TYPE_CHECKING:
    from database.qa_generator.facts import Fact
    from database.qa_generator.passages import Passage


class FactPassage(Base):
    """Which passages a bridge fact rests on.

    Written for `kind = 'bridge'` only, one row per passage including the
    anchor in facts.passage_id. Every other kind rests on that anchor alone
    and has no row here.
    """

    __tablename__ = "fact_passages"
    __table_args__ = (
        {
            "comment": "Which passages a bridge fact rests on, one row per passage "
            "including the anchor in facts.passage_id. Empty for every other kind, "
            "which rests on that anchor alone."
        },
    )

    fact_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("facts.id", ondelete="CASCADE"),
        primary_key=True,
        comment="The bridge fact.",
    )
    passage_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("passages.id", ondelete="CASCADE"),
        primary_key=True,
        # The primary key indexes (fact_id, passage_id), which is no help to
        # the lookup by passage_id alone that re-extraction runs.
        index=True,
        comment="A passage the claim rests on. Deleting the passage deletes this "
        "row, and the fact with it when the passage is the anchor.",
    )
    position: Mapped[int] = mapped_column(
        Integer,
        comment="Where this passage sat in the excerpt the model was shown, from "
        "0. Position 0 is the anchor.",
    )

    fact: Mapped[Fact] = relationship(back_populates="passage_links")
    passage: Mapped[Passage] = relationship(back_populates="fact_links")


#: Deletes a bridge fact once it rests on fewer than two passages. Re-chunking
#: one of them cascades to this table and not to the fact, which would
#: otherwise leave a claim that bridges nothing. A foreign key cascades parent
#: to child, never the reverse.
#:
#: Declared here, beside the table it belongs to, and executed by the
#: migration that creates that table: autogenerate sees tables and columns,
#: never a trigger.
DELETE_ORPHAN_FUNCTION = """
CREATE OR REPLACE FUNCTION delete_bridge_without_passages() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    DELETE FROM facts f
     WHERE f.id = OLD.fact_id
       AND (SELECT count(*) FROM fact_passages fp WHERE fp.fact_id = f.id) < 2;
    RETURN NULL;
END;
$$;
"""

DELETE_ORPHAN_TRIGGER = """
CREATE TRIGGER fact_passages_delete_orphan_bridge
AFTER DELETE ON fact_passages
FOR EACH ROW EXECUTE FUNCTION delete_bridge_without_passages();
"""

DROP_DELETE_ORPHAN = """
DROP TRIGGER IF EXISTS fact_passages_delete_orphan_bridge ON fact_passages;
DROP FUNCTION IF EXISTS delete_bridge_without_passages();
"""
