"""The archived_rows table, and the trigger that fills it."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, DateTime, Index, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from database.qa_generator.base import Base


class ArchivedRow(Base):
    """One row as it was when something deleted it.

    Not a soft delete. The live tables still lose the row, so no query
    anywhere has to learn to exclude one, no index widens, and nothing that
    already works reads differently. This is the copy taken on the way out,
    and deleting a question twice - once from the corpus and once from here
    - is what the two steps buy.
    """

    __tablename__ = "archived_rows"
    __table_args__ = (
        # A purge names a table, an age, or both. Two indexes rather than one
        # composite, so either alone is served.
        Index("ix_archived_rows_table_name", "table_name"),
        Index("ix_archived_rows_archived_at", "archived_at"),
        {
            "comment": "Every row deleted from every other table, as it was. "
            "Filled by an AFTER DELETE trigger rather than by the application, so "
            "a foreign key cascade, an orphan trigger and a statement typed into "
            "psql are archived alike. Nothing archives this table: emptying it is "
            "the second deletion."
        },
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    table_name: Mapped[str] = mapped_column(
        Text, comment="The table the row was deleted from."
    )
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        comment="The whole row, by column name, minus `embedding`. A vector "
        "renders as about 13 kB of text per row - ten times everything else the "
        "row holds - and is recomputed from the text it belongs to, so keeping it "
        "would cost ten archives and buy one. Every other column is here, the "
        "primary key included, so jsonb_populate_record puts a row back by hand.",
    )
    archived_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="When the deletion happened.",
    )


#: Copies a row into `archived_rows` as it is deleted.
#:
#: One function for all ten tables, because `to_jsonb(OLD)` names no column:
#: a table gains one and this keeps working. `embedding` is dropped by key,
#: and `jsonb - text` on a row without that key is the row unchanged, so the
#: same function serves the four tables carrying a vector and the six that
#: are not.
#:
#: Declared here, beside the table it fills, and executed by the migration
#: that creates that table: autogenerate sees tables and columns, never a
#: trigger.
ARCHIVE_FUNCTION = """
CREATE OR REPLACE FUNCTION archive_deleted_row() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO archived_rows (table_name, payload)
    VALUES (TG_TABLE_NAME, to_jsonb(OLD) - 'embedding');
    RETURN NULL;
END;
$$;
"""

DROP_ARCHIVE_FUNCTION = "DROP FUNCTION IF EXISTS archive_deleted_row();"


def archive_trigger(table: str) -> str:
    """Builds the AFTER DELETE trigger that archives one table's rows."""
    return f"""
CREATE TRIGGER {table}_archive_deleted_row
AFTER DELETE ON {table}
FOR EACH ROW EXECUTE FUNCTION archive_deleted_row();
"""


def drop_archive_trigger(table: str) -> str:
    """Builds the statement that takes it back off."""
    return f"DROP TRIGGER IF EXISTS {table}_archive_deleted_row ON {table};"
