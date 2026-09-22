"""A deletion leaves the row behind.

Every deletion in this schema was final, and most of them are not the one
somebody typed. `make delete SHA=...` removes a document, and the foreign
keys take its passages, the memberships on them and the citations under
them; then `fact_passages_delete_orphan_fact` takes every fact resting on
a passage that is gone, and `question_facts_delete_orphan_question` takes
every question left with no fact to rest on, and the self-referential
cascade on `follows_id` takes the follow-ups of those. One command, nine
tables, and the only record of what was there was the count it printed.

So every table gets an AFTER DELETE trigger that copies the row into
`archived_rows` first. A trigger rather than the services, because the
services are not where most of the deleting happens: a cascade runs inside
the engine, an orphan trigger runs inside another trigger, and
`make wipe`, a re-chunk, a re-extraction and a statement typed into
Adminer all destroy rows without passing through any Python that could
have taken a copy. What the row was is known in exactly one place, which
is the moment Postgres removes it.

Not a soft delete, deliberately. A `deleted_at` on ten tables is a
predicate every query in the repository layer has to grow and one of them
will forget, and it is a predicate the partial indexes on the queue
columns would have to carry too. The live tables still lose the row here,
so nothing that already works reads differently, and the archive is a
separate table nothing joins to.

`embedding` is the one column not kept. It renders to about 13 kB of text
a row, which is ten times the rest of a fact put together - 90 MB against
8.6 MB over the corpus this was written on - and it is recomputed from the
statement it belongs to. The key is dropped from the jsonb rather than the
column named per table, and `jsonb - text` on a row without that key is
the row unchanged, so one function serves the four tables that carry a
vector and the six that do not.

Nothing archives `archived_rows`. Emptying it is the second deletion, and
that is the whole point of it being here.

Revision: d511a6ee5767
Parent:   5b2e9c74a1f6
Created:  2026-09-22
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from database.qa_generator.archived_rows import (
    ARCHIVE_FUNCTION,
    DROP_ARCHIVE_FUNCTION,
    archive_trigger,
    drop_archive_trigger,
)

revision: str = "d511a6ee5767"
down_revision: str | None = "5b2e9c74a1f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: Every table there is, as of this revision, and named here rather than read
#: off `Base.metadata`: a revision describes one point in the history, and a
#: list read from the models would silently start attaching triggers to
#: tables added after it.
_ARCHIVED = (
    "documents",
    "passages",
    "facts",
    "fact_passages",
    "questions",
    "question_facts",
    "topics",
    "passage_topics",
    "ingest_events",
    "service_settings",
)


def upgrade() -> None:
    """Creates the archive, and fills it from every other table."""
    op.create_table(
        "archived_rows",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column(
            "table_name",
            sa.Text(),
            nullable=False,
            comment="The table the row was deleted from.",
        ),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            comment="The whole row, by column name, minus `embedding`. A vector renders as about 13 kB of text per row - ten times everything else the row holds - and is recomputed from the text it belongs to, so keeping it would cost ten archives and buy one. Every other column is here, the primary key included, so jsonb_populate_record puts a row back by hand.",
        ),
        sa.Column(
            "archived_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
            comment="When the deletion happened.",
        ),
        sa.PrimaryKeyConstraint("id"),
        comment="Every row deleted from every other table, as it was. Filled by an AFTER DELETE trigger rather than by the application, so a foreign key cascade, an orphan trigger and a statement typed into psql are archived alike. Nothing archives this table: emptying it is the second deletion.",
    )
    op.create_index(
        "ix_archived_rows_archived_at", "archived_rows", ["archived_at"], unique=False
    )
    op.create_index(
        "ix_archived_rows_table_name", "archived_rows", ["table_name"], unique=False
    )

    op.execute(ARCHIVE_FUNCTION)
    for table in _ARCHIVED:
        op.execute(archive_trigger(table))


def downgrade() -> None:
    """Takes the triggers off, then the archive with everything in it."""
    for table in _ARCHIVED:
        op.execute(drop_archive_trigger(table))
    op.execute(DROP_ARCHIVE_FUNCTION)

    op.drop_index("ix_archived_rows_table_name", table_name="archived_rows")
    op.drop_index("ix_archived_rows_archived_at", table_name="archived_rows")
    op.drop_table("archived_rows")
