"""A passage, a fact and a question share one space.

Adds `passages.embedding` and `facts.embedding` beside the question embedding
that was already here, each with the HNSW index its search needs, and gives
the fact checker the `duplicate` code the new gate rejects with.

Revision: 8ffd0b7409df
Parent:   b8d41e7a02c5
Created:  2026-09-20 13:33:31.962811
"""

from __future__ import annotations

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

revision: str = "8ffd0b7409df"
down_revision: str | None = "b8d41e7a02c5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_FACTS_EMBEDDING = (
    "Statement embedding, used by the dedup gate. That gate cosine-searches "
    "every validated fact, which the HNSW index above serves. The same width "
    "and the same model as questions.embedding and passages.embedding: one "
    "corpus is measured in one space, so a fact and the question resting on it "
    "are comparable. NULL on a fact extracted before the column existed, and "
    "on one no deployment asked to embed - EXTRACTION_DUPLICATE_COSINE at 0 "
    "turns the gate and the embedding off together."
)

_PASSAGES_EMBEDDING = (
    "Passage embedding, in the same space as facts.embedding and "
    "questions.embedding. What a nearest-passage search reads. Written by "
    "extraction rather than by chunking, which owns every other column here: "
    "the extraction worker loads the model anyway for the fact dedup gate, and "
    "a second worker loading two gigabytes to write one column is the cost "
    "this avoids. NULL until a passage has been through extraction under a "
    "deployment that embeds."
)

_REJECTION = "facts_rejection_code_valid"

_REJECTION_WAS = (
    "rejection_code IS NULL OR rejection_code IN ('evidence_absent', 'copied', "
    "'asserts_nothing', 'not_atomic', 'unsupported_addition', "
    "'unresolved_reference', 'not_condensed', 'not_listed', 'not_bridging', "
    "'over_cap')"
)

_REJECTION_NOW = (
    "rejection_code IS NULL OR rejection_code IN ('evidence_absent', 'copied', "
    "'asserts_nothing', 'not_atomic', 'unsupported_addition', "
    "'unresolved_reference', 'not_condensed', 'not_listed', 'not_bridging', "
    "'over_cap', 'duplicate')"
)


def upgrade() -> None:
    """Applies the change."""
    op.add_column(
        "facts",
        sa.Column(
            "embedding",
            pgvector.sqlalchemy.vector.VECTOR(dim=1024),
            nullable=True,
            comment=_FACTS_EMBEDDING,
        ),
    )
    op.create_index(
        "ix_facts_embedding_hnsw",
        "facts",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.add_column(
        "passages",
        sa.Column(
            "embedding",
            pgvector.sqlalchemy.vector.VECTOR(dim=1024),
            nullable=True,
            comment=_PASSAGES_EMBEDDING,
        ),
    )
    op.create_index(
        "ix_passages_embedding_hnsw",
        "passages",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.drop_constraint(_REJECTION, "facts", type_="check")
    op.create_check_constraint(_REJECTION, "facts", _REJECTION_NOW)


def downgrade() -> None:
    """Takes it back out."""
    # The rows first: a fact the new gate rejected names a code the old
    # constraint does not allow, and re-adding it would fail on them.
    op.execute(
        "UPDATE facts SET rejection_code = NULL, validation_error = NULL "
        "WHERE rejection_code = 'duplicate'"
    )
    op.drop_constraint(_REJECTION, "facts", type_="check")
    op.create_check_constraint(_REJECTION, "facts", _REJECTION_WAS)
    op.drop_index(
        "ix_passages_embedding_hnsw",
        table_name="passages",
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.drop_column("passages", "embedding")
    op.drop_index(
        "ix_facts_embedding_hnsw",
        table_name="facts",
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.drop_column("facts", "embedding")
