"""A topic sits where its passages do.

Adds `topics.embedding`, the mean of the vectors of the passages a topic
holds. Derived in SQL from `passages.embedding` rather than asked of a model,
so nothing loads one to write it.

No HNSW index: a fit is tens of topics, and the only query over this column
is a self-join finding the pairs that sit too close together. An index over
38 rows is read more slowly than the rows are.

Revision: 3c91a4e17b02
Parent:   8ffd0b7409df
Created:  2026-09-20 15:20:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

revision: str = "3c91a4e17b02"
down_revision: str | None = "8ffd0b7409df"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_COMMENT = (
    "Where this topic sits, as the membership-weighted mean of its passages' "
    "vectors, normalised back to length 1. Derived rather than asked for: it "
    "is an average of passages.embedding computed in SQL, so no worker loads "
    "a model to write it and it cannot disagree with the passages it is the "
    "mean of. What it is for is telling two topics apart - a factorisation "
    "splits one subject in two often enough that the coverage report needs to "
    "say so. NULL until the passages are embedded."
)


def upgrade() -> None:
    """Applies the change."""
    op.add_column(
        "topics",
        sa.Column(
            "embedding",
            pgvector.sqlalchemy.vector.VECTOR(dim=1024),
            nullable=True,
            comment=_COMMENT,
        ),
    )


def downgrade() -> None:
    """Takes it back out."""
    op.drop_column("topics", "embedding")
