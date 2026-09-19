"""A passage is read in its own language, not its document's.

`passages.language` is detected per passage and says so in its own comment:
one file carries a German report and its English summary, so a
document-wide label reads half the passages with the wrong pipeline.
Chunking segments each passage under that language, and topic modelling
and question generation both read the column.

Extraction did not. It joined `documents.language` and judged every
statement with that pipeline, so for a mixed-language document the two
halves of every check disagreed: `evidence_predicates` came off sentences
one pipeline found and `statement_predicates`, the units and the pronouns
came from another, with `facts.spacy_model` recording the wrong one. The
gates that suffered are the ones that matter - `not_atomic` counts finite
verbs, `unsupported_addition` compares vocabularies, and
`unresolved_reference` reads morphology that differs by language.

Nothing here changes data. The fix is in the queries, and the verdicts
already stored are recovered by `make extract-revalidate`, which re-reads
every fact under today's pipelines without calling a model. This revision
carries only the comment on `documents.language`, which claimed extraction
read it and now says what it actually is: the fallback for a passage too
short to tell its own.

Revision: e5a02c7b1946
Parent:   a17f4c93e2b8
Created:  2026-09-19
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e5a02c7b1946"
down_revision: str | None = "a17f4c93e2b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_WAS = (
    "ISO 639-1, NULL until parsing detects it. Selects the spaCy pipeline "
    "chunking and extraction read this document with."
)

_NOW = (
    "ISO 639-1, NULL until parsing detects it. The fallback pipeline for a "
    "passage of this document too short to tell its own language; "
    "`passages.language` is what selects one where it has it."
)


def upgrade() -> None:
    """Says what the document's language is actually for."""
    op.alter_column(
        "documents",
        "language",
        existing_type=sa.CHAR(length=2),
        existing_nullable=True,
        existing_comment=_WAS,
        comment=_NOW,
    )


def downgrade() -> None:
    """Restores the comment, which is all this revision carries."""
    op.alter_column(
        "documents",
        "language",
        existing_type=sa.CHAR(length=2),
        existing_nullable=True,
        existing_comment=_NOW,
        comment=_WAS,
    )
