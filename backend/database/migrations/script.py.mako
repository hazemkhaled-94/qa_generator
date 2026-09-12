"""${message}

Revision: ${up_revision}
Parent:   ${down_revision | comma,n}
Created:  ${create_date}
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
${imports if imports else ""}
revision: str = ${repr(up_revision)}
down_revision: str | None = ${repr(down_revision)}
branch_labels: str | Sequence[str] | None = ${repr(branch_labels)}
depends_on: str | Sequence[str] | None = ${repr(depends_on)}


def upgrade() -> None:
    """Applies the change."""
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    """Takes it back out."""
    ${downgrades if downgrades else "pass"}
