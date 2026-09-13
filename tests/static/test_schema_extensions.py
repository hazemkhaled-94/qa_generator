"""The migrations install every extension the database bootstrap installs.

configs/postgres/init.sh runs once, on the first boot of an empty volume, and
only under compose. A schema that also needs it cannot be built by
`alembic upgrade head` alone - which is every test database and every
deployment that is not this compose file.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = ROOT / "configs/postgres/init.sh"
VERSIONS = ROOT / "backend/database/migrations/versions"

_EXTENSION = re.compile(
    r"CREATE\s+EXTENSION\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z0-9_]+)", re.IGNORECASE
)


def test_the_migrations_create_what_the_bootstrap_creates() -> None:
    """Every extension init.sh creates is created by a migration too."""
    bootstrap = set(_EXTENSION.findall(BOOTSTRAP.read_text()))
    assert bootstrap, f"{BOOTSTRAP.name} creates no extension; has it moved?"

    migrated = set(
        _EXTENSION.findall("\n".join(p.read_text() for p in VERSIONS.glob("*.py")))
    )
    missing = bootstrap - migrated
    assert not missing, f"no migration creates {', '.join(sorted(missing))}"
