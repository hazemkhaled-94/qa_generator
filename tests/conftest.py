"""The settings the backend reads as it is imported.

Set here because pytest loads this file before it collects a test module.

The tuning values are read from configs/env/backend.env, the file the
services read, rather than copied. Credentials and addresses are not: no test
reaches a running service.
"""

from __future__ import annotations

import os
from pathlib import Path

TUNING = Path(__file__).resolve().parents[1] / "configs/env/backend.env"

for line in TUNING.read_text().splitlines():
    name, sign, value = line.partition("=")
    if sign and not name.lstrip().startswith("#"):
        os.environ.setdefault(name.strip(), value.strip())

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://unused:unused@localhost/x")
