"""The settings the frontend reads out of its environment."""

from __future__ import annotations

import os

#: The one address the frontend holds. The database, the object store and the
#: pipeline stages all live behind it.
BACKEND_URL = os.environ["BACKEND_URL"].rstrip("/")


def _integer(name: str) -> int:
    """Reads a required whole number, naming it when it is missing.

    Raises:
        RuntimeError: If it is unset, empty, or not a whole number.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        raise RuntimeError(
            f"{name} must be set for the frontend. It comes from .env and has "
            f"to be listed in the streamlit service's environment in "
            f"compose.yaml - being in .env alone is not enough."
        )
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(f"{name}={raw!r} is not a whole number") from None


#: Rows per page in the listings.
PAGE_SIZE = _integer("PAGE_SIZE")
