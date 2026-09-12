"""Reading configuration out of the environment, and nowhere else.

There are no defaults in code: a missing variable stops the service at
start-up, naming itself. Two files list what must be set:
`configs/env/backend.env` for how the pipeline behaves, and `.env` - see
`.env.example` - for credentials, ports and addresses. Two exceptions: a
setting whose absence is itself meaningful uses `optional`, and `telemetry`
falls back to INFO because it is configured before a service has read its
settings.
"""

from __future__ import annotations

import os

#: Accepted spellings of true. Anything else is false, including nonsense: a
#: boolean that guesses is worse than one that is simply off.
_TRUE = frozenset({"1", "true", "yes", "on"})


def required(name: str) -> str:
    """Reads a setting that must be present.

    Raises:
        KeyError: If it is unset or empty. Empty counts as missing, because
            compose turns an unset variable into an empty string rather than
            leaving it out.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        raise KeyError(
            f"{name} must be set in the environment. How the pipeline behaves "
            f"comes from configs/env/backend.env; credentials, ports and "
            f"addresses come from .env, which .env.example lists."
        )
    return value


def optional(name: str) -> str | None:
    """Reads a setting whose absence means something."""
    return os.environ.get(name, "").strip() or None


def integer(name: str) -> int:
    """Reads a required whole number.

    Raises:
        KeyError: If it is unset or empty.
        ValueError: If it is not a whole number, naming the variable.
    """
    value = required(name)
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name}={value!r} is not a whole number") from None


def decimal(name: str) -> float:
    """Reads a required number.

    Raises:
        KeyError: If it is unset or empty.
        ValueError: If it is not a number, naming the variable.
    """
    value = required(name)
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{name}={value!r} is not a number") from None


def boolean(name: str) -> bool:
    """Reads a required flag.

    Raises:
        KeyError: If it is unset or empty.
    """
    return required(name).lower() in _TRUE


def csv(name: str) -> tuple[str, ...]:
    """Reads a required comma-separated list.

    Raises:
        KeyError: If it is unset or empty.
    """
    return tuple(part.strip() for part in required(name).split(",") if part.strip())
