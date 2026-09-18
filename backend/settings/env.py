"""Reading configuration out of the environment, and out of an override.

There are no defaults in code: a missing variable stops the service at
start-up, naming itself. Two files list what must be set:
`configs/env/backend.env` for how the pipeline behaves, and `.env` - see
`.env.example` - for credentials, ports and addresses. Two exceptions: a
setting whose absence is itself meaningful uses `optional`, and `telemetry`
falls back to INFO because it is configured before a service has read its
settings.

Every reader takes an optional `source`. Absent, it is the process
environment. Given, it is whatever the caller resolved - the environment
overlaid with the stored rows. The environment stays required either way: a
source overrides a setting, it never supplies one.

Passed rather than ambient, so a candidate value can be parsed without
touching the environment the process runs under.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

#: Accepted spellings of true. Anything else is false, including nonsense: a
#: boolean that guesses is worse than one that is simply off.
_TRUE = frozenset({"1", "true", "yes", "on"})

#: Where a reader looks. None is the process environment.
Source = Mapping[str, str] | None


def required(name: str, source: Mapping[str, str] | None = None) -> str:
    """Reads a setting that must be present.

    Args:
        name: The variable to read.
        source: Where to read it, or None for the process environment.

    Raises:
        KeyError: If it is unset or empty. Empty counts as missing, because
            compose turns an unset variable into an empty string rather than
            leaving it out.
    """
    value = (os.environ if source is None else source).get(name, "").strip()
    if not value:
        raise KeyError(
            f"{name} must be set in the environment. How the pipeline behaves "
            f"comes from configs/env/backend.env; credentials, ports and "
            f"addresses come from .env, which .env.example lists."
        )
    return value


def optional(name: str, source: Mapping[str, str] | None = None) -> str | None:
    """Reads a setting whose absence means something."""
    return (os.environ if source is None else source).get(name, "").strip() or None


def integer(name: str, source: Mapping[str, str] | None = None) -> int:
    """Reads a required whole number.

    Raises:
        KeyError: If it is unset or empty.
        ValueError: If it is not a whole number, naming the variable.
    """
    value = required(name, source)
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name}={value!r} is not a whole number") from None


def decimal(name: str, source: Mapping[str, str] | None = None) -> float:
    """Reads a required number.

    Raises:
        KeyError: If it is unset or empty.
        ValueError: If it is not a number, naming the variable.
    """
    value = required(name, source)
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{name}={value!r} is not a number") from None


def boolean(name: str, source: Mapping[str, str] | None = None) -> bool:
    """Reads a required flag.

    Raises:
        KeyError: If it is unset or empty.
    """
    return required(name, source).lower() in _TRUE


def csv(name: str, source: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Reads a required comma-separated list.

    Raises:
        KeyError: If it is unset or empty.
    """
    return tuple(
        part.strip() for part in required(name, source).split(",") if part.strip()
    )


def mapping(name: str, source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Reads a required comma-separated list of `key:value` pairs.

    Raises:
        KeyError: If it is unset or empty.
        ValueError: If an entry carries no colon, naming the variable.
    """
    found = {}
    for entry in csv(name, source):
        key, colon, value = entry.partition(":")
        if not colon or not key.strip():
            raise ValueError(
                f"{name} takes KEY:VALUE entries separated by commas; {entry!r} "
                f"is not one"
            )
        found[key.strip()] = value.strip()
    return found
