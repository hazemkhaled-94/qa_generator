"""What the orchestrator reads from the environment.

The same bargain the backend makes in backend/settings/env.py: no defaults
in code for anything that is a deployment's choice, so a missing value
stops the code location at load time naming itself rather than at three in
the morning halfway through a run.

Separate from that module rather than importing it, because this runs in an
image of its own - dagster, requests and nothing else - and `settings` is a
backend package that image does not carry. The reader is named `required`
for the same reason it is there: tests/static/test_settings_documented.py
finds settings by the name of the function that reads them, so a private
spelling here would exempt this package from the gate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def required(name: str) -> str:
    """Reads a setting that must be present.

    Raises:
        KeyError: If it is unset or empty. Empty counts as missing, because
            compose turns an unset variable into an empty string rather
            than leaving it out.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        raise KeyError(
            f"{name} must be set in the environment. The orchestrator's "
            f"settings are in configs/env/orchestration.env; the address it "
            f"calls and its credentials are in .env."
        )
    return value


@dataclass(frozen=True)
class Settings:
    """Where the backend is, and how patiently to watch it."""

    #: The stage routes. The orchestrator talks to nothing else.
    backend_url: str
    #: How long an asset waits for a stage to drain before giving up on
    #: watching. Giving up is not failing the rows: they are still queued
    #: and a worker still has them.
    drain_timeout: float
    #: How often to ask. A stage whose unit costs a model call is not worth
    #: asking about every second.
    poll_seconds: float

    @classmethod
    def load(cls) -> Settings:
        """Reads settings from the environment."""
        return cls(
            backend_url=required("BACKEND_URL"),
            drain_timeout=float(required("ORCHESTRATION_DRAIN_TIMEOUT_SECONDS")),
            poll_seconds=float(required("ORCHESTRATION_POLL_SECONDS")),
        )
