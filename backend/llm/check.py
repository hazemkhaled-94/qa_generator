"""Whether the configured model answers, asked before a corpus finds out.

An unknown model id, an unreachable address, a missing credential and a
structured mode the provider will not accept all fail the same way: at the
first call, which on a real run is hours in and a queue of failed rows
behind it. This is that first call, made once and on purpose.

    python -m llm.check

Wrapped by `make doctor`. It asks for the smallest structured answer there
is, so what it proves is the whole path a stage depends on - the address
resolves, the credential is accepted, the model exists, and it can return
an object rather than prose.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
from dataclasses import replace
from datetime import UTC, datetime

import litellm
from pydantic import BaseModel, Field

import telemetry
from llm.client import Client
from llm.config import Settings

log = logging.getLogger(__name__)

#: Silenced for the length of this check. Each writes a refusal whole - the
#: client with its traceback, instructor once per attempt, azure-identity
#: with every credential it tried - which is what a worker's log wants and
#: not what somebody running a first check does. The refusal is reported
#: here instead, in one line.
QUIET = ("llm.client", "LiteLLM", "instructor", "azure.identity", "azure.core")

#: A token minted by hand, which is the one Entra ID credential that goes
#: stale on its own. Checked before the call because the refusal it causes
#: says only "missing, invalid, audience is incorrect, or have expired",
#: and which of those it was is the whole question.
MINTED = "AZURE_OPENAI_AD_TOKEN"

#: Long enough for a cold local model to load, short enough that a wrong
#: address is a failure rather than a wait. LLM_TIMEOUT_SECONDS is the
#: pipeline's patience with a 473 s passage and is not this.
TIMEOUT_SECONDS = 90.0


class Reachable(BaseModel):
    """The smallest structured answer a provider can be asked for."""

    ok: bool = Field(description="true")


def expired(token: str) -> datetime | None:
    """When a JWT expired, or None if it has not or cannot be read.

    The `exp` claim, read without verifying the signature: this is not
    authenticating anybody, it is reading a date out of a string the
    provider is about to reject.
    """
    try:
        payload = token.split(".")[1]
        claims = json.loads(
            base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
        )
        when = datetime.fromtimestamp(claims["exp"], UTC)
    except (IndexError, KeyError, TypeError, ValueError, binascii.Error):
        return None
    return when if when < datetime.now(UTC) else None


def main() -> int:
    """Asks the configured model one question.

    Returns:
        The process exit code: 0 if the model answered in shape.
    """
    telemetry.configure("doctor")
    for noisy in QUIET:
        logging.getLogger(noisy).setLevel(logging.CRITICAL)
    litellm.suppress_debug_info = True

    settings = Settings.load()
    log.info("model    %s", settings.model)
    log.info("address  %s", settings.base_url or "(the provider's own)")
    log.info("mode     %s", settings.structured_mode)

    if stale := expired(os.environ.get(MINTED, "")):
        days = (datetime.now(UTC) - stale).days
        log.error(
            "%s expired %s (%d day(s) ago) and is taken in preference to a "
            "working credential, so every call is refused. Comment it out to "
            "fall through to `az login`, or mint another one.",
            MINTED,
            stale.date(),
            days,
        )
        return 1

    # One attempt: a doctor reports what happened rather than working around
    # it, and the backoff would hide an address that is simply wrong.
    once = replace(
        settings, timeout_seconds=TIMEOUT_SECONDS, max_attempts=1, num_ctx=None
    )
    try:
        answer = Client(once).answer(
            system="You answer with a JSON object and nothing else.",
            user="Set ok to true.",
            shape=Reachable,
        )
    except Exception as refusal:  # noqa: BLE001 - a doctor reports any of them
        log.error("the model did not answer: %s: %s", type(refusal).__name__, refusal)
        return 1

    log.info("the model answered in shape: ok=%s", answer.ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
