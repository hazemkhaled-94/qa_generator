"""Whether a configured model answers, asked before a corpus finds out.

An unknown model id, an unreachable address, a missing credential and a
structured mode the provider will not accept all fail the same way: at the
first call. On a real run that is hours in, with a queue of failed rows
behind it. This is that first call, made once and on purpose.

    python -m llm.check        # which is what `make doctor` runs

A worker asks the same question before it claims anything, through
`reachable`. **A process that cannot call its model does not take work**,
because the alternative is a stage draining its queue into `failed` one
expensive timeout at a time.

Nothing here knows which provider is configured. The check is one ordinary
call through the same client a stage uses, so whatever authenticates a
stage authenticates this.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

import litellm
from pydantic import BaseModel, Field

import telemetry
from llm.client import Client
from llm.config import Settings

log = logging.getLogger(__name__)

#: Silenced while the check runs. Each of these narrates a refusal at
#: length - the client with its traceback, instructor once per attempt, a
#: credential library with every identity it tried - which is what a
#: worker's log wants and not what somebody reading a one-line verdict
#: does. The refusal is reported here instead, in one line.
QUIET = ("llm.client", "LiteLLM", "instructor", "azure.identity", "azure.core")


@contextmanager
def quiet() -> Iterator[None]:
    """Silences the libraries that narrate a refusal, for one call.

    Applied around the CHECK rather than only around `make doctor`, which
    is where it used to be. A worker asks this same question before every
    poll, and a worker whose Ollama is down produced 836 log lines in four
    minutes - 20 tracebacks, 96 litellm lines and every identity
    DefaultAzureCredential tried - for a fact that fits in one sentence
    and that `stages.cli` was already writing.

    Restored afterwards, so the narration a real call needs survives: this
    is about the health check, not about the pipeline.
    """
    held = {name: logging.getLogger(name).level for name in QUIET}
    was = litellm.suppress_debug_info
    for name in QUIET:
        logging.getLogger(name).setLevel(logging.CRITICAL)
    litellm.suppress_debug_info = True
    try:
        yield
    finally:
        for name, level in held.items():
            logging.getLogger(name).setLevel(level)
        litellm.suppress_debug_info = was


#: Long enough for a cold local model to load, short enough that a wrong
#: address is a failure rather than a wait. LLM_TIMEOUT_SECONDS is the
#: pipeline's patience with a 473 s passage and is not this.
TIMEOUT_SECONDS = 90.0


class Unreachable(Exception):
    """A configured model did not answer, and a stage must not start."""


class Reachable(BaseModel):
    """The smallest structured answer a provider can be asked for."""

    ok: bool = Field(description="true")


def reachable(settings: Settings) -> None:
    """Asks one model a trivial question, proving the path a stage needs.

    The address resolves, the credential is accepted, the model exists, and
    it can return an object rather than prose. One attempt and a short
    patience: this reports what happened rather than working around it, and
    the usual backoff would hide an address that is simply wrong.

    Args:
        settings: The model to ask, as the caller will call it.

    Raises:
        Unreachable: If it did not answer in shape.
    """
    once = replace(
        settings, timeout_seconds=TIMEOUT_SECONDS, max_attempts=1, num_ctx=None
    )
    try:
        with quiet():
            Client(once).answer(
                system="You answer with a JSON object and nothing else.",
                user="Set ok to true.",
                shape=Reachable,
            )
    except Exception as refusal:
        where = settings.base_url or "the provider's own address"
        raise Unreachable(
            f"{settings.model} at {where} did not answer: "
            f"{type(refusal).__name__}: {refusal}"
        ) from refusal


def before_work(*models: Settings) -> None:
    """Refuses to start when a model a stage is about to call will not answer.

    Passed to `stages.cli.queue_main` as its preflight, by the three stages
    that call a model. Runs before the watch loop rather than inside it,
    because that loop logs an exception and retries - which is right for a
    drain that failed and wrong for a deployment that can never work.

    Args:
        *models: Every model this stage calls, writer and verifier alike.
            Repeats are asked once.

    Raises:
        Unreachable: If any of them did not answer.
    """
    for settings in {(m.model, m.base_url): m for m in models}.values():
        reachable(settings)
        log.info("%s answered; claiming work", settings.model)


def main() -> int:
    """Asks the configured model one question, and says what happened.

    Returns:
        The process exit code: 0 if the model answered in shape.
    """
    telemetry.configure("doctor")

    settings = Settings.load()
    log.info("model    %s", settings.model)
    log.info("address  %s", settings.base_url or "(the provider's own)")
    log.info("mode     %s", settings.structured_mode)

    try:
        reachable(settings)
    except Unreachable as refusal:
        log.error("%s", refusal)
        return 1

    log.info("the model answered in shape")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
