"""Asking a served model for an answer in a shape.

Importing this loads litellm. Nothing outside a worker should name it.
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar

import instructor
import litellm
from instructor.core import InstructorRetryException
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llm.config import Settings

log = logging.getLogger(__name__)

#: Failures worth another attempt. An authentication failure, an unknown
#: model or a malformed schema is none of these and is raised at once.
_TRANSIENT = (
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.Timeout,
    litellm.exceptions.RateLimitError,
    litellm.exceptions.InternalServerError,
    litellm.exceptions.ServiceUnavailableError,
    InstructorRetryException,
)

Shape = TypeVar("Shape", bound=BaseModel)


class ModelUnavailable(Exception):
    """Raised when the model could not be reached or would not answer."""


def mode(name: str) -> instructor.Mode:
    """Turns a mode name from the environment into the mode itself."""
    try:
        return instructor.Mode[name.strip().upper()]
    except KeyError:
        raise ValueError(
            f"LLM_STRUCTURED_MODE={name!r} is not an instructor mode; expected "
            f"one of {', '.join(sorted(m.name for m in instructor.Mode))}"
        ) from None


class Client:
    """One served model, asked for answers in a declared shape."""

    def __init__(self, settings: Settings) -> None:
        """Builds the client and the retry policy around one call."""
        self._settings = settings
        # Any: instructor replaces create() at run time, so a checker would
        # match these keywords against the unpatched signature.
        self._client: Any = instructor.from_litellm(
            litellm.completion, mode=mode(settings.structured_mode)
        )
        # Only _TRANSIENT: retrying a bad model name or schema costs the
        # backoff on every call and buries the real error.
        self._attempt = retry(
            retry=retry_if_exception_type(_TRANSIENT),
            stop=stop_after_attempt(settings.max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )(self._ask)

    @property
    def model(self) -> str:
        """The model identifier, recorded on whatever the answer becomes."""
        return self._settings.model

    @property
    def temperature(self) -> float:
        """The sampling temperature, recorded alongside the model."""
        return self._settings.temperature

    def answer(self, *, system: str, user: str, shape: type[Shape]) -> Shape:
        """Asks the model one question and parses the answer into `shape`.

        Raises:
            ModelUnavailable: If it could not be reached, or did not return
                the requested shape after every attempt.
        """
        try:
            return self._attempt(system, user, shape)
        except Exception as exc:
            # Logged with the traceback before it is rewrapped: what the
            # caller records against the row is one line, and litellm's own
            # cause is the only thing that says which of the layers below
            # failed.
            log.exception("%s did not answer", self._settings.model)
            raise ModelUnavailable(f"{type(exc).__name__}: {exc}") from exc

    def _ask(self, system: str, user: str, shape: type[Shape]) -> Shape:
        """Sends one request."""
        return self._client.chat.completions.create(
            model=self._settings.model,
            # Omitted when unset: a provider with its own address would be
            # sent to the wrong one by a base URL meant for Ollama.
            **(
                {"api_base": self._settings.base_url} if self._settings.base_url else {}
            ),
            # Also omitted when unset: a hosted provider has no such parameter
            # and sizes its own window.
            **({"num_ctx": self._settings.num_ctx} if self._settings.num_ctx else {}),
            # A thinking model asked for a structured answer spends its whole
            # window thinking and returns nothing: one 12B model produced 7,469
            # tokens of reasoning, hit the length limit and answered with an
            # empty string, in 307 seconds. With thinking off the same call
            # took 9 seconds. Ollama reads this as `think`.
            **(
                {"reasoning_effort": self._settings.reasoning_effort}
                if self._settings.reasoning_effort
                else {}
            ),
            temperature=self._settings.temperature,
            timeout=self._settings.timeout_seconds,
            response_model=shape,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
