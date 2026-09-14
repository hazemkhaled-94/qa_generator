"""Asking a served model, and what happens when it will not answer.

Only a transient failure is worth another attempt: retrying a bad model name
or a malformed schema costs the backoff on every call and buries the error.
"""

from __future__ import annotations

import instructor
import litellm
import pytest
from pydantic import BaseModel

from llm.client import Client, ModelUnavailable, mode
from llm.config import Settings


class Shape(BaseModel):
    """The shape a caller asks for."""

    value: str = "answered"


def settings(**overrides) -> Settings:
    """Settings for a model nothing actually calls."""
    return Settings(
        **{
            "model": "ollama/test-model",
            "base_url": None,
            "structured_mode": "JSON_SCHEMA",
            "temperature": 0.0,
            "timeout_seconds": 1.0,
            "max_attempts": 2,
            **overrides,
        }
    )


def transient() -> Exception:
    """A failure worth another attempt."""
    return litellm.exceptions.RateLimitError(
        message="slow down", llm_provider="ollama", model="test-model"
    )


def permanent() -> Exception:
    """A failure no number of attempts fixes."""
    return litellm.exceptions.AuthenticationError(
        message="no key", llm_provider="ollama", model="test-model"
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("JSON_SCHEMA", instructor.Mode.JSON_SCHEMA),
        ("json_schema", instructor.Mode.JSON_SCHEMA),
        ("  tools  ", instructor.Mode.TOOLS),
    ],
)
def test_a_mode_name_becomes_the_mode(name: str, expected) -> None:
    """Case and surrounding whitespace do not matter."""
    assert mode(name) == expected


def test_an_unknown_mode_names_the_setting_and_the_choices() -> None:
    """The message says what to put in LLM_STRUCTURED_MODE instead."""
    with pytest.raises(ValueError, match="LLM_STRUCTURED_MODE"):
        mode("not_a_mode")


def test_the_model_and_temperature_are_published_for_provenance() -> None:
    """Recorded on whatever the answer becomes."""
    client = Client(settings(temperature=0.7))

    assert client.model == "ollama/test-model"
    assert client.temperature == 0.7


def test_a_transient_failure_is_retried_and_can_succeed(monkeypatch) -> None:
    """The second attempt's answer is the answer."""
    attempts = []

    def flaky(self, system, user, shape):
        """Fails once, then answers."""
        attempts.append(user)
        if len(attempts) == 1:
            raise transient()
        return shape()

    monkeypatch.setattr(Client, "_ask", flaky)
    answer = Client(settings()).answer(system="s", user="u", shape=Shape)

    assert answer.value == "answered"
    assert len(attempts) == 2, attempts


def test_a_transient_failure_that_never_clears_gives_up(monkeypatch) -> None:
    """Every attempt is spent, then the caller is told."""
    attempts = []

    def always(self, system, user, shape):
        """Never answers."""
        attempts.append(user)
        raise transient()

    monkeypatch.setattr(Client, "_ask", always)
    with pytest.raises(ModelUnavailable, match="RateLimitError"):
        Client(settings(max_attempts=3)).answer(system="s", user="u", shape=Shape)

    assert len(attempts) == 3, attempts


@pytest.mark.parametrize("failure", [permanent, lambda: ValueError("bad schema")])
def test_a_permanent_failure_is_raised_at_once(monkeypatch, failure) -> None:
    """One attempt, not three: the backoff would only bury the reason."""
    attempts = []

    def refuses(self, system, user, shape):
        """Fails in a way no retry fixes."""
        attempts.append(user)
        raise failure()

    monkeypatch.setattr(Client, "_ask", refuses)
    with pytest.raises(ModelUnavailable):
        Client(settings(max_attempts=3)).answer(system="s", user="u", shape=Shape)

    assert len(attempts) == 1, attempts


def test_the_failure_is_rewrapped_with_its_type_and_message(monkeypatch) -> None:
    """One line for the row, with the cause kept on the exception."""
    monkeypatch.setattr(
        Client,
        "_ask",
        lambda self, system, user, shape: (_ for _ in ()).throw(permanent()),
    )

    with pytest.raises(ModelUnavailable) as raised:
        Client(settings()).answer(system="s", user="u", shape=Shape)

    assert "AuthenticationError" in str(raised.value)
    assert isinstance(raised.value.__cause__, litellm.exceptions.AuthenticationError)


def test_the_lease_outlasts_every_attempt() -> None:
    """A healthy worker may take the timeout on each one."""
    lease = settings(timeout_seconds=900, max_attempts=3).lease

    assert lease.total_seconds() >= 900 * 3
