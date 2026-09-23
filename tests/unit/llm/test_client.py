"""Asking a served model, and what happens when it will not answer.

Only a transient failure is worth another attempt: retrying a bad model name
or a malformed schema costs the backoff on every call and buries the error.

A rate limit is transient and is still not retried here. It is retried by
litellm, which reads the provider's Retry-After; the policy in this module
cannot see that header and would give up inside three seconds.
"""

from __future__ import annotations

import instructor
import litellm
import pytest
from pydantic import BaseModel

from llm.client import Client, ModelUnavailable, mode, priced, spend
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
            "num_ctx": None,
            "reasoning_effort": None,
            **overrides,
        }
    )


def transient() -> Exception:
    """A failure worth another attempt here."""
    return litellm.exceptions.Timeout(
        message="too slow", llm_provider="ollama", model="test-model"
    )


def rate_limited() -> Exception:
    """A failure worth another attempt, but not one this module makes."""
    return litellm.exceptions.RateLimitError(
        message="slow down", llm_provider="ollama", model="test-model"
    )


def answered(shape):
    """What `_ask` hands back: the parsed answer and the response it came in."""
    return shape(), None


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
        return answered(shape)

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
    with pytest.raises(ModelUnavailable, match="Timeout"):
        Client(settings(max_attempts=3)).answer(system="s", user="u", shape=Shape)

    assert len(attempts) == 3, attempts


@pytest.mark.parametrize(
    "failure", [permanent, rate_limited, lambda: ValueError("bad schema")]
)
def test_a_failure_this_module_does_not_retry_is_raised_at_once(
    monkeypatch, failure
) -> None:
    """One attempt, not three.

    For a bad key or a bad schema the backoff would only bury the reason. For
    a rate limit litellm has already spent the retries, each of them waiting
    as long as the provider asked, so trying again here adds nothing.
    """
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


# ── What reaches the runtime, and what is left out ────────────────────────


def sent(**overrides) -> dict:
    """The keywords one call passes to the completion, with the model stubbed."""
    client = Client(settings(**overrides))
    seen: dict = {}

    def record(**kwargs):
        """Keeps the call and answers in the shape asked for."""
        seen.update(kwargs)
        return Shape(), None

    client._client.chat.completions.create_with_completion = record
    client._attempt = lambda system, user, shape: client._ask(system, user, shape)
    client.answer(system="s", user="u", shape=Shape)
    return seen


def test_a_context_window_reaches_the_runtime_when_one_is_set() -> None:
    """A self-hosted runtime reserves the whole window before it reads.

    A model advertising 131,072 tokens holds gigabytes of key-value cache for
    a prompt of two thousand, which on one machine was the difference between
    generating and swapping.
    """
    assert sent(num_ctx=8192)["num_ctx"] == 8192


def test_nothing_is_sent_about_the_window_to_a_hosted_provider() -> None:
    """It has no such parameter and refuses a request carrying it."""
    assert "num_ctx" not in sent(num_ctx=None, model="azure/gpt-4.1")


def test_a_self_hosted_model_is_given_a_window_it_can_serve() -> None:
    """Left to its own, a runtime reserves the model's advertised one.

    Measured here: gemma4:12b loaded at 131,072 tokens, which is several
    gigabytes of cache for a prompt of 1,441 and one request served at a
    time - so a worker holding a call blocked every other process.
    """
    assert sent(num_ctx=None, model="ollama_chat/gemma4:12b")["num_ctx"] == 8192


def test_the_reasoning_effort_reaches_the_runtime_when_one_is_set() -> None:
    """A thinking model asked for a structured answer answers with nothing.

    Measured on a 12B model: 7,469 tokens of reasoning, the length limit, and
    an empty string back after 307 seconds. The same call with thinking off
    took 9 seconds. Ollama reads this parameter as `think`.
    """
    assert sent(reasoning_effort="off")["reasoning_effort"] == "off"


def test_a_hosted_model_is_left_its_own_default() -> None:
    """`off` is Ollama's spelling and a hosted provider refuses it."""
    assert "reasoning_effort" not in sent(reasoning_effort=None, model="azure/gpt-4.1")


def test_a_self_hosted_model_is_told_not_to_think() -> None:
    """Without it a writer call took 414.7 seconds instead of 12.6.

    8,555 output tokens, of which 32,436 characters were reasoning and
    283 were the answer. Nothing in the deployment had to ask for this:
    every call here wants a structured answer, and a thinking model given
    one reasons instead of answering.
    """
    assert (
        sent(reasoning_effort=None, model="ollama_chat/gemma4:12b")["reasoning_effort"]
        == "off"
    )


def test_the_rate_limit_retries_are_handed_to_the_runtime() -> None:
    """One budget covers both paths, so a 429 gets the tries a timeout does.

    litellm reads the provider's Retry-After and waits that long; a hosted
    provider asks for twenty to sixty seconds, which is why this is not the
    exponential backoff around `_ask`.
    """
    assert sent(max_attempts=3)["num_retries"] == 2


def test_a_single_attempt_asks_the_runtime_for_no_retries() -> None:
    """Nothing to spend, rather than one retry nobody asked for."""
    assert sent(max_attempts=1)["num_retries"] == 0


# ── What a call is recorded as having cost ────────────────────────────────


class Usage:
    """What a provider reports it billed for."""

    def __init__(self, prompt_tokens=None, completion_tokens=None) -> None:
        """Initialises the counts one response carries."""
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class Response:
    """A completion, as far as the fields read off one are concerned."""

    def __init__(self, usage=None) -> None:
        """Initialises the response with the usage it reports, if any."""
        self.usage = usage


def test_the_tokens_a_provider_reports_are_recorded(monkeypatch) -> None:
    """Read off the response: the provider is what knows what it billed for."""
    monkeypatch.setattr(litellm, "completion_cost", lambda **_: 0.0042)
    fields = spend(Response(Usage(prompt_tokens=1204, completion_tokens=88)))

    assert fields["llm.tokens.input"] == 1204
    assert fields["llm.tokens.output"] == 88
    assert fields["llm.cost_usd"] == pytest.approx(0.0042)


def test_a_model_with_no_published_price_records_no_cost(monkeypatch) -> None:
    """Absent, not zero.

    Zero is what litellm costs a self-hosted model at, and a zero is a number
    a dashboard sums and reports as free. The tokens are still recorded.
    """
    monkeypatch.setattr(litellm, "completion_cost", lambda **_: 0.0)
    fields = spend(Response(Usage(prompt_tokens=10, completion_tokens=2)))

    assert "llm.cost_usd" not in fields
    assert fields["llm.tokens.input"] == 10


def test_a_response_the_pricing_cannot_read_costs_nothing(monkeypatch) -> None:
    """A paid-for answer is not lost to the arithmetic about what it cost."""

    def unreadable(**_):
        """What litellm raises for a response it cannot read a model off."""
        raise ValueError("Model is None and does not exist in completion_response")

    monkeypatch.setattr(litellm, "completion_cost", unreadable)

    assert spend(Response(Usage(prompt_tokens=10, completion_tokens=2))) == {
        "llm.tokens.input": 10,
        "llm.tokens.output": 2,
    }


def test_a_response_carrying_no_usage_records_nothing(monkeypatch) -> None:
    """A field with nothing behind it is worse than no field."""
    monkeypatch.setattr(litellm, "completion_cost", lambda **_: 0.0)

    assert spend(Response()) == {}


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({}, ""),
        ({"llm.tokens.input": 1204}, " (1,204 in)"),
        (
            {
                "llm.tokens.input": 1204,
                "llm.tokens.output": 88,
                "llm.cost_usd": 0.0042,
            },
            " (1,204 in, 88 out, $0.0042)",
        ),
    ],
)
def test_what_a_person_watching_the_logs_reads(fields, expected) -> None:
    """The JSON copy carries the fields whether or not this renders them."""
    assert priced(fields) == expected


def test_the_shape_and_the_prompt_version_reach_the_log_line(
    monkeypatch, caplog
) -> None:
    """What a call was and which prompt asked it, on the line it writes.

    `llm.shape` is what `make spend-by-shape` reads. `llm.prompt_version`
    is what says two runs are two datasets - eight versions of the
    question prompt were declared before anything recorded which one a
    call had used.
    """
    monkeypatch.setattr(
        Client, "_ask", lambda self, system, user, shape: answered(shape)
    )

    with caplog.at_level("INFO"):
        Client(settings()).answer(system="s", user="u", shape=Shape, prompt_version="8")

    written = [one for one in caplog.records if one.name == "llm.client"]
    assert written, [one.name for one in caplog.records]
    assert written[-1].__dict__["llm.shape"] == "Shape"
    assert written[-1].__dict__["llm.prompt_version"] == "8"


def test_a_prompt_with_no_version_claims_none(monkeypatch, caplog) -> None:
    """Absent says something: this prompt has never been versioned.

    An empty string would be a version, and would group with every other
    unversioned call as though they were one prompt.
    """
    monkeypatch.setattr(
        Client, "_ask", lambda self, system, user, shape: answered(shape)
    )

    with caplog.at_level("INFO"):
        Client(settings()).answer(system="s", user="u", shape=Shape)

    written = [one for one in caplog.records if one.name == "llm.client"]
    assert "llm.prompt_version" not in written[-1].__dict__


def test_the_span_is_told_the_same_two_things(monkeypatch) -> None:
    """One fact said twice, spelled the same in the trace and the log.

    The instrumentor names every model call `completion`, so without these
    a run is two hundred identical spans. Read off the context the way the
    instrumentor reads it, rather than by exporting a span: what is being
    checked is that the context carries them at all.
    """
    from openinference.instrumentation import get_attributes_from_context

    seen: dict[str, object] = {}

    def record(self, system, user, shape):
        """Reads what the instrumentor would read, mid-call."""
        seen.update(dict(get_attributes_from_context()))
        return answered(shape)

    monkeypatch.setattr(Client, "_ask", record)
    Client(settings()).answer(system="s", user="u", shape=Shape, prompt_version="8")

    assert seen["tag.tags"] == ["Shape"]
    assert seen["llm.prompt_template.version"] == "8"
