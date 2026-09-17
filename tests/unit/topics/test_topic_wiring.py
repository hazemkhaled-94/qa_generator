"""What the factory builds, and what it reads to build it.

The settings are the deployment's only way in, so a value the fitter would
refuse has to stop the process here rather than at the first fit.
"""

from __future__ import annotations

import pytest

from topic_modelling.config import Settings
from topic_modelling.factory import build_service


@pytest.fixture
def unserved(monkeypatch):
    """A deployment with no model to name a topic with."""
    monkeypatch.delenv("LLM_MODEL", raising=False)
    return Settings.load()


# ── Reading the environment ───────────────────────────────────────────────


def test_the_fitter_settings_are_read_from_the_environment(unserved) -> None:
    """The values in configs/env/backend.env, which the tests load."""
    assert unserved.num_topics == 40
    assert unserved.passages_per_topic == 40
    assert unserved.passes == 10
    assert unserved.random_state == 42
    assert unserved.top_terms == 12
    assert unserved.min_weight == 0.05
    assert unserved.no_below == 3
    assert unserved.no_above == 0.5


def test_the_language_names_are_read_as_pairs(unserved) -> None:
    """The naming prompt says which language to answer in by name."""
    assert unserved.languages == {"de": "German", "en": "English"}


def test_no_model_configured_is_a_valid_deployment(unserved) -> None:
    """Topics are still fitted; they are just left unnamed."""
    assert unserved.model is None


def test_a_served_model_is_read_when_one_is_configured(monkeypatch) -> None:
    """The same LLM_MODEL every other stage reads."""
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-4o-mini")

    assert Settings.load().model is not None


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("TOPIC_NUM_TOPICS", "twelve"),
        ("TOPIC_MIN_WEIGHT", "a lot"),
        ("TOPIC_NO_BELOW", "3.5"),
        ("TOPIC_NO_ABOVE", "half"),
    ],
)
def test_a_setting_that_is_not_a_number_stops_the_process(
    monkeypatch, name, value
) -> None:
    """Read at start-up, so a bad value fails the container and not a fit."""
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        Settings.load()


@pytest.mark.parametrize(
    "name",
    [
        "TOPIC_NUM_TOPICS",
        "TOPIC_PASSES",
        "TOPIC_RANDOM_STATE",
        "TOPIC_TOP_TERMS",
        "TOPIC_MIN_WEIGHT",
        "TOPIC_NO_BELOW",
        "TOPIC_NO_ABOVE",
        "TOPIC_LANGUAGE_NAMES",
    ],
)
def test_every_setting_is_required(monkeypatch, name) -> None:
    """None of them has a default: compose empties an unset one.

    Raises KeyError rather than ValueError, which is what `required` answers
    an absent setting with.
    """
    monkeypatch.setenv(name, "")

    with pytest.raises(KeyError, match=name):
        Settings.load()


def test_a_language_pair_without_a_name_is_refused(monkeypatch) -> None:
    """`de:German`, not `de`: the prompt has nothing to say otherwise."""
    monkeypatch.setenv("TOPIC_LANGUAGE_NAMES", "de")

    with pytest.raises(ValueError):
        Settings.load()


# ── Wiring it up ──────────────────────────────────────────────────────────


def test_the_service_is_built_without_a_labeller_when_nothing_is_served(
    unserved, caplog
) -> None:
    """A fit without names is still a fit, and the log says so."""
    with caplog.at_level("WARNING"):
        built = build_service(unserved)

    assert built._labeller is None
    assert "no model configured" in caplog.text


def test_the_service_is_built_with_a_labeller_when_a_model_is_served(
    monkeypatch,
) -> None:
    """The one collaborator a deployment can do without."""
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-4o-mini")

    built = build_service(Settings.load())

    assert built._labeller is not None
    assert built._labeller.model == "openai/gpt-4o-mini"


def test_the_built_service_carries_the_settings_into_the_fitter(unserved) -> None:
    """A setting that reached nothing would be a setting nobody could change."""
    built = build_service(unserved)

    assert built._fitter._num_topics == unserved.num_topics
    assert built._fitter._no_above == unserved.no_above


def test_the_built_service_names_itself_as_a_stage(unserved) -> None:
    """Every span and log line carries it."""
    built = build_service(unserved)

    assert built.name == "topic_modelling"
    assert built.unit == "fit"


def test_a_setting_the_fitter_refuses_stops_the_build(monkeypatch) -> None:
    """Not at the first fit, an hour into a run."""
    monkeypatch.setenv("TOPIC_NUM_TOPICS", "1")

    with pytest.raises(ValueError, match="TOPIC_NUM_TOPICS"):
        build_service(Settings.load())
