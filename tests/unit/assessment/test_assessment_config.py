"""The flag, the judge and the two cost dials.

The flag is the one setting in this package with a behaviour of its own:
off, the command line says so and stops before it claims anything, which
is a different thing from finding an empty queue and has to read as one.
"""

from __future__ import annotations

import pytest

from assessment import run as cli
from assessment.config import Settings, judged_kinds
from assessment.factory import judge_model
from llm.config import Settings as ModelSettings

#: A complete set of values, for the reader to vary one of.
VALUES = {
    "ASSESSMENT_ENABLED": "true",
    "ASSESSMENT_JUDGE_MODEL": "ollama_chat/qwen3:14b",
    "ASSESSMENT_KINDS": "fact,topic,question",
    "ASSESSMENT_SAMPLE": "200",
}


def settings(**changed: str) -> Settings:
    """The settings, read from VALUES with some values changed."""
    return Settings.load({**VALUES, **changed})


def model() -> ModelSettings:
    """Model settings a lease can be derived from."""
    return ModelSettings.load(
        {
            "LLM_MODEL": "ollama_chat/gemma4:31b",
            "LLM_STRUCTURED_MODE": "JSON",
            "LLM_TEMPERATURE": "0",
            "LLM_TIMEOUT_SECONDS": "60",
            "LLM_MAX_ATTEMPTS": "3",
        }
    )


def test_the_flag_is_read() -> None:
    """Both ways, because off is the default and has to be reachable."""
    assert settings().enabled is True
    assert settings(ASSESSMENT_ENABLED="false").enabled is False


def test_a_sample_of_zero_means_every_artefact() -> None:
    """Which on a large corpus is a model call per fact, so it is opt-in."""
    assert settings(ASSESSMENT_SAMPLE="0").unlimited is True
    assert settings().unlimited is False


def test_the_kinds_are_checked_against_the_templates() -> None:
    """A kind with no template is a queue row nothing could judge."""
    assert judged_kinds(settings()) == ("fact", "topic", "question")

    with pytest.raises(ValueError, match="passage"):
        judged_kinds(settings(ASSESSMENT_KINDS="fact,passage"))


def test_the_judge_is_the_model_the_setting_names() -> None:
    """And is the shared model when it names nothing, which is warned about."""
    assert judge_model(settings(), model()).model == "ollama_chat/qwen3:14b"
    assert (
        judge_model(settings(ASSESSMENT_JUDGE_MODEL=""), model()).model
        == "ollama_chat/gemma4:31b"
    )


def test_the_command_line_stops_when_the_phase_is_off(monkeypatch, caplog) -> None:
    """Saying so, and without building a service or reaching a database.

    A deployment that has not turned this on and cannot tell "off" from
    "nothing to judge" will read the second as a bug in the first.
    """
    monkeypatch.setattr(
        "assessment.run.snapshot",
        lambda: ({**VALUES, "ASSESSMENT_ENABLED": "false"}, "environment"),
    )
    built = []
    monkeypatch.setattr(
        "assessment.run.build_service", lambda *a: built.append(a) or None
    )

    with caplog.at_level("INFO"):
        assert cli.main(["--status"]) == 0

    assert not built, "nothing is built when the phase is off"
    assert "ASSESSMENT_ENABLED" in caplog.text
