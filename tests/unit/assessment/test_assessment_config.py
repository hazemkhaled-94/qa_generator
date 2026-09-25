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
    "ASSESSMENT_JUDGE_MODEL": "azure/gpt-5.4",
    "ASSESSMENT_KINDS": "fact,topic,question",
    "ASSESSMENT_SAMPLE": "200",
}


def settings(**changed: str) -> Settings:
    """The settings, read from VALUES with some values changed."""
    return Settings.load({**VALUES, **changed})


#: A self-hosted writer, which is what makes the judge below a
#: CROSS-PROVIDER override rather than a change of model.
_SHARED = {
    "LLM_MODEL": "ollama_chat/gemma4:31b",
    "LLM_BASE_URL": "http://localhost:11434",
    "OLLAMA_BASE_URL": "http://localhost:11434",
    "LLM_STRUCTURED_MODE": "JSON",
    "LLM_TEMPERATURE": "0",
    "LLM_TIMEOUT_SECONDS": "60",
    "LLM_MAX_ATTEMPTS": "3",
}


def model(**changed: str) -> ModelSettings:
    """The shared model settings, with whatever a test needs changed."""
    return ModelSettings.load({**_SHARED, **changed})


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
    assert judge_model(settings(), model()).model == "azure/gpt-5.4"
    assert (
        judge_model(settings(ASSESSMENT_JUDGE_MODEL=""), model()).model
        == "ollama_chat/gemma4:31b"
    )


def test_a_hosted_judge_does_not_keep_the_local_runtime_address() -> None:
    """The deployment this is pointed at judges on Azure and writes locally.

    `overridden` drops a base URL it can prove belongs to the runtime being
    left, which is when it is the one OLLAMA_BASE_URL names. Kept, an Azure
    request would be sent to Ollama's port - which is the failure that
    argument exists to stop, in the other direction.
    """
    asked = judge_model(settings(), model())

    assert asked.model == "azure/gpt-5.4"
    assert asked.base_url is None, "litellm reads AZURE_API_BASE instead"
    assert asked.window is None, "a hosted provider sizes its own context"


def test_a_judge_on_the_writers_own_provider_keeps_the_address() -> None:
    """A same-provider judge is the one-thing change this always was.

    Naming another deployment of the provider already configured moves the
    model and nothing else.
    """
    asked = judge_model(
        settings(),
        model(
            LLM_MODEL="azure/gpt-4.1",
            LLM_BASE_URL="https://example.openai.azure.com",
            OLLAMA_BASE_URL="",
        ),
    )

    assert asked.model == "azure/gpt-5.4"
    assert asked.base_url == "https://example.openai.azure.com"


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
