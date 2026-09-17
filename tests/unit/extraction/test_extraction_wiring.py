"""How the service is assembled, and what its command line offers.

Nothing here calls a model. What is under test is that a deployment's
settings reach the objects that read them, and that every operation this
stage has is reachable from the command line.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from database.qa_generator import FactKind
from extraction.config import Settings
from extraction.factory import build_bridge, build_service
from llm.config import Settings as ModelSettings

MODEL = ModelSettings(
    model="ollama/test-model",
    base_url="http://model.invalid",
    structured_mode="JSON_SCHEMA",
    temperature=0.0,
    timeout_seconds=1.0,
    max_attempts=1,
    num_ctx=None,
    reasoning_effort="off",
)


@pytest.fixture(autouse=True)
def _no_engine_left_behind():
    """Forgets the engine building a repository cached, whatever it was.

    A queue binds to the session factory as it is constructed, and that
    factory is cached for the process against whatever DATABASE_URL held at
    the time. Building one here with no database would otherwise hand the
    placeholder engine to whichever layer runs next.
    """
    from database.qa_generator.engine import engine, sessions

    yield
    engine.cache_clear()
    sessions.cache_clear()


def settings(*kinds: str, share: float = 0.6, other: float = 0.0) -> Settings:
    """The extraction settings, built rather than read.

    `other` is off by default, so a test about the wiring is not also a
    test of what the atomic cap works out to.
    """
    return Settings(
        kinds=frozenset(kinds),
        digest_share=share,
        min_other_share=other,
        bridges_per_topic=5,
        bridge_passages=2,
    )


def test_a_deployment_writing_digests_gets_a_digest_reader() -> None:
    """The kinds it asked for, and no others."""
    built = build_service(MODEL, settings(FactKind.ATOMIC, FactKind.SUMMARY))

    assert built._digest is not None
    assert built._digest._kinds == (FactKind.SUMMARY,)


def test_a_deployment_writing_only_atomic_facts_gets_no_digest_reader() -> None:
    """No reader means no second call per passage."""
    built = build_service(MODEL, settings(FactKind.ATOMIC))

    assert built._digest is None


def test_the_table_reader_is_claimed_by_name_and_the_model_is_the_default() -> None:
    """A cell grid needs no model; everything else does."""
    built = build_service(MODEL, settings(FactKind.ATOMIC))

    assert built._extractors.block_types == ("table",)
    assert built._extractors.for_block_type("table").method == "deterministic"
    assert built._extractors.for_block_type("text").method == "llm"


def test_the_lease_is_derived_from_the_model_settings() -> None:
    """A healthy worker may take the timeout on every attempt."""
    built = build_service(MODEL, settings(FactKind.ATOMIC))

    assert built._repository.lease == timedelta(seconds=2)


def test_the_digest_share_reaches_the_checker() -> None:
    """The one setting the checks themselves read."""
    from drivers import passage

    from extraction.models import CandidateFact

    strict = build_service(MODEL, settings(FactKind.ATOMIC, share=0.01))
    digest = CandidateFact("The device weighs 4 kg.", (0, 1), kind=FactKind.SUMMARY)

    judged = strict._checker.check(passage(), digest, "llm")
    assert judged.rejection_code == "not_condensed", judged.validation_error


def test_the_bridge_reader_is_built_apart_from_the_queue() -> None:
    """Its unit of work is a group of passages, so it is not routed."""
    assert build_bridge(MODEL).provenance.model == "ollama/test-model"


def test_the_settings_are_read_from_the_environment_when_none_are_given() -> None:
    """A worker starting up reads what the deployment declared."""
    built = build_service(MODEL)

    assert built._checker is not None
    assert built._extractors.block_types == ("table",)


class TestCommandLine:
    """Every operation this stage has is reachable from one command."""

    @pytest.fixture(autouse=True)
    def served(self, monkeypatch) -> None:
        """Which model to call lives in .env, which no test layer reads."""
        monkeypatch.setenv("LLM_MODEL", "ollama/test-model")

    @pytest.fixture
    def usage(self, capsys) -> str:
        """What `--help` prints."""
        from extraction.run import main

        with pytest.raises(SystemExit):
            main(["--help"])
        return capsys.readouterr().out

    @pytest.mark.parametrize(
        "flag",
        ["--status", "--start", "--stop", "--retry", "--rerun", "--watch", "--only"],
    )
    def test_every_shared_queue_verb_is_offered(self, usage, flag) -> None:
        """The flags mirror the stage's HTTP surface one for one."""
        assert flag in usage, usage

    @pytest.mark.parametrize("flag", ["--revalidate", "--bridge"])
    def test_this_stage_own_operations_are_offered(self, usage, flag) -> None:
        """Neither is a queue verb, and neither has a route."""
        assert flag in usage, usage

    def test_two_operations_in_one_command_are_refused(self, capsys) -> None:
        """They are mutually exclusive: the second used to be ignored."""
        from extraction.run import main

        with pytest.raises(SystemExit):
            main(["--bridge", "--revalidate"])
        assert "not allowed with" in capsys.readouterr().err
