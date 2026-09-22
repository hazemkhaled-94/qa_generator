"""The golden-set command line.

It calls a real model, so nothing here does. What is covered is which
operation each flag selects and the two exit codes that are not failures:
a set with no task, and a run with nothing accepted in it.
"""

from __future__ import annotations

import pytest

from evaluation import experiments
from evaluation.run import parser


class Driver:
    """The command line with Phoenix and the model replaced."""

    def __init__(self, monkeypatch) -> None:
        """Points every collaborator at something that records."""
        from evaluation import run as module

        self.uploaded: list[str] = []
        self.scored: list[str] = []
        self.judged: list[tuple[str, int | None]] = []
        self.refuse: Exception | None = None

        monkeypatch.setattr(module.telemetry, "configure", lambda *_, **__: None)
        monkeypatch.setattr(module.Settings, "load", staticmethod(lambda: object()))
        monkeypatch.setattr(
            module.experiments, "upload", lambda name, settings: self._upload(name)
        )
        monkeypatch.setattr(
            module.experiments, "run", lambda name, settings: self._score(name)
        )
        monkeypatch.setattr(module, "_second_opinion", self._opinion)
        self._main = module.main

    def _upload(self, name: str) -> str:
        """Records one upload."""
        self.uploaded.append(name)
        return "id"

    def _score(self, name: str) -> str:
        """Records one experiment, or raises what it was told to."""
        if self.refuse is not None:
            raise self.refuse
        self.scored.append(name)
        return "http://phoenix/experiment"

    def _opinion(self, run: str, limit: int | None) -> int:
        """Records one second opinion."""
        self.judged.append((run, limit))
        return 0

    def run(self, *argv: str) -> int:
        """Runs the command line."""
        return self._main(list(argv))


@pytest.fixture
def cli(monkeypatch) -> Driver:
    """The command line, wired to recorders."""
    return Driver(monkeypatch)


@pytest.mark.parametrize("name", experiments.NAMES)
def test_each_golden_set_can_be_uploaded(cli, name: str) -> None:
    """Every set is browsable in Phoenix, scored here or not."""
    assert cli.run("--upload", name) == 0
    assert cli.uploaded == [name]


def test_a_set_with_a_task_is_scored(cli) -> None:
    """Extraction's numbers are a measurement rather than a gate."""
    assert cli.run("--score", experiments.EXTRACTION) == 0
    assert cli.scored == [experiments.EXTRACTION]


def test_a_set_with_no_task_exits_two_rather_than_crashing(cli) -> None:
    """It has none on purpose, and the message says what measures it."""
    cli.refuse = NotImplementedError("a gate rather than a measurement")

    assert cli.run("--score", experiments.QUESTIONS) == 2


def test_a_second_opinion_is_asked_over_one_run(cli) -> None:
    """Never a verdict: the output is a queue for review."""
    assert cli.run("--second-opinion", "run-7") == 0
    assert cli.judged == [("run-7", None)]


def test_a_second_opinion_takes_a_limit(cli) -> None:
    """One model call each, so a person decides how many."""
    assert cli.run("--second-opinion", "run-7", "--limit", "25") == 0
    assert cli.judged == [("run-7", 25)]


def test_an_action_is_required(cli) -> None:
    """A bare run would otherwise read the settings and stop."""
    with pytest.raises(SystemExit):
        cli.run()


def test_two_actions_in_one_command_are_refused(cli) -> None:
    """They are mutually exclusive."""
    with pytest.raises(SystemExit):
        cli.run("--upload", experiments.EXTRACTION, "--score", experiments.EXTRACTION)


def test_a_set_nothing_answers_to_is_refused(cli) -> None:
    """Before a dataset of that name is created in Phoenix."""
    with pytest.raises(SystemExit):
        cli.run("--upload", "passages-golden")


def test_every_set_the_module_names_is_offered_by_the_parser() -> None:
    """A set the tool cannot name is a set nobody scores."""
    for name in experiments.NAMES:
        assert parser().parse_args(["--score", name]).score == name


# ── The second opinion, which asks the verifier's model ───────────────────


@pytest.fixture
def opinion(monkeypatch):
    """The real `_second_opinion`, over a judge that answers from a script."""
    from evaluation import run as module
    from evaluation import second_opinion

    monkeypatch.setattr(module.telemetry, "configure", lambda *_, **__: None)
    monkeypatch.setattr(module.Settings, "load", staticmethod(lambda: object()))
    # Model identity, which compose gives a worker and backend.env does not.
    monkeypatch.setenv("LLM_MODEL", "ollama_chat/test-model")
    asked: list[tuple] = []

    def judging(answer):
        """Points `judge` at whatever a test wants it to do."""

        def judge(run: str, model: str, limit: int):
            asked.append((run, model, limit))
            if isinstance(answer, Exception):
                raise answer
            return answer

        monkeypatch.setattr(second_opinion, "judge", judge)
        monkeypatch.setattr(second_opinion, "report", lambda judged: ["a line"])
        return asked

    return module.main, judging


def test_a_second_opinion_asks_the_verifier_s_model(opinion) -> None:
    """A model marking its own work agrees with itself."""
    main, judging = opinion
    asked = judging([])

    assert main(["--second-opinion", "run-7"]) == 0
    run, model, limit = asked[0]
    assert (run, limit) == ("run-7", 200)
    assert model, "no model was named"


def test_a_limit_reaches_the_judge(opinion) -> None:
    """One model call each."""
    main, judging = opinion
    asked = judging([])

    assert main(["--second-opinion", "run-7", "--limit", "5"]) == 0
    assert asked[0][2] == 5


def test_a_run_with_nothing_accepted_exits_two(opinion) -> None:
    """Not a crash: there is nothing to have a second opinion about."""
    main, judging = opinion
    judging(RuntimeError("no such run"))

    assert main(["--second-opinion", "run-nothing"]) == 2
