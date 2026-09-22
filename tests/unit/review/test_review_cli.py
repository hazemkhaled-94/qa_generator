"""The review command line.

Three flags and one thing that is not a flag: `--ids`, which replaces the
sample with a queue somebody else built. What the parser refuses matters
more here than what it accepts - a typo that pushes a sample instead of
the rows asked for looks like it worked.
"""

from __future__ import annotations

import pytest

pytest.importorskip("argilla", reason="the observability group is not installed")

from review import datasets
from review.run import _ids, parser


class Driver:
    """The command line with its collaborators replaced."""

    def __init__(self, monkeypatch) -> None:
        """Points every collaborator at something that records."""
        from review import run as module

        self.pushed: list[tuple] = []
        self.pulled: list[str] = []
        self.counted = 0

        monkeypatch.setattr(module.telemetry, "configure", lambda *_, **__: None)
        monkeypatch.setattr(module.telemetry, "trace_engine", lambda *_: None)
        monkeypatch.setattr(module, "engine", lambda: None)
        monkeypatch.setattr(module, "ReviewRepository", lambda: self)
        monkeypatch.setattr(module, "QuestionCatalog", lambda: object())
        monkeypatch.setattr(module, "TopicCatalog", lambda: object())
        monkeypatch.setattr(module.Settings, "load", staticmethod(lambda: object()))
        monkeypatch.setattr(
            module,
            "push",
            lambda name, settings, catalogs, ids=None: self.pushed.append((name, ids)),
        )
        monkeypatch.setattr(
            module, "pull", lambda name, settings, catalogs: self.pulled.append(name)
        )
        self._main = module.main

    def counts(self) -> dict:
        """What the repository reports."""
        self.counted += 1
        return {"facts": 3, "reviewed": {}}

    def run(self, *argv: str) -> int:
        """Runs the command line."""
        return self._main(list(argv))


@pytest.fixture
def cli(monkeypatch) -> Driver:
    """The command line, wired to recorders."""
    return Driver(monkeypatch)


def test_status_reads_the_counts_and_nothing_else(cli) -> None:
    """The cheap flag: it never reaches Argilla."""
    assert cli.run("--status") == 0
    assert cli.counted == 1
    assert cli.pushed == [] and cli.pulled == []


@pytest.mark.parametrize("name", datasets.NAMES)
def test_each_dataset_can_be_pushed(cli, name: str) -> None:
    """One per thing a model decided and a person may disagree with."""
    assert cli.run("--push", name) == 0
    assert cli.pushed == [(name, None)]


@pytest.mark.parametrize("name", datasets.NAMES)
def test_each_dataset_can_be_pulled(cli, name: str) -> None:
    """The verdicts come home through the same three names."""
    assert cli.run("--pull", name) == 0
    assert cli.pulled == [name]


def test_a_named_queue_of_ids_replaces_the_sample(cli) -> None:
    """What `make second-opinion` prints reaches somebody who can settle it."""
    assert cli.run("--push", "questions", "--ids", "4,9,11") == 0
    assert cli.pushed == [("questions", [4, 9, 11])]


def test_an_action_is_required(cli) -> None:
    """A bare run would otherwise do nothing and exit 0."""
    with pytest.raises(SystemExit):
        cli.run()


def test_two_actions_in_one_command_are_refused(cli) -> None:
    """They are mutually exclusive."""
    with pytest.raises(SystemExit):
        cli.run("--push", "facts", "--pull", "facts")


def test_a_dataset_nothing_answers_to_is_refused(cli) -> None:
    """Before it reaches Argilla and creates one."""
    with pytest.raises(SystemExit):
        cli.run("--push", "passages")


# ── The queue `--ids` names ───────────────────────────────────────────────


def test_no_ids_named_is_a_sample(cli) -> None:
    """Which is the ordinary way to review."""
    assert _ids(None, datasets.QUESTIONS) is None
    assert _ids("", datasets.QUESTIONS) is None


@pytest.mark.parametrize("named", ["4,9,11", "4 9 11", "4, 9,11"])
def test_ids_are_read_however_they_are_separated(named: str) -> None:
    """A person pasting a line of output should not have to reformat it."""
    assert _ids(named, datasets.QUESTIONS) == [4, 9, 11]


def test_ids_that_are_not_numbers_are_refused() -> None:
    """A typo would otherwise push a sample and look like it worked."""
    with pytest.raises(SystemExit, match="not a list"):
        _ids("4,nine", datasets.QUESTIONS)


@pytest.mark.parametrize("name", [datasets.FACTS, datasets.TOPIC_LABELS])
def test_ids_are_refused_for_a_set_with_no_queue_behind_it(name: str) -> None:
    """The other two are sampled, so naming ids means something went wrong."""
    with pytest.raises(SystemExit, match="is not that set"):
        _ids("4,9", name)


def test_every_dataset_name_is_offered_by_the_parser() -> None:
    """A set the tool cannot name is a set nobody reviews."""
    for name in datasets.NAMES:
        assert parser().parse_args(["--push", name]).push == name
