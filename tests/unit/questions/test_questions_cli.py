"""The question generation command line.

Every flag is the same operation as a route under /questions. Most of them
are `stages.cli.queue_main`'s and are covered there; what is this stage's
own is the two operations no other stage has - the re-check and the release
draw - and the lease, which is derived from how much a topic costs rather
than from how much one row does.
"""

from __future__ import annotations

import os
from typing import ClassVar

import pytest


class Queue:
    """A question queue that records what was asked of it."""

    scopes: ClassVar[dict[str, None]] = {"topic": None}

    def __init__(self, lease=None, **ignored) -> None:
        """Records the lease it was built with."""
        del ignored
        self.lease = lease
        self.calls: list[str] = []

    def counts_by_status(self, within=None) -> dict[str, int]:
        """The queue depth."""
        self.calls.append("counts")
        return {"new": 2}

    def start(self, within=None, *, trigger: str | None = None) -> int:
        """Queues what was never asked for."""
        self.calls.append("start")
        return 1

    def stop(self, within=None) -> int:
        """Takes back what has not begun."""
        self.calls.append("stop")
        return 1

    def retry(self, within=None, *, trigger: str | None = None) -> int:
        """Returns failed rows."""
        self.calls.append("retry")
        return 1

    def reset(self, within=None, *, trigger: str | None = None) -> int:
        """Queues every row again."""
        self.calls.append("rerun")
        return 1

    def narrowed(self, scope: str, value: str):
        """The condition one scope selects."""
        from stages.queue import Unnarrowable

        if scope != "topic":
            raise Unnarrowable("unknown_scope", "this stage narrows to topic")
        return f"{scope}={value}"


class Service:
    """A service that records each drain."""

    def __init__(self) -> None:
        """Initialises with nothing drained."""
        self.drains = 0

    def drain(self, stopping=None) -> int:
        """Records one drain."""
        del stopping
        self.drains += 1
        return 0


class Driver:
    """The command line with its collaborators replaced."""

    def __init__(self, monkeypatch) -> None:
        """Points every collaborator at something that records."""
        from question_generation import run as module
        from stages import cli

        self.queue = Queue()
        self.service = Service()
        self.built = 0
        self.rechecked: list = []
        self.balanced: list = []
        self.watched: list[float] = []
        #: The models the preflight was asked to prove, in order. Replaced
        #: because the real one calls them, and a unit test serves nothing.
        self.proved: list[str] = []
        self.refuse: Exception | None = None
        #: How many refusals are left before the preflight relents, and
        #: what a watching worker slept between them. None means forever.
        self.refusals: int | None = None
        self.slept: list[float] = []

        def build():
            """Counts the builds and hands back the recording service."""
            self.built += 1
            return self.service

        # Model identity, which compose gives a worker and .env gives the
        # host. conftest loads configs/env/backend.env, which carries the
        # rest of the model settings and deliberately not this one.
        monkeypatch.setenv("LLM_MODEL", "ollama_chat/test-model")

        # The settings the command reads, without the database they are
        # stored in. Settings.load is still the real one.
        monkeypatch.setattr(
            module, "snapshot", lambda: (dict(os.environ), "test-settings")
        )
        monkeypatch.setattr(module, "QuestionQueue", lambda **kwargs: self.queue)
        monkeypatch.setattr(module, "QuestionCatalog", lambda: object())
        monkeypatch.setattr(module, "build_service", lambda *_: build())
        monkeypatch.setattr(
            module, "reverify", lambda catalog, settings, within: self._recheck(within)
        )
        monkeypatch.setattr(
            module, "balance", lambda catalog, settings, within: self._balance(within)
        )

        def proving(*models):
            """Records what was asked, and refuses when the test says to."""
            self.proved.extend(one.model for one in models)
            if self.refuse is None:
                return
            if self.refusals is None or self.refusals > 0:
                if self.refusals is not None:
                    self.refusals -= 1
                raise self.refuse

        monkeypatch.setattr(module, "before_work", proving)
        # Recorded rather than waited out: the retry is what is under test,
        # and the real ceiling is a minute.
        monkeypatch.setattr(cli.time, "sleep", self.slept.append)
        monkeypatch.setattr(cli.telemetry, "configure", lambda *_, **__: None)
        monkeypatch.setattr(cli.telemetry, "trace_engine", lambda *_: None)
        monkeypatch.setattr(cli, "engine", lambda: None)
        monkeypatch.setattr(
            cli, "watch", lambda _, seconds: self.watched.append(seconds)
        )
        self._main = module.main

    def _recheck(self, within) -> int:
        """Records one re-check and what it was narrowed to."""
        self.rechecked.append(within)
        return 3

    def _balance(self, within) -> int:
        """Records one draw and what it was narrowed to."""
        self.balanced.append(within)
        return 7

    def run(self, *argv: str) -> int:
        """Runs the command line."""
        return self._main(list(argv))


@pytest.fixture
def cli(monkeypatch) -> Driver:
    """The command line, wired to recorders."""
    return Driver(monkeypatch)


def test_status_reports_the_queue_and_builds_no_service(cli) -> None:
    """The cheap flag: one query, and no model named."""
    assert cli.run("--status") == 0
    assert cli.queue.calls == ["counts"]
    assert cli.built == 0, "a status read loaded the writer"


@pytest.mark.parametrize("flag", ["start", "stop", "retry"])
def test_each_queue_verb_is_the_route_beside_it(cli, flag: str) -> None:
    """The same operation POST /questions/{flag} makes."""
    assert cli.run(f"--{flag}") == 0
    assert cli.queue.calls == [flag]
    assert cli.service.drains == 0


def test_no_flag_at_all_drains_what_is_already_queued(cli) -> None:
    """A bare run works the queue once and returns."""
    assert cli.run() == 0
    assert cli.service.drains == 1


def test_rerun_queues_everything_again_and_then_drains(cli) -> None:
    """Both, in that order."""
    assert cli.run("--rerun") == 0
    assert cli.queue.calls == ["rerun"]
    assert cli.service.drains == 1


def test_watch_keeps_draining_instead_of_returning(cli) -> None:
    """What the worker container runs."""
    assert cli.run("--watch") == 0
    assert cli.service.drains == 0, "watch drains on its own schedule"
    assert cli.watched, "the watch loop was never entered"


def test_every_model_this_stage_calls_is_proved_before_anything_is_claimed(
    cli,
) -> None:
    """The verifier and the phrasing judge as well as the writer.

    A run whose verifier will not answer is a run of ungated questions,
    which is worse than no run at all. QUESTIONS_PHRASING_MODEL may name a
    model neither of the other two does, and an unreachable one abstains
    once per question instead of failing the start.
    """
    assert cli.run("--watch") == 0
    assert len(cli.proved) == 3, "the writer, the verifier and the phrasing judge"


def test_a_model_that_will_not_answer_stops_a_drain(cli) -> None:
    """A one-off drain gives up, because a person holds the exit code.

    `stages.worker.watch` logs an exception and polls again, which is right
    for a drain that failed and wrong for a deployment that can never work:
    a worker with no usable credential would claim a topic every poll and
    fail it, emptying the queue into `failed` while reporting itself up.
    """
    cli.refuse = RuntimeError("no credential")

    assert cli.run() == 1, "a refused preflight is a non-zero exit"
    assert not cli.watched, "the watch loop must not be entered"
    assert cli.queue.calls == [], "nothing was claimed"
    assert cli.slept == [], "a drain does not wait for a model"


def test_a_watching_worker_waits_for_its_model_rather_than_exiting(cli) -> None:
    """The process stays, and the model coming back is enough.

    Exiting costs a process, and under `restart: unless-stopped` a process
    is a restart, a fresh `run_id` and a Phoenix project holding the one
    call that failed. An expired credential once cost 2,019 restarts and
    2,116 such projects.
    """
    cli.refuse = RuntimeError("ollama is not up yet")
    cli.refusals = 3

    assert cli.run("--watch") == 0, "the worker started once the model answered"
    assert cli.slept == [5.0, 10.0, 20.0], "it backs off between attempts"
    assert cli.watched, "and then watches, having claimed nothing before"


def test_the_wait_between_attempts_has_a_ceiling(cli) -> None:
    """A minute, so a model down overnight is not a log line every second."""
    cli.refuse = RuntimeError("still down")
    cli.refusals = 8

    cli.run("--watch")

    assert max(cli.slept) == 60.0, cli.slept


def test_two_actions_in_one_command_are_refused(cli) -> None:
    """They are mutually exclusive, so the parser rejects the pair."""
    with pytest.raises(SystemExit):
        cli.run("--reverify", "--balance")


# ── The two operations no other stage has ─────────────────────────────────


def test_reverify_puts_the_stored_questions_through_the_free_gates(cli) -> None:
    """No model is called, so a re-check costs a pass over the table."""
    assert cli.run("--reverify") == 0
    assert cli.rechecked == [None]
    assert cli.built == 0, "the re-check calls no model and needs no service"


def test_balance_draws_a_release_without_building_a_service(cli) -> None:
    """The draw is a column on rows already there."""
    assert cli.run("--balance") == 0
    assert cli.balanced == [None]
    assert cli.built == 0


def test_a_re_check_can_be_narrowed_to_one_topic(cli) -> None:
    """`--only` reaches this stage's own operations too."""
    assert cli.run("--reverify", "--only", "topic=3") == 0
    assert cli.rechecked == ["topic=3"]


def test_the_only_scope_this_stage_takes_is_a_topic(cli) -> None:
    """A topic is already the smallest subject there is."""
    with pytest.raises(SystemExit, match="narrows to topic"):
        cli.run("--status", "--only", "document=abc")


def test_a_narrowing_without_a_value_is_refused(cli) -> None:
    """The message names the form."""
    with pytest.raises(SystemExit, match="SCOPE=VALUE"):
        cli.run("--status", "--only", "topic")


# ── The lease a topic needs ───────────────────────────────────────────────


def test_the_queue_is_built_without_a_lease_of_its_own(monkeypatch) -> None:
    """A topic is held by the beat, not by a lease sized for the work.

    This stage used to derive one from what a topic costs, which made a
    killed worker's topic unreachable for as long as the work might have
    taken. The queue takes the shared default now.
    """
    from question_generation import run as module
    from stages import cli as shared

    held = {}

    def queue(**kwargs):
        """Records what the run asked the queue for."""
        held.update(kwargs)
        return Queue(**kwargs)

    monkeypatch.setenv("LLM_MODEL", "ollama_chat/test-model")
    monkeypatch.setattr(module, "snapshot", lambda: (dict(os.environ), "test-settings"))
    monkeypatch.setattr(module, "QuestionQueue", queue)
    monkeypatch.setattr(module, "build_service", lambda *_: Service())
    monkeypatch.setattr(shared.telemetry, "configure", lambda *_, **__: None)
    monkeypatch.setattr(shared.telemetry, "trace_engine", lambda *_: None)
    monkeypatch.setattr(shared, "engine", lambda: None)

    module.main(["--status"])

    assert "lease" not in held, held
