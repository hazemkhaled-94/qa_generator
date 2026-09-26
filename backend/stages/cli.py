"""The command line every pipeline stage shares.

One drain on the host, in the foreground, against the same database the
workers use. The flags mirror the stage's HTTP surface one for one.

A watching worker builds its service again when the settings change. The
check sits between drains, where nothing is claimed.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable
from typing import Any

import telemetry
from database.qa_generator import engine
from settings import decimal
from settings.runs import named_run, run_id
from stages.queue import StageQueue, Unnarrowable
from stages.service import StageService
from stages.worker import watch

log = logging.getLogger(__name__)

#: The queue actions every stage answers, and the route each mirrors.
_ACTIONS = {
    "status": "report the queue depth",
    "start": "queue the rows never asked for",
    "stop": "take back what has not begun",
    "retry": "return failed rows to the queue",
    "reclaim": "return a row a dead worker still holds, without waiting "
    "out its lease. Narrow it with --only unless the stage is stopped",
    "rerun": "queue every row again, finished ones included",
}

#: One operation only a single stage has, as (help, verb, run). `run` takes
#: whatever `--only` narrowed to and returns how many rows it changed.
Extra = tuple[str, str, Callable[[Any], int]]

#: How long a watching worker waits before asking the model again, and the
#: ceiling it doubles towards. Unchanged by the quietening below: how often
#: a worker TRIES decides how fast it recovers, and a model that comes back
#: should be noticed within the minute.
_RETRY_SECONDS = 5.0
_RETRY_CEILING = 60.0

#: How often a worker that still cannot reach its model says so again.
#:
#: How often it SHOUTS is a different question from how often it tries, and
#: they used to be the same one. A worker polling every 60s wrote a line
#: every 60s, and around it the client wrote a traceback, instructor wrote
#: its attempts and a credential library listed every identity it had: 836
#: lines in four minutes, from one worker, about one unreachable address.
#:
#: So the first refusal is reported in full and the repeats are counted.
#: Ten minutes is long enough that an overnight outage is a handful of
#: lines and short enough that `make logs` on a broken deployment still
#: says what is wrong without scrolling.
_RETRY_REPORT_SECONDS = 600.0


def parser(module: str, actions: dict[str, str]) -> argparse.ArgumentParser:
    """Builds a stage's argument parser.

    The actions are mutually exclusive: two of them in one command silently
    ran only the first.
    """
    built = argparse.ArgumentParser(prog=f"python -m {module}")
    group = built.add_mutually_exclusive_group()
    for flag, description in actions.items():
        group.add_argument(f"--{flag}", action="store_true", help=description)
    built.add_argument(
        "--only",
        metavar="SCOPE=VALUE",
        help="narrow the action to one item, as document=<sha256> or "
        "passage=<id>. Which scopes a stage takes is its own; --only on a "
        "stage that takes none is refused.",
    )
    built.add_argument("--watch", action="store_true", help="drain and keep draining")
    return built


def reloading(
    name: str, build_service: Callable[[], StageService]
) -> Callable[[Callable[[], bool]], int]:
    """Wraps a drain so it rebuilds the service when the settings change.

    One small query per poll - the version is a digest of the stored
    overrides - and a rebuild only when something moved.

    ponytail: checked between drains, not per row, so a change made during a
    long run waits for that run to finish. The ceiling is that a rebuild
    reloads 2.2 GB of embedding weights in one stage; the upgrade is to
    check inside `StageService.drain`'s loop and rebuild only the
    collaborators whose settings differ.
    """
    from settings.store import current

    held: dict[str, Any] = {}

    def drain(stopping: Callable[[], bool]) -> int:
        """Drains the queue, on a service built from the current settings."""
        version = current()
        if held.get("version") != version:
            if held:
                log.info(
                    "%s: settings are now %s; building the service again",
                    name,
                    version,
                )
            held["service"] = build_service()
            held["version"] = version
        return held["service"].drain(stopping)

    return drain


def narrowing(queue: StageQueue, only: str | None):
    """Reads `--only` into the condition it selects, or nothing.

    Raises:
        SystemExit: If it is not `scope=value`, names a scope this stage does
            not take, or carries a value that column cannot hold. The last
            two are the queue's own wording, which the route answers with too.
    """
    if only is None:
        return None
    scope, _, value = only.partition("=")
    if not value:
        accepted = ", ".join(queue.scopes) or "nothing"
        raise SystemExit(f"--only takes SCOPE=VALUE, where SCOPE is one of {accepted}")
    try:
        return queue.narrowed(scope, value)
    except Unnarrowable as exc:
        raise SystemExit(str(exc)) from None


def queue_main(
    *,
    name: str,
    module: str,
    repository: Callable[[], StageQueue],
    build_service: Callable[[], StageService],
    argv: list[str],
    extra: dict[str, Extra] | None = None,
    preflight: Callable[[], None] | None = None,
    prompts: Callable[[], int] | None = None,
) -> int:
    """Runs one stage's command line.

    The repository and the service are built only when a flag needs them, so
    `--status` costs a query and not a converter. `extra` adds the operations
    only this stage has, in the same mutually exclusive group as the rest.

    `preflight` is what a stage proves before it claims anything - for the
    three that call a model, that the model answers. It raises, and the
    process stops with nothing taken.

    `prompts` records what this stage sends, so the version on each row it
    writes can be resolved to the text that produced it. Beside preflight
    because both are things done once before the first claim; unlike
    preflight it cannot fail a start, since a prompt nobody can read back
    is worse than a run that did not happen only if the run happened.
    """
    extra = extra or {}
    actions = {**_ACTIONS, **{flag: help for flag, (help, _, _) in extra.items()}}
    args = parser(module, actions).parse_args(argv)

    # Logs now, tracing once this process knows it is going to work.
    #
    # The exporter is what creates a Phoenix project, and the first span a
    # stage produces is the preflight's - the call it makes to prove the
    # model answers. A deployment whose credential has expired therefore
    # left a project per attempt holding one failed call, which under
    # `restart: unless-stopped` is a project per restart. That is the same
    # failure the comment in `_ready` describes from the other side, and
    # this is the half of it the retry loop could not fix.
    #
    # A provider can be installed once per process, so this cannot be done
    # by configuring twice; `tracing=False` leaves it uninstalled.
    telemetry.configure(name, run=run_id(), named=named_run(), tracing=False)

    def act(run):
        """Runs one queue operation, narrowed to whatever --only names."""
        queue = repository()
        return run(queue, narrowing(queue, args.only))

    if args.status:
        log.info("%s queue: %s", name, act(lambda q, w: q.counts_by_status(w)))
        return 0

    # Every verb that queues rows stamps them with this process's run, so
    # the drain that picks them up - a different process, or a container
    # started weeks ago - writes what it produces under the id of whatever
    # asked. RUN_ID names it where somebody named it; see settings.runs.
    asking = run_id()

    for chosen, verb, run in (
        (args.start, "queued", lambda q, w: q.start(w, trigger=asking)),
        (args.stop, "taken off the queue", lambda q, w: q.stop(w)),
        (args.retry, "returned to the queue", lambda q, w: q.retry(w, trigger=asking)),
        (args.reclaim, "reclaimed", lambda q, w: q.reclaim(w, trigger=asking)),
    ):
        if chosen:
            log.info("%s: %d row(s) %s", name, act(run), verb)
            return 0

    for flag, (_, verb, run) in extra.items():
        if getattr(args, flag):
            log.info("%s: %d row(s) %s", name, act(lambda q, w, do=run: do(w)), verb)
            return 0

    if args.rerun:
        log.info(
            "%s: %d row(s) queued again",
            name,
            act(lambda q, w: q.reset(w, trigger=asking)),
        )

    # Here rather than inside the loop below, which logs an exception and
    # polls again: that is right for a drain that failed and wrong for a
    # deployment that can never work. A worker with no usable credential
    # would otherwise claim a row every poll and fail it, and the queue
    # would empty into `failed` while the container reported itself up.
    if preflight is not None and not _ready(name, preflight, args.watch):
        return 1

    # Past here the process is going to claim rows, so its spans are worth
    # a project. Named with the run only when somebody named the run: an
    # unnamed one appends to the stage's own project and is told apart by
    # the `run.id` attribute. See telemetry/pipeline.py.
    telemetry.configure(name, run=run_id(), named=named_run())
    telemetry.trace_engine(engine())

    # After preflight, so a deployment that cannot work does not leave a
    # record of prompts it never sent. Never raises; see stages.prompts.
    if prompts is not None:
        prompts()

    if not args.watch:
        # Built once: one drain cannot outlive a change to the settings.
        build_service().drain()
        return 0

    watch(reloading(name, build_service), decimal("WORKER_POLL_SECONDS"))
    return 0


def _ready(name: str, preflight: Callable[[], None], watching: bool) -> bool:
    """Whether the model answered, waiting for it where this is a worker.

    A one-off drain gives up, because a person is holding the exit code.
    A **watching worker waits in place** instead, and that is the whole
    point of this function: exiting costs a process, and under
    `restart: unless-stopped` a process is a restart, a new `run_id`, and
    a Phoenix project holding the one call that failed. An expired
    credential once cost 2,019 restarts and 2,116 such projects, which is
    how a pipeline that was down for a day looked like a pipeline that had
    run two thousand times.

    Waiting also recovers by itself. A model that is down at boot and up
    ten minutes later is the ordinary case for one served off a laptop,
    and it needs nobody to run `make up` again.
    """
    delay = _RETRY_SECONDS
    attempts = 0
    began = time.monotonic()
    said = 0.0
    while True:
        try:
            preflight()
            if attempts:
                # The recovery is worth a line of its own. Without it the
                # log goes quiet and a reader cannot tell a worker that
                # started working from one that stopped complaining.
                log.info(
                    "%s reached its model after %d attempt(s) over %s, and is "
                    "claiming work",
                    name,
                    attempts + 1,
                    _for(time.monotonic() - began),
                )
            return True
        except Exception as refusal:  # noqa: BLE001 - any of them means wait
            # One line rather than a traceback: this is a deployment that is
            # wrong, not a bug, and the person reading `podman logs` needs
            # the reason and not the call stack.
            if not watching:
                log.error("%s will not start: %s", name, refusal)
                return False
            waited = time.monotonic() - began
            if not attempts:
                log.error(
                    "%s cannot reach its model and will keep trying every "
                    "%.0fs, reporting again every %.0f minutes until it can: "
                    "%s",
                    name,
                    _RETRY_CEILING,
                    _RETRY_REPORT_SECONDS / 60,
                    refusal,
                )
                said = waited
            elif waited - said >= _RETRY_REPORT_SECONDS:
                # Counted rather than repeated. The reason is in the first
                # line; what a reader needs now is whether it is still
                # happening and for how long.
                log.warning(
                    "%s still cannot reach its model: %d attempt(s) over %s",
                    name,
                    attempts + 1,
                    _for(waited),
                )
                said = waited
            attempts += 1
            time.sleep(delay)
            delay = min(delay * 2, _RETRY_CEILING)


def _for(seconds: float) -> str:
    """How long something has been going on, for a person to read."""
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"
