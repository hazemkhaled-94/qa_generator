"""The command line every pipeline stage shares.

One drain on the host, in the foreground, against the same database the
workers use. The flags mirror the stage's HTTP surface one for one.

A watching worker builds its service again when the settings change, so a
value written through a page or a route reaches it without a restart. It
never rebuilds part-way through a row: the check sits between drains, which
is where nothing is claimed.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from typing import Any

import telemetry
from database.qa_generator import engine
from settings import decimal
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
    "rerun": "queue every row again, finished ones included",
}

#: One operation only a single stage has, as (help, verb, run). `run` takes
#: whatever `--only` narrowed to and returns how many rows it changed.
Extra = tuple[str, str, Callable[[Any], int]]


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

    The service is built from settings, and holds what they decided: which
    model to call, what a passage is measured against, what a gate lets
    through. A worker that read them once answers to the file it started
    with for as long as it runs, which is what made the UI unable to
    configure anything.

    The version is a digest of the stored overrides, so this is one small
    query per poll and a rebuild only when something actually moved.

    ponytail: checked between drains rather than per row. A drain works the
    queue until it is empty, so a change made during a long run is picked up
    when that run finishes rather than part-way through - which is the right
    trade for question generation, where rebuilding means loading 2.2 GB of
    embedding weights again. The upgrade, if a long run ever has to turn on
    a sixpence, is to check in `StageService.drain`'s loop and rebuild only
    the collaborators whose settings differ.
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
) -> int:
    """Runs one stage's command line.

    The repository and the service are built only when a flag needs them, so
    `--status` costs a query and not a converter. `extra` adds the operations
    only this stage has, in the same mutually exclusive group as the rest.
    """
    extra = extra or {}
    actions = {**_ACTIONS, **{flag: help for flag, (help, _, _) in extra.items()}}
    args = parser(module, actions).parse_args(argv)

    telemetry.configure(name)
    telemetry.trace_engine(engine())

    def act(run):
        """Runs one queue operation, narrowed to whatever --only names."""
        queue = repository()
        return run(queue, narrowing(queue, args.only))

    if args.status:
        log.info("%s queue: %s", name, act(lambda q, w: q.counts_by_status(w)))
        return 0

    for chosen, verb, run in (
        (args.start, "queued", lambda q, w: q.start(w)),
        (args.stop, "taken off the queue", lambda q, w: q.stop(w)),
        (args.retry, "returned to the queue", lambda q, w: q.retry(w)),
    ):
        if chosen:
            log.info("%s: %d row(s) %s", name, act(run), verb)
            return 0

    for flag, (_, verb, run) in extra.items():
        if getattr(args, flag):
            log.info("%s: %d row(s) %s", name, act(lambda q, w, do=run: do(w)), verb)
            return 0

    if args.rerun:
        log.info("%s: %d row(s) queued again", name, act(lambda q, w: q.reset(w)))

    if not args.watch:
        # Built once, from the settings as they stand: one drain cannot
        # outlive a change to them.
        build_service().drain()
        return 0

    watch(reloading(name, build_service), decimal("WORKER_POLL_SECONDS"))
    return 0
