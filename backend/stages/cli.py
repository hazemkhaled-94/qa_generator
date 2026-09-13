"""The command line every pipeline stage shares.

One drain on the host, in the foreground, against the same database the
workers use. The flags mirror the stage's HTTP surface one for one.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable

import telemetry
from database.qa_generator import engine
from settings import decimal
from stages.queue import StageQueue
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


def narrowing(queue: StageQueue, only: str | None):
    """Reads `--only` into the condition it selects, or nothing.

    Raises:
        SystemExit: If it is not `scope=value`, names a scope this stage does
            not take, or carries a value that column cannot hold.
    """
    if only is None:
        return None
    scope, _, value = only.partition("=")
    accepted = ", ".join(queue.scopes) or "nothing"
    if not value:
        raise SystemExit(f"--only takes SCOPE=VALUE, where SCOPE is one of {accepted}")
    try:
        return queue.narrow(scope, value)
    except KeyError:
        raise SystemExit(f"this stage narrows to {accepted}, not {scope!r}") from None
    except ValueError:
        raise SystemExit(f"{value!r} is not a valid {scope}") from None


def queue_main(
    *,
    name: str,
    module: str,
    log_level: str,
    repository: Callable[[], StageQueue],
    build_service: Callable[[], StageService],
    argv: list[str],
) -> int:
    """Runs one stage's command line.

    The repository and the service are built only when a flag needs them, so
    `--status` costs a query and not a converter.
    """
    args = parser(module, _ACTIONS).parse_args(argv)

    telemetry.configure(name, log_level)
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

    if args.rerun:
        log.info("%s: %d row(s) queued again", name, act(lambda q, w: q.reset(w)))

    service = build_service()
    if not args.watch:
        service.drain()
        return 0

    watch(service.drain, decimal("WORKER_POLL_SECONDS"))
    return 0
