"""Command-line entry point for reading and changing settings.

Every flag is the same operation as the route beside it under /settings:

  python -m settings.run --service topics                GET  /settings/topics
  python -m settings.run --service topics --set K=V      PATCH
  python -m settings.run --service topics --unset K      PATCH, with a null

One command for all seven services rather than a flag on each stage's own
line, because `platform` has no stage and no worker to hang a flag off. The
Makefile wraps it per service, which is where the per-stage wording lives.

Nothing here runs a stage. A setting written reaches a worker when that
worker next claims a row; what a change staled is printed, and rebuilding it
is the stage's own rerun.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import telemetry
from database.qa_generator import engine
from settings import catalog, changes
from settings.store import Settings

log = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    """Builds the argument parser."""
    built = argparse.ArgumentParser(prog="python -m settings.run")
    built.add_argument(
        "--service",
        required=True,
        choices=sorted({one.service for one in catalog.SETTINGS}),
        help="which service to read or change. One service per command, "
        "because one page configures one service.",
    )
    built.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        dest="written",
        help="change one setting. Repeatable; every change in one command is "
        "written together, and the whole command is refused if any one value "
        "is.",
    )
    built.add_argument(
        "--unset",
        action="append",
        default=[],
        metavar="NAME",
        dest="cleared",
        help="return one setting to whatever configs/env/backend.env or .env "
        "says. Repeatable.",
    )
    return built


def proposed(written: list[str], cleared: list[str]) -> dict[str, str | None]:
    """Reads the changes a command asked for.

    Raises:
        SystemExit: If a `--set` carries no `=`, or a name is given twice.
    """
    values: dict[str, str | None] = {}
    for entry in written:
        name, sign, value = entry.partition("=")
        if not sign:
            raise SystemExit(f"--set takes NAME=VALUE; {entry!r} has no '='")
        values[name.strip()] = value.strip()
    for name in cleared:
        if name.strip() in values:
            raise SystemExit(
                f"{name.strip()} is both set and unset in one command, and "
                f"only one of those can be what was meant"
            )
        values[name.strip()] = None
    return values


def report(service: str, store: Settings) -> list[str]:
    """Reads what one service is configured to do, one line per setting.

    The value it reads now beside what the files say, so the two are
    comparable at a glance and a changed setting is obvious without having
    to know which file to open.

    Returns the lines rather than logging them, so what a person sees is
    what a test can read.
    """
    resolved = store.resolved()
    overrides = store.overrides()
    lines = [f"{service} settings, at version {store.version()}:"]
    for setting in catalog.of(service):
        value = resolved.get(setting.name) or ""
        default = os.environ.get(setting.name) or ""
        where = (
            "deployment"
            if setting.fixed
            else ("stored" if setting.name in overrides else "file")
        )
        said = f", file says {default or '(unset)'}" if where == "stored" else ""
        lines.append(f"  {setting.name:<32} {value or '(unset)':<28} {where}{said}")
    return lines


def change(service: str, values: dict[str, str | None], store: Settings) -> list[str]:
    """Writes what a command asked for, refusing what the stage would not read.

    Through `settings.changes`, which is what the route uses too, so a
    terminal is refused for the same reasons a page is and in the same
    words. No version is given: a person at a terminal is deciding against
    what they just read, and there is no page that went stale in between.

    Returns the lines rather than logging them, for the same reason `report`
    does.

    Raises:
        SystemExit: If a setting is not this service's, is one the deployment
            owns, or would stop a service that is working now.
    """
    try:
        moved = changes.apply(service, values, store=store)
    except changes.Refused as refusal:
        raise SystemExit(refusal.detail) from None

    lines = [f"{service}: {moved.detail}"]
    if moved.stale:
        lines.append(
            f"what {', '.join(moved.stale)} already produced was made under "
            f"the old values; rerun that stage to rebuild it"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    """Runs the settings command line.

    Args:
        argv: The arguments to parse, or None to read them from sys.argv.

    Returns:
        The process exit code.
    """
    args = parser().parse_args(sys.argv[1:] if argv is None else argv)

    telemetry.configure("settings")
    telemetry.trace_engine(engine())

    store = Settings()
    values = proposed(args.written, args.cleared)
    said = (
        change(args.service, values, store) if values else report(args.service, store)
    )
    for line in said:
        log.info("%s", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
