"""Command-line entry point for the archive.

What the deletions left behind, and the second deletion that empties it.
Run with --help for the flags.
"""

from __future__ import annotations

import argparse
import logging
import sys

import telemetry
from archive.store import Archive
from database.qa_generator import engine

log = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    """Builds the argument parser. The two actions are mutually exclusive."""
    built = argparse.ArgumentParser(prog="python -m archive.run")
    group = built.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--status", action="store_true", help="what the archive holds, by table"
    )
    group.add_argument(
        "--purge",
        action="store_true",
        help="delete what it holds. Irreversible, unlike the first deletion",
    )
    built.add_argument(
        "--table", metavar="TABLE", help="purge only rows deleted from this table"
    )
    built.add_argument(
        "--older-than",
        metavar="DAYS",
        type=int,
        help="purge only what was archived more than this many days ago",
    )
    built.add_argument(
        "--all",
        action="store_true",
        help="purge everything, rows and objects alike",
    )
    return built


def _sized(count: int) -> str:
    """Renders a byte count in the largest unit that leaves it above one."""
    size = float(count)
    for unit in ("B", "kB", "MB"):
        if size < 1024:
            return f"{size:,.0f} {unit}"
        size /= 1024
    return f"{size:,.1f} GB"


def _status(archive: Archive) -> int:
    """Reports what is held, by table, and what the bucket holds."""
    held = archive.rows()
    for table in held:
        log.info(
            "%-18s %8d row(s)  %10s  %s to %s",
            table.table_name,
            table.rows,
            _sized(table.bytes),
            table.oldest.date(),
            table.newest.date(),
        )
    if not held:
        log.info("no archived rows")
    else:
        log.info(
            "%-18s %8d row(s)  %10s",
            "total",
            sum(table.rows for table in held),
            _sized(sum(table.bytes for table in held)),
        )

    objects, size = archive.objects()
    log.info("%-18s %8d object(s) %10s", "archive bucket", objects, _sized(size))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Reports what the archive holds, or empties it.

    A purge says what it is purging: a table, an age, or `--all`. Nothing
    is the shape of a typo rather than a request, and this is the deletion
    with nothing behind it.

    Args:
        argv: Arguments to parse, defaulting to the process's own.

    Returns:
        0, or 2 when a purge named nothing to purge.
    """
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)

    telemetry.configure("archive")
    telemetry.trace_engine(engine())
    archive = Archive()

    if args.status:
        return _status(archive)

    if not (args.table or args.older_than or args.all):
        log.error(
            "a purge names what it purges: --table, --older-than, or --all. "
            "This is the deletion the archive was protecting against"
        )
        return 2

    purged = archive.purge(table=args.table, older_than_days=args.older_than)
    log.info(
        "purged %d row(s) and %d object(s)%s",
        purged.rows,
        purged.objects,
        f" from {args.table}" if args.table else "",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
