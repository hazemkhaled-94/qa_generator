"""Command-line entry point for the ingestion service.

The same operations are on the API under /documents. Run with --help for the
flags.
"""

from __future__ import annotations

import argparse
import logging
import sys

import telemetry
from database.qa_generator import engine
from ingestion.config import Settings
from ingestion.factory import build_removal, build_service
from settings.runs import run_id

log = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    """Builds the argument parser. The flags are mutually exclusive."""
    built = argparse.ArgumentParser(prog="python -m ingestion.run")
    group = built.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="list every document")
    group.add_argument(
        "--delete",
        metavar="SHA256",
        help="remove a document, its file and everything derived from it",
    )
    group.add_argument(
        "--delete-derived",
        metavar="SHA256",
        help="drop only the passages and facts, keeping the document",
    )
    group.add_argument(
        "--delete-all",
        action="store_true",
        help="remove every document, its files and the upload history. Topics "
        "are not a document's and go separately; `make wipe` runs both",
    )
    return built


def main(argv: list[str] | None = None) -> int:
    """Lists stored documents, or deletes one.

    A digest is taken in full rather than as a prefix.

    Args:
        argv: Arguments to parse, defaulting to the process's own.

    Returns:
        0, or 1 when no document has the given digest.
    """
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)

    settings = Settings.load()
    telemetry.configure("ingestion", run=run_id())
    telemetry.trace_engine(engine())

    if args.list:
        _, documents = build_service(settings).documents(limit=1000)
        for document in documents:
            log.info(
                "%s  %-9s %s",
                document.sha256,
                document.parse_status,
                document.filename or "-",
            )
        return 0

    removal = build_removal()
    if args.delete_all:
        for gone in removal.delete_every():
            log.info("%s: %d passage(s) removed", gone.sha256, gone.passages)
        return 0

    digest, remove = (
        (args.delete, removal.delete)
        if args.delete
        else (args.delete_derived, removal.delete_derived)
    )
    removed = remove(digest)
    if removed is None:
        log.error("no document with digest %s", digest)
        return 1
    log.info(
        "%s: %d passage(s) removed, document=%s file=%s parsed=%s",
        removed.sha256,
        removed.passages,
        removed.document,
        removed.file,
        removed.parsed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
