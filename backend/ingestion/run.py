"""Command-line entry point for the ingestion service.

Deleting is here rather than under a stage because a document belongs to
ingestion. The same operations are on the API under /documents. Run with
--help for the flags.
"""

from __future__ import annotations

import argparse
import logging
import sys

import telemetry
from database.qa_generator import engine
from ingestion.config import Settings
from ingestion.factory import build_removal, build_service

log = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    """Builds the argument parser.

    The three are mutually exclusive: two in one command silently ran only
    the first, and one of them is irreversible.
    """
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
    return built


def main(argv: list[str] | None = None) -> int:
    """Lists stored documents, or deletes one.

    A digest is taken in full rather than as a prefix: this is the operation
    where guessing wrong cannot be undone.
    """
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)

    settings = Settings.load()
    telemetry.configure("ingestion", settings.log_level)
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
