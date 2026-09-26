"""Command-line entry point for recalibrating the stored confidences."""

from __future__ import annotations

import logging
import sys

from settings import decimal
from stages.recalibrate import main

log = logging.getLogger(__name__)


def run() -> int:
    """Recalibrates both tables from the floors the settings name."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    written = main(
        questions_floor=decimal("QUESTIONS_DUPLICATE_FLOOR"),
        facts_floor=decimal("EXTRACTION_DUPLICATE_FLOOR"),
    )
    log.info("recalibrated %d row(s) in total", written)
    return 0


if __name__ == "__main__":
    sys.exit(run())
