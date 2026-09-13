"""The JSON line the log shipper reads.

An `@timestamp` Filebeat cannot parse is replaced with the time the line was
read, and an exception rendered without `error.stack_trace` reaches the
dashboard as a failure with no cause.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime

import pytest

from telemetry.logs import JsonFormatter, _file


@pytest.fixture
def line() -> dict:
    """One formatted record: an error with a traceback and a caller's field."""
    record = logging.LogRecord(
        name="extraction.service",
        level=logging.ERROR,
        pathname="extraction/service.py",
        lineno=104,
        msg="failed passage %d",
        args=(41,),
        exc_info=None,
        func="process_next",
    )
    try:
        raise ValueError("the model would not answer")
    except ValueError:
        record.exc_info = sys.exc_info()
    # As telemetry.add_trace_fields puts them on: zeros outside a span.
    record.otelTraceID = "0" * 32
    record.otelSpanID = "0" * 16
    record.passage_id = 41  # a caller's own field, passed as extra=

    return json.loads(JsonFormatter("extraction").format(record))


def test_the_ecs_fields_are_named_and_filled(line: dict) -> None:
    """The level, the logger, the service, the message and the origin."""
    assert line["log.level"] == "error", line["log.level"]
    assert line["log.logger"] == "extraction.service", line["log.logger"]
    assert line["service.name"] == "extraction", line["service.name"]
    assert line["message"] == "failed passage 41", line["message"]
    assert line["log.origin.file.line"] == 104, line["log.origin.file.line"]
    assert line["trace.id"] == "0" * 32, line["trace.id"]


def test_a_callers_own_fields_survive(line: dict) -> None:
    """What was passed as extra= is carried beside the rest."""
    assert line["passage_id"] == 41


def test_the_whole_traceback_is_carried(line: dict) -> None:
    """The type, the message and the stack, not just the message."""
    assert line["error.type"] == "ValueError", line["error.type"]
    assert line["error.message"] == "the model would not answer"
    assert "Traceback" in line["error.stack_trace"], line["error.stack_trace"]
    assert "ValueError" in line["error.stack_trace"]


def test_the_timestamp_is_parseable_offset_aware_and_utc(line: dict) -> None:
    """What Filebeat needs to keep it rather than substitute its own."""
    assert line["@timestamp"].endswith("Z"), line["@timestamp"]
    when = datetime.fromisoformat(line["@timestamp"])
    assert when.tzinfo is not None, line["@timestamp"]
    assert when.utcoffset().total_seconds() == 0, line["@timestamp"]


def test_a_record_renders_as_one_line() -> None:
    """The shipper reads one event per line."""
    record = logging.LogRecord(
        name="extraction.service",
        level=logging.ERROR,
        pathname="extraction/service.py",
        lineno=104,
        msg="failed passage %d",
        args=(41,),
        exc_info=None,
        func="process_next",
    )
    try:
        raise ValueError("the model would not answer")
    except ValueError:
        record.exc_info = sys.exc_info()

    assert "\n" not in JsonFormatter("extraction").format(record)


def test_no_log_dir_means_no_file_handler(monkeypatch) -> None:
    """Unset is every host command."""
    monkeypatch.delenv("LOG_DIR", raising=False)
    assert _file("extraction") is None


def test_a_writable_log_dir_is_created_and_named_per_container(
    monkeypatch, tmp_path
) -> None:
    """A scaled stage runs several writers over one volume."""
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "nested"))

    handler = _file("extraction")
    assert handler is not None, "a writable LOG_DIR must give a handler"
    assert handler.baseFilename.startswith(f"{tmp_path / 'nested'}/extraction-")
    handler.close()


def test_an_unwritable_log_dir_costs_the_shipped_copy_and_nothing_else(
    monkeypatch,
) -> None:
    """Raising here would stop a worker over a mounted volume."""
    monkeypatch.setenv("LOG_DIR", "/proc/nowhere/qa")
    assert _file("extraction") is None
