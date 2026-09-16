"""The three command lines, with the services they build stood in for.

What these cover is the wiring and the exit codes: which flags exist, which
combinations are refused, what each one runs and what it returns.
"""

from __future__ import annotations

from typing import Any

import pytest

from ingestion import run as ingestion_run
from ingestion.models import Removal, StoredDocument
from preprocessing.chunking import run as chunking_run
from preprocessing.parsing import run as parsing_run

SHA = "a" * 64


@pytest.fixture(autouse=True)
def quiet(monkeypatch) -> None:
    """Keeps the entry points from configuring logging or an engine."""
    for module in (ingestion_run, parsing_run, chunking_run):
        monkeypatch.setattr(module, "telemetry", _Telemetry(), raising=False)
        monkeypatch.setattr(module, "engine", lambda: None, raising=False)


class _Telemetry:
    """Stands in for the telemetry module."""

    @staticmethod
    def configure(name: str) -> None:
        """Configures nothing."""

    @staticmethod
    def trace_engine(engine: Any) -> None:
        """Instruments nothing."""


# ── Ingestion ──────────────────────────────────────────────────────────────


class _Catalogue:
    """Stands in for the ingest service, listing what it was given."""

    def __init__(self, *documents: StoredDocument) -> None:
        """Takes the documents to list."""
        self.documents_held = list(documents)
        self.limits: list[int] = []

    def documents(self, search=None, limit=100, offset=0):
        """Answers with every document, recording the page asked for."""
        self.limits.append(limit)
        return len(self.documents_held), self.documents_held


class _Removal:
    """Stands in for the removal service, recording what it was asked."""

    def __init__(self, *, removes: bool = True) -> None:
        """Takes whether a digest is known."""
        self.removes = removes
        self.deleted: list[str] = []
        self.derived: list[str] = []

    def delete(self, sha256: str) -> Removal | None:
        """Removes a document completely."""
        self.deleted.append(sha256)
        return self._answer(sha256, document=True)

    def delete_derived(self, sha256: str) -> Removal | None:
        """Drops only what the pipeline built from it."""
        self.derived.append(sha256)
        return self._answer(sha256, document=False)

    def _answer(self, sha256: str, *, document: bool) -> Removal | None:
        """What either deletion reports."""
        if not self.removes:
            return None
        return Removal(
            sha256=sha256,
            document=document,
            passages=3,
            file=document,
            parsed=document,
        )


def stored(sha256: str = SHA, filename: str | None = "report.pdf") -> StoredDocument:
    """One document as the catalogue reports it."""
    return StoredDocument(
        sha256=sha256,
        filename=filename,
        first_seen=None,
        page_count=1,
        title=None,
        language=None,
        parse_status="parsed",
        parse_error=None,
        chunk_status="new",
        chunk_error=None,
        extracted_passages=0,
        total_passages=0,
    )


def test_one_of_the_three_operations_is_required() -> None:
    """Every one of them is a different thing to do."""
    with pytest.raises(SystemExit) as refused:
        ingestion_run.main([])

    assert refused.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["--list", "--delete", SHA],
        ["--delete", SHA, "--delete-derived", SHA],
        ["--list", "--delete-derived", SHA],
    ],
)
def test_two_operations_in_one_command_are_refused(argv: list[str]) -> None:
    """One of them is irreversible, and only the first used to run."""
    with pytest.raises(SystemExit) as refused:
        ingestion_run.main(argv)

    assert refused.value.code == 2


def test_a_deletion_with_no_digest_is_refused() -> None:
    """A digest is taken in full rather than as a prefix."""
    with pytest.raises(SystemExit) as refused:
        ingestion_run.main(["--delete"])

    assert refused.value.code == 2


def test_listing_reports_every_document(monkeypatch, caplog) -> None:
    """Which is the read-only half of this command line."""
    catalogue = _Catalogue(stored(), stored("b" * 64, filename=None))
    monkeypatch.setattr(ingestion_run, "build_service", lambda settings: catalogue)

    with caplog.at_level("INFO"):
        assert ingestion_run.main(["--list"]) == 0

    assert SHA in caplog.text
    assert "report.pdf" in caplog.text
    assert "parsed" in caplog.text
    assert catalogue.limits == [1000]


def test_deleting_a_document_reports_what_went_with_it(monkeypatch, caplog) -> None:
    """The counts, per store, so nothing is assumed."""
    removal = _Removal()
    monkeypatch.setattr(ingestion_run, "build_removal", lambda: removal)

    with caplog.at_level("INFO"):
        assert ingestion_run.main(["--delete", SHA]) == 0

    assert removal.deleted == [SHA]
    assert "3 passage(s) removed" in caplog.text


def test_dropping_the_derived_data_runs_the_other_deletion(monkeypatch) -> None:
    """The two are different operations, not one with a flag."""
    removal = _Removal()
    monkeypatch.setattr(ingestion_run, "build_removal", lambda: removal)

    assert ingestion_run.main(["--delete-derived", SHA]) == 0
    assert (removal.derived, removal.deleted) == ([SHA], [])


def test_a_digest_no_document_has_fails_the_command(monkeypatch, caplog) -> None:
    """A non-zero exit, so a script notices."""
    monkeypatch.setattr(ingestion_run, "build_removal", lambda: _Removal(removes=False))

    with caplog.at_level("ERROR"):
        assert ingestion_run.main(["--delete", SHA]) == 1

    assert f"no document with digest {SHA}" in caplog.text


# ── The two stage command lines ────────────────────────────────────────────


class _QueueMain:
    """Stands in for the shared stage command line, recording its wiring."""

    def __init__(self) -> None:
        """Starts with nothing recorded."""
        self.called: dict[str, Any] = {}

    def __call__(self, **kwargs: Any) -> int:
        """Records what the entry point asked for."""
        self.called = kwargs
        return 0


@pytest.mark.parametrize(
    ("module", "name", "queue"),
    [
        (parsing_run, "parsing", "ParseQueue"),
        (chunking_run, "chunking", "ChunkQueue"),
    ],
)
def test_a_stage_names_its_queue_and_its_own_module(
    monkeypatch, module, name, queue
) -> None:
    """The module name is what the --help line prints."""
    recorder = _QueueMain()
    monkeypatch.setattr(module, "queue_main", recorder)

    assert module.main([]) == 0

    assert recorder.called["name"] == name
    assert recorder.called["module"] == module.__name__
    assert recorder.called["repository"].__name__ == queue


@pytest.mark.parametrize("module", [parsing_run, chunking_run])
def test_the_service_is_not_built_until_a_flag_needs_it(monkeypatch, module) -> None:
    """`--status` costs a query and not a converter."""
    recorder = _QueueMain()
    monkeypatch.setattr(module, "queue_main", recorder)
    monkeypatch.setattr(
        module,
        "build_service",
        lambda settings: pytest.fail("the service was built eagerly"),
    )

    module.main([])

    assert callable(recorder.called["build_service"])


@pytest.mark.parametrize("module", [parsing_run, chunking_run])
def test_the_arguments_are_taken_from_the_process_when_none_are_given(
    monkeypatch, module
) -> None:
    """Which is how the worker's command line reaches it."""
    recorder = _QueueMain()
    monkeypatch.setattr(module, "queue_main", recorder)
    monkeypatch.setattr(module.sys, "argv", ["run", "--status"])

    module.main()

    assert recorder.called["argv"] == ["--status"]


def test_parsing_declares_no_operation_of_its_own(monkeypatch) -> None:
    """Every flag it has is one every stage has."""
    recorder = _QueueMain()
    monkeypatch.setattr(parsing_run, "queue_main", recorder)

    parsing_run.main([])

    assert "extra" not in recorder.called


def test_chunking_declares_the_re_read_as_its_own_operation(monkeypatch) -> None:
    """A change to how vocabulary is read must reach the topic model."""
    recorder = _QueueMain()
    monkeypatch.setattr(chunking_run, "queue_main", recorder)

    chunking_run.main([])

    extra = recorder.called["extra"]
    assert list(extra) == ["revocabulary"]
    help_text, verb, run = extra["revocabulary"]
    assert "vocabulary" in help_text
    assert verb == "read again"
    assert callable(run)
