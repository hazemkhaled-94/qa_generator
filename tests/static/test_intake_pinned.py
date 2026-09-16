"""Ingestion, parsing and chunking: their surface, pinned.

Nothing here stops the code being changed - it makes a change to the surface
fail a test, so that it is a decision somebody took rather than a drift
nobody noticed.

Four things are pinned: the settings they read, the routes they serve, the
fields they answer with, and the modules they are made of. A change to any of
them is a change every reader of these stages sees, so making one means
editing this file in the same commit and saying why in the message.

Each of the three describes itself, in a README beside its code: they share
one story - a file becomes a document becomes passages - and they are three
packages with three queues and three command lines.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: The three packages, by the name their modules are pinned under.
SERVICES = {
    "ingestion": ROOT / "backend/ingestion",
    "parsing": ROOT / "backend/preprocessing/parsing",
    "chunking": ROOT / "backend/preprocessing/chunking",
}

#: The headings every service README carries, as in test_topics_pinned.py
#: and test_facts_pinned.py.
HEADINGS = ("## What it does", "## Configuration", "## Tests")

#: The readers in backend/settings/env.py, by name.
READERS = frozenset({"required", "optional", "integer", "decimal", "boolean", "csv"})

#: Every setting the three stages read, and nothing else. Connection settings
#: are absent because database and blob_store own theirs.
SETTINGS = frozenset(
    {
        "PIPELINE_VERSION",
        "MAX_FILE_SIZE_MB",
        "ALLOWED_MIME_TYPES",
        "PARSING_OCR_CHAR_THRESHOLD",
        "PARSING_MIN_CONFIDENCE",
        "PARSING_TABLE_MODE",
        "PARSING_HEADING_HIERARCHY",
        "PARSING_TIMEOUT_SECONDS",
        "DOCLING_ARTIFACTS_PATH",
        "EMBEDDING_MODEL",
        "EMBEDDING_MAX_TOKENS",
        "CHUNKING_MERGE_PEERS",
    }
)

#: Every route the three stages serve, as (method, path).
ROUTES = frozenset(
    {
        ("POST", "/documents"),
        ("GET", "/documents"),
        ("GET", "/documents/names"),
        ("GET", "/documents/{sha256}/file"),
        ("DELETE", "/documents/{sha256}"),
        ("DELETE", "/documents/{sha256}/derived"),
        ("GET", "/parsing/status"),
        ("POST", "/parsing/{action}"),
        ("GET", "/parsing/{scope}/{value}/status"),
        ("POST", "/parsing/{scope}/{value}/{action}"),
        ("GET", "/chunking/status"),
        ("POST", "/chunking/{action}"),
        ("GET", "/chunking/{scope}/{value}/status"),
        ("POST", "/chunking/{scope}/{value}/{action}"),
        ("GET", "/passages"),
        ("GET", "/passages/types"),
        ("GET", "/passages/{passage_id}"),
    }
)

#: The fields each value the three stages pass around or answer with carries.
SHAPES = {
    "ingestion.models": {
        "UploadedFile": {"filename", "data"},
        "PdfFacts": {"page_count", "char_count"},
        "DocumentName": {"sha256", "filename"},
        "StoredDocument": {
            "sha256",
            "filename",
            "first_seen",
            "page_count",
            "title",
            "language",
            "parse_status",
            "parse_error",
            "chunk_status",
            "chunk_error",
            "extracted_passages",
            "total_passages",
            "oversized",
        },
        "Removal": {"sha256", "document", "passages", "file", "parsed"},
        "IngestResult": {"outcome", "sha256", "detail"},
    },
    "preprocessing.parsing.models": {
        "ClaimedDocument": {"sha256", "media_type", "page_count", "char_count"},
        "SourceDocument": {"sha256", "data", "scanned"},
        "Conversion": {"document", "confidence", "confidence_low"},
        "ParsedDocument": {
            "title",
            "language",
            "content_sha256",
            "page_count",
            "confidence",
            "confidence_low",
        },
    },
    "preprocessing.chunking.models": {
        "ClaimedDocument": {"sha256", "language"},
        "Chunk": {
            "ordinal",
            "text",
            "page_from",
            "page_to",
            "section_path",
            "block_type",
            "doc_item_refs",
            "bbox",
            "table_cells",
            "language",
            "sentences",
            "lemmas",
        },
        "Chunking": {"passages", "oversized"},
        "StoredPassage": {
            "id",
            "doc_sha256",
            "ordinal",
            "text",
            "page_from",
            "page_to",
            "section_path",
            "block_type",
            "language",
            "doc_item_refs",
            "bbox",
            "table_count",
            "sentence_count",
        },
        "PassageDetail": {
            "passage",
            "table_cells",
            "sentences",
            "extract_status",
            "extract_error",
        },
    },
}

#: The modules each package is made of. A new one is a change of shape.
MODULES = {
    "ingestion": frozenset(
        {
            "__init__.py",
            "config.py",
            "factory.py",
            "models.py",
            "pdf.py",
            "removal.py",
            "repository.py",
            "run.py",
            "service.py",
            "stores.py",
        }
    ),
    "parsing": frozenset(
        {
            "__init__.py",
            "analysis.py",
            "config.py",
            "factory.py",
            "models.py",
            "repository.py",
            "run.py",
            "service.py",
        }
    ),
    "chunking": frozenset(
        {
            "__init__.py",
            "config.py",
            "factory.py",
            "models.py",
            "passages.py",
            "repository.py",
            "run.py",
            "service.py",
        }
    ),
}

#: The pipelines package, which is parsing's one subpackage.
PIPELINES = frozenset({"__init__.py", "base.py", "pdf.py"})


def _read_by(source: str) -> set[str]:
    """Finds the settings one module reads, by literal name."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        called = node.func
        name = (
            called.id if isinstance(called, ast.Name) else getattr(called, "attr", None)
        )
        first = node.args[0]
        if (
            name in READERS
            and isinstance(first, ast.Constant)
            and isinstance(first.value, str)
        ):
            found.add(first.value)
    return found


def test_the_stages_read_exactly_the_settings_named_here() -> None:
    """A new setting is a new thing to configure, document and default."""
    found: set[str] = set()
    for path in SERVICES.values():
        for module in path.glob("**/*.py"):
            if "__pycache__" not in module.parts:
                found |= _read_by(module.read_text())

    assert found == SETTINGS, (
        f"added: {sorted(found - SETTINGS)}, gone: {sorted(SETTINGS - found)}"
    )


def test_every_setting_is_declared_where_a_deployment_reads_it() -> None:
    """A setting with nowhere to be set is unreachable."""
    declared = (ROOT / "configs/env/backend.env").read_text() + (
        ROOT / ".env.example"
    ).read_text()

    missing = [name for name in SETTINGS if f"{name}=" not in declared]
    assert not missing, f"read but never declared: {sorted(missing)}"


#: The prefixes these three stages own on the API.
PREFIXES = ("/documents", "/parsing", "/chunking", "/passages")


def test_the_stages_serve_exactly_the_routes_named_here() -> None:
    """The frontend and the Makefile both address these by hand.

    Read off the committed OpenAPI snapshot rather than off a live
    application: importing a route module imports the composition root,
    which binds every service to the environment as it is read, and doing
    that before the containers are up leaves the API pointed at nothing for
    the rest of the session. tests/contract holds that snapshot to the
    running application, so this needs neither.
    """
    published = json.loads((ROOT / "tests/contract/openapi.json").read_text())
    found = {
        (verb.upper(), path)
        for path, operations in published.items()
        for verb in operations
        if path.startswith(PREFIXES)
    }

    assert found == ROUTES, (
        f"added: {sorted(found - ROUTES)}, gone: {sorted(ROUTES - found)}"
    )


@pytest.mark.parametrize(
    ("module", "name"),
    [(module, name) for module, shapes in SHAPES.items() for name in sorted(shapes)],
)
def test_the_values_the_stages_pass_around_keep_their_fields(
    module: str, name: str
) -> None:
    """A field added or removed changes every reader of it."""
    from importlib import import_module

    fields = set(getattr(import_module(module), name).__dataclass_fields__)
    expected = SHAPES[module][name]

    assert fields == expected, (
        f"added: {sorted(fields - expected)}, gone: {sorted(expected - fields)}"
    )


@pytest.mark.parametrize("service", sorted(MODULES))
def test_each_stage_is_made_of_exactly_these_modules(service: str) -> None:
    """A module added here is a part of the stage nothing else describes."""
    found = {path.name for path in SERVICES[service].glob("*.py")}

    assert found == MODULES[service], (
        f"added: {sorted(found - MODULES[service])}, "
        f"gone: {sorted(MODULES[service] - found)}"
    )


def test_parsing_holds_one_pipeline_per_format_and_no_more() -> None:
    """Adding a format is a module here and an entry in the registry."""
    found = {path.name for path in (SERVICES["parsing"] / "pipelines").glob("*.py")}

    assert found == PIPELINES, (
        f"added: {sorted(found - PIPELINES)}, gone: {sorted(PIPELINES - found)}"
    )


def test_the_registry_dispatches_exactly_the_types_the_allowlist_may_name() -> None:
    """A type ingestion accepts and no pipeline handles fails at parsing."""
    from ingestion.models import DETECTABLE
    from preprocessing.parsing.pipelines import PdfPipeline

    assert PdfPipeline.media_types == tuple(sorted(DETECTABLE))


@pytest.mark.parametrize("service", sorted(SERVICES))
def test_each_stage_documents_itself(service: str) -> None:
    """The README is the stage's description, so it has to be there."""
    readme = SERVICES[service] / "README.md"

    assert readme.exists(), f"{readme} is missing"
    written = readme.read_text()
    for heading in HEADINGS:
        assert heading in written, f"{service} has no {heading!r} section"
