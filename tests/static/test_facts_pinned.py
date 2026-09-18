"""The facts service's surface, pinned.

Nothing here stops the code being changed - it makes a change to the surface
fail a test, so that it is a decision somebody took rather than a drift
nobody noticed.

Six things are pinned: the settings it reads, the routes it serves, the
fields it answers with, the modules it is made of, the vocabularies it
writes to the database, and which checks each kind of fact faces. A change to
any of them is a change every reader of this service sees, so making one
means editing this file in the same commit and saying why in the message.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SERVICE = ROOT / "backend/extraction"

#: One of this service's settings named in a string, refusals included.
_NAMED = re.compile(r"EXTRACTION_[A-Z][A-Z0-9_]*")

#: Every setting the service reads, and nothing else. Which model to call and
#: how patiently is LLM_*, shared with every other stage that calls one.
SETTINGS = frozenset(
    {
        "EXTRACTION_KINDS",
        "EXTRACTION_DIGEST_MAX_SHARE",
        "EXTRACTION_MIN_OTHER_SHARE",
        "EXTRACTION_BRIDGES_PER_TOPIC",
        "EXTRACTION_BRIDGE_PASSAGES",
        "EXTRACTION_MODEL",
    }
)

#: The queue routes, as (method, path). The four verbs come twice: once for
#: the whole queue and once for the part of it a scope names.
QUEUE_ROUTES = frozenset(
    {
        ("GET", "/extraction/status"),
        ("POST", "/extraction/{action}"),
        ("GET", "/extraction/{scope}/{value}/status"),
        ("POST", "/extraction/{scope}/{value}/{action}"),
    }
)

#: The routes that read back what the service produced.
FACT_ROUTES = frozenset(
    {
        ("GET", "/facts"),
        ("GET", "/facts/quality"),
        ("GET", "/facts/{fact_id}/passages"),
    }
)

#: The fields each value the service passes around or answers with carries.
SHAPES = {
    "Provenance": {"model", "prompt_version", "temperature"},
    "PassageToExtract": {
        "id",
        "text",
        "section_path",
        "block_type",
        "language",
        "sentences",
        "table_cells",
        "doc_sha256",
    },
    "CandidateFact": {"statement", "sentences", "kind", "passages"},
    "Citation": {"passage_id", "sentence_ids", "start", "end"},
    "CheckedFact": {
        "statement",
        "evidence_text",
        "extraction_method",
        "validated",
        "rejection_code",
        "validation_error",
        "kind",
        "citations",
        "statement_predicates",
        "evidence_predicates",
        "units_statement",
        "units_added",
        "unresolved_references",
        "extraction_model",
        "prompt_version",
        "extraction_temperature",
        "spacy_model",
        "spacy_version",
    },
    "FactSource": {
        "passage_id",
        "doc_sha256",
        "ordinal",
        "page_from",
        "position",
    },
    "StoredFact": {
        "id",
        "statement",
        "evidence_text",
        "kind",
        "extraction_method",
        "validated",
        "rejection_code",
        "validation_error",
        "statement_predicates",
        "evidence_predicates",
        "units_added",
        "unresolved_references",
        "passages",
    },
    "FactQuality": {
        "total",
        "validated",
        "mean_statement_chars",
        "mean_evidence_chars",
        "mean_statement_predicates",
        "mean_evidence_predicates",
        "facts_per_passage",
        "rejected",
        "kinds",
    },
}

#: The modules the service is made of. A new one is a change of shape.
MODULES = frozenset(
    {
        "__init__.py",
        "config.py",
        "factory.py",
        "models.py",
        "repository.py",
        "run.py",
        "service.py",
        "validation.py",
    }
)

#: One reader per kind of passage, plus the two whose unit is not a passage.
EXTRACTORS = frozenset(
    {"__init__.py", "base.py", "bridge.py", "digest.py", "llm.py", "table.py"}
)

#: Every kind of fact the service writes.
KINDS = frozenset({"atomic", "summary", "outline", "bridge"})

#: Every reason a candidate is refused. A rejected fact's code is the quality
#: signal this project reports, so a new one is a new column on that report.
REJECTIONS = frozenset(
    {
        "evidence_absent",
        "copied",
        "asserts_nothing",
        "not_atomic",
        "unsupported_addition",
        "unresolved_reference",
        "not_condensed",
        "not_listed",
        "not_bridging",
        # Not a check's: the service refuses what the checks passed, because
        # the passage yielded more atomic facts than
        # EXTRACTION_MIN_OTHER_SHARE leaves room for.
        "over_cap",
    }
)

#: Which checks each kind faces, in the order they are applied. This is the
#: whole of what "kind" means, so it is pinned rather than described.
CHECKS = {
    "atomic": ("_copied", "_asserts", "_atomic", "_supported", "_self_contained"),
    "summary": ("_asserts", "_supported", "_condensed"),
    "outline": ("_listed", "_supported", "_condensed"),
    "bridge": ("_asserts", "_atomic", "_supported", "_self_contained", "_bridging"),
}

#: The prompt version recorded on every fact each model-backed reader draws.
#: Bumped whenever the prompt changes what counts as a fact: two prompts are
#: two datasets, and a corpus read under both is neither.
PROMPTS = {"llm": "6", "digest": "1", "bridge": "2"}


def test_the_service_reads_exactly_the_settings_named_here() -> None:
    """A new setting is a new thing to configure, document and default."""
    found: set[str] = set()
    for path in (*SERVICE.glob("*.py"), *(SERVICE / "extractors").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                found |= set(_NAMED.findall(node.value))

    assert found == SETTINGS, (
        f"added: {sorted(found - SETTINGS)}, gone: {sorted(SETTINGS - found)}"
    )


def test_every_setting_is_declared_where_a_deployment_reads_it() -> None:
    """A setting with nowhere to be set is unreachable."""
    declared = (ROOT / "configs/env/backend.env").read_text()

    missing = [name for name in SETTINGS if f"{name}=" not in declared]
    assert not missing, f"read but never declared: {sorted(missing)}"


@pytest.mark.parametrize(
    ("module", "attribute", "pinned"),
    [
        ("api.routes.extraction", "router", QUEUE_ROUTES),
        ("api.routes.facts", "router", FACT_ROUTES),
    ],
)
def test_the_service_serves_exactly_the_routes_named_here(
    module, attribute, pinned
) -> None:
    """The frontend and the Makefile both address these by hand."""
    from importlib import import_module

    router = getattr(import_module(module), attribute)
    found = {
        (method, route.path)
        for route in router.routes
        for method in getattr(route, "methods", ())
        if method != "HEAD"
    }

    assert found == pinned, (
        f"added: {sorted(found - pinned)}, gone: {sorted(pinned - found)}"
    )


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_the_values_the_service_passes_around_keep_their_fields(name: str) -> None:
    """A field added or removed changes every page and client that reads it."""
    from extraction import models

    fields = set(getattr(models, name).__dataclass_fields__)

    assert fields == SHAPES[name], (
        f"added: {sorted(fields - SHAPES[name])}, gone: {sorted(SHAPES[name] - fields)}"
    )


def test_the_service_is_made_of_exactly_these_modules() -> None:
    """A module added here is a part of the service nothing else describes."""
    found = {path.name for path in SERVICE.glob("*.py")}

    assert found == MODULES, (
        f"added: {sorted(found - MODULES)}, gone: {sorted(MODULES - found)}"
    )


def test_there_is_exactly_one_reader_per_kind_of_passage() -> None:
    """A new extractor is a new way a fact can come to exist."""
    found = {path.name for path in (SERVICE / "extractors").glob("*.py")}

    assert found == EXTRACTORS, (
        f"added: {sorted(found - EXTRACTORS)}, gone: {sorted(EXTRACTORS - found)}"
    )


def test_the_kinds_written_to_the_database_are_these() -> None:
    """The column's CHECK constraint holds the same list."""
    from database.qa_generator import FactKind

    assert {str(kind) for kind in FactKind} == KINDS


def test_the_reasons_a_fact_is_refused_are_these() -> None:
    """Each failure reaches its own code; a shared bucket reports nothing."""
    from database.qa_generator import Rejection

    assert {str(code) for code in Rejection} == REJECTIONS


def test_each_kind_faces_exactly_the_checks_named_here() -> None:
    """Which checks a kind faces is the whole of what its kind means."""
    from extraction.validation import FactChecker

    applied = {
        kind: tuple(
            getattr(check, "func", check).__name__
            for check in FactChecker(0.6)._by_kind[kind]
        )
        for kind in KINDS
    }

    assert applied == CHECKS


def test_a_statement_nobody_wrote_faces_one_check() -> None:
    """A grid is neither written nor a sentence, so nothing else applies."""
    from extraction.validation import _COMPOSED

    assert tuple(check.__name__ for check in _COMPOSED) == ("_copied",)


@pytest.mark.parametrize(("reader", "version"), sorted(PROMPTS.items()))
def test_each_prompt_version_is_the_one_recorded_on_its_facts(reader, version) -> None:
    """Two prompts are two datasets; the version is what tells them apart."""
    from importlib import import_module

    module = import_module(f"extraction.extractors.{reader}")
    assert module.PROMPT_VERSION == version


def test_the_service_documents_itself() -> None:
    """The README is the service's description, so it has to be there."""
    readme = SERVICE / "README.md"

    assert readme.exists(), f"{readme} is missing"
    written = readme.read_text()
    for heading in ("## What it does", "## Configuration", "## Tests"):
        assert heading in written, f"the README has no {heading!r} section"


def test_the_readme_describes_every_kind_and_every_refusal() -> None:
    """A vocabulary the reader cannot look up is a vocabulary nobody uses."""
    written = (SERVICE / "README.md").read_text()

    missing = [name for name in (*KINDS, *REJECTIONS) if name not in written]
    assert not missing, f"undocumented: {sorted(missing)}"
