"""The topic modelling service's surface, pinned.

Nothing here stops the code being changed - it makes a change to the surface
fail a test, so that it is a decision somebody took rather than a drift
nobody noticed.

Three things are pinned: the settings it reads, the routes it serves, and the
fields it answers with. A change to any of them is a change every reader of
this service sees, so making one means editing this file in the same commit
and saying why in the message.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SERVICE = ROOT / "backend/topic_modelling"

#: One of this service's settings named in a string, refusals included.
_NAMED = re.compile(r"TOPIC_[A-Z][A-Z0-9_]*")

#: Every setting the service reads, and nothing else.
SETTINGS = frozenset(
    {
        "TOPIC_LANGUAGE_NAMES",
        "TOPIC_NUM_TOPICS",
        "TOPIC_PASSAGES_PER_TOPIC",
        "TOPIC_PASSES",
        "TOPIC_RANDOM_STATE",
        "TOPIC_TOP_TERMS",
        "TOPIC_MIN_WEIGHT",
        "TOPIC_NO_BELOW",
        "TOPIC_NO_ABOVE",
    }
)

#: Every route under /topics, as (method, path).
ROUTES = frozenset(
    {
        ("GET", "/topics/status"),
        ("GET", "/topics"),
        ("GET", "/topics/fit"),
        ("GET", "/topics/visualisation/{language}"),
        ("PATCH", "/topics/{topic_id}"),
        ("POST", "/topics/discover"),
        ("POST", "/topics/stop"),
        ("POST", "/topics/retry"),
        ("DELETE", "/topics"),
    }
)

#: The fields each value the service answers with carries.
SHAPES = {
    "StoredTopic": {
        "id",
        "language",
        "topic_index",
        "top_terms",
        "label",
        "labelled_by",
        "include_in_coverage",
        "passages",
        "dominant_passages",
        "mean_weight",
        "documents",
        "table_passages",
        "validated_facts",
    },
    "LanguageFit": {
        "language",
        "topics",
        "corpus_passages",
        "corpus_vocabulary",
        "passages_without_topics",
        "fitted_at",
        "live_passages",
        "memberships",
    },
    "TopicFit": {
        "status",
        "error",
        "requested_at",
        "topics",
        "languages",
        "passages_without_language",
    },
    "TopicRemoval": {"topics", "memberships", "labels", "languages"},
}

#: The modules the service is made of. A new one is a change of shape.
MODULES = frozenset(
    {
        "__init__.py",
        "config.py",
        "factory.py",
        "labels.py",
        "models.py",
        "repository.py",
        "run.py",
        "service.py",
        "topics.py",
        "visualisation.py",
    }
)


def test_the_service_reads_exactly_the_settings_named_here() -> None:
    """A new setting is a new thing to configure, document and default.

    Over every string the service holds, not only the ones it passes to a
    reader, so a name that appears in a refusal is checked against this list
    too - one of them was wrong on the Topics page for a while.
    """
    found: set[str] = set()
    for path in SERVICE.glob("*.py"):
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


def test_the_service_serves_exactly_the_routes_named_here() -> None:
    """The frontend and the Makefile both address these by hand."""
    from api.routes.topics import router

    found = {
        (method, route.path)
        for route in router.routes
        for method in getattr(route, "methods", ())
        if method != "HEAD"
    }

    assert found == ROUTES, (
        f"added: {sorted(found - ROUTES)}, gone: {sorted(ROUTES - found)}"
    )


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_the_values_the_service_answers_with_keep_their_fields(name: str) -> None:
    """A field added or removed changes every page and client that reads it."""
    from topic_modelling import models

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


def test_the_service_documents_itself() -> None:
    """The README is the service's description, so it has to be there."""
    readme = SERVICE / "README.md"

    assert readme.exists(), f"{readme} is missing"
    written = readme.read_text()
    for heading in ("## What it does", "## Configuration", "## Tests"):
        assert heading in written, f"the README has no {heading!r} section"
