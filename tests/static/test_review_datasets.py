"""Every name in an Argilla dataset's settings, checked for collisions.

Argilla requires a name to be unique across a dataset's FIELDS, QUESTIONS
and METADATA **together**, not within each group. Two of them called
`judge` is refused at dataset creation with `SettingsError: names of
dataset settings must be unique`, and the dataset is created and then
deleted again - so the failure reaches a person as `make review-push-...`
exiting non-zero with no dataset and no records.

Nothing caught that. The unit tests build records rather than settings,
and building an `rg.` object at all reaches for a default client, so a
test that called `fact_settings()` would need a running Argilla and could
not run in this layer.

So this reads the module as SYNTAX: every `name=` keyword inside each
`rg.Settings(...)` call, whatever kind of thing it declares. No client, no
import, and a failure that names the duplicate.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "review/datasets.py"


def settings_functions() -> dict[str, ast.FunctionDef]:
    """Every function in `review/datasets.py` that builds an rg.Settings."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    found = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "Settings"
            ):
                found[node.name] = node
    return found


def declared_names(function: ast.FunctionDef) -> list[str]:
    """Every literal `name=` a settings function declares, in order.

    Includes the ones reached through a helper called inside it - the
    judge's field and metadata are built by `_judge_field` and
    `_judge_metadata`, and a check that did not follow them would pass
    while the dataset still refused to be created.
    """
    names = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                names.append(keyword.value.value)
    return names


def helper_names() -> list[str]:
    """The names the shared judge helpers declare."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in (
            "_judge_field",
            "_judge_metadata",
        ):
            names.extend(declared_names(node))
    return names


def test_there_are_settings_to_check() -> None:
    """A rename that moved them would otherwise make this vacuous."""
    assert set(settings_functions()) == {
        "fact_settings",
        "topic_settings",
        "question_settings",
    }


@pytest.mark.parametrize("name", sorted(settings_functions()))
def test_no_two_settings_of_one_dataset_share_a_name(name: str) -> None:
    """Fields, questions and metadata share one namespace in Argilla."""
    declared = declared_names(settings_functions()[name]) + helper_names()
    duplicated = sorted({one for one in declared if declared.count(one) > 1})

    assert not duplicated, (
        f"{name} declares {duplicated} more than once. Argilla refuses a "
        f"dataset whose fields, questions and metadata do not have distinct "
        f"names, and the push fails at creation."
    )


def test_the_judge_reaches_every_dataset() -> None:
    """All three, or a reviewer sees two machine verdicts on some rows only."""
    # The call sites, not the definitions: `def _judge_field()` carries the
    # same characters and would make this read one too many.
    source = SOURCE.read_text(encoding="utf-8")

    assert source.count("            _judge_field(),") == 3
    assert source.count("            *_judge_metadata(),") == 3


def test_the_metadata_a_record_carries_is_the_metadata_declared() -> None:
    """A term a record sets and the dataset does not declare is dropped."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    written = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "judge_metadata":
            for inner in ast.walk(node):
                if isinstance(inner, ast.Dict):
                    written = {
                        key.value for key in inner.keys if isinstance(key, ast.Constant)
                    }

    assert written == set(helper_names()) - {"judge"}, (
        "the terms judge_metadata writes and the properties _judge_metadata "
        "declares have drifted apart"
    )
