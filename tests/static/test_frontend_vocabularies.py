"""The vocabularies the frontend spells out against the ones the database holds.

A page that lists every gate a question is put through holds a second copy of
`QuestionRejection`, keyed by code. `test_api_vocabularies` checks the route's
copies; this checks the page's, which is the one nothing else would notice
going stale — a gate the checker no longer emits is simply a row that never
has a count beside it, and a gate it does emit is missing from a panel whose
whole claim is that it lists every one.

Which is what happened. `wrong_type` was removed from the enum by
`20260916_1215`, and the page went on offering it as a gate while `compound`,
`off_topic` and `answerable_elsewhere` — all three of which the checker
emits — were listed nowhere.

Read as syntax rather than by importing, so the test costs no Streamlit
import and a failure names the codes.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from database.qa_generator import QuestionRejection

ROOT = Path(__file__).resolve().parents[2]

#: The page holding the gates, and the name it binds them to.
PAGE = ROOT / "frontend/views/questions.py"
DICT = "_GATES"

#: The phase each gate names, which the checker runs it in.
PHASES = ROOT / "backend/question_generation/types.py"

#: Where the views live, and the every-page explainer content in each.
VIEWS = ROOT / "frontend/views"

#: Every page carrying an explainer, by the module it draws a service for.
#: A page absent from here draws none, which `test_views.py` allows and
#: this file therefore cannot check.
EXPLAINED = (
    "documents",
    "passages",
    "facts",
    "topics",
    "questions",
    "assessment",
)


def _bound(path: Path, name: str) -> ast.expr:
    """The value of one module-level assignment."""
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.Assign):
            continue
        if name in [one.id for one in node.targets if isinstance(one, ast.Name)]:
            return node.value
    pytest.fail(f"{path.name} binds no module-level {name}")


def _argument(call: ast.expr, position: int) -> str | None:
    """One positional string argument of a call, if it is a literal."""
    if not isinstance(call, ast.Call) or len(call.args) <= position:
        return None
    given = call.args[position]
    if isinstance(given, ast.Constant) and isinstance(given.value, str):
        return given.value
    return None


def _keys(path: Path, name: str) -> list[str]:
    """The code each check is rejected under, in the order the page lists them.

    Two shapes, because the two pages hold two. Questions binds a tuple of
    `explain.Gate(...)` calls, where the code is the first argument;
    Facts binds a dict keyed by the code. Both are read here rather than
    normalised in the pages, because what each holds beside the code
    differs and neither shape is wrong.
    """
    listed = _bound(path, name)
    if isinstance(listed, ast.Dict):
        return [
            key.value
            for key in listed.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        ]
    assert isinstance(listed, ast.Tuple), f"{name} is neither a dict nor a tuple"
    found = [_argument(one, 0) for one in listed.elts]
    assert all(one is not None for one in found), (
        f"{name} holds an entry whose code is not a string literal"
    )
    return [one for one in found if one is not None]


def _phases(path: Path, name: str) -> list[str]:
    """The phase each gate declares, which is its second argument."""
    listed = _bound(path, name)
    assert isinstance(listed, ast.Tuple)
    return [one for one in (_argument(one, 1) for one in listed.elts) if one]


def test_the_page_lists_exactly_the_gates_the_checker_can_reject_under() -> None:
    """Neither a gate that was removed nor one the checker still emits."""
    listed = set(_keys(PAGE, DICT))
    held = {str(one) for one in QuestionRejection}

    assert listed == held, (
        f"{DICT}: the page lists {sorted(listed - held)} that no question "
        f"can be rejected under, and is missing {sorted(held - listed)} "
        f"that one can"
    )


def test_every_gate_is_described_once() -> None:
    """A duplicated key would silently drop a description."""
    keys = _keys(PAGE, DICT)

    assert len(keys) == len(set(keys)), "a gate is listed twice"


def test_every_phase_a_gate_names_is_one_the_checker_runs() -> None:
    """The second vocabulary, which the page now reconciles against the first.

    A rejection code and a gate phase are two names for one event seen from
    two sides: the table filters on `malformed`, a Phoenix trace records
    `structural`. The page carries both so a reader can match them, and a
    phase invented here would be a mapping that looks authoritative and is
    not.

    `re-check only` is the one name with no phase behind it, and correctly
    so: `source_changed` is not a verdict any run of the checker reaches -
    it is what `questions-reverify` writes.
    """
    declared = set(_bound(PHASES, "GATES").elts)  # type: ignore[attr-defined]
    known = {
        one.value
        for one in declared
        if isinstance(one, ast.Constant) and isinstance(one.value, str)
    } | {"re-check only"}

    # A gate that can be reached from two phases names both, as
    # `answerable_after_all` does.
    named = {
        part.strip()
        for phase in _phases(PAGE, DICT)
        for part in phase.replace(" or ", ",").split(",")
        if part.strip()
    }

    assert named <= known, (
        f"the page puts a gate in {sorted(named - known)}, which is not a "
        f"phase the checker runs"
    )


# ── What every page's explainer claims ────────────────────────────────────


def _steps(name: str) -> list[ast.Call]:
    """The `explain.Step(...)` calls one view binds to `_STEPS`."""
    listed = _bound(VIEWS / f"{name}.py", "_STEPS")
    assert isinstance(listed, ast.Tuple), f"{name}: _STEPS is not a tuple literal"
    return [one for one in listed.elts if isinstance(one, ast.Call)]


def _strings(call: ast.Call, position: int) -> list[str]:
    """The string literals of one positional tuple argument."""
    if len(call.args) <= position:
        return []
    given = call.args[position]
    if not isinstance(given, ast.Tuple):
        return []
    return [
        one.value
        for one in given.elts
        if isinstance(one, ast.Constant) and isinstance(one.value, str)
    ]


@pytest.mark.parametrize("name", EXPLAINED)
def test_every_setting_a_step_names_is_one_the_catalogue_describes(name: str) -> None:
    """A step naming a setting nothing reads is a page inventing a dial.

    The explainer tells a reader which settings change what a step does, so
    a name that has gone stale sends them to a control that is not there.
    The Topics page carried this check alone; every page names settings now.
    """
    from settings.catalog import BY_NAME

    named = {one for call in _steps(name) for one in _strings(call, 3)}
    unknown = named - set(BY_NAME)

    assert not unknown, (
        f"{name} names settings the catalogue does not describe: {sorted(unknown)}"
    )


@pytest.mark.parametrize("name", EXPLAINED)
def test_every_module_a_step_names_is_one_that_exists(name: str) -> None:
    """A step pointing at a module nobody can open is worse than none.

    Matched by filename anywhere under `backend/`, because a page says
    `service.py` rather than the package path a reader would have to know
    already.
    """
    named = {one for call in _steps(name) for one in _strings(call, 2)}
    paths = [
        str(one.relative_to(ROOT / "backend"))
        for one in (ROOT / "backend").rglob("*.py")
    ]
    # By suffix, so `service.py` matches and `pipelines/pdf.py` does too:
    # a page names the module a reader would open, not the package path
    # they would have to know already to find it.
    missing = {
        one
        for one in named
        if not any(path == one or path.endswith(f"/{one}") for path in paths)
    }

    assert not missing, f"{name} names modules that do not exist: {sorted(missing)}"


def test_the_facts_page_lists_exactly_the_codes_a_fact_can_be_rejected_under() -> None:
    """The Questions guarantee, for the page that had none.

    It had drifted: `over_cap` and `duplicate` were in the enum and on
    nothing in the UI, and `duplicate` is the commonest rejection there is
    — 2,292 of 3,098 on one corpus, so three quarters of the refusals were
    unexplained by a panel claiming to list every check.
    """
    from database.qa_generator import Rejection

    listed = set(_keys(VIEWS / "facts.py", "_CHECKS"))
    held = {str(one) for one in Rejection}

    assert listed == held, (
        f"the Facts page lists {sorted(listed - held)} that no fact can be "
        f"rejected under, and is missing {sorted(held - listed)} that one can"
    )
