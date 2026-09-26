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
    """The code each gate is rejected under, in the order the page lists them.

    Read off a tuple of `explain.Gate(...)` calls rather than a dict, which
    is what the page held until each gate gained the phase it runs in and
    what it costs. The code is still the first thing declared.
    """
    listed = _bound(path, name)
    assert isinstance(listed, ast.Tuple), f"{name} is not a tuple literal"
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
