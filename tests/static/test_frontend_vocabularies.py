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

#: The page holding the dict, and the name it binds it to.
PAGE = ROOT / "frontend/views/questions.py"
DICT = "_GATES"


def _keys(path: Path, name: str) -> list[str]:
    """Reads the string keys of one module-level dict literal."""
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [one.id for one in node.targets if isinstance(one, ast.Name)]
        if name not in targets:
            continue
        assert isinstance(node.value, ast.Dict), f"{name} is not a dict literal"
        return [
            key.value
            for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        ]
    pytest.fail(f"{path.name} binds no module-level {name}")


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
