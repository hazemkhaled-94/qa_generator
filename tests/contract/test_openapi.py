"""The published surface, and the two things that read it.

The API is the only address the frontend holds, and the OpenAPI document is
the only description of it. These compare that document against a committed
copy, against the paths the frontend actually calls, and against the codes
the routes raise.
"""

from __future__ import annotations

import ast
import inspect
import json
import re
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = Path(__file__).parent / "openapi.json"
CLIENTS = ROOT / "frontend/lib/backend"

pytestmark = pytest.mark.integration


def summarised(document: dict) -> dict:
    """The shape of the surface: every path, its verbs and their answers.

    The whole document would fail on a reflowed docstring. This is what a
    caller can observe: which paths exist, which verbs they take, which
    status codes come back and which parameters each accepts.
    """
    return {
        path: {
            verb: {
                "responses": sorted(operation.get("responses", {})),
                "parameters": sorted(
                    parameter["name"] for parameter in operation.get("parameters", [])
                ),
            }
            for verb, operation in verbs.items()
        }
        for path, verbs in sorted(document["paths"].items())
    }


def templated(path: str) -> str:
    """Reduces a path to its shape, so a parameter's name does not matter."""
    return re.sub(r"\{[^}]+\}", "{}", path.split("?")[0])


@pytest.fixture(scope="module")
def published(application) -> dict:
    """The document the application generates."""
    return application.openapi()


def test_the_surface_has_not_changed(published) -> None:
    """A route, a verb, a parameter or a status code moving is deliberate.

    When it is, delete tests/contract/openapi.json and run this again: the
    first run writes it, and the diff is what gets reviewed.
    """
    current = summarised(published)
    if not SNAPSHOT.exists():
        SNAPSHOT.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"wrote the first {SNAPSHOT.name}")

    assert current == json.loads(SNAPSHOT.read_text())


def _called_paths() -> set[str]:
    """Every path the frontend's clients build, read off their `_request` calls.

    Read as syntax rather than as strings in a file: `delete` builds its
    path by concatenation, so the literal "/derived" appears on its own and
    is not a path anybody calls.
    """
    called = set()

    def literal(node: ast.AST) -> str | None:
        """The path one argument spells out, as far as it is knowable."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            return "".join(
                part.value if isinstance(part, ast.Constant) else "{}"
                for part in node.values
            )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = literal(node.left), literal(node.right)
            return None if left is None else left + (right or "")
        return None

    for source in CLIENTS.glob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_request"
                and len(node.args) >= 2
            ):
                found = literal(node.args[1])
                if found and found.startswith("/"):
                    called.add(found)
    return called


def test_the_frontend_calls_something(published) -> None:
    """A refactor that renamed `_request` would otherwise pass silently."""
    called = _called_paths()

    assert len(called) > 8, called
    assert "/documents/names" in called


def test_every_path_the_frontend_calls_exists(published) -> None:
    """The frontend holds these as literals and finds out at run time.

    The stage client composes its path from a stage name it is given, so
    those are checked by the test below rather than compared as literals.
    """
    served = {templated(path) for path in published["paths"]}
    fixed = {path for path in _called_paths() if not path.startswith("/{}")}

    missing = {path for path in fixed if templated(path) not in served}

    assert not missing, f"the frontend calls paths the API does not serve: {missing}"


#: The stages the frontend names when it composes a queue path, and the
#: scopes it narrows them with.
STAGES = ("parsing", "chunking", "extraction", "topics", "questions")
SCOPES = ("", "/document/{}", "/passage/{}")


@pytest.mark.parametrize("stage", STAGES)
def test_every_stage_the_frontend_names_answers_for_its_queue(published, stage) -> None:
    """`/{stage}/status`, whichever stage a page is drawing."""
    served = {templated(path) for path in published["paths"]}

    assert f"/{stage}/status" in served, sorted(served)


@pytest.mark.parametrize("stage", ("parsing", "chunking", "extraction", "questions"))
@pytest.mark.parametrize("scope", SCOPES)
def test_every_narrowed_stage_path_the_frontend_builds_exists(
    published, stage, scope
) -> None:
    """The narrowed pair, which the per-item controls call."""
    served = {templated(path) for path in published["paths"]}
    built = f"/{stage}/{{}}/{{}}/status" if scope else f"/{stage}/status"

    assert built in served, sorted(served)


def _raised_by(endpoint) -> set[int]:
    """The statuses one handler raises an ApiError with."""
    try:
        source = textwrap.dedent(inspect.getsource(endpoint))
    except OSError:  # pragma: no cover - every endpoint has source here
        return set()
    return {
        call.args[0].value
        for call in ast.walk(ast.parse(source))
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "ApiError"
        and call.args
        and isinstance(call.args[0], ast.Constant)
    }


def test_every_refusal_a_route_raises_is_declared(application) -> None:
    """A caller branches on the code, and reads the codes from the document.

    Matched through the application's own route objects rather than through
    operation ids, which are generated from the handler name and the path
    and so collide between /passages/{id} and /passages/types.
    """
    from fastapi.routing import APIRoute

    undeclared = []
    for route in application.routes:
        if not isinstance(route, APIRoute):
            continue
        declared = set(route.responses)
        for status in _raised_by(route.endpoint) - declared:
            undeclared.append(
                f"{min(route.methods)} {route.path} raises {status}, "
                f"declaring {sorted(declared) or 'nothing'}"
            )

    assert not undeclared, "\n".join(undeclared)


def test_every_declared_refusal_carries_the_error_body(published) -> None:
    """One shape for every deliberate refusal, so a caller parses one thing."""
    wrong = []
    for path, verbs in published["paths"].items():
        for verb, operation in verbs.items():
            for status, answer in operation.get("responses", {}).items():
                if status in ("422", "200", "202", "204"):
                    continue
                schema = (
                    answer.get("content", {})
                    .get("application/json", {})
                    .get("schema", {})
                )
                if schema.get("$ref", "").endswith("ErrorBody"):
                    continue
                wrong.append(f"{verb.upper()} {path} -> {status}: {schema}")

    assert not wrong, "\n".join(wrong)
