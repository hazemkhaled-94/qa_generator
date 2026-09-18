"""The API process loads no model, no converter and no inference library.

One image serves the api and all five workers, so every stage's libraries are
installed in the container the api runs in. What keeps them out of the *process*
is that a service package re-exports nothing and a caller names the submodule it
wants: the api imports each stage's catalogue and never its service.

That is an import-graph property, and an import graph is easy to break by
accident. Moving one value object into the module that loads torch is enough,
and nothing about the change looks like it costs anything - the api still
starts, still answers, and is now several seconds slower to boot and carrying
two gigabytes it never uses against a 512 MB limit.

Read as syntax rather than by importing, so it needs no database, no
environment and no container, and so a failure can name the edge that
introduced the weight rather than just the library.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: Where the api starts. `main` assembles the application and `dependencies`
#: is the composition root, which is the module that decides what exists.
ROOTS = ("api.main", "api.dependencies")

#: Libraries that belong to a worker and not to the process serving JSON.
#: Each is either a model, a thing that loads one, or a converter.
HEAVY = {
    "torch": "the tensor library, and two gigabytes of it",
    "transformers": "the embedding model's loader",
    "litellm": "the served-model client",
    "instructor": "structured output around that client",
    "gensim": "the topic factorisation",
    "spacy": "the language pipelines",
    "docling": "the PDF converter",
    "docling_core": "the converter's document model",
    "pyLDAvis": "the topic figure",
    "pyldavis": "the topic figure",
    "sklearn": "brought in by the topic model",
    "lingua": "the language detector",
}


def _first_party() -> dict[str, Path]:
    """Every module a container can import, by the name it imports it under.

    The image puts each `backend/` package at the top level and `telemetry`
    beside them, so those are the names that resolve at run time.
    """
    found: dict[str, Path] = {}
    for base, prefix in ((ROOT / "backend", None), (ROOT / "telemetry", "telemetry")):
        for path in base.glob("**/*.py"):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(base if prefix is None else base.parent)
            parts = list(relative.with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            if parts:
                found[".".join(parts)] = path
    return found


def _imports(path: Path) -> set[str]:
    """What one module imports when it is imported, and nothing more.

    Two kinds of import are left out because they do not run at import time:
    one inside a function, which is how the topic service defers pyLDAvis,
    and one under `if TYPE_CHECKING`, which is how the models refer to each
    other without a cycle. Counting either would fail this on code that
    never loads.
    """
    found: set[str] = set()

    def walk(node: ast.AST, deferred: bool) -> None:
        """Collects the imports that run, descending into what runs."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                walk(child, True)
                continue
            if isinstance(child, ast.If) and "TYPE_CHECKING" in ast.dump(child.test):
                continue
            if not deferred and isinstance(child, ast.Import):
                found.update(alias.name for alias in child.names)
            elif (
                not deferred
                and isinstance(child, ast.ImportFrom)
                and child.level == 0
                and child.module
            ):
                found.add(child.module)
                # `from x import y` where y is itself a module. Names that are
                # not modules simply do not resolve below.
                found.update(f"{child.module}.{one.name}" for one in child.names)
            walk(child, deferred)

    walk(ast.parse(path.read_text()), False)
    return found


MODULES = _first_party()


def _reached() -> tuple[set[str], dict[str, str]]:
    """Every module the api loads, and which module first imported each.

    Breadth-first, so the parent recorded for a module is on a shortest path
    from the api to it, which is the chain worth printing.
    """
    parent: dict[str, str] = {}
    seen = set(ROOTS)
    queue = list(ROOTS)
    while queue:
        name = queue.pop(0)
        path = MODULES.get(name)
        if path is None:
            continue
        for imported in sorted(_imports(path)):
            if imported in MODULES and imported not in seen:
                seen.add(imported)
                parent[imported] = name
                queue.append(imported)
    return seen, parent


def _chain(name: str, parent: dict[str, str]) -> str:
    """How the api came to load one module, read from the api outwards."""
    walked = [name]
    while parent.get(walked[-1]):
        walked.append(parent[walked[-1]])
    return " -> ".join(reversed(walked))


REACHED, PARENT = _reached()


def test_the_walk_finds_the_modules_it_is_meant_to_guard() -> None:
    """A refactor that moved the composition root would pass silently.

    The api reaches dozens of first-party modules and every read route's
    catalogue; a walk that found a handful is a walk that stopped early.
    """
    assert len(REACHED) > 30, sorted(REACHED)
    assert "api.dependencies" in REACHED
    assert "question_generation.catalog" in REACHED, (
        "the api serves /questions, so it must reach that catalogue"
    )
    assert "extraction.repository" in REACHED


@pytest.mark.parametrize("library", sorted(HEAVY))
def test_the_api_never_imports(library: str) -> None:
    """Named one at a time, so a failure says which one and how."""
    guilty = [
        f"{_chain(name, PARENT)} imports {library}"
        for name in sorted(REACHED)
        if any(
            found == library or found.startswith(f"{library}.")
            for found in _imports(MODULES[name])
        )
    ]

    assert not guilty, (
        f"the api process would load {library} ({HEAVY[library]}):\n"
        + "\n".join(guilty)
        + "\n\nA catalogue the api holds must not import the module that "
        "loads one. Move the value object it wanted into models.py."
    )


#: The stages that run in a worker. Each splits its database access in two -
#: a queue and a catalogue - and the api imports only those. Ingestion is not
#: here: it has no queue and no worker, and the api is what runs it.
WORKER_STAGES = (
    "preprocessing.parsing",
    "preprocessing.chunking",
    "extraction",
    "topic_modelling",
    "question_generation",
)


@pytest.mark.parametrize("stage", WORKER_STAGES)
def test_the_api_reaches_no_worker_stages_working_half(stage: str) -> None:
    """A catalogue is the api's half; a service is the worker's.

    The service holds the converter, the model client and the gates, so a
    catalogue that imports it drags all of that in whether or not anything
    calls it. That is the shape every failure above takes, caught one edge
    earlier and without having to name the library it would have pulled.
    """
    working = {
        name
        for name in REACHED
        if name.startswith(f"{stage}.")
        and name.rsplit(".", 1)[-1]
        in ("service", "factory", "run", "generation", "embedding", "verification")
    }

    assert not working, f"the api reaches the working half of {stage}:\n" + "\n".join(
        _chain(name, PARENT) for name in sorted(working)
    )
