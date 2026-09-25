"""The names one stage goes by in four tools, held against each other.

`telemetry/pipeline.py` names every stage's Dagster asset and Argilla
dataset because the two places those names really live cannot be imported
from where they are needed: `orchestration` is a code location in an image
of its own, and no container carries the Argilla client.

So there are two copies of each, and nothing would notice one going stale.
A renamed asset would leave the application linking to a Dagster page that
404s, and a renamed dataset would leave it naming one Argilla does not
hold - both of which look exactly like the tool being down.

`orchestration` is read as syntax rather than imported: importing it
configures telemetry and builds a Dagster code location, and this needs
neither.
"""

from __future__ import annotations

import ast
from pathlib import Path

from telemetry import pipeline

ROOT = Path(__file__).resolve().parents[2]


#: What `_value` returns for a right-hand side built from anything but
#: literals and names the module already bound - a call, an f-string, a
#: comprehension. Those are skipped rather than guessed at.
_UNREADABLE = object()


def _value(node: ast.expr, bound: dict[str, object]) -> object:
    """Evaluates one assignment's right-hand side, or reports it unreadable.

    No builtins and no imports: nothing but what the file already bound
    above it is in scope, so this can resolve `ROW_STAGES + (...)` and a
    dict keyed by `FACTS` and nothing that runs.
    """
    try:
        return eval(
            compile(ast.Expression(node), "<names>", "eval"),
            {"__builtins__": {}},
            bound,
        )
    except Exception:  # noqa: BLE001 - unreadable is an answer, not a failure
        return _UNREADABLE


def _assigned(module: Path, name: str) -> object:
    """Reads one assignment out of a module, without importing it.

    Every module-level binding this can work out is resolved in order, so
    a value naming an earlier one reads the same as a literal would.
    """
    bound: dict[str, object] = {}
    for node in ast.parse(module.read_text()).body:
        if isinstance(node, ast.Assign):
            value = _value(node.value, bound)
            for target in node.targets:
                if value is not _UNREADABLE and isinstance(target, ast.Name):
                    bound[target.id] = value

    assert name in bound, f"{module.name} binds no {name} this could read"
    return bound[name]


def test_every_stage_with_an_asset_is_one_dagster_defines() -> None:
    """The asset keys, and the order they are in.

    Compared as a list rather than keyed by stage, because ALL_STAGES
    holds the ROUTE prefix and not the stage name - `topics` for
    `topic_modelling`, `questions` for `question_generation` - and the
    asset order is what this has to keep anyway.
    """
    declared = _assigned(ROOT / "orchestration" / "stages.py", "ALL_STAGES")

    assert list(pipeline.ASSET.values()) == [asset for asset, _, _ in declared], (
        "telemetry.pipeline.ASSET and orchestration.stages.ALL_STAGES "
        "disagree about the assets, or about the order they run in"
    )


def test_every_stage_but_ingestion_has_an_asset() -> None:
    """Ingestion owns no queue: an upload writes a row and stops."""
    assert set(pipeline.ASSET) == set(pipeline.STAGES) - {"ingestion"}


def test_every_stage_with_a_dataset_is_one_review_pushes() -> None:
    """The dataset name each stage links to is the name Argilla holds."""
    declared = _assigned(ROOT / "review" / "datasets.py", "STAGES")
    by_stage = {stage: name for name, stage in declared.items()}

    assert pipeline.DATASET == by_stage, (
        "telemetry.pipeline.DATASET and review.datasets.STAGES disagree "
        "about which dataset a stage is reviewed through"
    )


def test_every_named_stage_is_a_stage() -> None:
    """A name for a stage the pipeline does not have is a link to nowhere."""
    named = set(pipeline.ASSET) | set(pipeline.DATASET)

    assert named <= set(pipeline.STAGES), sorted(named - set(pipeline.STAGES))
