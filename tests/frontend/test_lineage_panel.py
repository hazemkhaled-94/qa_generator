"""The links the lineage panel offers into the other four tools.

Every one of these is a URL nobody would notice breaking: a wrong Dagster
asset key or a wrong Grafana variable renders as a page that loads and
says nothing, which is what a tool with no data looks like.

The link building is pure and is tested as such - it takes the chain and
the addresses and returns markdown - so none of this needs Streamlit or a
running deployment.
"""

from __future__ import annotations

import pytest

from lib.lineage import elsewhere

pytestmark = pytest.mark.frontend

#: Where a browser opens each tool, as `GET /services` reports it out of
#: SERVICE_URLS. Grafana is published on 3001 and serves 3000, which is
#: the reason that route exists rather than the port being derived.
PUBLISHED = {
    "grafana": "http://localhost:3001",
    "dagster-webserver": "http://localhost:3000",
    "argilla": "http://localhost:6900",
    "phoenix": "http://localhost:6006",
}


def chain(kind: str = "question", run: str | None = "a-run", **artifact) -> dict:
    """One chain as `GET /lineage` answers it, of one artefact."""
    stage = {
        "document": "parsing",
        "passage": "chunking",
        "fact": "extraction",
        "topic": "topic_modelling",
        "question": "question_generation",
    }[kind]
    return {
        "kind": kind,
        "id": "7",
        "gates": [],
        "gates_recorded": False,
        "steps": [
            {
                "position": 6,
                "stage": stage,
                "kind": kind,
                "total": 1,
                "artifacts": [{"id": "7", "run_id": run, **artifact}],
            }
        ],
    }


def test_a_question_links_to_all_three() -> None:
    """Grafana, Dagster and Argilla each get a line of their own."""
    lines = elsewhere(chain(), PUBLISHED)

    assert len(lines) == 3, lines
    assert "**Grafana**" in lines[0]
    assert "**Dagster**" in lines[1]
    assert "**Argilla**" in lines[2]


def test_grafana_opens_the_run_dashboard_narrowed_to_this_run() -> None:
    """The variable name is the one the dashboard declares, `var-run`."""
    (grafana, *_) = elsewhere(chain(run="full-20260922"), PUBLISHED)

    assert "http://localhost:3001/d/qa-pipeline-runs/?var-run=full-20260922" in grafana


def test_a_row_naming_no_run_still_links_to_grafana() -> None:
    """Every row written before run_id existed carries NULL and cannot."""
    (grafana, *_) = elsewhere(chain(run=None), PUBLISHED)

    assert "var-run" not in grafana
    assert "names no run" in grafana


def test_dagster_opens_the_asset_that_stage_materialises() -> None:
    """`questions`, which is the key, not `question_generation`, the stage."""
    _, dagster, _ = elsewhere(chain(), PUBLISHED)

    assert "http://localhost:3000/assets/questions" in dagster


def test_argilla_names_the_numbered_dataset() -> None:
    """The reviewer has to be told which of three datasets to open."""
    *_, argilla = elsewhere(chain(), PUBLISHED)

    assert "`6-questions`" in argilla
    assert "http://localhost:6900/datasets" in argilla


def test_a_fact_points_at_its_own_stage_not_the_questions_one() -> None:
    """Extraction's asset and extraction's dataset, numbered 4."""
    _, dagster, argilla = elsewhere(chain(kind="fact"), PUBLISHED)

    assert "/assets/facts" in dagster
    assert "`4-facts`" in argilla


def test_a_document_has_an_asset_and_no_dataset() -> None:
    """Nobody reviews a conversion: judging one needs the source page."""
    lines = elsewhere(chain(kind="document"), PUBLISHED)

    assert [one.split("**")[1] for one in lines] == ["Grafana", "Dagster"]
    assert "/assets/parsed_documents" in lines[1]


def test_a_tool_this_deployment_does_not_publish_is_left_out() -> None:
    """An unfollowable link is worse than none, as with PHOENIX_BASE_URL."""
    assert elsewhere(chain(), {}) == []
    assert len(elsewhere(chain(), {"argilla": "http://localhost:6900"})) == 1
