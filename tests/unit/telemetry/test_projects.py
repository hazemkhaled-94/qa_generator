"""Which Phoenix projects are leftovers, and which are somebody's data.

The whole risk of this tool is in one predicate. A project holding only
model calls is a preflight that never became a run and is worth taking
back; a project whose first page happens to be model calls may be a
corpus of labelling. `5-topic_modelling` reads exactly the second way and
held fifty spans of it.
"""

from __future__ import annotations

import pytest

from telemetry import projects


def project(name: str, spans: list[str]) -> projects.Project:
    """One project holding spans with these names."""
    return projects.Project(
        id=name,
        name=name,
        spans=len(spans),
        work=any(one != projects.CALL for one in spans),
        whole=len(spans) < projects.SAMPLE,
    )


def test_a_project_of_nothing_but_model_calls_is_spent() -> None:
    """A preflight proved the model answers and the process then died."""
    assert project("7-assessment-abc", [projects.CALL]).spent


def test_a_project_holding_a_stage_span_is_not() -> None:
    """Something claimed a row in it."""
    assert not project("7-assessment-demo", [projects.CALL, "assess"]).spent


def test_an_empty_project_is_spent() -> None:
    """Nothing was ever filed under it."""
    assert project("stale", []).spent


def test_a_project_too_big_to_read_in_one_page_is_left_alone() -> None:
    """The failure this exists to avoid.

    `5-topic_modelling` held fifty model calls in its first page and a
    corpus of labelling behind them. Read to the sample limit, every span
    seen is a model call - and deleting on that reading destroys the run.
    """
    filled = project("5-topic_modelling", [projects.CALL] * projects.SAMPLE)

    assert not filled.work, "every span read was a model call"
    assert not filled.whole, "and the page was full, so there may be more"
    assert not filled.spent, "so it must not be taken"


def test_default_is_never_taken() -> None:
    """It is where a misconfigured span lands, which is the evidence."""
    held = [
        project("default", [projects.CALL]),
        project("7-parsing-abc", [projects.CALL]),
    ]

    assert [one.name for one in projects.spent(held)] == ["7-parsing-abc"]


def test_the_report_counts_both_sides() -> None:
    """So somebody can see what a prune would do before running one."""
    held = [
        project("7-assessment-a", [projects.CALL]),
        project("7-assessment-b", [projects.CALL]),
        project("7-assessment-demo", ["assess"]),
    ]

    lines = "\n".join(projects.report(held))

    assert "3 project(s)" in lines
    assert "2 of them hold no work" in lines
    assert "1 hold something" in lines


def test_pruning_nothing_deletes_nothing() -> None:
    """A second run of the same command is not a second deletion."""

    class _Client:
        """A client that would record a delete, if one were asked for."""

        def __init__(self) -> None:
            """Holds nothing yet."""
            self.deleted: list[str] = []

        def delete(self, path: str):
            """Records the delete."""
            self.deleted.append(path)
            raise AssertionError("nothing should have been deleted")

    client = _Client()

    assert projects.prune(client, [project("7-assessment-demo", ["assess"])]) == 0
    assert client.deleted == []


def test_pruning_takes_each_spent_project_once() -> None:
    """One DELETE per project, and the count is what went."""

    class _Answer:
        """A successful response, as `requests` reports one."""

        ok = True

    class _Client:
        """A client that succeeds at every delete."""

        def __init__(self) -> None:
            """Holds nothing yet."""
            self.deleted: list[str] = []

        def delete(self, path: str):
            """Records the delete."""
            self.deleted.append(path)
            return _Answer()

    client = _Client()
    held = [project("7-parsing-a", [projects.CALL]), project("7-parsing-b", [])]

    assert projects.prune(client, held) == 2
    assert client.deleted == ["/v1/projects/7-parsing-a", "/v1/projects/7-parsing-b"]


def test_purging_takes_the_projects_that_hold_work_too() -> None:
    """Which is the whole difference from pruning.

    `make wipe` deletes the corpus, and a trace of a run over documents
    that no longer exist is not worth keeping. The spent-only rule would
    leave every project that recorded a real run, which after a wipe is
    all of them.
    """

    class _Answer:
        """A successful response, as `requests` reports one."""

        ok = True

    class _Client:
        """A client that succeeds at every delete."""

        def __init__(self) -> None:
            """Holds nothing yet."""
            self.deleted: list[str] = []

        def delete(self, path: str):
            """Records the delete."""
            self.deleted.append(path)
            return _Answer()

    held = [project("spent", [projects.CALL]), project("worked", ["extract"])]

    pruning = _Client()
    assert projects.prune(pruning, held) == 1
    assert pruning.deleted == ["/v1/projects/spent"], "pruning leaves the run"

    purging = _Client()
    assert projects.prune(purging, held, everything=True) == 2
    assert purging.deleted == ["/v1/projects/spent", "/v1/projects/worked"]


def test_purging_an_empty_phoenix_deletes_nothing() -> None:
    """A wipe run twice is not an error."""
    assert projects.prune(None, [], everything=True) == 0


@pytest.mark.parametrize("action", ["--list", "--prune", "--purge"])
def test_the_three_actions_are_exclusive_and_one_is_required(action: str) -> None:
    """The shape every command line in this repository takes."""
    assert projects.parser().parse_args([action])

    with pytest.raises(SystemExit):
        projects.parser().parse_args([])
    with pytest.raises(SystemExit):
        projects.parser().parse_args(["--list", "--prune"])
    with pytest.raises(SystemExit):
        projects.parser().parse_args(["--prune", "--purge"])


# ── Reading past the first page ────────────────────────────────────────────


class _Paged:
    """A Phoenix holding more projects than one page returns."""

    def __init__(self, pages: list[list[str]]) -> None:
        """Answers each listing request with the next page."""
        self.pages = pages
        self.asked: list[dict] = []

    def get(self, path: str, params: dict | None = None):
        """Answers a listing or a span query."""
        if path.endswith("/spans"):
            return _Json({"data": [{"name": projects.CALL}]})
        self.asked.append(dict(params or {}))
        at = 0 if not params or "cursor" not in params else int(params["cursor"])
        names = self.pages[at]
        return _Json(
            {
                "data": [{"id": one, "name": one} for one in names],
                "next_cursor": str(at + 1) if at + 1 < len(self.pages) else None,
            }
        )


class _Json:
    """A response carrying a decoded body."""

    def __init__(self, body: dict) -> None:
        """Holds the body."""
        self.body = body

    def raise_for_status(self) -> None:
        """It succeeded."""

    def json(self) -> dict:
        """The body."""
        return self.body


def test_every_page_is_read_not_just_the_first() -> None:
    """A cursor ignored is a count that is really the page size.

    Phoenix answers with a page and a `next_cursor`. Reading one page
    against a deployment holding 696 projects reported 500, pruned 497,
    and then reported 500 again from the next page - which is
    indistinguishable from a delete that silently failed, and is what
    happened.
    """
    client = _Paged([["a", "b"], ["c", "d"], ["e"]])

    found = projects.held(client)

    assert [one.name for one in found] == ["a", "b", "c", "d", "e"]
    assert len(client.asked) == 3, "one request per page, to the end"
    assert "cursor" not in client.asked[0], "the first page asks for no cursor"
    assert client.asked[1]["cursor"] == "1"


def test_a_single_page_asks_once() -> None:
    """The common case does not pay for a second request."""
    client = _Paged([["only"]])

    assert len(projects.held(client)) == 1
    assert len(client.asked) == 1
