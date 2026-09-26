"""The topic modelling command line.

Every flag is the same operation as a route under /topics, so what each one
has to do is call one collaborator and answer with an exit status.
"""

from __future__ import annotations

import pytest
from topic_drivers import CommandDriver


@pytest.fixture
def cli(monkeypatch, tmp_path) -> CommandDriver:
    """The command line with its collaborators replaced."""
    return CommandDriver(monkeypatch, tmp_path / "topics")


def test_status_reports_the_queue_and_runs_nothing(cli) -> None:
    """The cheap flag: one query, no worker."""
    assert cli.run("--status") == 0
    assert cli.calls == ["counts"], cli.calls
    assert cli.drains == 0


def test_stop_withdraws_a_queued_fit(cli) -> None:
    """The same as POST /topics/stop."""
    assert cli.run("--stop") == 0
    assert cli.calls == ["stop"], cli.calls


def test_retry_returns_a_failed_fit_to_the_queue(cli) -> None:
    """The same as POST /topics/retry."""
    assert cli.run("--retry") == 0
    assert cli.calls == ["retry"], cli.calls


def test_delete_removes_every_topic(cli) -> None:
    """The same as DELETE /topics."""
    assert cli.run("--delete") == 0
    assert cli.calls == ["delete_all"], cli.calls


def test_discover_queues_a_fit_and_drains_it(cli) -> None:
    """Passed on its own it queues and works in one go."""
    assert cli.run("--discover") == 0
    assert cli.queued == [1], cli.queued
    assert cli.drains == 1, cli.drains


def test_no_flag_at_all_drains_without_queueing(cli) -> None:
    """A bare run works what is already there."""
    assert cli.run() == 0
    assert cli.queued == []
    assert cli.drains == 1


def test_watch_keeps_draining_instead_of_returning(cli) -> None:
    """What the worker container runs."""
    assert cli.run("--watch") == 0
    assert cli.drains == 0, "watch drains on its own schedule"
    assert cli.watched, "the watch loop was never entered"


def test_discover_with_watch_queues_then_watches(cli) -> None:
    """Both, in that order."""
    assert cli.run("--discover", "--watch") == 0
    assert cli.queued == [1]
    assert cli.watched


def test_two_actions_in_one_command_are_refused(cli) -> None:
    """They are mutually exclusive, so the parser rejects the pair."""
    with pytest.raises(SystemExit):
        cli.run("--stop", "--retry")


# ── Writing the figures out ───────────────────────────────────────────────


def test_visualise_writes_one_file_per_modelled_language(cli) -> None:
    """One page per language, named by its code."""
    cli.fits = ["de", "en"]
    cli.drew("de")
    cli.drew("en", b"<html>en</html>")

    assert cli.run("--visualise") == 0
    assert set(cli.files()) == {"de.html", "en.html"}, cli.files()
    assert cli.files()["en.html"] == b"<html>en</html>"


def test_render_redraws_each_language_from_its_stored_model(cli) -> None:
    """The figure comes back without a re-fit of the corpus.

    `export` holds a view and `models` holds what it is a view of, so a
    lost page costs a render. Before the models were kept this was the one
    thing that could not be done.
    """
    cli.fits = ["de", "en"]
    cli.fitted("de")
    cli.fitted("en")

    assert cli.run("--render") == 0
    assert set(cli.figures) == {"topics/de.html", "topics/en.html"}, cli.figures
    assert cli.figures["topics/de.html"].startswith(b"<"), "a page, not a sentinel"


def test_render_replaces_the_figure_already_there(cli) -> None:
    """One key per language, so the redraw overwrites rather than adds.

    Idempotent rather than lossy: the same stored model through the same
    `prepare` draws the same page.
    """
    cli.fits = ["de"]
    cli.fitted("de")
    cli.drew("de", b"<html>the old figure</html>")

    assert cli.run("--render") == 0
    assert cli.figures["topics/de.html"] != b"<html>the old figure</html>"
    # Away before the new one goes in, so a render that raises leaves no
    # figure rather than the previous language's.
    assert "remove topics/de.html" in cli.calls
    assert cli.calls.index("remove topics/de.html") < cli.calls.index(
        "put topics/de.html"
    )


def test_render_can_be_narrowed_to_one_language(cli) -> None:
    """`--language` picks which LANGUAGE, not which fit.

    There is one model per language because the database holds one fit's
    topics, so there is no older model to choose.
    """
    cli.fits = ["de", "en"]
    cli.fitted("de")
    cli.fitted("en")

    assert cli.run("--render", "--language", "en") == 0
    assert set(cli.figures) == {"topics/en.html"}, cli.figures


def test_render_for_a_language_with_no_topics_is_an_error(cli) -> None:
    """Naming a language the corpus does not hold is a typo, not a no-op."""
    cli.fits = ["de"]
    cli.fitted("de")

    assert cli.run("--render", "--language", "fr") == 1
    assert cli.figures == {}


def test_render_with_no_model_stored_is_an_error(cli) -> None:
    """Topics fitted before this bucket existed have no model to draw from."""
    cli.fits = ["de"]

    assert cli.run("--render") == 1
    assert cli.figures == {}


def test_visualise_with_no_topics_stored_is_an_error(cli) -> None:
    """There is nothing to draw, and the exit status has to say so."""
    cli.fits = []

    assert cli.run("--visualise") == 1
    assert cli.files() == {}


def test_visualise_where_no_language_has_a_figure_is_an_error(cli) -> None:
    """Topics fitted before a fit that draws leave nothing to write."""
    cli.fits = ["de"]

    assert cli.run("--visualise") == 1
    assert cli.files() == {}


def test_visualise_writes_what_it_has_when_one_language_is_missing(cli) -> None:
    """A partial answer is still worth writing, so this succeeds."""
    cli.fits = ["de", "en"]
    cli.drew("de")

    assert cli.run("--visualise") == 0
    assert set(cli.files()) == {"de.html"}, cli.files()
