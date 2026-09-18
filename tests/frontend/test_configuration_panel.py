"""The configuration panel, on the page that runs the service it configures.

The panel holds no list of settings: every control is derived from what the
API says about the setting, so what these cover is the deriving. A page test
is the only place that reaches it, because what a control does with a type
and a pair of bounds is not visible from either side on its own.

The backend is stubbed, including the answers that are refusals - which is
the half no integration test reaches, and the half a person actually needs
to read.
"""

from __future__ import annotations

import pytest
import requests
from conftest import Answers
from pages import SETTINGS, answers, settings

pytestmark = pytest.mark.frontend

#: Every page and the service its panel configures. One page configures one
#: service, and this is that mapping as the views declare it.
PAGES = [
    ("upload", "ingestion"),
    ("documents", "parsing"),
    ("passages", "chunking"),
    ("facts", "extraction"),
    ("topics", "topics"),
    ("questions", "questions"),
    ("health", "platform"),
]


def _clients(configured: Answers) -> dict:
    """Every client any page reaches for, so one test can open all of them."""
    return {
        "catalog_api": Answers(**answers()),
        "settings_api": configured,
        "upload_api": Answers(counts={"documents": 3, "upload_attempts": 5}),
        "health_api": Answers(reachable=(True, "Reachable."), components={}),
    }


def _save(app):
    """The panel's Save button, by its label.

    By label and not by position: the stage's own controls are drawn above
    it, and which index it lands on differs from page to page.
    """
    return next(one for one in app.button if one.label == "Save")


@pytest.mark.parametrize(("name", "service"), PAGES)
def test_every_page_asks_for_the_service_it_runs(run_view, name, service) -> None:
    """One page configures one service, and never another page's."""
    configured = Answers(**settings())
    app = run_view(name, **_clients(configured))

    assert app.exception == [], app.exception
    asked = [args[0] for method, args, _ in configured.asked if method == "settings"]
    assert asked == [service], asked


def test_a_number_is_drawn_as_a_number(open_view) -> None:
    """Typed from what the API said it is, not from the page's own list."""
    view = open_view("topics")

    drawn = {one.label: one for one in view.app.number_input}
    assert "A_COUNT" in drawn
    assert drawn["A_COUNT"].value == 5


def test_a_share_is_drawn_with_the_bounds_it_has(open_view) -> None:
    """A share above one is a gate nothing passes, so the control refuses it."""
    view = open_view("topics")

    drawn = {one.label: one for one in view.app.number_input}
    assert drawn["A_SHARE"].value == 0.5


def test_a_flag_is_drawn_as_a_checkbox(open_view) -> None:
    """Read the way the backend's own reader reads it."""
    view = open_view("topics")

    drawn = {one.label: one for one in view.app.checkbox}
    assert drawn["A_FLAG"].value is True


def test_a_closed_set_is_drawn_as_a_picker(open_view) -> None:
    """A list of two is a picker, not a box somebody can mistype into."""
    view = open_view("topics")

    drawn = {one.label: one for one in view.app.selectbox}
    assert drawn["A_MODE"].value == "fast"
    assert list(drawn["A_MODE"].options) == ["fast", "accurate"]


def test_a_list_with_a_closed_set_is_drawn_as_a_multiselect(open_view) -> None:
    """Each entry is checked against the list, so each entry is picked."""
    view = open_view("topics")

    drawn = {one.label: one for one in view.app.multiselect}
    assert drawn["A_LIST"].value == ["one", "two"]


def test_a_setting_the_deployment_owns_is_drawn_and_disabled(open_view) -> None:
    """Visible without opening a shell, and not writable from here."""
    view = open_view("topics")

    drawn = {one.label: one for one in view.app.text_input}
    assert drawn["A_POOL"].disabled is True


def test_a_setting_whose_absence_means_something_is_drawn_empty(open_view) -> None:
    """An empty box is how one is turned off, which a number box cannot say."""
    view = open_view("topics")

    drawn = {one.label: one for one in view.app.text_input}
    assert drawn["A_MODEL"].value == ""


def test_a_changed_setting_is_marked(open_view) -> None:
    """A reader should see at a glance that the file no longer decides it."""
    changed = [{**one, "stored": True} for one in SETTINGS["settings"]]
    view = open_view("topics", configured=Answers(**settings(settings=changed)))

    assert "A_COUNT ·" in [one.label for one in view.app.number_input]


def test_saving_sends_the_version_the_panel_was_drawn_from(open_view) -> None:
    """So a save that would land on somebody else's change is refused."""
    configured = Answers(**settings())
    view = open_view("topics", configured=configured)

    _save(view.app).click().run()

    sent = [call for call in configured.asked if call[0] == "change_settings"]
    assert sent, "the save button sent nothing"
    assert sent[0][1][2] == SETTINGS["version"]


def test_saving_sends_every_setting_it_may_write(open_view) -> None:
    """And not the one the deployment owns, which it must not try to write."""
    configured = Answers(**settings())
    view = open_view("topics", configured=configured)

    _save(view.app).click().run()

    sent = next(call for call in configured.asked if call[0] == "change_settings")
    written = sent[1][1]
    assert "A_COUNT" in written
    assert "A_POOL" not in written


def test_a_refused_change_is_shown_in_the_words_the_api_refused_it(
    open_view,
) -> None:
    """The one thing a person can act on, and the reason the API writes it.

    An HTTPError stringifies as its status line, so a page that showed the
    exception would show `400 Client Error` and nothing about which setting
    or what was wrong with it.
    """
    refused = requests.exceptions.HTTPError("400 Client Error")
    refused.response = _refusal(
        "would_not_load", "topics would not start with that: TOPIC_PASSES=many"
    )
    configured = Answers(settings=lambda service: SETTINGS, change_settings=refused)
    view = open_view("topics", configured=configured)

    after = _save(view.app).click().run()

    assert "TOPIC_PASSES" in after.error[0].value


def test_a_change_that_stales_something_says_so_and_it_stays(open_view) -> None:
    """A toast disappears; what this says to do costs a corpus-sized run."""
    configured = Answers(
        settings=lambda service: SETTINGS,
        change_settings=lambda service, values, version: {
            "service": service,
            "version": "changed",
            "changed": ["A_COUNT"],
            "cleared": [],
            "stale": ["chunking", "questions"],
            "detail": "1 setting(s) changed. What chunking, questions already "
            "produced was made under the old values; rerun that stage.",
        },
    )
    view = open_view("topics", configured=configured)

    after = _save(view.app).click().run()

    assert "rerun" in after.warning[0].value


def test_an_unreachable_backend_leaves_the_rest_of_the_page_working(
    open_view,
) -> None:
    """The panel is one section of a page that has a job of its own.

    A page that would not draw because its configuration could not be read
    would be a page where nothing could be looked at either.
    """
    view = open_view(
        "topics",
        configured=Answers(settings=requests.exceptions.ConnectionError("refused")),
    )

    assert view.raised == [], view.raised
    assert "Configuration unavailable" in view.text()


def _refusal(code: str, detail: str):
    """A response carrying the API's refusal body."""

    class Response:
        """Just enough of a response for the panel to read the reason."""

        @staticmethod
        def json() -> dict:
            """The body every deliberate refusal carries."""
            return {"code": code, "detail": detail}

    return Response()
