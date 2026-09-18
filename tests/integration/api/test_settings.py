"""Reading and changing what a service is configured to do, over HTTP.

The routes are a trust boundary: what they accept becomes what every worker
does on the row it claims next, and what they refuse is the only thing
between a typed value and a worker that will not start.

Every test runs against the real wiring - the real store, the real
catalogue, and each stage's real `Settings.load` - because what is being
checked is that the validation a page is refused by is the validation the
worker would have failed at start-up with.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration

#: A service, a setting of its own, and a value that setting accepts.
OWNED = [
    ("ingestion", "MAX_FILE_SIZE_MB", "7"),
    ("parsing", "PARSING_TABLE_MODE", "fast"),
    ("chunking", "CHUNKING_MERGE_PEERS", "false"),
    ("extraction", "EXTRACTION_BRIDGE_PASSAGES", "3"),
    ("topics", "TOPIC_PASSES", "42"),
    ("questions", "QUESTIONS_PER_TOPIC", "9"),
    ("platform", "LLM_TEMPERATURE", "0.5"),
]


def test_every_service_reports_its_settings(client) -> None:
    """One page configures one service, so each must have something to draw."""
    for service, _, _ in OWNED:
        answered = client.get(f"/settings/{service}")

        assert answered.status_code == 200, service
        body = answered.json()
        assert body["service"] == service
        assert body["settings"], service


def test_a_service_nothing_configures_is_refused(client) -> None:
    """The list is closed, so a typo in a path is not an empty page."""
    assert client.get("/settings/nonsense").status_code == 422


def test_a_setting_says_what_it_is_and_what_it_holds(client) -> None:
    """What a page draws a control from, holding no list of its own."""
    body = client.get("/settings/topics").json()
    found = {one["name"]: one for one in body["settings"]}

    passes = found["TOPIC_PASSES"]
    assert passes["kind"] == "integer"
    assert passes["help"]
    assert passes["value"] == os.environ["TOPIC_PASSES"]
    assert passes["default"] == os.environ["TOPIC_PASSES"]
    assert passes["stored"] is False
    assert passes["low"] == 1
    assert passes["invalidates"] == ["topics"]


def test_a_closed_set_of_values_is_published(client) -> None:
    """A page draws a list, not a text box, when there is a list."""
    body = client.get("/settings/parsing").json()
    found = {one["name"]: one for one in body["settings"]}

    assert found["PARSING_TABLE_MODE"]["choices"] == ["fast", "accurate"]


def test_a_setting_the_deployment_owns_is_shown_and_marked(client) -> None:
    """Visible without opening a shell, and marked so nobody tries to write it."""
    body = client.get("/settings/platform").json()
    found = {one["name"]: one for one in body["settings"]}

    assert found["DATABASE_POOL_SIZE"]["fixed"] is True
    assert found["LLM_TEMPERATURE"]["fixed"] is False


@pytest.mark.parametrize(("service", "name", "value"), OWNED)
def test_a_setting_can_be_changed_and_read_back(client, service, name, value) -> None:
    """Every service, one at a time, so a failure says which one."""
    answered = client.patch(f"/settings/{service}", json={"values": {name: value}})

    assert answered.status_code == 200, answered.text
    assert answered.json()["changed"] == [name]

    found = {
        one["name"]: one
        for one in client.get(f"/settings/{service}").json()["settings"]
    }
    assert found[name]["value"] == value
    assert found[name]["stored"] is True
    assert found[name]["default"] == os.environ.get(name)
    assert found[name]["changed_at"]


def test_a_change_reaches_the_stage_that_reads_it(client) -> None:
    """The point of all of it: a value written is a value a worker loads."""
    from settings.store import Settings as Store
    from topic_modelling.config import Settings

    client.patch("/settings/topics", json={"values": {"TOPIC_PASSES": "42"}})

    assert Settings.load(Store().resolved()).passes == 42


def test_a_setting_can_be_returned_to_the_files(client) -> None:
    """Null is the reset, and there is no stored copy of a default to restore."""
    client.patch("/settings/topics", json={"values": {"TOPIC_PASSES": "42"}})

    answered = client.patch("/settings/topics", json={"values": {"TOPIC_PASSES": None}})

    assert answered.status_code == 200, answered.text
    assert answered.json()["cleared"] == ["TOPIC_PASSES"]

    found = {
        one["name"]: one for one in client.get("/settings/topics").json()["settings"]
    }
    assert found["TOPIC_PASSES"]["value"] == os.environ["TOPIC_PASSES"]
    assert found["TOPIC_PASSES"]["stored"] is False


def test_writing_the_value_already_set_changes_nothing(client) -> None:
    """A page saving a form it did not edit should not look like a change."""
    answered = client.patch(
        "/settings/topics",
        json={"values": {"TOPIC_PASSES": os.environ["TOPIC_PASSES"]}},
    )

    assert answered.json()["changed"] == []
    assert "nothing changed" in answered.json()["detail"]


def test_a_setting_another_page_owns_is_refused(client) -> None:
    """One page configures one service, and the route holds that line."""
    answered = client.patch(
        "/settings/topics", json={"values": {"QUESTIONS_PER_TOPIC": "9"}}
    )

    assert answered.status_code == 400
    assert answered.json()["code"] == "wrong_service"


def test_a_setting_nothing_configures_is_refused(client) -> None:
    """A typo cannot be stored and then wondered about."""
    answered = client.patch("/settings/topics", json={"values": {"TOPIC_PASSESS": "9"}})

    assert answered.status_code == 404
    assert answered.json()["code"] == "unknown_setting"


def test_a_setting_the_deployment_owns_cannot_be_written(client) -> None:
    """A pool size is read when a process starts; storing one would do nothing."""
    answered = client.patch(
        "/settings/platform", json={"values": {"DATABASE_POOL_SIZE": "50"}}
    )

    assert answered.status_code == 400
    assert answered.json()["code"] == "fixed_setting"


def test_a_value_off_the_list_is_refused_with_the_list(client) -> None:
    """What the stage's parser would take and the catalogue knows better of."""
    answered = client.patch(
        "/settings/parsing", json={"values": {"PARSING_TABLE_MODE": "medium"}}
    )

    assert answered.status_code == 400
    assert answered.json()["code"] == "not_a_choice"
    assert "accurate" in answered.json()["detail"]


def test_a_list_of_values_is_checked_one_entry_at_a_time(client) -> None:
    """A closed set constrains each entry, not the whole string.

    Checking the string against the list would refuse every setting that
    takes more than one thing: these are two kinds that are both on the
    list, and `atomic,summary` is not itself on it.
    """
    answered = client.patch(
        "/settings/extraction",
        json={"values": {"EXTRACTION_KINDS": "atomic,summary"}},
    )

    assert answered.status_code == 200, answered.text
    assert answered.json()["changed"] == ["EXTRACTION_KINDS"]


def test_one_bad_entry_in_a_list_is_named_on_its_own(client) -> None:
    """Which entry was wrong, rather than that the whole value was."""
    answered = client.patch(
        "/settings/extraction",
        json={"values": {"EXTRACTION_KINDS": "atomic,nonsense"}},
    )

    assert answered.status_code == 400
    assert answered.json()["code"] == "not_a_choice"
    assert "nonsense" in answered.json()["detail"]


@pytest.mark.parametrize("value", ["2", "-0.5"])
def test_a_share_outside_its_bounds_is_refused(client, value) -> None:
    """A number that parses and still cannot be what it claims to be.

    An overlap of 2 is a gate no answer passes, and the stage's own parser
    has no opinion about it: it reads a float and gets one.
    """
    answered = client.patch(
        "/settings/questions", json={"values": {"QUESTIONS_ANSWER_OVERLAP": value}}
    )

    assert answered.status_code == 400
    assert answered.json()["code"] == "out_of_range"


def test_a_value_the_stage_cannot_parse_is_refused_in_the_stages_words(
    client,
) -> None:
    """The message a worker would have failed at start-up with."""
    answered = client.patch(
        "/settings/topics", json={"values": {"TOPIC_PASSES": "many"}}
    )

    assert answered.status_code == 400
    assert answered.json()["code"] == "would_not_load"
    assert "TOPIC_PASSES" in answered.json()["detail"]


def test_a_mix_naming_something_unaskable_is_refused(client) -> None:
    """The stage's own validation, reached through the route.

    Written once, in question_generation.config, and this is that message
    arriving at a page rather than at a worker's first start.
    """
    answered = client.patch(
        "/settings/questions", json={"values": {"QUESTIONS_TYPE_MIX": "nonsense:1"}}
    )

    assert answered.status_code == 400
    assert answered.json()["code"] == "would_not_load"
    assert "QUESTIONS_TYPE_MIX" in answered.json()["detail"]


def test_extraction_still_insists_on_the_atomic_kind(client) -> None:
    """Every passage is read for its claims, whoever asks otherwise."""
    answered = client.patch(
        "/settings/extraction", json={"values": {"EXTRACTION_KINDS": "summary,outline"}}
    )

    assert answered.status_code == 400
    assert answered.json()["code"] == "would_not_load"


def test_a_change_breaking_another_service_is_refused(client) -> None:
    """EMBEDDING_MAX_TOKENS is the platform's and chunking reads it.

    A change validated only against the service that owns it would be a
    change that stops a worker that never appeared in the request.
    """
    answered = client.patch(
        "/settings/platform", json={"values": {"EMBEDDING_MAX_TOKENS": "not a number"}}
    )

    assert answered.status_code == 400
    assert answered.json()["code"] == "would_not_load"


def test_nothing_is_written_when_one_value_in_a_request_is_refused(client) -> None:
    """Half a change is a configuration nobody asked for."""
    answered = client.patch(
        "/settings/topics",
        json={"values": {"TOPIC_PASSES": "42", "TOPIC_TOP_TERMS": "many"}},
    )

    assert answered.status_code == 400

    found = {
        one["name"]: one for one in client.get("/settings/topics").json()["settings"]
    }
    assert found["TOPIC_PASSES"]["stored"] is False


def test_a_change_names_what_it_staled(client) -> None:
    """The remedy is the stage's own rerun, on the page that already offers it."""
    answered = client.patch(
        "/settings/platform", json={"values": {"EMBEDDING_MODEL": "intfloat/other"}}
    ).json()

    assert answered["stale"] == ["chunking", "questions"]
    assert "rerun" in answered["detail"]


def test_a_change_that_stales_nothing_says_nothing_about_rebuilding(client) -> None:
    """What the next run writes is not what the last one wrote."""
    answered = client.patch(
        "/settings/questions", json={"values": {"QUESTIONS_PER_TOPIC": "9"}}
    ).json()

    assert answered["stale"] == []
    assert "rerun" not in answered["detail"]


def test_the_version_moves_when_a_setting_changes(client) -> None:
    """What a page hands back to write safely."""
    before = client.get("/settings/topics").json()["version"]

    after = client.patch(
        "/settings/topics", json={"values": {"TOPIC_PASSES": "42"}}
    ).json()["version"]

    assert after != before


def test_a_write_against_the_current_version_lands(client) -> None:
    """The ordinary case: a page drawn, edited and saved."""
    version = client.get("/settings/topics").json()["version"]

    answered = client.patch(
        "/settings/topics",
        json={"values": {"TOPIC_PASSES": "42"}, "version": version},
    )

    assert answered.status_code == 200, answered.text


def test_a_write_against_a_version_that_has_moved_is_refused(client) -> None:
    """Two people on one page, and the second does not silently win."""
    version = client.get("/settings/topics").json()["version"]
    client.patch("/settings/topics", json={"values": {"TOPIC_TOP_TERMS": "5"}})

    answered = client.patch(
        "/settings/topics",
        json={"values": {"TOPIC_PASSES": "42"}, "version": version},
    )

    assert answered.status_code == 409
    assert answered.json()["code"] == "version_moved"


def test_a_setting_whose_absence_means_something_can_be_turned_off(client) -> None:
    """Empty is how a value the files name is overridden back to absent.

    Set first, because that is the only state there is anything to turn off
    in: the test environment names no verifier, and writing empty over an
    absence is the no-op below.
    """
    client.patch(
        "/settings/questions",
        json={"values": {"QUESTIONS_VERIFIER_MODEL": "ollama_chat/checker"}},
    )

    answered = client.patch(
        "/settings/questions", json={"values": {"QUESTIONS_VERIFIER_MODEL": ""}}
    )

    assert answered.status_code == 200, answered.text
    assert answered.json()["changed"] == ["QUESTIONS_VERIFIER_MODEL"]

    found = {
        one["name"]: one for one in client.get("/settings/questions").json()["settings"]
    }
    assert found["QUESTIONS_VERIFIER_MODEL"]["value"] is None
    assert found["QUESTIONS_VERIFIER_MODEL"]["stored"] is True

    from question_generation.config import Settings
    from settings.store import Settings as Store

    assert Settings.load(Store().resolved()).verifier_model is None


def test_writing_empty_over_an_absence_changes_nothing(client, monkeypatch) -> None:
    """Absent already, so there is nothing to turn off and no row to write.

    The absence is made rather than assumed. `.env` carries a verifier on a
    developer's machine and not in CI, so a test that read whichever the
    shell happened to export passed on one and failed on the other - and
    the environment is half of what `resolved` compares a new value
    against, which is the thing under test here.
    """
    monkeypatch.delenv("QUESTIONS_VERIFIER_MODEL", raising=False)

    answered = client.patch(
        "/settings/questions", json={"values": {"QUESTIONS_VERIFIER_MODEL": ""}}
    )

    assert answered.status_code == 200, answered.text
    assert answered.json()["changed"] == []


def test_the_upload_limit_takes_effect_without_a_restart(client, pdf) -> None:
    """Ingestion is the one service with no worker, so the API is the worker.

    The limit was baked into a service built at import, which is why this is
    the one setting a change could not have reached.
    """
    client.patch("/settings/ingestion", json={"values": {"MAX_FILE_SIZE_MB": "1"}})

    refused = client.post(
        "/documents",
        files={
            "file": (
                "big.pdf",
                b"%PDF-1.4" + b"0" * (2 * 1024 * 1024),
                "application/pdf",
            )
        },
    )

    assert refused.status_code == 413, refused.text


def test_raising_the_upload_limit_takes_effect_too(client, pdf) -> None:
    """The same change in the other direction, so it is the limit that moved."""
    client.patch("/settings/ingestion", json={"values": {"MAX_FILE_SIZE_MB": "1"}})
    client.patch("/settings/ingestion", json={"values": {"MAX_FILE_SIZE_MB": "100"}})

    accepted = client.post(
        "/documents", files={"file": ("small.pdf", pdf, "application/pdf")}
    )

    assert accepted.status_code == 200, accepted.text


def test_the_generation_plan_reports_a_changed_mix(client) -> None:
    """What was asked for, beside what came out, has to be what is asked now.

    The plan was read at start-up, so a mix changed through a page would have
    been reported as whatever the API booted with.
    """
    client.patch("/settings/questions", json={"values": {"QUESTIONS_PER_TOPIC": "3"}})

    assert client.get("/questions/plan").json()["per_topic"] == 3


def test_the_model_litellm_calls_can_be_changed(client) -> None:
    """The setting this was all for, and the one furthest from a database.

    The api calls no model and serves this anyway: which model the workers
    call is a setting like any other, and the page that runs a stage is
    where it is chosen.
    """
    answered = client.patch(
        "/settings/platform", json={"values": {"LLM_MODEL": "ollama_chat/other"}}
    )

    assert answered.status_code == 200, answered.text

    from llm.config import Settings
    from settings.store import Settings as Store

    assert Settings.load(Store().resolved()).model == "ollama_chat/other"
