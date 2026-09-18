"""The configuration panel, beside the controls that run the stage.

Drawn inside the page's service panel, where the Redo button that rebuilds
what a change staled already is.

Every control is derived from what the API says about the setting - its
type, its bounds, the closed set of values where there is one - so there is
no list of settings here.

Collapsed by default.
"""

from __future__ import annotations

import json
from typing import Any

import requests
import streamlit as st

from lib import backend

#: How a flag is written back. Everything goes to the API as text.
_TRUE, _FALSE = "true", "false"

#: Where a message waits out the rerun that a save causes.
_SAID = "settings-said-{service}"


def refusal(error: requests.exceptions.RequestException) -> str:
    """Reads the sentence the API refused a change with.

    An HTTPError stringifies as its status line alone.
    """
    response = getattr(error, "response", None)
    if response is None:
        return str(error)
    try:
        body = response.json()
    except (ValueError, json.JSONDecodeError):
        return str(error)
    return body.get("detail") or str(error)


def panel(service: str) -> None:
    """Draws one service's configuration, inside the panel that runs it.

    Builds its own client: two of the pages that draw this hold no other.
    """
    client = backend.settings_api()
    try:
        held = client.settings(service)
    except requests.exceptions.RequestException as error:
        st.caption(f"Configuration unavailable: {refusal(error)}")
        return

    # Outside the fold, so it is read without opening the panel again.
    said = st.session_state.pop(_SAID.format(service=service), None)
    if said:
        st.warning(said)

    changed = sum(one["stored"] for one in held["settings"])
    label = "Configuration" + (f" · {changed} changed" if changed else "")
    with st.expander(label):
        _form(client, service, held)
        if changed:
            _reset(client, service, held)


def _form(client, service: str, held: dict) -> None:
    """Draws the settings as a form, and saves what it is submitted with."""
    with st.form(f"settings-{service}"):
        values: dict[str, str | None] = {}
        for setting in held["settings"]:
            drawn = _control(service, setting)
            if drawn is not None:
                values[setting["name"]] = drawn
        submitted = st.form_submit_button("Save", type="primary", width="stretch")

    if submitted:
        _save(client, service, values, held["version"])


def _reset(client, service: str, held: dict) -> None:
    """Offers to return every changed setting to what the files say."""
    stored = [one["name"] for one in held["settings"] if one["stored"]]
    if st.button(
        "Return to the file",
        key=f"reset-{service}",
        width="stretch",
        help=f"Deletes the {len(stored)} stored value(s), so "
        f"configs/env/backend.env and .env decide again.",
    ):
        _save(client, service, dict.fromkeys(stored), held["version"])


def _save(client, service: str, values: dict[str, str | None], version: str) -> None:
    """Sends one change and says what it did, or why it was refused."""
    try:
        answer = client.change_settings(service, values, version)
    except requests.exceptions.RequestException as error:
        st.error(refusal(error))
        return

    st.toast(answer.get("detail") or f"{service}: settings saved")
    moved = answer.get("changed") or answer.get("cleared")
    if answer.get("stale"):
        # Held, not drawn: it has to outlive the rerun below.
        st.session_state[_SAID.format(service=service)] = answer["detail"]
    if moved:
        st.rerun(scope="app")


def _control(service: str, setting: dict) -> str | None:
    """Draws one setting's control, and reads back what it holds as text.

    Returns None for a setting the deployment owns, which is drawn disabled
    and never written.
    """
    name = setting["name"]
    key = f"setting-{service}-{name}"
    value = setting["value"]
    label = _label(setting)
    help = _help(setting)

    if setting["fixed"]:
        st.text_input(label, value=value or "", key=key, help=help, disabled=True)
        return None

    if setting["kind"] == "boolean":
        return (
            _TRUE
            if st.checkbox(label, value=_flag(value), key=key, help=help)
            else _FALSE
        )

    if setting["choices"]:
        return _chosen(setting, label, key, help)

    if setting["optional"]:
        # Typed by hand even for a number: a number box cannot say nothing,
        # and an empty box is how one is turned off.
        return st.text_input(label, value=value or "", key=key, help=help)

    if setting["kind"] == "integer":
        return str(
            int(
                st.number_input(
                    label,
                    value=int(value or 0),
                    step=1,
                    min_value=_bound(setting["low"], int),
                    max_value=_bound(setting["high"], int),
                    key=key,
                    help=help,
                )
            )
        )

    if setting["kind"] == "decimal":
        return _written(
            st.number_input(
                label,
                value=float(value or 0),
                step=0.01,
                min_value=_bound(setting["low"], float),
                max_value=_bound(setting["high"], float),
                key=key,
                help=help,
            )
        )

    return st.text_input(label, value=value or "", key=key, help=help)


def _chosen(setting: dict, label: str, key: str, help: str) -> str:
    """Draws the control for a setting with a closed set of values."""
    choices = setting["choices"]
    if setting["kind"] in ("csv", "mapping"):
        held = [
            one.strip() for one in (setting["value"] or "").split(",") if one.strip()
        ]
        picked = st.multiselect(
            label,
            choices,
            default=[one for one in held if one in choices],
            key=key,
            help=help,
        )
        return ",".join(picked)

    # A blank first entry when absence means something.
    offered = ["", *choices] if setting["optional"] else list(choices)
    current = setting["value"] or ""
    # `or ""`: nothing chosen reads as absent.
    return (
        st.selectbox(
            label,
            offered,
            index=offered.index(current) if current in offered else 0,
            key=key,
            help=help,
        )
        or ""
    )


def _label(setting: dict) -> str:
    """The setting's name, marked when the files no longer decide it."""
    return f"{setting['name']} ·" if setting["stored"] else setting["name"]


def _help(setting: dict) -> str:
    """What the setting does, and what changing it leaves behind."""
    said = setting["help"]
    if setting["fixed"]:
        said = f"{said} Set in .env; changing it needs a restart."
    if setting["stored"]:
        said = f"{said} Changed from {setting['default'] or 'unset'}."
    if setting["invalidates"]:
        said = (
            f"{said} What {', '.join(setting['invalidates'])} already produced "
            f"was made under the current value."
        )
    return said


def _flag(value: str | None) -> bool:
    """Reads a stored flag the way the backend's own reader reads it."""
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _bound(value: Any, kind: type) -> Any:
    """One bound, in the type its control wants, or nothing."""
    return None if value is None else kind(value)


def _written(value: float) -> str:
    """Writes a number back without the trailing zeros a widget adds."""
    return f"{value:g}"
