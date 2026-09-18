"""The configuration panel, beside the controls that run the stage.

One page configures one service, which is the same service that page runs, so
this is drawn inside the page's service panel: the remedy for a change that
stales something is the Redo button already there.

Every control is derived from what the API says about the setting - its type,
its bounds, the closed set of values where there is one - so there is no list
of settings here and adding one to the catalogue adds it to the page. Nothing
about which settings exist is known on this side.

Collapsed by default. Configuring a stage is not what anybody arrives to do,
and a page that opens with forty numbers on it is a page nobody reads.
"""

from __future__ import annotations

import json
from typing import Any

import requests
import streamlit as st

from lib import backend

#: How a value is written back. Everything goes to the API as text, because
#: that is what the environment would have handed the stage, and the stage's
#: own reader is what parses it.
_TRUE, _FALSE = "true", "false"

#: Where a message waits out the rerun that a save causes.
_SAID = "settings-said-{service}"


def refusal(error: requests.exceptions.RequestException) -> str:
    """Reads what the API refused a change with.

    Every deliberate refusal carries a code and a sentence saying what to
    fix; an HTTPError stringifies as the status line alone, which is the one
    thing a person cannot act on.
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

    Builds its own client: every page draws one of these and two of them
    hold no other client, so threading one through would mean the Upload
    page reaching for a client named after the Documents page.
    """
    client = backend.settings_api()
    try:
        held = client.settings(service)
    except requests.exceptions.RequestException as error:
        st.caption(f"Configuration unavailable: {refusal(error)}")
        return

    # Drawn before the fold and outside it, so it is read whether or not
    # anybody opens the panel again. A save reruns the page to redraw what
    # the change moved, and the rerun is what would otherwise take this
    # away on the way past.
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
        # Held rather than drawn: what this says to do costs a corpus-sized
        # run and is the person's decision, so it has to outlive the rerun
        # below rather than flash past with it.
        st.session_state[_SAID.format(service=service)] = answer["detail"]
    if moved:
        st.rerun(scope="app")


def _control(service: str, setting: dict) -> str | None:
    """Draws one setting's control, and reads back what it holds as text.

    Returns None for a setting this page will not write, which is one the
    deployment owns: it is drawn so somebody can see it without opening a
    shell, and disabled so nobody tries.
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
        # Typed by hand even when it is a number: an empty box is how a
        # setting whose absence means something is turned off, and a number
        # box has no way to say nothing.
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

    # A blank first entry when absence means something, so the control can
    # say nothing as well as say a value.
    offered = ["", *choices] if setting["optional"] else list(choices)
    current = setting["value"] or ""
    # `or ""`: a picker with an index always hands one back, and nothing
    # chosen reads as absent, which is what the blank entry above means.
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
