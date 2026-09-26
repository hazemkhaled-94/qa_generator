"""What a service does, and how to read the numbers it leaves behind.

Two things a page cannot say in a tooltip: the order a service works in,
and what one measurement means. Both are drawn here so a view supplies the
content and none of the layout.

Everything this draws is folded away. The rule the pages were made minimal
for still holds - nothing not needed on arrival is shown by default - and a
reference nobody has opened yet is exactly that.

A view supplies `Step`s and `Gate`s. Neither is derived from the backend:
they are the prose beside the code, and a service's own README is where
they come from.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import streamlit as st

#: How a reading's direction reads, by whether a high value is the safe one.
_DIRECTION = {True: "higher is safer", False: "lower is safer"}

#: The gates a HIGH reading refuses, for rows written before the direction
#: was stored beside the reading. Every row written since carries its own,
#: and this is only ever the fallback - the same fallback, and the same two
#: names, as `stages/recalibrate.py`. A similarity gate is the only kind
#: that runs this way.
_REFUSES_HIGH = frozenset({"near_duplicate", "duplicate"})


@dataclass(frozen=True)
class Step:
    """One step of a service, in the order the service takes them.

    Attributes:
        name: What the step is called.
        what: What it does, in one sentence.
        parts: The modules that do it, as a reader would open them.
        settings: The settings that change what it does.
    """

    name: str
    what: str
    parts: tuple[str, ...] = ()
    settings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Gate:
    """One gate, as the page explains rather than counts it.

    Attributes:
        code: The rejection code a row carries when this gate refuses it.
        phase: The gate group it runs in, which is what a trace records.
        tests: What has to be true for a row to get past it.
        costs: What running it spends - nothing, a probe, or a call.
        kind: `rule`, `measurement` or `opinion`.
    """

    code: str
    phase: str
    tests: str
    costs: str
    kind: str


@dataclass(frozen=True)
class Reading:
    """What one measuring gate's number is, for the panel that shows it.

    Attributes:
        reads: What the number is a measurement of.
        refuses: What the gate does when the reading crosses its threshold.
    """

    reads: str
    refuses: str
    scale: str = ""


@dataclass(frozen=True)
class Service:
    """Everything a page can say about the one service it runs."""

    what: str
    steps: Sequence[Step] = field(default_factory=tuple)
    gates: Sequence[Gate] = field(default_factory=tuple)
    readings: Mapping[str, Reading] = field(default_factory=dict)


def high_is_safe(reading: Mapping[str, object]) -> bool:
    """Which way one stored reading runs.

    Off the reading where it carries one. A row written before the
    direction was stored falls back to the gate's name, which matters:
    defaulting those to `True` would tell a reader that a question scoring
    0.93 against a 0.93 duplicate ceiling was as safe as it gets.
    """
    recorded = reading.get("high_is_safe")
    if recorded is not None:
        return bool(recorded)
    return reading.get("gate") not in _REFUSES_HIGH


def direction(reading: Mapping[str, object]) -> str:
    """How a reading's direction reads on a page."""
    return _DIRECTION[high_is_safe(reading)]


def reachable(threshold: float, safe_end: float) -> str:
    """The stretch of a scale a reading can actually land in."""
    low, high = sorted((float(threshold), float(safe_end)))
    return f"{low:.2f}–{high:.2f}"


def panel(service: Service, label: str = "How this works") -> None:
    """Draws one service's workings, in a single fold.

    One fold and not three, split into tabs inside it. A page that opens
    with five collapsed sections is as unreadable as one that opens with
    forty figures, and the three things in here are read by the same
    person at the same moment: what the service does, what can refuse a
    row, and what the numbers on that row mean.

    Tabs and not nested folds, because Streamlit refuses an expander
    inside an expander.
    """
    with st.expander(label):
        st.caption(service.what)
        counted = f"Gates · {len(service.gates)}" if service.gates else "Gates"
        steps, gates, numbers = st.tabs(["What it does", counted, "The numbers"])
        with steps:
            _steps(service.steps)
        with gates:
            _gates(service.gates)
        with numbers:
            _readings(service.readings)


def _steps(steps: Sequence[Step]) -> None:
    """The order the service works in, numbered as it runs."""
    if not steps:
        return
    st.dataframe(
        [
            {
                "Step": f"{position}. {one.name}",
                "What it does": one.what,
                "What runs it": ", ".join(one.parts) or "—",
                "Settings": ", ".join(one.settings) or "—",
            }
            for position, one in enumerate(steps, start=1)
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "What it does": st.column_config.TextColumn(width="large"),
            "What runs it": st.column_config.TextColumn(width="medium"),
        },
    )


def _gates(gates: Sequence[Gate]) -> None:
    """Every gate, in the order they are applied.

    The phase is carried beside the code because they are two names for
    one thing seen from two sides, and a page showing only one of them
    leaves a reader unable to match a trace to a rejection.

    A service with no gates says so. An empty table would read as a table
    that failed to load, where "nothing here refuses a row" is a fact worth
    stating - it is the whole of what the evaluation phase is.
    """
    if not gates:
        st.caption(
            "This service has no gates: nothing it does refuses a row or "
            "changes a verdict another stage reached."
        )
        return
    st.caption(
        "Applied cheapest first, and the first one to refuse stops the rest. "
        "A gate after the one that refused a question never read it, which is "
        "not the same as passing it."
    )
    st.dataframe(
        [
            {
                "Rejected as": one.code,
                "In phase": one.phase,
                "Passes when": one.tests,
                "Kind": one.kind,
                "Costs": one.costs,
            }
            for one in gates
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "Passes when": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption(
        "**Rule** reads what is already there and calls nothing. "
        "**Measurement** takes a number and compares it to a threshold — only "
        "these carry a margin. **Opinion** is a model's judgement, which has "
        "no number behind it and so contributes no confidence."
    )


def confidence(row: Mapping[str, Any], readings: Mapping[str, Reading]) -> None:
    """What the measuring gates read on one row, and how close the weakest came.

    A verdict says a gate let this through; these say by how much. A row
    0.92 from an accepted twin and one 0.01 from it are both `accepted`
    against the same ceiling, and this panel is the difference.

    Absent where nothing measured anything - an opinion has no number
    behind it and a structural rule has no scale - and saying so is better
    than a row of dashes nobody can read a meaning into.

    Shared by every page whose rows carry `gate_scores`, so a fact and a
    question are read the same way.
    """
    scores: list[dict] = list(row.get("gate_scores") or [])
    if not scores:
        st.caption(
            "No gate that read this measured anything, so it carries no "
            "confidence. That is not a low one — most gates are rules, "
            "which either fire or do not, and opinions, which have no scale "
            "behind them."
        )
        return

    st.caption(_weakest(row, scores, readings))
    st.dataframe(
        [
            {
                "Gate": one["gate"],
                "Reads": readings[one["gate"]].reads
                if one["gate"] in readings
                else "—",
                "Measured": one["value"],
                "Refuses at": one["threshold"],
                "Safe when": direction(one),
                "Margin": one["margin"],
            }
            for one in scores
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "Reads": st.column_config.TextColumn(
                width="large", help="What the measured number is, on this gate's scale."
            ),
            "Measured": st.column_config.NumberColumn(
                format="%.3f", help="What this gate read for this row."
            ),
            "Refuses at": st.column_config.NumberColumn(
                format="%.2f",
                help="The value at which the verdict changes. Whether crossing "
                "it means going above or below is the next column.",
            ),
            "Margin": st.column_config.ProgressColumn(
                "Margin",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
                help="How much room the reading had, as a share of the room "
                "its scale offers. 1.00 is as far from refusal as that gate "
                "goes; 0.00 is on the line. Comparable between gates, which "
                "the raw numbers are not.",
            ),
        },
    )
    if any("safe_end" not in one for one in scores):
        st.caption(
            ":orange[This row was scored before the margins were calibrated,] "
            "so its confidence is measured against room its scale never had "
            "and reads lower than it should. `make confidence-recalibrate` "
            "takes every stored margin again."
        )


def _weakest(
    row: Mapping[str, Any],
    scores: list[dict],
    readings: Mapping[str, Reading],
) -> str:
    """The confidence as a sentence, naming what is weakest about the row.

    The number alone is the thing nobody could read: it is the SMALLEST of
    several readings on different scales, so which gate it came from is
    most of what it says.
    """
    held = row.get("confidence")
    if held is None:
        return "Nothing this row was measured against carries a margin."

    worst = min(scores, key=lambda one: one["margin"])
    reading = readings.get(worst["gate"])
    comfort = "comfortable" if held >= 0.5 else "close" if held < 0.2 else "adequate"
    said = (
        f"**Confidence {held:.2f} — {comfort}.** The weakest of the "
        f"{len(scores)} reading(s) below is `{worst['gate']}`, which measured "
        f"{worst['value']:.3f} against a threshold of {worst['threshold']:.2f} "
        f"({direction(worst)})."
    )
    if reading:
        said += f" {reading.reads}"
    return (
        f"{said} A margin, not a probability — it is the smallest reading of "
        f"the several below, which sit on different scales and cannot be "
        f"averaged."
    )


def _readings(readings: Mapping[str, Reading]) -> None:
    """What each measuring gate's number is, and which way it runs.

    Absent where nothing measures anything, which is most stages: a
    citation resolves or it does not, and a conversion raises or it does
    not. Neither carries a number to be close to a threshold on.
    """
    if not readings:
        st.caption(
            "Nothing in this service is a measurement, so no row it writes "
            "carries a confidence. Its checks are rules, which either fire "
            "or do not."
        )
        return
    st.dataframe(
        [
            {
                "Gate": name,
                "The number is": one.reads,
                "Scale": one.scale or "—",
                "Refused when": one.refuses,
            }
            for name, one in readings.items()
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "The number is": st.column_config.TextColumn(width="large"),
        },
    )
