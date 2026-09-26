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
        steps, gates, numbers = st.tabs(
            ["What it does", f"Gates · {len(service.gates)}", "The numbers"]
        )
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
    """
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


def _readings(readings: Mapping[str, Reading]) -> None:
    """What each measuring gate's number is, and which way it runs."""
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
