"""The search panel, the filter panel and the pager they share.

Two panels, not one. Searching is typing words and choosing where to look for
them; filtering is narrowing to a value a column already holds. They were one
toolbar and read as five equal boxes, four of which were not the search.

The pager is drawn in the filter panel because it narrows what is shown
without changing what matched, and because the total it counts through is
only known after the rows are fetched.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from lib import config, page

#: Filters to a row before the next wraps. Wide enough that the common pages
#: draw one row, narrow enough that Questions' six do not squeeze to nothing.
_PER_ROW = 4


def label_for(document: dict) -> str:
    """Names a document the way a person would recognise it."""
    return document.get("filename") or f"{document['sha256'][:12]}…"


def search(key: str, fields: dict[str, str]) -> tuple[str, str]:
    """Draws the search panel and returns the words and the column.

    `fields` maps the heading a column carries in the table to the name the
    route takes for it.
    """
    with page.panel("Search"):
        # The column picker is drawn only where there is a choice: one page
        # searches filename, title and digest together, because a person
        # looking for a document does not know which of the three they
        # remember.
        words, column = st.columns([3, 2]) if len(fields) > 1 else (st, None)
        typed = words.text_input(
            "Words",
            key=f"{key}-search",
            placeholder="Contains…",
            help="A case-insensitive substring. Not a word search.",
        )
        heading = (
            next(iter(fields))
            if column is None
            else column.selectbox(
                "In column",
                list(fields),
                key=f"{key}-field",
                help="Where the words are looked for.",
            )
        )
    return typed, fields[heading]


def field_name(fields: dict[str, str], field: str | None) -> str:
    """Names a searched column the way the search panel showed it."""
    heading = next((k for k, v in fields.items() if v == field), next(iter(fields)))
    return heading.lower()


def filters(
    key: str,
    choices: dict[str, tuple[str, Sequence[str], str]],
    documents: list[dict] | None = None,
) -> tuple[dict[str, str | None], DeltaGenerator]:
    """Draws the filter panel and reserves the slot the pager fills.

    `choices` maps the name the route takes to the heading, the values and
    the tooltip. Each returns the chosen value, or None for the `All` entry
    every picker carries first.

    The document picker is drawn even when the corpus is empty, disabled
    rather than absent, so nothing beside it moves.
    """
    slots: list[tuple[str, str, list[str], dict | None, str]] = []
    if documents is not None:
        named = {
            f"{label_for(one)} ({one['sha256'][:8]})": one["sha256"]
            for one in documents
        }
        slots.append(
            ("document", "Document", ["All documents", *named], named, "One document.")
        )
    slots += [
        (name, heading, [f"All {heading.lower()}", *values], None, explanation)
        for name, (heading, values, explanation) in choices.items()
    ]

    chosen: dict[str, str | None] = {}
    with page.panel("Filters"):
        # One column past the pickers, which is where the pager goes.
        columns = _laid_out(len(slots) + 1)
        for column, (name, heading, options, named, explanation) in zip(
            columns, slots, strict=False
        ):
            picked = column.selectbox(
                heading,
                options,
                key=f"{key}-{name}",
                disabled=len(options) == 1,
                help=explanation,
            )
            chosen[name] = (
                named.get(picked)
                if named is not None
                else picked
                if picked in options[1:]
                else None
            )
    return chosen, columns[len(slots)]


def _laid_out(count: int) -> list[DeltaGenerator]:
    """Splits that many pickers across as many rows of columns as they need."""
    drawn: list[DeltaGenerator] = []
    while len(drawn) < count:
        # A short last row keeps the width a full row has, rather than
        # spreading two pickers across the page.
        drawn += list(st.columns(_PER_ROW))[: min(count - len(drawn), _PER_ROW)]
    return drawn


def paged(
    key: str,
    into: DeltaGenerator,
    fetch: Callable[[int, int], dict[str, Any]],
    rows_key: str,
) -> tuple[int, list[Any]]:
    """Fetches one page, draws the pager into its slot, and returns both.

    The page number is read from session state rather than from the widget,
    which is drawn afterwards: asking the backend for the total first cost a
    second request, and its own count query, on every rerun.
    """
    size = config.PAGE_SIZE
    slot = f"{key}-page"
    number = int(st.session_state.get(slot, 1))

    answer = fetch(size, (number - 1) * size)
    total, rows = answer["total"], answer[rows_key]
    if not rows and total and number > 1:
        # The corpus shrank under a page somebody was already on.
        st.session_state[slot] = 1
        answer = fetch(size, 0)
        total, rows = answer["total"], answer[rows_key]

    pages = max(1, math.ceil(total / size))
    if int(st.session_state.get(slot, 1)) > pages:
        st.session_state[slot] = 1
    into.number_input(
        f"Page of {pages:,}",
        min_value=1,
        max_value=pages,
        value=1,
        step=1,
        key=slot,
        disabled=pages == 1,
        help="Which page of the rows that matched.",
    )
    return total, rows
