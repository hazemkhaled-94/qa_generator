"""The toolbar and pager the pages that list pipeline output share.

One row of five fixed slots on every listing page: the search box, the column
that search looks in, the document picker, the type filter and the page
number. A page that has no use for a slot leaves it blank rather than closing
the gap.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, NamedTuple

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from lib import config

#: Relative widths of the five toolbar slots, in the order they are drawn.
_SLOTS = (2.9, 2.0, 2.4, 2.0, 1.3)


class Toolbar(NamedTuple):
    """What the toolbar was set to, and where the pager goes."""

    document: str | None
    search: str
    field: str | None
    block_type: str | None
    page: DeltaGenerator


def page_size() -> int:
    """Rows per page, from the environment."""
    return config.PAGE_SIZE


def label_for(document: dict) -> str:
    """Names a document the way a person would recognise it."""
    return document.get("filename") or f"{document['sha256'][:12]}…"


#: What the type slot filters, by the heading the page gives it. Passages are
#: filtered on the kind of block the parser found; facts on how the fact was
#: drawn.
_TYPE_HELP = {
    "Type": "The kind of block the parser found this passage in - a "
    "paragraph, a heading, a list, a table. Narrows every figure and row "
    "below to one kind.",
    "Method": "How the fact was drawn: `llm` by the served model, "
    "`deterministic` by a rule that needs no model. Narrows every figure and "
    "row below to one method.",
}


def filters(
    key: str,
    documents: list[dict] | None = None,
    types: list[str] | None = None,
    type_label: str = "Type",
    fields: dict[str, str] | None = None,
) -> Toolbar:
    """Renders the toolbar and reserves the slot the pager fills.

    The document picker is drawn even when the corpus is empty, disabled
    rather than absent, so nothing beside it moves.
    """
    search_slot, field_slot, document_slot, type_slot, page_slot = st.columns(_SLOTS)

    search = search_slot.text_input(
        "Search",
        key=f"{key}-search",
        placeholder="Contains…",
        help="A case-insensitive substring, matched against the column "
        "chosen beside this. Not a word search and not a regular "
        "expression: `risk` matches `risks` and `Risikos` alike. Leave it "
        "empty to match everything.",
    )

    field = None
    if fields:
        heading = field_slot.selectbox(
            "in column",
            list(fields),
            key=f"{key}-field",
            help="Which column the words above are looked for in. Nothing "
            "outside it is searched.",
        )
        field = fields[heading]

    chosen = None
    if documents is not None:
        options = {
            f"{label_for(d)} ({d['sha256'][:8]})": d["sha256"] for d in documents
        }
        selected = document_slot.selectbox(
            "Document",
            ["All documents", *options],
            key=f"{key}-document",
            disabled=not options,
            help="Narrow every figure and row below to one document.",
        )
        chosen = options.get(selected)

    block_type = None
    if types:
        picked = type_slot.selectbox(
            type_label,
            [f"All {type_label.lower()}s", *types],
            key=f"{key}-type",
            help=_TYPE_HELP.get(
                type_label,
                f"Narrows every figure and row below to one {type_label.lower()}.",
            ),
        )
        block_type = picked if picked in types else None

    return Toolbar(chosen, search, field, block_type, page_slot)


def field_name(fields: dict[str, str], field: str | None) -> str:
    """Names a searched column the way the toolbar showed it."""
    heading = next((k for k, v in fields.items() if v == field), next(iter(fields)))
    return heading.lower()


def paged(
    key: str,
    into: DeltaGenerator,
    fetch: Callable[[int, int], tuple[int, list[Any]]],
    rows_key: str,
) -> tuple[int, list[Any]]:
    """Fetches one page, draws the pager into its slot, and returns both.

    The page number is read from session state rather than from the widget,
    which is drawn afterwards: asking the backend for the total first cost a
    second request, and its own count query, on every rerun.
    """
    size = page_size()
    slot = f"{key}-page"
    page = int(st.session_state.get(slot, 1))

    answer = fetch(size, (page - 1) * size)
    total, rows = answer["total"], answer[rows_key]
    if not rows and total and page > 1:
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
        help="Type a page number to go straight to it."
        if pages > 1
        else "Everything matching fits on one page.",
    )
    return total, rows
