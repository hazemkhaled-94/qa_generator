"""Passages view."""

from __future__ import annotations

import streamlit as st

from lib import backend, catalog, page, stage

#: The columns the search box can look in, by the heading each one carries
#: in the table below, mapped to the name /passages takes.
_FIELDS = {"Text": "text", "Section": "section", "Text and section": "both"}


def view() -> None:
    """Renders the passages page."""
    page.header(
        "Passages",
        "Where the chunker cut each document, the heading every passage sits "
        "under and the block it came from.",
    )
    stage.panel("chunking", "documents", "chunked")

    client = backend.catalog_api()
    bar = catalog.filters(
        "passages",
        client.document_names(),
        types=client.passage_types(),
        fields=_FIELDS,
    )

    where = {
        "document": bar.document,
        "q": bar.search,
        "field": bar.field,
        "block_type": bar.block_type,
    }
    total, rows = catalog.paged(
        "passages",
        bar.page,
        lambda limit, offset: client.passages(**where, limit=limit, offset=offset),
        "passages",
    )
    if not total:
        st.info(
            f"No passages whose {catalog.field_name(_FIELDS, bar.field)} matches."
            if bar.search
            else "No passages match."
            if bar.document or bar.block_type
            else "No passages yet. Chunk some documents from the strip above."
        )
        return

    lengths = [len(row["text"]) for row in rows]
    page.metrics(
        {
            "Passages matching": f"{total:,}",
            "On this page": f"{len(rows):,}",
            "Shortest here": f"{min(lengths):,}",
            "Sentences here": f"{sum(r['sentence_count'] for r in rows):,}",
        }
    )

    st.dataframe(
        [
            {
                "#": row["ordinal"],
                "Pages": _pages(row),
                "Type": row["block_type"] or "—",
                "Lang": (row["language"] or "—").upper(),
                "Section": row["section_path"] or "—",
                "Text": row["text"],
            }
            for row in rows
        ],
        width="stretch",
        hide_index=True,
        column_config={"Text": st.column_config.TextColumn(width="large")},
    )

    st.divider()
    chosen = st.selectbox(
        "Inspect",
        rows,
        format_func=lambda r: f"#{r['ordinal']} · {(r['section_path'] or '—')[:60]}",
    )
    _detail(client, chosen)


def _detail(client, row: dict) -> None:
    """Shows one passage in full, its numbered sentences and its cell grids.

    The numbers are what a fact cites, so this is where a citation can be
    traced back to the text it names.
    """
    st.text(row["text"])
    detail = client.passage(row["id"])

    with st.expander(f"{len(detail['sentences'])} numbered sentence(s)"):
        st.dataframe(
            [
                {
                    "#": sentence["i"],
                    "Claims": sentence["predicates"],
                    "Text": row["text"][sentence["start"] : sentence["end"]],
                }
                for sentence in detail["sentences"]
            ],
            width="stretch",
            hide_index=True,
            column_config={"Text": st.column_config.TextColumn(width="large")},
        )

    if not row["table_count"]:
        return

    for index, grid in enumerate(detail["table_cells"], start=1):
        st.caption(
            f"Table {index} of {len(detail['table_cells'])}"
            + (f" · {grid['caption']}" if grid.get("caption") else "")
            + f" · {grid['num_rows']}x{grid['num_cols']}, "
            f"{len(grid['cells'])} cell(s) rendered here"
        )
        st.dataframe(
            [
                {
                    "Row": cell["row"],
                    "Col": cell["col"],
                    "Header": "column"
                    if cell["column_header"]
                    else "row"
                    if cell["row_header"]
                    else "—",
                    "Text": cell["text"],
                    "Cites row": "—" if cell["line"] is None else cell["line"],
                }
                for cell in grid["cells"]
            ],
            width="stretch",
            hide_index=True,
            column_config={"Text": st.column_config.TextColumn(width="large")},
        )


def _pages(row: dict) -> str:
    """Describes the page range a passage covers."""
    first, last = row["page_from"], row["page_to"]
    if first is None:
        return "—"
    return str(first) if first == last else f"{first}–{last}"


page.render(view)
