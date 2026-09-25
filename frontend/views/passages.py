"""Passages view. Runs chunking, and no other stage."""

from __future__ import annotations

import streamlit as st

from lib import backend, catalog, configure, lineage, page, stage

#: The one stage this page runs. It queues over a document and replaces all
#: of its passages at once, so a passage picked below is run by its document.
_CHUNKING = stage.Queue("chunking", "Chunking", "documents", "chunked")

#: The columns the search box can look in, by the heading each carries in
#: the table below, mapped to the name /passages takes.
_FIELDS = {"Text": "text", "Section": "section", "Text and section": "both"}

#: The chunk states a document passes through.
_STATES = ("new", "pending", "in_progress", "chunked", "failed")


def view() -> None:
    """Renders the passages page."""
    page.header("Passages")

    client = backend.catalog_api()
    counts = client.stage_status("chunking")["rows"]
    documents = sum(counts.values())
    passages = sum(client.stage_status("extraction")["rows"].values())

    with page.panel("Overview"):
        page.stats(
            {
                "Passages": (f"{passages:,}", "Passages chunking has produced."),
                "Documents chunked": (
                    page.share(counts.get("chunked", 0), documents),
                    "Documents split into passages.",
                ),
                "Not started": (f"{counts.get('new', 0):,}", page.STATUS_HELP["new"]),
                "Failed": (f"{counts.get('failed', 0):,}", page.STATUS_HELP["failed"]),
            }
        )

    # Held open above the search box and filled below it: the fold measures
    # the page of rows those filters select, and a panel is drawn where it is
    # created rather than where it is filled.
    analysis, service = st.container(), st.container()

    words, field = catalog.search("passages", _FIELDS)
    chosen, pager = catalog.filters(
        "passages",
        {
            "block_type": (
                "Types",
                client.passage_types(),
                "The kind of block the parser found this passage in.",
            )
        },
        documents=client.document_names(),
    )

    where = {
        "document": chosen["document"],
        "q": words,
        "field": field,
        "block_type": chosen["block_type"],
    }
    total, rows = catalog.paged(
        "passages",
        pager,
        lambda limit, offset: client.passages(**where, limit=limit, offset=offset),
        "passages",
    )

    with analysis, st.expander("Analysis"):
        page.findings(_analysis(counts, rows))

    with service, page.panel("Chunking"):
        stage.service(client, _CHUNKING)
        configure.panel("chunking")

    with page.panel(f"Passages · {total:,}"):
        if not rows:
            st.caption("Nothing matches.")
            return
        picked = page.table(
            [
                {
                    "#": row["ordinal"],
                    "Pages": _pages(row),
                    "Type": row["block_type"] or "—",
                    "Language": (row["language"] or "—").upper(),
                    "Section": row["section_path"] or "—",
                    "Sentences": row["sentence_count"],
                    "Text": row["text"],
                }
                for row in rows
            ],
            key="passages-table",
            column_config={"Text": st.column_config.TextColumn(width="large")},
        )

    if picked is not None:
        _detail(client, rows[picked])


def _analysis(counts: dict[str, int], rows: list[dict]) -> list[dict]:
    """Builds the fold: the chunk queue, then the page of rows measured."""
    queue = page.queue_rows("Documents", counts, _STATES)
    if not rows:
        return queue
    lengths = [len(row["text"]) for row in rows]
    measured = {
        "Shortest on this page": (
            f"{min(lengths):,} characters",
            "A very short passage often carries no detectable language.",
        ),
        "Longest on this page": (
            f"{max(lengths):,} characters",
            ("One far above the rest means the chunker found no boundary to cut on."),
        ),
        "Sentences on this page": (
            f"{sum(row['sentence_count'] for row in rows):,}",
            "The numbers a fact cites.",
        ),
        "Rows carrying a table": (
            f"{sum(1 for row in rows if row['table_count']):,}",
            "Passages holding a cell grid rather than prose.",
        ),
    }
    return queue + [
        {
            "Check": check,
            "Value": value,
            "Should be": "—",
            "State": "—",
            "What it means": what,
        }
        for check, (value, what) in measured.items()
    ]


def _detail(client, row: dict) -> None:
    """Shows everything held about one passage, and chunks its document."""
    detail = client.passage(row["id"])
    scope = ("document", row["doc_sha256"])

    with page.panel(f"Passage #{row['ordinal']}"):
        page.attributes(
            {
                "Id": row["id"],
                "Document": row["doc_sha256"],
                "Position": row["ordinal"],
                "Pages": _pages(row),
                "Section": row["section_path"],
                "Block type": row["block_type"],
                "Language": row["language"],
                "Characters": len(row["text"]),
                "Sentences": row["sentence_count"],
                "Tables": row["table_count"],
                "Converter items": row["doc_item_refs"],
                "Boxes": len(row["bbox"]),
                "Extraction": detail["extract_status"],
                "Extraction error": detail["extract_error"],
                "Text": row["text"],
            }
        )

        st.caption(
            "Sentences",
            help="The number in the first column is what a fact's citation refers to.",
        )
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

        for index, grid in enumerate(detail["table_cells"], start=1):
            st.caption(
                f"Table {index} of {len(detail['table_cells'])}"
                + (f" · {grid['caption']}" if grid.get("caption") else "")
                + f" · {grid['num_rows']}x{grid['num_cols']}"
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
                        # A string either way: a column mixing one with a
                        # number is not a column the table can render.
                        "Cites row": page.written(cell["line"]),
                    }
                    for cell in grid["cells"]
                ],
                width="stretch",
                hide_index=True,
                column_config={"Text": st.column_config.TextColumn(width="large")},
            )

        st.caption(
            "Chunking runs over a document and replaces all of its "
            "passages, this one included."
        )
        stage.service(client, _CHUNKING, scope)

        with st.expander("How this was produced"):
            lineage.panel("passage", row["id"])


def _pages(row: dict) -> str:
    """Describes the page range a passage covers."""
    first, last = row["page_from"], row["page_to"]
    if first is None:
        return "—"
    return str(first) if first == last else f"{first}–{last}"


page.render(view)
