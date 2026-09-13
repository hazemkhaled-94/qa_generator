"""Passages view."""

from __future__ import annotations

import streamlit as st

from lib import backend, catalog, page, stage

#: Chunking produces the passages this page lists; extraction consumes them.
#: Both appear, because the state a passage is in belongs to one or the other.
_CHUNKING = stage.Queue("chunking", "Chunking", "documents", "chunked")
_EXTRACTION = stage.Queue("extraction", "Extraction", "passages", "extracted")

#: The columns the search box can look in, by the heading each one carries
#: in the table below, mapped to the name /passages takes.
_FIELDS = {"Text": "text", "Section": "section", "Text and section": "both"}


def view() -> None:
    """Renders the passages page."""
    page.header(
        "Passages",
        "Where the chunker cut each document, the heading every passage sits "
        "under and the block it came from. Pick a passage from the table to "
        "read it in full and to re-run extraction over that passage alone.",
    )

    client = backend.catalog_api()
    page.section(
        "Chunking and extraction across the whole corpus",
        "Every passage the corpus holds, not only the ones matching the "
        "filters below. Chunking counts documents, because it queues over a "
        "document and replaces all of its passages at once; extraction "
        "counts passages. These figures refresh on their own every few "
        "seconds.",
    )
    stage.overview(client, [_CHUNKING, _EXTRACTION], _corpus_figures)

    st.divider()
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
            else "No passages yet. Chunk some documents from the Documents page."
        )
        return

    page.section(
        "The passages these filters select",
        "Measured over every passage matching the search, the document and "
        "the type chosen above - not over the corpus, and not only over the "
        "page shown below. The lengths are the exception and say so: they "
        "are measured on this page of rows, because the backend returns the "
        "text of these rows and not of the rest.",
    )
    lengths = [len(row["text"]) for row in rows]
    page.metrics(_filtered_figures(total, rows, lengths))

    st.dataframe(
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
        width="stretch",
        hide_index=True,
        column_config={"Text": st.column_config.TextColumn(width="large")},
    )

    st.divider()
    chosen = st.selectbox(
        "Passage to work on",
        rows,
        format_func=lambda r: f"#{r['ordinal']} · {(r['section_path'] or '—')[:60]}",
        help="Everything below this point - the figures, the text and the "
        "extraction controls - applies to this passage alone.",
    )
    _detail(client, chosen)


def _corpus_figures(
    counts: dict[str, dict[str, int]],
) -> dict[str, tuple[object, str]]:
    """Names the corpus-wide figures the top of this page carries."""
    passages = sum(counts["extraction"].values())
    documents = sum(counts["chunking"].values())
    return {
        "Passages in corpus": (
            f"{passages:,}",
            (
                "Every passage chunking has produced, across every document. "
                "Re-chunking a document replaces its passages, so this figure "
                "falls and rises as documents are rebuilt."
            ),
        ),
        "Documents chunked": (
            *page.portion(counts["chunking"].get("chunked", 0), documents),
            (
                "Documents that have been split into passages, as a share of "
                "every document stored. A document that is not chunked "
                "contributes no passages here."
            ),
        ),
        "Passages per document": (
            f"{passages / documents:.0f}" if documents else "—",
            (
                "Passages in the corpus divided by documents stored. A rough "
                "figure: it counts documents that have not been chunked yet, so "
                "it reads low while chunking is still running."
            ),
        ),
        "Read for facts": (
            *page.portion(counts["extraction"].get("extracted", 0), passages),
            (
                "Passages extraction has read, as a share of every passage. A "
                "passage counts as read whether or not the facts it yielded "
                "passed their checks."
            ),
        ),
        "Extraction failed": (
            f"{counts['extraction'].get('failed', 0):,}",
            page.STATUS_HELP["failed"]
            + " Retry from the Documents page, or from the controls below "
            "for one passage at a time.",
        ),
    }


def _filtered_figures(
    total: int, rows: list[dict], lengths: list[int]
) -> dict[str, tuple[object, str]]:
    """Names the figures for whatever the filters currently select."""
    sentences = sum(row["sentence_count"] for row in rows)
    tables = sum(1 for row in rows if row["table_count"])
    return {
        "Passages matching": (
            f"{total:,}",
            (
                "Passages matching every filter above at once: the search text "
                "in the chosen column, the chosen document and the chosen block "
                "type. This is what the pager counts through."
            ),
        ),
        "Shown on this page": (
            f"{len(rows):,}",
            (
                "Rows fetched for the page number above. PAGE_SIZE sets how "
                "many; the four figures to the right are measured on these rows."
            ),
        ),
        "Shortest on page": (
            f"{min(lengths):,}",
            "characters",
            (
                "The shortest passage among the rows shown. Very short passages "
                "often carry no detectable language, which puts them outside "
                "every topic model."
            ),
        ),
        "Longest on page": (
            f"{max(lengths):,}",
            "characters",
            (
                "The longest passage among the rows shown. One far above the "
                "rest usually means the chunker could not find a heading or a "
                "sentence boundary to cut on."
            ),
        ),
        "Sentences on page": (
            f"{sentences:,}",
            (
                "Numbered sentences across the rows shown. These numbers are "
                "what a fact cites, so a fact can be traced back to the sentence "
                f"it names. {tables:,} of these rows carry a table."
            ),
        ),
    }


def _detail(client, row: dict) -> None:
    """Shows one passage in full, its numbered sentences and its cell grids.

    The numbers are what a fact cites, so this is where a citation can be
    traced back to the text it names.
    """
    detail = client.passage(row["id"])
    scope = ("passage", str(row["id"]))

    page.section(
        f"Passage #{row['ordinal']}",
        "Measurements of this passage alone, and of the facts extraction drew from it.",
    )
    page.metrics(
        {
            "Characters": (
                f"{len(row['text']):,}",
                (
                    "Length of this passage's text. The chunker aims at a size "
                    "budget; one far above it means no boundary was found to cut "
                    "on."
                ),
            ),
            "Sentences": (
                f"{row['sentence_count']:,}",
                (
                    "Numbered sentences spaCy found. A fact cites one or more of "
                    "these numbers, which is how a claim is traced back to the "
                    "text that supports it."
                ),
            ),
            "Tables": (
                f"{row['table_count']:,}",
                (
                    "Cell grids the parser recovered from this passage. Their "
                    "cells are listed below and are read for facts alongside the "
                    "prose."
                ),
            ),
            "Pages": (
                _pages(row),
                "Which pages of the source PDF this passage was drawn from.",
            ),
            "Extraction": (
                detail["extract_status"],
                page.STATUS_HELP.get(
                    detail["extract_status"], "This passage's extraction state."
                ),
            ),
        }
    )

    if detail["extract_status"] == "failed" and detail.get("extract_error"):
        st.error(f"Extraction failed for this passage: {detail['extract_error']}")

    page.section(
        "Run this passage",
        "Queues, withdraws or repeats extraction for this passage alone. "
        "Chunking has no row here because it queues over a document and "
        "replaces all of its passages at once - to re-chunk, use the "
        "Documents page. Deleting a single passage is the same: passages "
        "belong to their document, so the Documents page deletes them "
        "together. " + stage.COLOUR_KEY,
    )
    stage.controls(client, _EXTRACTION, scope)

    st.text(row["text"])

    with st.expander(f"{len(detail['sentences'])} numbered sentence(s)"):
        st.caption(
            "The number in the first column is what a fact's citation refers "
            "to. Claims is how many predicates spaCy found in the sentence: a "
            "sentence carrying several should yield several facts, not one."
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
