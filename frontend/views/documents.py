"""Documents view. Runs parsing, and no other stage.

Also where the whole pipeline is set going, which is not the same thing and
is not a stage this page owns: the Pipeline panel asks the orchestrator to
decide that every stage should run, and the orchestrator is what then starts
each in turn. It is here because this is the page somebody lands on after an
upload, and "I have added a document, now take it through" is the question
they arrive with. See `lib/pipeline.py`.
"""

from __future__ import annotations

import streamlit as st

from lib import backend, catalog, configure, lineage, page, pipeline, stage

#: The one stage this page runs. Chunking is the Passages page's and
#: extraction is the Facts page's; neither can be reached from here.
_PARSING = stage.Queue("parsing", "Parsing", "documents", "parsed")

#: The column the search box looks in. One entry, because the backend
#: searches filename, title and digest together.
_FIELDS = {"Document, title or digest": "any"}

#: The parse states a document can be narrowed to.
_STATES = ("new", "pending", "in_progress", "parsed", "failed")

#: Everything a document row holds, by the heading the detail table gives it.
_ATTRIBUTES = {
    "Document": "filename",
    "Title": "title",
    "Digest": "sha256",
    "Language": "language",
    "Pages": "page_count",
    "Uploaded": "first_seen",
    "Parsing": "parse_status",
    "Parsing error": "parse_error",
    "Chunking": "chunk_status",
    "Chunking error": "chunk_error",
    "Passages": "total_passages",
    "Passages read": "extracted_passages",
    "Oversized passages": "oversized",
}


def view() -> None:
    """Renders the documents page."""
    page.header("Documents")

    client = backend.catalog_api()
    counts = client.stage_status("parsing")["rows"]
    documents = sum(counts.values())

    with page.panel("Overview"):
        page.stats(
            {
                "Documents": (
                    f"{documents:,}",
                    "PDFs held, whatever stage each is at.",
                ),
                "Parsed": (
                    page.share(counts.get("parsed", 0), documents),
                    "Converted into a structured document.",
                ),
                "Not started": (
                    f"{counts.get('new', 0):,}",
                    page.STATUS_HELP["new"],
                ),
                "Failed": (f"{counts.get('failed', 0):,}", page.STATUS_HELP["failed"]),
            }
        )

    with st.expander("Analysis"):
        page.findings(page.queue_rows("Documents", counts, _STATES))

    # Before Parsing, because it is the broader of the two: this takes the
    # corpus through every stage, and the panel below runs the one stage
    # this page owns.
    with page.panel("Pipeline"):
        pipeline.panel(backend.pipeline_api())

    with page.panel("Parsing"):
        stage.service(client, _PARSING)
        configure.panel("parsing")

    words, _ = catalog.search("documents", _FIELDS)
    chosen, pager = catalog.filters(
        "documents",
        {"parse_status": ("Parse states", _STATES, "One parse state.")},
    )

    total, rows = catalog.paged(
        "documents",
        pager,
        lambda limit, offset: client.documents(
            q=words, parse_status=chosen["parse_status"], limit=limit, offset=offset
        ),
        "documents",
    )

    with page.panel(f"Documents · {total:,}"):
        if not rows:
            st.caption("Nothing matches.")
            return
        picked = page.table(
            [
                {
                    "Document": catalog.label_for(document),
                    "Title": document["title"] or "—",
                    "Pages": page.written(document["page_count"]),
                    "Language": (document["language"] or "—").upper(),
                    "Parsing": document["parse_status"],
                    "Passages": document["total_passages"],
                    "Uploaded": (document["first_seen"] or "")[:16].replace("T", " "),
                }
                for document in rows
            ],
            key="documents-table",
        )

    if picked is not None:
        _detail(client, rows[picked])


@st.cache_data(show_spinner=False, max_entries=4)
def _file(_client, sha256: str) -> bytes:
    """Fetches one document, remembering the last few.

    Cached because clicking the download button reruns the script, which
    would otherwise pull the file again to redraw the viewer beside it.
    `_client` is underscored so Streamlit leaves it out of the cache key.
    """
    return _client.file(sha256)


def _detail(client, document: dict) -> None:
    """Shows everything held about one document, and runs parsing over it."""
    sha = document["sha256"]
    name = catalog.label_for(document)
    scope = ("document", sha)

    with page.panel(name):
        page.attributes(
            {label: document.get(key) for label, key in _ATTRIBUTES.items()}
        )
        stage.service(client, _PARSING, scope)

        # A toggle, not an expander: an expander runs its body whichever way
        # it is folded, so the fetch would happen on every rerun regardless.
        if st.toggle("View the file", key=f"documents-file-{sha}"):
            with st.spinner(f"Loading {name}…"):
                data = _file(client, sha)
            st.download_button(
                "Download", data=data, file_name=name, mime="application/pdf"
            )
            st.pdf(data, height=700)

        with st.expander("How this was produced"):
            lineage.panel("document", sha)

    _removal(client, document, name, sha)


def _removal(client, document: dict, name: str, sha: str) -> None:
    """Offers the two deletions, each behind its own confirmation."""
    chosen = stage.removal(
        "documents",
        "Acts on this document only. The rows and the file are archived "
        "rather than destroyed, and nothing here puts them back.",
        {
            "derived": (
                "Delete passages and facts",
                (
                    "Deletes its passages and the facts drawn from them. The "
                    "document, its file and its parsed form stay, and "
                    "chunking returns to new."
                ),
                f"Delete every passage and fact drawn from {name}?",
            ),
            "whole": (
                "Delete document",
                (
                    "Deletes the document row, the file, the parsed form and "
                    "everything drawn from it."
                ),
                f"Delete {name} and everything drawn from it?",
            ),
        },
        subject=sha,
    )
    if chosen is None:
        return
    removed = client.delete(sha, derived_only=chosen == "derived")
    st.toast(
        f"Removed {removed['passages']} passage(s)"
        + (", the file and its parsed form." if removed["document"] else ".")
    )
    st.rerun()


page.render(view)
