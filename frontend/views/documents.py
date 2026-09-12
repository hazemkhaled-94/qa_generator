"""Documents view."""

from __future__ import annotations

import streamlit as st

from lib import backend, catalog, page, stage

#: The stages a document passes through, in order, with the status and error
#: columns each one reports under.
_STAGES = (
    ("Parsing", "parse_status", "parse_error"),
    ("Chunking", "chunk_status", "chunk_error"),
)

#: The columns the search box can look in. One entry, because the backend
#: searches filename, title and digest together: a person looking for a
#: document does not know which of the three they remember.
_FIELDS = {"Document, title or digest": "any"}


def view() -> None:
    """Renders the documents page."""
    page.header(
        "Documents",
        "Every PDF the corpus holds, and how far each one has got through "
        "the pipeline.",
    )
    stage.panel("parsing", "documents", "parsed")

    client = backend.catalog_api()
    bar = catalog.filters("documents", fields=_FIELDS)

    total, rows = catalog.paged(
        "documents",
        bar.page,
        lambda limit, offset: client.documents(
            q=bar.search, limit=limit, offset=offset
        ),
        "documents",
    )
    if not total:
        st.info(
            "No documents match that search."
            if bar.search
            else "No documents yet. Upload a PDF to begin."
        )
        return

    page.metrics(
        {
            "Documents matching": f"{total:,}",
            "Parsed here": f"{sum(1 for d in rows if d['parse_status'] == 'parsed'):,}",
            "Chunked here": f"{sum(1 for d in rows if d['chunk_status'] == 'chunked'):,}",
            "Passages here": f"{sum(d['total_passages'] for d in rows):,}",
        }
    )

    st.dataframe(
        [
            {
                "Document": catalog.label_for(document),
                "Title": document["title"] or "—",
                "Pages": document["page_count"] or "—",
                "Lang": (document["language"] or "—").upper(),
                "Parsed": document["parse_status"],
                "Chunked": document["chunk_status"],
                "Extracted": f"{document['extracted_passages']}/"
                f"{document['total_passages']}",
                "Uploaded": (document["first_seen"] or "")[:16].replace("T", " "),
            }
            for document in rows
        ],
        width="stretch",
        hide_index=True,
    )

    st.divider()
    chosen = st.selectbox("Inspect", rows, format_func=catalog.label_for)
    _detail(client, chosen)


@st.cache_data(show_spinner=False, max_entries=4)
def _file(_client, sha256: str) -> bytes:
    """Fetches one document, remembering the last few.

    Cached because clicking the download button reruns the script, which would
    otherwise pull the file again to redraw the viewer beside it. `_client` is
    underscored so Streamlit leaves it out of the cache key.
    """
    return _client.file(sha256)


def _removal(client, document: dict) -> None:
    """Offers the two deletions, each behind its own confirmation.

    The first click only arms the choice; a second, separately labelled one
    carries it out. Neither can be undone.
    """
    name = catalog.label_for(document)
    sha = document["sha256"]
    armed = st.session_state.get("documents-armed")

    derived, whole, *_ = st.columns(stage.CONTROLS)
    if derived.button("Delete passages and facts", key="arm-derived", width="stretch"):
        st.session_state["documents-armed"] = ("derived", sha)
        st.rerun()
    if whole.button("Delete document", key="arm-whole", width="stretch"):
        st.session_state["documents-armed"] = ("whole", sha)
        st.rerun()

    if not armed or armed[1] != sha:
        return

    kind = armed[0]
    if kind == "derived":
        st.warning(
            f"Delete every passage and fact drawn from **{name}**? The document "
            "and its file stay, and chunking goes back to not started."
        )
        label = "Yes, delete the derived data"
    else:
        st.error(
            f"Delete **{name}** entirely? This removes the uploaded PDF from the "
            "object store, its converted form, and every passage and fact drawn "
            "from it. The upload record is kept. This cannot be undone."
        )
        label = "Yes, delete it permanently"

    confirm, cancel, *_ = st.columns(stage.CONTROLS)
    if confirm.button(label, key="confirm-delete", type="primary", width="stretch"):
        removed = client.delete(sha, derived_only=kind == "derived")
        del st.session_state["documents-armed"]
        st.success(
            f"Removed {removed['passages']} passage(s)"
            + (", the file and its parsed form." if removed["document"] else ".")
        )
        st.rerun()
    if cancel.button("Cancel", key="cancel-delete", width="stretch"):
        del st.session_state["documents-armed"]
        st.rerun()


def _detail(client, document: dict) -> None:
    """Shows one document's stage results and the file itself."""
    columns = st.columns(len(_STAGES))
    for column, (label, field, _) in zip(columns, _STAGES, strict=True):
        column.html(
            f"<div class='qa-muted'>{label}</div>"
            f"<div style='margin:.3rem 0'>{catalog.stage_pill(document[field])}</div>"
        )

    for label, field, error_field in _STAGES:
        # Only while the stage is failed: an error left over from a run that
        # has since been reset is not this document's state.
        if document[field] == "failed" and document.get(error_field):
            st.error(f"{label}: {document[error_field]}")

    oversized = document.get("oversized") or 0
    if oversized:
        st.warning(
            f"{oversized:,} passage(s) of this document are above "
            "EMBEDDING_MAX_TOKENS. The chunker splits on that budget, so this "
            "means the split did not happen; those passages will be truncated "
            "when they are embedded."
        )

    st.caption(f"sha256 {document['sha256']}")

    # A toggle, not an expander: an expander runs its body whichever way it
    # is folded, so the fetch would happen on every rerun regardless.
    if st.toggle("View the file"):
        # A document takes long enough to arrive that without the spinner the
        # page looks broken rather than busy.
        with st.spinner(f"Loading {catalog.label_for(document)}…"):
            data = _file(client, document["sha256"])

        st.download_button(
            "Download",
            data=data,
            file_name=catalog.label_for(document),
            mime="application/pdf",
        )
        st.pdf(data, height=700)

    st.divider()
    _removal(client, document)


page.render(view)
