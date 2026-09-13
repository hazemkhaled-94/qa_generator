"""Documents view."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from lib import backend, catalog, page, stage

#: What the polling fragment hands a page, and what the page returns for it.
Figures = Callable[[dict[str, dict[str, int]]], dict[str, tuple[object, str]]]

#: The stages a document passes through, in order. Parsing and chunking queue
#: over the document itself; extraction queues over its passages, which is why
#: its unit differs.
_PARSING = stage.Queue("parsing", "Parsing", "documents", "parsed")
_CHUNKING = stage.Queue("chunking", "Chunking", "documents", "chunked")
_EXTRACTION = stage.Queue("extraction", "Extraction", "passages", "extracted")
_STAGES = (_PARSING, _CHUNKING, _EXTRACTION)

#: The two stages that record their state on the document row itself, with
#: the status and error column each reports under.
_ON_ROW = (
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
        "the pipeline. Pick a document from the table to see its own figures "
        "and to run, stop or delete that document alone.",
    )

    client = backend.catalog_api()
    page.section(
        "Pipeline across the whole corpus",
        "Every document and passage the corpus holds, not only the ones "
        "matching the search below. Parsing and chunking count documents; "
        "extraction counts passages, because that is the unit it reads. "
        "These figures refresh on their own every few seconds.",
    )
    stage.overview(client, list(_STAGES), _corpus_figures)

    st.divider()
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

    st.caption(
        f"{total:,} document(s) match the search; {len(rows):,} shown on this page."
    )
    st.dataframe(
        [
            {
                "Document": catalog.label_for(document),
                "Title": document["title"] or "—",
                "Pages": document["page_count"] or "—",
                "Language": (document["language"] or "—").upper(),
                "Parsing": document["parse_status"],
                "Chunking": document["chunk_status"],
                "Passages read": f"{document['extracted_passages']}"
                f"/{document['total_passages']}",
                "Uploaded": (document["first_seen"] or "")[:16].replace("T", " "),
            }
            for document in rows
        ],
        width="stretch",
        hide_index=True,
    )

    st.divider()
    chosen = st.selectbox(
        "Document to work on",
        rows,
        format_func=catalog.label_for,
        help="Everything below this point - the figures, the stage controls "
        "and the deletions - applies to this document alone and to nothing "
        "else in the corpus.",
    )
    _detail(client, chosen)


def _corpus_figures(
    counts: dict[str, dict[str, int]],
) -> dict[str, tuple[object, str]]:
    """Names the corpus-wide figures the top of this page carries."""
    documents = sum(counts["parsing"].values())
    passages = sum(counts["extraction"].values())
    return {
        "Documents stored": (
            f"{documents:,}",
            (
                "PDFs held in the object store, whatever stage each has reached. "
                "Uploading adds one; deleting a document removes one."
            ),
        ),
        "Parsed": (
            *page.portion(counts["parsing"].get("parsed", 0), documents),
            (
                "Documents converted from the uploaded PDF into a structured "
                "document, with a title, a language and a page layout. The share "
                "is of every document stored."
            ),
        ),
        "Chunked": (
            *page.portion(counts["chunking"].get("chunked", 0), documents),
            (
                "Documents split into passages, each passage carrying its "
                "sentences and lemmas. The share is of every document stored."
            ),
        ),
        "Passages": (
            f"{passages:,}",
            (
                "Passages chunking has produced across every document. This is "
                "the unit extraction queues over, so it is the size of the "
                "extraction queue when everything is asked for."
            ),
        ),
        "Passages read": (
            *page.portion(counts["extraction"].get("extracted", 0), passages),
            (
                "Passages extraction has read for facts. The share is of every "
                "passage in the corpus. A passage counts as read whether or not "
                "the facts it yielded passed their checks."
            ),
        ),
    }


@st.cache_data(show_spinner=False, max_entries=4)
def _file(_client, sha256: str) -> bytes:
    """Fetches one document, remembering the last few.

    Cached because clicking the download button reruns the script, which would
    otherwise pull the file again to redraw the viewer beside it. `_client` is
    underscored so Streamlit leaves it out of the cache key.
    """
    return _client.file(sha256)


def _detail(client, document: dict) -> None:
    """Shows one document's own figures, its controls and the file itself."""
    sha = document["sha256"]
    name = catalog.label_for(document)
    scope = ("document", sha)

    page.section(
        f"Figures for {name}",
        "Everything below counts this document and the passages chunking "
        "drew from it, and nothing else in the corpus. The bars are the same "
        "three stages as above, narrowed to this one document.",
    )
    stage.overview(client, list(_STAGES), _document_figures(document), scope=scope)

    for label, field, error_field in _ON_ROW:
        # Only while the stage is failed: an error left over from a run that
        # has since been reset is not this document's state.
        if document[field] == "failed" and document.get(error_field):
            st.error(f"{label} failed for this document: {document[error_field]}")

    page.section(
        "Run this document",
        "Each row queues, withdraws or repeats one stage for this document "
        "alone. None of these buttons does the work: they move rows between "
        "statuses, and the stage's worker picks up whatever has become "
        "claimable on its next poll. Nothing here touches any other document. "
        + stage.COLOUR_KEY,
    )
    for queue in _STAGES:
        stage.controls(client, queue, scope)

    st.caption(f"sha256 {sha}")

    # A toggle, not an expander: an expander runs its body whichever way it
    # is folded, so the fetch would happen on every rerun regardless.
    if st.toggle(
        "View the file",
        help="Fetches the PDF as it was uploaded and shows it below. It is "
        "not fetched until this is on.",
    ):
        # A document takes long enough to arrive that without the spinner the
        # page looks broken rather than busy.
        with st.spinner(f"Loading {name}…"):
            data = _file(client, sha)

        st.download_button(
            "Download", data=data, file_name=name, mime="application/pdf"
        )
        st.pdf(data, height=700)

    st.divider()
    _removal(client, document)


def _document_figures(document: dict) -> Figures:
    """Builds the figures for one document, closing over its own row.

    A closure because the page's polling passes in the queue counts and the
    rest - pages, language, oversized passages - comes from the row already
    fetched.
    """

    def figures(counts: dict[str, dict[str, int]]) -> dict[str, tuple[object, str]]:
        """Names the per-document figures."""
        passages = sum(counts["extraction"].values())
        oversized = document.get("oversized") or 0
        return {
            "Pages": (
                document["page_count"] or "—",
                "Pages in the uploaded PDF, as ingestion counted them.",
            ),
            "Language": (
                (document["language"] or "—").upper(),
                (
                    "The language parsing detected for this document. It decides "
                    "which spaCy pipeline reads it and which topic model covers "
                    "it; a document with none belongs to no topic model."
                ),
            ),
            "Passages": (
                f"{passages:,}",
                (
                    "Passages chunking drew from this document. Re-chunking "
                    "replaces every one of them, which deletes the facts and "
                    "topic memberships they held."
                ),
            ),
            "Passages read": (
                *page.portion(counts["extraction"].get("extracted", 0), passages),
                (
                    "Passages of this document extraction has read for facts, as "
                    "a share of its passages."
                ),
            ),
            "Oversized passages": (
                f"{oversized:,}",
                (
                    "Passages longer than EMBEDDING_MAX_TOKENS. The chunker "
                    "splits on that budget, so any above it means the split did "
                    "not happen and those passages will be truncated when they "
                    "are embedded. Zero is what this should read."
                ),
            ),
        }

    return figures


def _removal(client, document: dict) -> None:
    """Offers the two deletions, each behind its own confirmation.

    The first click only arms the choice; a second, separately labelled one
    carries it out. Neither can be undone.
    """
    name = catalog.label_for(document)
    sha = document["sha256"]
    armed = st.session_state.get("documents-armed")

    st.html(
        "<div class='qa-danger-zone'>"
        "<div class='qa-danger-title'>Delete</div>"
        "<div class='qa-danger-detail'>Actions are done on this document "
        "only, and can not be undone!</div></div>"
    )

    derived, whole, *_ = st.columns(stage.DANGER)
    if derived.button(
        "Delete passages and facts",
        key="danger-arm-derived",
        width="stretch",
        help="Deletes every passage of this document and every fact drawn "
        "from those passages, along with the topic memberships they held. "
        "The document, its uploaded file and its parsed form all stay, and "
        "chunking returns to `new` so the passages can be rebuilt.",
    ):
        st.session_state["documents-armed"] = ("derived", sha)
        st.rerun()
    if whole.button(
        "Delete document",
        key="danger-arm-whole",
        width="stretch",
        help="Deletes the document row, the uploaded PDF, its parsed form "
        "and everything the pipeline drew from it. Only the upload record in "
        "ingest_events survives.",
    ):
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

    confirm, cancel, *_ = st.columns(stage.DANGER)
    if confirm.button(label, key="danger-confirm-delete", width="stretch"):
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


page.render(view)
