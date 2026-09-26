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

from lib import backend, catalog, configure, explain, lineage, page, pipeline, stage

#: The one stage this page runs. Chunking is the Passages page's and
#: extraction is the Facts page's; neither can be reached from here.
_PARSING = stage.Queue("parsing", "Parsing", "documents", "parsed")

#: What refuses a document, by the failure the row carries. Not gates in
#: the sense the later stages have them - nothing here judges quality - but
#: they are what a `failed` row means, and the page lists them for the same
#: reason: a refusal with no name cannot be looked up.
_GATES = (
    explain.Gate(
        "unsupported type",
        "dispatch",
        "The format is one a pipeline here handles.",
        "nothing",
        "rule",
    ),
    explain.Gate(
        "scanned document",
        "ocr check",
        "The document carries text rather than being images of it.",
        "nothing",
        "measurement",
    ),
    explain.Gate(
        "conversion failed",
        "convert",
        "The converter read it without raising, inside its timeout.",
        "the conversion",
        "rule",
    ),
    explain.Gate(
        "empty document",
        "read values",
        "The conversion produced body text to read a title and language off.",
        "nothing",
        "rule",
    ),
)

#: The order this service works in, and what runs each step.
_STEPS = (
    explain.Step(
        "Dispatch on the media type",
        "The pipeline that owns the format ingestion detected from the "
        "leading bytes. A type nothing handles fails here, naming the types "
        "that are handled.",
        ("pipelines/__init__.py",),
        (),
    ),
    explain.Step(
        "Decide whether it is scanned",
        "Characters per page against a floor, read from counts that already "
        "exist. OCR is not implemented, so a scanned document is refused "
        "here, where the reason is still known.",
        ("service.py",),
        ("PARSING_OCR_CHAR_THRESHOLD",),
    ),
    explain.Step(
        "Convert",
        "Docling, with heading hierarchy and table structure on and OCR off. "
        "This is the expensive step — layout and table-structure models over "
        "every page — which is why the stage runs in a worker off a queue.",
        ("pipelines/pdf.py",),
        ("PARSING_TIMEOUT_SECONDS",),
    ),
    explain.Step(
        "Repair the text",
        "Every string is rewritten before anything reads it: composed to "
        "NFC, words the page broke across two lines rejoined, control and "
        "zero-width characters dropped, every other kind of space folded. A "
        "character a model cannot reproduce is one no quote of it can match.",
        ("pipelines/pdf.py",),
        (),
    ),
    explain.Step(
        "Store the converted document",
        "One JSON object in the `parsed` bucket, keyed by the source digest, "
        "written BEFORE anything reads values out of it — so a bug in the "
        "reading does not mean paying for the conversion twice. This is what "
        "chunking reads, so re-chunking never re-parses.",
        ("service.py",),
        (),
    ),
    explain.Step(
        "Read the values the row holds",
        "The body rendered as text — table values included, running headers "
        "and footers excluded — then the title, the language, and a hash of "
        "the body with whitespace collapsed and case folded.",
        ("analysis.py",),
        (),
    ),
)

#: Everything this page can say about the service it runs.
_SERVICE = explain.Service(
    what=(
        "Turns the uploaded file into a structured document: headings in a "
        "hierarchy, tables as cell grids, text in reading order. Nothing it "
        "does starts the next stage — a re-parse leaves the old passages "
        "standing until chunking is asked to run."
    ),
    steps=_STEPS,
    gates=_GATES,
)

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
        explain.panel(_SERVICE, "How parsing works")
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
