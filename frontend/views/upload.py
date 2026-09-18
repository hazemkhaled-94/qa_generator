"""Upload view. The ingestion service, and nothing else."""

from __future__ import annotations

import requests
import streamlit as st

from lib import backend, configure, page
from lib.backend.upload import UploadApi


def view() -> None:
    """Renders the upload page, submitting every chosen file at once."""
    page.header("Upload")

    client = backend.upload_api()
    held = client.counts()

    with page.panel("Overview"):
        page.stats(
            {
                "Documents": (
                    f"{held.get('documents', 0):,}",
                    "PDFs held in the object store.",
                ),
                "Uploads": (
                    f"{held.get('upload_attempts', 0):,}",
                    "Uploads recorded, accepted and refused alike.",
                ),
            }
        )

    with page.panel("Ingestion"):
        with st.form("upload", border=False):
            files = st.file_uploader(
                "PDFs",
                type=["pdf"],
                accept_multiple_files=True,
                help="Stored under the digest of its own bytes. A file the "
                "corpus already holds is recorded and stores nothing.",
            )
            submit, *_ = st.columns(4)
            submitted = submit.form_submit_button(
                "Upload all", type="primary", width="stretch"
            )
        # Outside the upload form and inside the panel: a form cannot hold
        # another, and the configuration belongs beside the service it
        # configures.
        configure.panel("ingestion")

    if not submitted:
        return
    if not files:
        st.warning("Select at least one PDF.")
        return

    with st.spinner(f"Uploading {len(files)} file(s)…"):
        for uploaded in files:
            _report(uploaded.name, client, uploaded.getvalue())


def _report(filename: str, client: UploadApi, data: bytes) -> None:
    """Uploads one file and shows what became of it."""
    try:
        outcome = client.add_document(filename, data)["outcome"]
    except requests.exceptions.RequestException as exc:
        # The reason for a 413 or 415 is in the response body.
        body = getattr(exc.response, "text", None)
        st.error(f"{filename}: {body or exc}")
        return

    if outcome == "stored":
        st.success(f"{filename}: stored.")
    else:
        st.info(f"{filename}: {outcome.replace('_', ' ')} — already held.")


page.render(view)
