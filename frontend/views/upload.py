"""Upload view."""

from __future__ import annotations

import requests
import streamlit as st

from lib import backend, page
from lib.backend.upload import UploadApi


def view() -> None:
    """Renders the upload page, submitting each chosen file in turn."""
    page.header(
        "Upload",
        "Add PDFs to the corpus. Nothing runs on its own: a stored file waits "
        "at `new` until somebody presses Start on the Documents page.",
    )

    client = backend.upload_api()

    with st.form("upload", border=False):
        files = st.file_uploader(
            "Documents",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        submit, _ = st.columns([1, 3])
        submitted = submit.form_submit_button("Submit", type="primary", width="stretch")

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
