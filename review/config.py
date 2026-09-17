"""What the review tool reads from the environment.

Runs on the host, like `make schema` and the bare stage targets, so it
reads the same two files the Makefile sources: configs/env/review.env for
how a sample is drawn, and .env for where Argilla is and how to sign in.
"""

from __future__ import annotations

from dataclasses import dataclass

from settings import integer, optional, required


@dataclass(frozen=True)
class Settings:
    """Where Argilla is, and how much to put in front of a person."""

    api_url: str
    api_key: str
    #: The Argilla workspace the datasets live in. One per deployment, so
    #: two people reviewing two corpora do not annotate each other's rows.
    workspace: str
    #: How many records one push puts in a dataset. A review is a sample -
    #: nobody is reading twenty thousand facts - and the sample is
    #: stratified over the verdicts, so this is split across them.
    sample: int
    #: What to call the reviewer in the row's provenance. Optional: a
    #: deployment with one annotator does not need to say who each time.
    reviewer: str | None

    @classmethod
    def load(cls) -> Settings:
        """Reads settings from the environment."""
        return cls(
            api_url=required("ARGILLA_API_URL"),
            api_key=required("ARGILLA_API_KEY"),
            workspace=required("ARGILLA_WORKSPACE"),
            sample=integer("REVIEW_SAMPLE_SIZE"),
            reviewer=optional("REVIEW_REVIEWER"),
        )
