"""Each stage's settings, read out of the environment.

One `load` per stage, each a trust boundary: the values arrive as strings
from a file nobody type-checks, and a stage that starts with the wrong one
fails a corpus rather than a request.

The values here are the ones in configs/env/backend.env, which
tests/conftest.py has already put in the environment.
"""

from __future__ import annotations

import pytest


def test_ingestion_reads_its_limit_and_its_allowlist() -> None:
    """What an upload is measured and typed against."""
    from ingestion.config import Settings

    loaded = Settings.load()

    assert loaded.pipeline_version
    assert loaded.max_file_size_mb == 100
    assert loaded.allowed_mime_types == ("application/pdf",)


def test_the_upload_limit_is_offered_in_bytes() -> None:
    """Derived, so megabytes and bytes cannot disagree."""
    from ingestion.config import Settings

    loaded = Settings.load()

    assert loaded.max_file_size_bytes == loaded.max_file_size_mb * 1024 * 1024


def test_an_allowlist_naming_an_undetectable_type_is_refused(monkeypatch) -> None:
    """It would read as support that does not exist."""
    from ingestion.repository import DocumentRepository
    from ingestion.service import IngestService

    with pytest.raises(ValueError, match="this build cannot detect"):
        IngestService(
            repository=DocumentRepository(),
            store=None,
            max_file_size_bytes=1,
            allowed_media_types=("application/pdf", "image/png"),
            pipeline_version="0.0.1",
        )


def test_chunking_reads_the_model_it_sizes_passages_against() -> None:
    """One name for the tokenizer and the embedder, so they cannot differ."""
    from preprocessing.chunking.config import Settings

    loaded = Settings.load()

    assert loaded.embedding_model
    assert loaded.max_tokens == 512
    assert loaded.merge_peers is True


def test_parsing_reads_its_thresholds_and_its_timeout() -> None:
    """What decides an OCR fallback and what stops a runaway conversion."""
    from preprocessing.parsing.config import Settings

    loaded = Settings.load()

    assert loaded.ocr_char_threshold == 100
    assert loaded.min_confidence == 0.5
    assert loaded.table_mode == "accurate"
    assert loaded.heading_hierarchy is True
    assert loaded.document_timeout_seconds == 1800


def test_parsing_treats_an_unset_artifacts_path_as_absent(monkeypatch) -> None:
    """Absence means docling fetches its own models."""
    from preprocessing.parsing.config import Settings

    monkeypatch.delenv("DOCLING_ARTIFACTS_PATH", raising=False)

    assert Settings.load().artifacts_path is None


def test_topic_modelling_reads_the_fitter_and_the_language_names() -> None:
    """The names are what a report is read by."""
    from topic_modelling.config import Settings

    loaded = Settings.load()

    assert loaded.languages == {"de": "German", "en": "English"}
    assert loaded.num_topics == 40
    assert loaded.passages_per_topic == 40
    assert loaded.passes == 10
    assert loaded.random_state == 42
    assert loaded.top_terms == 12
    assert loaded.min_weight == 0.05
    assert loaded.no_below == 3
    assert loaded.no_above == 0.5


def test_topic_modelling_names_no_model_when_none_is_served(monkeypatch) -> None:
    """Topics are still fitted; they are just left unnamed."""
    from topic_modelling.config import Settings

    monkeypatch.delenv("LLM_MODEL", raising=False)

    assert Settings.load().model is None


def test_topic_modelling_reads_the_model_when_one_is_served(monkeypatch) -> None:
    """The same settings extraction uses, so the two cannot disagree."""
    from topic_modelling.config import Settings

    monkeypatch.setenv("LLM_MODEL", "ollama/qwen3")
    # The fallback is what is under test, so the override has to be absent
    # whatever the shipped configuration happens to name.
    monkeypatch.delenv("TOPIC_MODEL", raising=False)

    loaded = Settings.load()

    assert loaded.model is not None
    assert loaded.model.model == "ollama/qwen3"


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("ingestion.config", "MAX_FILE_SIZE_MB"),
        ("preprocessing.chunking.config", "EMBEDDING_MAX_TOKENS"),
        ("preprocessing.parsing.config", "PARSING_MIN_CONFIDENCE"),
        ("topic_modelling.config", "TOPIC_NUM_TOPICS"),
    ],
)
def test_a_stage_will_not_start_without_its_settings(monkeypatch, module, name) -> None:
    """No defaults in code: the process stops naming the variable."""
    import importlib

    monkeypatch.delenv(name, raising=False)
    settings = importlib.import_module(module).Settings

    with pytest.raises(KeyError, match=name):
        settings.load()


@pytest.mark.parametrize(
    ("module", "name", "value"),
    [
        ("ingestion.config", "MAX_FILE_SIZE_MB", "a hundred"),
        ("preprocessing.parsing.config", "PARSING_MIN_CONFIDENCE", "half"),
        ("topic_modelling.config", "TOPIC_NUM_TOPICS", "twelve"),
    ],
)
def test_a_setting_of_the_wrong_shape_names_itself(
    monkeypatch, module, name, value
) -> None:
    """The message says which variable to go and fix."""
    import importlib

    monkeypatch.setenv(name, value)
    settings = importlib.import_module(module).Settings

    with pytest.raises(ValueError, match=name):
        settings.load()


def test_a_fact_kind_this_stage_cannot_use_is_refused_at_start_up() -> None:
    """Rather than selecting nothing and writing no questions at all.

    All four of the kinds EXTRACTION_KINDS produces are askable now - an
    outline's lines are indented under their own number, so it no longer
    breaks the numbering the writer cites by - so what is left to refuse is
    a name that is not a fact kind. Read as a filter it would match no row,
    and a topic with no facts is not a failure: the stage would report a
    clean run over an empty corpus.
    """
    import os

    from question_generation.config import Settings

    previous = os.environ.get("QUESTIONS_FACT_KINDS")
    os.environ["QUESTIONS_FACT_KINDS"] = "atomic,paraphrase"
    try:
        with pytest.raises(ValueError, match="QUESTIONS_FACT_KINDS"):
            Settings.load()
    finally:
        if previous is None:
            os.environ.pop("QUESTIONS_FACT_KINDS", None)
        else:
            os.environ["QUESTIONS_FACT_KINDS"] = previous
