"""Loading the spaCy pipeline for a language."""

from __future__ import annotations

from functools import lru_cache

import spacy
from spacy.language import Language

from settings import csv, required


def _models() -> dict[str, str]:
    """Reads NLP_MODELS, a comma-separated list of `language:model`."""
    models = {}
    for pair in csv("NLP_MODELS"):
        language, _, model = pair.partition(":")
        if not model:
            raise ValueError(f"NLP_MODELS entry {pair!r} is not language:model")
        models[language.strip().lower()] = model.strip()
    return models


@lru_cache(maxsize=1)
def _configured() -> tuple[dict[str, str], str]:
    """Returns the configured models and the fallback language."""
    models = _models()
    default = required("NLP_DEFAULT_LANGUAGE").lower()
    if default not in models:
        raise ValueError(
            f"NLP_DEFAULT_LANGUAGE={default!r} is not one of the languages "
            f"NLP_MODELS names ({', '.join(sorted(models))})"
        )
    return models, default


@lru_cache(maxsize=8)
def _loaded(model: str) -> Language:
    """Loads one pipeline, once per process."""
    try:
        return spacy.load(model)
    except OSError as exc:
        raise RuntimeError(
            f"the spaCy model {model!r} is not installed. It is named in "
            f"NLP_MODELS and must be present in the image; nothing downloads "
            f"it at run time."
        ) from exc


def languages() -> tuple[str, ...]:
    """Every language this deployment has a pipeline for."""
    models, _ = _configured()
    return tuple(sorted(models))


def pipeline(language: str | None) -> Language:
    """Returns the pipeline for a language, falling back to the default."""
    models, default = _configured()
    return _loaded(models.get((language or "").strip().lower(), models[default]))


def name(language: str | None) -> str:
    """Returns the model identifier used for a language."""
    models, default = _configured()
    return models.get((language or "").strip().lower(), models[default])
