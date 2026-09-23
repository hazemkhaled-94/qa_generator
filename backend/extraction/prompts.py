"""Every prompt this stage sends, for the record that resolves a version.

`facts.prompt_version` names a version; this is what says what that version
asked for. Three extractors, three versions, bumped separately: what counts
as a fact, what a digest condenses, and what a bridge may claim are three
decisions.

Read by `run.py` at start-up and by the api. Nothing here calls a model.
"""

from __future__ import annotations

from extraction.extractors import bridge, digest, llm
from stages.prompts import Composed

#: What the stage records its prompts under.
SERVICE = "extraction"


def catalogue(cap: int | None = None) -> list[Composed]:
    """Every prompt this stage can send, composed as it would be sent.

    `cap` is what `EXTRACTION_MIN_OTHER_SHARE` works out to, as the
    factory resolved it, because the atomic extractor appends the cap
    instruction to its own prompt and a record of the uncapped text would
    be a record of something the model was never given. Absent means the
    same thing it means there: no cap, and the paragraph is left off
    entirely rather than written as a limitless one.
    """
    return [
        Composed("atomic", llm.PROMPT_VERSION, llm.composed(cap)),
        Composed("digest", digest.PROMPT_VERSION, digest._SYSTEM),
        Composed("bridge", bridge.PROMPT_VERSION, bridge._SYSTEM),
    ]
