"""Every prompt this stage sends, for the record that resolves a version.

One prompt, and until now it had neither a version nor a row: the labeller
sends a real call to a real model, and what it asked for existed only in the
source at whatever commit was checked out. A topic named under one wording
and a topic named under the next were the same column with nothing to tell
them apart.

Read by `run.py` where it checks the model answers. Nothing here calls one.
"""

from __future__ import annotations

from stages.prompts import Composed
from topic_modelling import labels

#: What the stage records its prompts under.
SERVICE = "topics"


def catalogue() -> list[Composed]:
    """Every prompt this stage can send, composed as it would be sent.

    Both halves and the shape: the system prompt, the user message as its
    template, and the JSON schema the answer has to come back in.
    """
    return [
        Composed(
            "label",
            labels.PROMPT_VERSION,
            labels._SYSTEM,
            labels._USER,
            labels._Label.model_json_schema(),
        )
    ]
