"""How long a pair one encoder may be given.

`ENCODER_MAX_TOKENS` is one setting and the encoders it configures do not
agree about what is possible: `bge-m3-zeroshot-v2.0` reads 8,192 tokens and
`xlm-roberta-large-squad2` reads 512. A deployment raising the setting for
the first would hand the second a length its position embeddings do not
have, which is an error at the first call rather than a slow degradation -
and a deployment leaving it at 512 for the second quietly gives up the
window that was the reason to choose the first.

So the setting is a CEILING a deployment asks for, and each model is held to
its own. Nothing here has to know which model is which.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

#: Above this, a declared length is a sentinel rather than a window.
#: A tokenizer reports `model_max_length` as a number near 1e30 when the
#: checkpoint says nothing, and truncating to 1e30 truncates nothing.
_ABSURD = 1_000_000


def window(tokenizer, wanted: int) -> int:
    """The longest pair this checkpoint can be given, at or under `wanted`.

    Read off the tokenizer rather than off `max_position_embeddings`,
    because the two disagree by the positions a model reserves and the
    tokenizer is the one that already accounts for them: `xlm-roberta`
    declares 514 embeddings and its tokenizer says 512, which is the number
    that matters here.

    Args:
        tokenizer: The loaded tokenizer.
        wanted: What the deployment asked for.

    Returns:
        `wanted`, or the model's own maximum where that is smaller.
    """
    declared = getattr(tokenizer, "model_max_length", None)
    if not isinstance(declared, int) or declared <= 0 or declared > _ABSURD:
        return wanted
    if declared < wanted:
        log.info(
            "%s reads %d tokens, so the %d asked for is capped at it",
            getattr(tokenizer, "name_or_path", "the model"),
            declared,
            wanted,
        )
    return min(wanted, declared)
