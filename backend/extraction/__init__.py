"""Extraction: passages become atomic facts citing a sentence of their source.

Claims passages, skips the ones that assert nothing, sends the rest to the
extractor their block type calls for - a cell grid to a deterministic
reader, everything else to a local model - and checks every proposed fact
with spaCy. A fact that fails a check is stored with the reason rather than
dropped.

Reads nothing but the database, and has no HTTP surface of its own.

Nothing is re-exported here: a caller naming a submodule pays for that
submodule alone, which is what keeps litellm out of the API.
"""
