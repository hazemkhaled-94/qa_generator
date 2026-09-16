"""Extraction: passages become facts that cite the text supporting them.

Claims passages, skips the ones that assert nothing, and reads each of the
rest four ways: the claims it carries one at a time, a summary of the whole
of it, an outline of its points, and - in a pass of its own over the topics -
the claims that need more than one passage. Every proposal is checked with
spaCy. One that fails a check is stored with the reason rather than dropped.

Reads nothing but the database, and has no HTTP surface of its own.

See README.md for what each kind is for, how it is checked, and what it
costs. Nothing is re-exported here: a caller naming a submodule pays for that
submodule alone, which is what keeps litellm out of the API.
"""
