"""Ingestion: an uploaded file becomes a stored document.

Decides what a file is from its leading bytes, refuses what it cannot parse,
stores the accepted bytes and records every attempt. Removing a document is
here too, in a service of its own.

No HTTP surface of its own. Nothing is re-exported here: a caller naming a
submodule pays for that submodule alone.
"""
