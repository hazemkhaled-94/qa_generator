"""Ingestion: an uploaded file becomes a stored document.

Detects the media type from the leading bytes, refuses what it cannot read,
stores the accepted bytes and records every attempt. Removal is a service of
its own.

Nothing is re-exported: a caller names the submodule it wants.
"""
