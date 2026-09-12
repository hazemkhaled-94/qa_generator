"""Parsing: a stored document becomes a structured one.

Claims documents, dispatches on the media type recorded at ingest, converts
with the pipeline for that format, writes the converted document to the
parsed bucket, then reads out of it the title, language, content digest and
confidence the database holds.

Nothing is re-exported here: a caller naming a submodule pays for that
submodule alone.
"""
