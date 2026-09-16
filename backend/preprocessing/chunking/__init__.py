"""Chunking: a structured document becomes retrievable passages.

Claims documents, reads the converted document back out of the parsed
bucket, cuts it into passages carrying their heading trail, pages, block
type, table cells, sentences and lemmas, and replaces whatever passages the
document had before.

Nothing is re-exported: a caller names the submodule it wants, which is what
keeps Docling out of the API process.
"""
