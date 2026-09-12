"""Sentence splitting, claim analysis, vocabulary and language detection.

Chunking writes a passage's language, sentences and lemmas; extraction reads
the sentences and analyses the statements a model writes; topic modelling
reads the lemmas. Loading a pipeline is this package's alone, as building an
engine is database's.

Nothing is re-exported here: `nlp.models` is plain values and `nlp.analysis`
loads spaCy, and a caller that needs only the first must not pay for the
second.
"""
