"""Topic modelling: each language becomes topics over its own vocabulary.

Claims a requested fit, reads the lemmas chunking stored for every passage,
factorises a tf-idf weighted term matrix with gensim, and replaces every row
in `topics` and `passage_topics`. One fit covers every language and is always
a full refit.

Nothing is re-exported: a caller names the submodule it wants.

README.md in this package describes the steps, the settings and the
reasoning behind them, and tests/static/test_topics_pinned.py pins the
surface this service presents.
"""
