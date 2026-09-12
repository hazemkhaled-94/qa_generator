"""Topic modelling: the corpus becomes topics over its own vocabulary.

Claims a requested fit, reads the lemmas chunking stored for every passage,
factorises a tf-idf weighted term matrix with gensim, and replaces every
row in `topics` and
`passage_topics`.

A fit is always a full refit, and always every language: the factorisation
estimates each topic relative to all the others over one vocabulary, so
adding a document moves every topic at once. Discovery is therefore asked
for rather than triggered.

Nothing is re-exported here: a caller naming a submodule pays for that
submodule alone, which is what keeps gensim out of the API.
"""
