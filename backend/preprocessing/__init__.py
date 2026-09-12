"""The stages between ingestion and fact extraction.

    parsing    a stored PDF becomes a structured document
    chunking   that document becomes passages

Each is a package with the same shape: config, models, repository, service,
factory, run. Neither imports the other; they meet at the parsed bucket and
hand work over through their own status column on the document.
"""
