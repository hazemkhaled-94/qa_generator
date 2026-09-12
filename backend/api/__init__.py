"""The backend's HTTP surface.

Everything the frontend is allowed to know. It has one base URL and no idea
what runs behind it: not the database, not the object store, not how a stage
does its work. A stage's queue routes are built from one shared factory, so
adding a stage is a repository here and a line in `routes/__init__`.
"""
