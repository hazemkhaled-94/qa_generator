"""What the deletions left behind, and the second deletion that empties it.

Nothing here archives anything. The rows are copied in by an AFTER DELETE
trigger on every table and the objects are moved in by the removal paths;
this reads what is held and purges it.
"""

from archive.store import Archive, Held

__all__ = ["Archive", "Held"]
