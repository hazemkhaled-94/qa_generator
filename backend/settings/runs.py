"""What names one run, so two of them can be told apart.

`settings_version` names a CONFIGURATION. It is a digest of the stored
overrides, so two runs under one configuration carry the same value and are
indistinguishable in the data - which is exactly the position anybody
comparing runs is in. The A/B in `evaluation/README.md` changed a model and
a rule together, and the only reason the two runs could be separated
afterwards is that the first one's questions had been deleted first.

This is the other half. `settings_version` says what the run was configured
as; `run_id` says which run it was.

Per PROCESS rather than per drain, which narrows what a run means on
purpose. A `--watch` worker drains whenever something arrives, so an id per
drain would put thousands of them in a week of data and as many projects in
Phoenix. One worker lifetime is one run, and `make questions` in the
foreground - the case the comparison exists for - is one run either way.

Named `runs` and not `run` because `run.py` is a command line in every
package here, this one included.
"""

from __future__ import annotations

import os
from functools import lru_cache
from uuid import uuid4

#: Set this to name a run rather than have it named. The A/B in
#: evaluation/README.md would have been
#: `make questions-rerun TOPIC=439 RUN_ID=b-gemma4-12b` against
#: `... RUN_ID=a-gpt-4.1`, and the comparison one command rather than two
#: tables read off a terminal and typed into a file by hand.
#:
#: Unset is the useful default, and its absence means something the way
#: EVAL_RUN_NAME's does: a uuid is unambiguous, and nothing has to be
#: decided in advance about whether a run will turn out to be worth
#: comparing.
VARIABLE = "RUN_ID"


@lru_cache(maxsize=1)
def run_id() -> str:
    """Names this process's run, the same answer every time it is asked.

    Cached rather than threaded down through the factories, which is the
    difference between this and `settings_version`. A version can change
    while a process runs - that is what `reloading` in `stages/cli.py`
    watches for - so it has to be re-read and handed to whatever writes a
    row. A run cannot change: it IS the process.
    """
    named = (os.environ.get(VARIABLE) or "").strip()
    return named or uuid4().hex
