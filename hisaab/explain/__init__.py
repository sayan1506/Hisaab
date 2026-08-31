"""The LLM layer: plain-language explanations for rows the matcher refused to resolve.

**This package is the only place in ``hisaab/`` allowed to reach a network**, and it is
allowed by exactly one clause of ``check_isolation.py`` check 8. Everything that clause
does *not* exempt still applies here, which is the interesting part:

  * it cannot read ``truth.json`` (check 1),
  * it cannot import ``hisaab.generator`` (check 6),
  * it cannot read the declared fee columns (check 7),
  * it cannot use ``subprocess``, ``importlib`` or ``ctypes`` (check 8's "reach" half),
  * and **nothing that ships may import it** (check 8b) -- it is a leaf, reached as
    ``python -m hisaab.explain``.

So the one component that talks to a model is the one component with no privileged
information at all. It explains rows from the same inputs a human reconciler would have.

**The layout exists to keep the network in one file.** ``client.py`` is the only module
that constructs an SDK client or sends a request; ``prompt.py``, ``schema.py`` and
``verify.py`` are pure functions over data. That is what lets gate 17 run the whole
pipeline against a recorded fixture with nothing installed and no network access -- the
untestable part is one seam, not the feature.
"""

from __future__ import annotations

#: Bumped with the phase, matching ``pyproject.toml``.
__all__ = ["EXPLAIN_SCHEMA_VERSION"]

#: The version of the explanation artifact this package writes. Phase 11 renders that
#: artifact rather than importing this package (check 8b forbids the import), so the file
#: is an interface between phases and gets a version like every other one here.
EXPLAIN_SCHEMA_VERSION = 1
