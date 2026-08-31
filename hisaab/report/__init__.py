"""Report — renders a run's four (or five) documents into one self-contained HTML page.

Deliberately unreserved. Not on ``MATCHER_PACKAGES`` in ``tools/check_isolation.py``: this
package renders a decision, it never makes one, so none of checks 1, 2, 6 or 7 have anything
to police here. Not on ``TRUTH_READERS`` either -- it never opens ``truth.json``, only what
``hisaab.scoring`` and ``hisaab.triage`` already wrote to disk, and both of those are
themselves barred from leaking the answer key into a ``--out`` document. Check 3 (tree-wide,
default-deny) covers this package from the moment its first file exists, the same as every
other package in ``hisaab/``.

It reads the explain artifact from its JSON file rather than importing ``hisaab.explain``:
that package is a leaf (``EXPLAIN_IMPORT_READERS`` is empty) precisely so that nothing
shipped under ``hisaab/`` can reach the model layer by a second door, and this package is no
exception to that rule despite rendering the artifact rather than matching anything.
"""
