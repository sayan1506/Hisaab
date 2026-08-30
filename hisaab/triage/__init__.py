"""Triage — the exception queue, grouped by cause and ranked by money at risk.

**The package with the least access in the project, and that is the design.** Triage answers
"what should a person work on first, and why" using only what an operator actually has: the
matcher's ``matches.json`` and the input files under ``data/``. It never reads ``truth.json``,
because an operator running this on their own month has no answer key -- a ranking tool that
needed one would be a demo, not a deliverable.

Three boundaries hold that up, all enforced by ``tools/check_isolation.py`` rather than by
intent, and ``hisaab/triage`` is listed in ``MATCHER_PACKAGES`` so all of them apply here:

  * **No truth** (checks 1 and 3). Check 3 is tree-wide and default-deny: any module not on
    ``TRUTH_READERS`` that shows read evidence fails, so this package was covered from the
    moment its first file existed. Verified by deliberate violation -- planting an import of
    ``hisaab.scoring.metrics`` here exits 1 and names the file -- and by the control that
    makes the guard mean something, an honest triage module reading only ``data/``, which
    exits 0. Without the second, the check would be a package ban rather than a truth ban.

  * **No generator** (check 6). This is the boundary Phase 9 had to *add*: check 6 was scoped
    to ``MATCHER_PACKAGES``, and until ``hisaab/triage`` joined it, this package could import
    the generator's fee rates, T+n cycle and narration templates with nothing to stop it.
    It matters more here than the truth case: triage's whole job is explaining **why** a row
    failed, which is exactly the job that tempts an author toward the generator's rate table,
    and the result would be a queue that *looks* right rather than one that is.

  * **No scoring** (check 1), which is the boundary with a consequence worth stating plainly
    rather than discovering in the next step. ``reads_truth()`` is a *static* check matching
    the ``hisaab.scoring`` prefix, so it is irrelevant that importing
    ``hisaab.scoring.verdict_io`` happens not to execute the loader today: triage may not
    import it, full stop. The allowlist's own comment gives the reasoning -- a module that
    can import the package can import the loader.

    So whatever reads ``matches.json`` on triage's behalf has to sit outside
    ``hisaab.scoring``. That file's *writer* (``write_verdicts``, ``MATCHES_JSON``) already
    lives in ``hisaab/common/verdict.py``; its validating reader currently does not.
    Resolving that -- share the reader by moving it down, or duplicate a narrow one here with
    a comment saying why, the way ``matcher/load.py`` duplicates the CSV headers -- is a
    decision for the step that first needs to read a verdict, not for this docstring to
    pre-empt.

What triage may therefore rely on: ``hisaab.common`` (money, ids, reasons, the verdict
contract) and the CSVs under ``data/``. Nothing else.
"""
