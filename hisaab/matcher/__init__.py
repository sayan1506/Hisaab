"""The matching engine — deterministic, integer paise, no LLM on the match path.

Reads the five CSVs under ``data/`` and emits one verdict per bank row. It never
reads ``truth.json``, and it never imports ``hisaab.generator`` or
``hisaab.scoring``: the first knows the answers, the second two know how the
answers were made and how they are graded. ``tools/check_isolation.py`` enforces
all three mechanically (checks 1, 2 and 6) rather than trusting this docstring.

Phase 3 is **Tier 1 only**: an exact join on ``(value_date, net_paise)`` inside a
±0 business-day window. Nothing here models fees, searches subsets, or applies a
tolerance -- those are Phases 4, 5 and 3-of-the-tier-list respectively, and each
arrives as a change to a parameter or a new stage rather than a rewrite.
"""
