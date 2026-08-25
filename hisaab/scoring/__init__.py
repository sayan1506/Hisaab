"""Scoring — the only package permitted to read ``truth.json``.

Nothing on the matching path may import this package. ``tools/check_isolation.py``
enforces that, and Phase 2's scoring harness is built on ``truth_io``.
"""
