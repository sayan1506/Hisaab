"""Synthetic data generator.

Generates one *story* forward (sales -> settlements -> bank credits) and then
splits it into impoverished views the matcher sees, plus a truth file it never
sees. Run it with:

    python -m hisaab.generator --seed 42 --n 60 --month 2026-08

Phase 1 is clean mode: every mess flag off, one payment -> one settlement -> one
bank credit, identical amounts, identical dates.
"""
