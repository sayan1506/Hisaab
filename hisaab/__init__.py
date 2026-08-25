"""Hisaab — multi-source reconciliation for the Razorpay AI Buildathon (Track 04).

Package layout:
    hisaab.common     shared vocabulary (ID formats, exception reason codes)
    hisaab.generator  synthetic data generator — emits data/ and truth/
    hisaab.scoring    the ONLY place truth.json is read (see hisaab/scoring/truth_io.py)

Structural rule, enforced by tools/check_isolation.py: nothing on the matching
path may import hisaab.scoring or otherwise reach truth.json.
"""

__version__ = "0.1.0"
