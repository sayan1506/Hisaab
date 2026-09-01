"""Where a triggered run lives on disk, and the one refusal every trigger path must pass.

Phase 12b's webapp shells out to the pipeline (see ``pipeline.py``) rather than importing
``hisaab.generator``/``hisaab.matcher``/``hisaab.scoring`` for computation — the same reason
``tools/acceptance.py`` does. Importing ``hisaab.scoring`` even for a type would put this
module on ``tools/check_isolation.py``'s ``TRUTH_READERS`` allowlist by that check's own
stated rule ("a module that can import the package can import the loader"), and a demo
server has no business anywhere near that list.

**Two structural guarantees, not conventions this module happens to follow:**

1. Every run directory is computed from ``ROOT / "out" / "runs" / ...`` with no caller-
   supplied base path anywhere in the call chain. There is no input to ``run_dir_for`` that
   can resolve to the repo-root ``data/``/``truth/`` — the committed reference run every
   gate and every cited number in ``ASSUMPTIONS.md`` depends on.
2. Seed 99 (the Phase 12 holdout) is refused here, at the one function every trigger path
   goes through — ``pipeline.trigger_run`` calls ``refuse_holdout_seed`` before anything
   else runs. A frontend that made re-running the holdout a one-click action would turn
   "the holdout is untouched" into "the holdout was touched twice, informally, off the
   record." Refusing only in the form would leave a second path (a direct POST, a script
   against the same endpoint) that never sees that check.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

#: Restated rather than imported from ``tools.acceptance`` — that module is a gate runner,
#: not a library this ships, and the matcher's own CLI already sets the precedent of
#: restating a constant rather than importing across a boundary that shouldn't be crossed
#: for one value (see ``hisaab/matcher/cli.py``'s ``DEFAULT_SEED`` comment).
HOLDOUT_SEED = 99

RUNS_ROOT = ROOT / "out" / "runs"


class HoldoutRefused(Exception):
    """Seed 99 was requested through the webapp."""


def refuse_holdout_seed(seed: int) -> None:
    if seed == HOLDOUT_SEED:
        raise HoldoutRefused(
            f"seed {HOLDOUT_SEED} is the Phase 12 holdout and is run exactly once, by hand, "
            f"outside this webapp -- see ASSUMPTIONS.md #28/#44. Pick a different seed."
        )


def run_dir_for(seed: int, n: int, flags: list[str]) -> Path:
    """A deterministic path under ``out/runs/``, structurally unable to reach the repo root.

    Built by string formatting onto ``RUNS_ROOT`` alone -- no argument here is itself a
    path, so there is no value of ``seed``/``n``/``flags`` that produces anything other
    than a child of ``out/runs/``. That is what "structurally unable" means, as opposed to
    "unlikely": it isn't a matter of validating the inputs correctly, there is nothing
    these inputs could be that escapes ``RUNS_ROOT``.
    """
    flags_key = ",".join(sorted(flags))
    flags_hash = hashlib.sha256(flags_key.encode("utf-8")).hexdigest()[:8]
    name = f"{seed}-{n}-{flags_hash}"
    return RUNS_ROOT / name


def list_runs() -> list[dict[str, object]]:
    """One row per existing run directory, newest first by directory mtime.

    Reads only directory names and the presence of ``report.html`` -- never opens any
    JSON document a run wrote, so this module stays outside every truth-adjacent check
    without needing an exemption from any of them.
    """
    if not RUNS_ROOT.exists():
        return []
    rows = []
    for d in RUNS_ROOT.iterdir():
        if not d.is_dir():
            continue
        report = d / "out" / "report.html"
        rows.append(
            {
                "id": d.name,
                "path": str(d),
                "has_report": report.exists(),
                "mtime": d.stat().st_mtime,
            }
        )
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows


if __name__ == "__main__":
    # --- run_dir_for cannot reach the repo root, for any input -----------------------
    for seed, n, flags in [(1, 60, []), (0, 0, []), (-1, -1, ["fees"]), (99, 1000, [])]:
        d = run_dir_for(seed, n, flags)
        assert d.is_relative_to(RUNS_ROOT), f"{d} escaped RUNS_ROOT for input {(seed, n, flags)}"
        assert d != ROOT / "data"
        assert d != ROOT / "truth"

    # --- same inputs, same path; different flags, different path ---------------------
    assert run_dir_for(1, 60, ["fees", "tds"]) == run_dir_for(1, 60, ["tds", "fees"])
    assert run_dir_for(1, 60, []) != run_dir_for(1, 60, ["fees"])
    assert run_dir_for(1, 60, []) != run_dir_for(2, 60, [])

    # --- the holdout seed is refused, by name, at the function layer -----------------
    try:
        refuse_holdout_seed(HOLDOUT_SEED)
    except HoldoutRefused as e:
        assert "99" in str(e) and "holdout" in str(e)
    else:
        raise AssertionError("seed 99 must be refused")
    refuse_holdout_seed(1)  # any other seed passes silently

    print("tools/webapp/run_registry.py self-check ok")
