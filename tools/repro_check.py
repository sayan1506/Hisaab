"""Acceptance gates 1, 2 and 6 — reproducibility, seed sensitivity, throughput.

    python tools/repro_check.py [--seed 42] [--n 60] [--sizes 60 200]

"Run with ``--seed 42`` and you get my exact numbers" is a claim we do not want to
walk back, so it is measured rather than assumed.

Three things this proves, and one it deliberately does the hard way:

  **Gate 1 — byte-identical output.** Two runs at the same seed produce identical
  sha256 for all six deterministic files. Diff the bytes; do not assume.

  **Gate 2 — seed sensitivity.** ``--seed 43`` produces different data with the
  same shape. A generator that ignored its seed would pass gate 1 perfectly.

  **Gate 6 — throughput.** ``--n 200`` finishes fast enough to rerun after every
  change. Also the Phase 12 scale check, arriving early and for free.

The hard way: each run is a **separate subprocess with a different
``PYTHONHASHSEED``**. In-process comparison would miss the failure mode that
actually matters here -- deriving a stream seed from the builtin ``hash()``, which
is randomised per process. That bug is invisible within one interpreter and fatal
across two, which is exactly the situation a judge reproducing our numbers is in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hisaab.generator.emit import (  # noqa: E402
    DATA_FILES,
    DETERMINISTIC_FILES,
    MANIFEST_JSON,
    TRUTH_JSON,
)

#: Deliberately different, and neither is 0. If any stream seed were derived from
#: the builtin hash(), these two runs would diverge.
HASH_SEED_A = "1"
HASH_SEED_B = "987654"

#: Gate 6: a run we would happily repeat after every change.
THROUGHPUT_BUDGET_SECONDS = 2.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_generator(
    out: Path, truth: Path, seed: int, n: int, hash_seed: str, month: str = "2026-08"
) -> tuple[float, str]:
    """Run the generator in a subprocess. Returns (wall_seconds, stdout_line_1)."""
    started = time.perf_counter()
    proc = subprocess.run(
        [
            sys.executable, "-m", "hisaab.generator",
            "--seed", str(seed), "--n", str(n), "--month", month,
            "--out", str(out), "--truth", str(truth), "--quiet",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**_base_env(), "PYTHONHASHSEED": hash_seed},
    )
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        raise SystemExit(
            f"generator failed (exit {proc.returncode}) with PYTHONHASHSEED={hash_seed}\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
    return elapsed, proc.stdout.splitlines()[0] if proc.stdout else ""


def _base_env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env.pop("PYTHONHASHSEED", None)
    return env


def hashes_of(data: Path, truth: Path) -> dict[str, str]:
    out = {name: sha256(data / name) for name in DATA_FILES}
    out[TRUTH_JSON] = sha256(truth / TRUTH_JSON)
    return out


def manifest_comparable(truth: Path) -> str:
    """The manifest with its run-location and wall-clock fields stripped.

    Two things in ``run_manifest.json`` legitimately vary between two runs of the
    same seed, and both are excluded here rather than being called failures:

      * ``timing`` -- the wall clock. Non-determinism is confined to this block on
        purpose, so everything outside it must be byte-stable.
      * ``config.out_dir`` / ``config.truth_dir`` -- where the run wrote. This
        harness gives each run its own directory deliberately, so that run b cannot
        read or overwrite anything run a produced. Comparing paths would flag that
        independence as a bug.

    Everything else is compared, including ``file_sha256`` -- which catches a
    manifest reporting stale hashes for files it did not actually write.
    """
    raw = json.loads((truth / MANIFEST_JSON).read_text("utf-8"))
    raw.pop("timing", None)
    config = raw.get("config")
    if isinstance(config, dict):
        for key in ("out_dir", "truth_dir"):
            config.pop(key, None)
    return json.dumps(raw, indent=2, sort_keys=True)


def gate_1_byte_identical(root: Path, seed: int, n: int) -> dict[str, str]:
    """Two runs, same seed, different PYTHONHASHSEED -> identical bytes."""
    print(f"gate 1 -- byte-identity at seed {seed}, n={n}, across two processes")
    runs: list[dict[str, str]] = []
    for label, hash_seed in (("a", HASH_SEED_A), ("b", HASH_SEED_B)):
        data, truth = root / f"{label}/data", root / f"{label}/truth"
        elapsed, config_line = run_generator(data, truth, seed, n, hash_seed)
        runs.append(hashes_of(data, truth))
        print(f"    run {label}: PYTHONHASHSEED={hash_seed:<7} {elapsed * 1000:6.0f} ms")
        json.loads(config_line)  # line 1 of stdout must be the resolved config

    a, b = runs
    mismatched = [name for name in DETERMINISTIC_FILES if a[name] != b[name]]
    if mismatched:
        for name in mismatched:
            print(f"    MISMATCH {name}\n      run a {a[name]}\n      run b {b[name]}")
        raise SystemExit(
            "GATE 1 FAILED -- output is not reproducible across processes.\n"
            "  The usual cause is a stream seed derived from the builtin hash(), "
            "which PYTHONHASHSEED randomises per process, or a module-level "
            "random.* call, or a wall-clock read outside the manifest's timing block."
        )
    for name in DETERMINISTIC_FILES:
        print(f"    {name:<22} {a[name][:16]}  identical")

    if manifest_comparable(root / "a/truth") != manifest_comparable(root / "b/truth"):
        raise SystemExit(
            "GATE 1 FAILED -- run_manifest.json differs outside its timing and "
            "run-location fields, so something non-deterministic escaped into run "
            "provenance."
        )
    print(f"    {MANIFEST_JSON:<22} identical outside timing/paths")
    return a


def gate_2_seed_sensitivity(root: Path, seed: int, other_seed: int, n: int) -> None:
    """A different seed must change the data without changing its shape."""
    print(f"\ngate 2 -- seed {other_seed} differs from seed {seed}, same shape")
    data, truth = root / "c/data", root / "c/truth"
    run_generator(data, truth, other_seed, n, HASH_SEED_A)
    base = hashes_of(root / "a/data", root / "a/truth")
    other = hashes_of(data, truth)

    # Every file carrying data must differ. refunds.csv is header-only in clean
    # mode, so it is identical by construction -- not a failure, and asserting
    # otherwise would be asserting the wrong thing.
    must_differ = [f for f in DETERMINISTIC_FILES if f != "refunds.csv"]
    same = [name for name in must_differ if base[name] == other[name]]
    if same:
        raise SystemExit(
            f"GATE 2 FAILED -- these files are identical at two different seeds, "
            f"so the seed is being ignored: {same}"
        )
    if base["refunds.csv"] != other["refunds.csv"]:
        raise SystemExit("GATE 2 FAILED -- header-only refunds.csv should not vary by seed")
    for name in must_differ:
        print(f"    {name:<22} differs")

    # Same shape: identical headers and row counts.
    for name in DATA_FILES:
        left = (root / "a/data" / name).read_text("utf-8").splitlines()
        right = (data / name).read_text("utf-8").splitlines()
        if left[0] != right[0]:
            raise SystemExit(f"GATE 2 FAILED -- {name} header changed with the seed")
        if len(left) != len(right):
            raise SystemExit(
                f"GATE 2 FAILED -- {name} has {len(left) - 1} rows at seed {seed} but "
                f"{len(right) - 1} at seed {other_seed}; shape must not vary"
            )
    print(f"    headers and row counts unchanged across {len(DATA_FILES)} files")


def gate_6_throughput(root: Path, seed: int, sizes: list[int]) -> None:
    """Generation must stay fast enough to rerun after every change."""
    print("\ngate 6 -- throughput")
    for n in sizes:
        data, truth = root / f"t{n}/data", root / f"t{n}/truth"
        elapsed, _ = run_generator(data, truth, seed, n, HASH_SEED_A)
        rate = n / elapsed if elapsed else 0
        verdict = "ok" if elapsed <= THROUGHPUT_BUDGET_SECONDS else "TOO SLOW"
        print(f"    n={n:<5} {elapsed * 1000:6.0f} ms  ({rate:,.0f} rec/s incl. interpreter start)  {verdict}")
        if elapsed > THROUGHPUT_BUDGET_SECONDS:
            raise SystemExit(
                f"GATE 6 FAILED -- n={n} took {elapsed:.1f}s, over the "
                f"{THROUGHPUT_BUDGET_SECONDS}s budget. Generation is O(n); an "
                f"accidental quadratic scan is the likely cause."
            )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Reproducibility and throughput gates.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--other-seed", type=int, default=43)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--sizes", type=int, nargs="+", default=[60, 200])
    p.add_argument("--keep", action="store_true", help="keep the temp runs for inspection")
    args = p.parse_args(argv)

    root = Path(tempfile.mkdtemp(prefix="hisaab-repro-"))
    try:
        gate_1_byte_identical(root, args.seed, args.n)
        gate_2_seed_sensitivity(root, args.seed, args.other_seed, args.n)
        gate_6_throughput(root, args.seed, args.sizes)
    finally:
        if args.keep:
            print(f"\n  temp runs kept at {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)

    print("\ngates 1, 2 and 6 pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
