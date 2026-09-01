"""Runs the five-stage CLI pipeline as subprocesses -- never imports the engine.

Mirrors the shape ``tools/acceptance.py``'s ``_run()``/``_matcher_and_score()`` already
demonstrate (subprocess, capture stdout/stderr, non-zero exit is a failure), read there for
pattern only -- ``tools.acceptance`` is a gate runner, not a library this ships, so it is
never imported here.

**Why five different argument lists, not one call repeated with a shared seed/window.**
Only ``hisaab.generator`` and ``hisaab.matcher`` take ``--seed``; ``hisaab.scoring``,
``hisaab.triage`` and ``hisaab.report`` take none -- they read what the matcher already
wrote and reconcile from that. ``hisaab.matcher`` needs ``--window 1`` for any composable
mess-flag run (``DEFAULT_WINDOW_DAYS = 0`` in ``hisaab/matcher/blocking.py``; ``--window 1``
is fixed always here, matching ``tools/acceptance.py``'s own ``MESS_WINDOW_DAYS = 1`` -- a
clean-mode run tolerates the wider window with no behavior change, since there is nothing
for it to slide into).

**Explain is an explicit per-run opt-in, never automatic.** Ticking it means spending one
live, paid Anthropic API call. Nothing in this module imports ``anthropic`` unconditionally
-- ``explain_available()`` imports ``hisaab.explain.client`` lazily, inside the function
body, so a checkout with the ``llm`` extra not installed still imports this module cleanly.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

from .run_registry import ROOT, refuse_holdout_seed, run_dir_for

#: Matches ``tools/acceptance.py``'s own constant, restated rather than imported for the
#: same reason ``run_registry.HOLDOUT_SEED`` is restated -- this module never imports
#: ``tools.acceptance``.
MESS_WINDOW_DAYS = 1


class PipelineError(Exception):
    """A stage exited non-zero. Carries the stage name and the stage's own stderr."""


def explain_available() -> tuple[bool, str]:
    """Whether ticking "explain" would actually work, and why not if it wouldn't.

    Calls the same precondition guard ``hisaab.explain.client._client()`` uses internally,
    so the webapp's refusal reason is never a second, drifting copy of that message.
    """
    try:
        from hisaab.explain import client as explain_client  # noqa: PLC0415
    except ModuleNotFoundError:
        return False, "the `llm` extra is not installed (pip install -e \".[llm]\")"
    try:
        explain_client._client()
    except explain_client.ExplainError as e:
        return False, str(e)
    return True, ""


def _run_stage(argv: list[str], label: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", *argv], cwd=ROOT, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise PipelineError(
            f"{label} failed (exit {proc.returncode})\n{proc.stdout.rstrip()}\n"
            f"{proc.stderr.rstrip()}"
        )
    return proc.stdout


def trigger_run(
    seed: int, n: int, flags: list[str], *, explain: bool = False
) -> Iterator[str]:
    """Yields one status line per stage. Any stage's non-zero exit stops the run --
    never retries, never falls back to a partial render."""
    refuse_holdout_seed(seed)

    from hisaab.generator.config import MessFlags  # noqa: PLC0415 -- reading a config

    known = set(MessFlags.names())
    unimplemented = set(MessFlags.unimplemented())
    for f in flags:
        if f not in known:
            raise PipelineError(f"unknown mess flag: {f!r}")
        if f in unimplemented:
            raise PipelineError(
                f"--{f.replace('_', '-')} is declared but not implemented yet -- "
                f"story.py does not read it, so the run would be labelled with a mess it "
                f"does not have"
            )

    run_dir = run_dir_for(seed, n, flags)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise PipelineError(f"{run_dir} already has a run -- refusing to overwrite")
    data_dir = run_dir / "data"
    truth_dir = run_dir / "truth"
    out_dir = run_dir / "out"

    yield "generator: starting"
    flag_args = [f"--{f.replace('_', '-')}" for f in flags]
    _run_stage(
        [
            "hisaab.generator", "--seed", str(seed), "--n", str(n),
            "--out", str(data_dir), "--truth", str(truth_dir), "--quiet", *flag_args,
        ],
        "generator",
    )
    yield "generator: done"

    yield "matcher: starting"
    matches_path = out_dir / "matches.json"
    _run_stage(
        [
            "hisaab.matcher", "--data", str(data_dir), "--out", str(matches_path),
            "--seed", str(seed), "--window", str(MESS_WINDOW_DAYS), "--quiet",
        ],
        "matcher",
    )
    yield "matcher: done"

    yield "scoring: starting"
    metrics_path = out_dir / "metrics.json"
    _run_stage(
        [
            "hisaab.scoring", "--matches", str(matches_path), "--truth", str(truth_dir),
            "--out", str(metrics_path), "--quiet",
        ],
        "scoring",
    )
    yield "scoring: done"

    yield "triage: starting"
    triage_path = out_dir / "triage.json"
    _run_stage(
        [
            "hisaab.triage", "--matches", str(matches_path), "--data", str(data_dir),
            "--out", str(triage_path), "--quiet",
        ],
        "triage",
    )
    yield "triage: done"

    explain_path: Path | None = None
    if explain:
        yield "explain: starting"
        explain_path = out_dir / "explain.json"
        _run_stage(
            [
                "hisaab.explain", "--matches", str(matches_path), "--data", str(data_dir),
                "--out", str(explain_path),
            ],
            "explain",
        )
        yield "explain: done"

    yield "report: starting"
    report_args = [
        "hisaab.report", "--matches", str(matches_path), "--metrics", str(metrics_path),
        "--triage", str(triage_path), "--out", str(out_dir / "report.html"), "--quiet",
    ]
    if explain_path is not None:
        report_args += ["--explain", str(explain_path)]
    _run_stage(report_args, "report")
    yield "report: done"

    yield f"run complete: {run_dir}"


if __name__ == "__main__":
    # --- unimplemented flag is refused before any subprocess runs --------------------
    try:
        list(trigger_run(1, 60, ["rounding_edge"]))
    except PipelineError as e:
        assert "not implemented" in str(e)
    else:
        raise AssertionError("an unimplemented flag must be refused")

    # --- seed 99 is refused before anything else, including flag validation ----------
    from .run_registry import HoldoutRefused

    try:
        list(trigger_run(99, 1000, ["not_a_real_flag"]))
    except HoldoutRefused as e:
        assert "99" in str(e)
    else:
        raise AssertionError("seed 99 must be refused ahead of flag validation")

    # --- unknown flag name is refused ------------------------------------------------
    try:
        list(trigger_run(1, 60, ["not_a_real_flag"]))
    except PipelineError as e:
        assert "unknown mess flag" in str(e)
    else:
        raise AssertionError("an unknown flag name must be refused")

    print("tools/webapp/pipeline.py self-check ok (subprocess stages not exercised here -- "
          "see the manual verification step for a real run)")
