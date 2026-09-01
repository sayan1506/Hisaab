"""The three pages: trigger a run, watch/view it, list past runs.

    pip install -e ".[web]"
    python -m tools.webapp.server [--port 8000]

**Local-only by design, and this is a real security property, not a formality.**
Triggering a run means this server accepts a seed/size/flag set from a form and spawns
four or five subprocesses with those arguments. The CLIs validate their own inputs and
refuse what they don't recognize, but that bound stops mattering the moment this binds to
anything but loopback. So the bind host is a literal below, never read from a request, an
env var, or an undocumented flag -- there is deliberately no ``--host`` argument. No
authentication exists, and none is added: the property being sold is "runs on your
machine only," not "safe to expose."

Every route serves or triggers a run under ``out/runs/`` (see ``run_registry.py``) and
never imports ``hisaab.generator``/``hisaab.matcher``/``hisaab.scoring`` for computation --
``pipeline.trigger_run`` shells out to the same CLIs a person would type by hand.
"""

from __future__ import annotations

import html
import threading
import uuid
from pathlib import Path
from typing import Any

from flask import Flask, redirect, request, send_file

from hisaab.generator.cli import FLAG_HELP
from hisaab.generator.config import MessFlags

from .pipeline import PipelineError, explain_available, trigger_run
from .run_registry import HoldoutRefused, RUNS_ROOT, list_runs, run_dir_for

app = Flask(__name__)

#: run_id (the uuid the form POST mints) -> live status, kept only in memory. Restarting
#: the server loses in-progress status but not the run directories themselves -- the
#: listing page (list_runs) reads the filesystem, not this dict.
_RUNS: dict[str, dict[str, Any]] = {}
_RUNS_LOCK = threading.Lock()


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font-family:sans-serif;margin:2rem;max-width:80ch}"
        "label{display:block;margin:.4rem 0}input[type=checkbox]{margin-right:.5rem}"
        ".hint{color:#666;font-size:.9em}.err{color:#a00}</style></head>"
        f"<body><h1>{html.escape(title)}</h1>{body}</body></html>"
    )


@app.route("/", methods=["GET"])
def index() -> str:
    unimplemented = set(MessFlags.unimplemented())
    flag_rows = []
    for name in MessFlags.names():
        disabled = "disabled" if name in unimplemented else ""
        note = " (declared but not implemented)" if name in unimplemented else ""
        flag_rows.append(
            f'<label><input type="checkbox" name="flags" value="{name}" {disabled}> '
            f'--{name.replace("_", "-")} '
            f'<span class="hint">{html.escape(FLAG_HELP.get(name, ""))}{note}</span></label>'
        )

    can_explain, why_not = explain_available()
    explain_disabled = "" if can_explain else "disabled"
    explain_note = "" if can_explain else f' <span class="hint">disabled: {html.escape(why_not)}</span>'

    body = f"""
    <form method="post" action="/run">
      <label>seed <input type="number" name="seed" value="1" required></label>
      <label>n <input type="number" name="n" value="60" required></label>
      <fieldset><legend>mess flags</legend>{"".join(flag_rows)}</fieldset>
      <label><input type="checkbox" name="explain" {explain_disabled}> generate explanations
      (spends one live model call){explain_note}</label>
      <button type="submit">run</button>
    </form>
    <p><a href="/runs">past runs</a></p>
    """
    return _page("Hisaab -- trigger a run", body)


@app.route("/run", methods=["POST"])
def run() -> Any:
    try:
        seed = int(request.form["seed"])
        n = int(request.form["n"])
    except (KeyError, ValueError):
        return _page("Bad input", '<p class="err">seed and n must be integers</p>'), 400
    flags = request.form.getlist("flags")
    explain = "explain" in request.form

    run_id = uuid.uuid4().hex[:12]
    with _RUNS_LOCK:
        _RUNS[run_id] = {"lines": [], "done": False, "error": None, "run_dir": None}

    def _worker() -> None:
        try:
            run_dir = run_dir_for(seed, n, flags)
            with _RUNS_LOCK:
                _RUNS[run_id]["run_dir"] = str(run_dir)
            for line in trigger_run(seed, n, flags, explain=explain):
                with _RUNS_LOCK:
                    _RUNS[run_id]["lines"].append(line)
        except (HoldoutRefused, PipelineError) as e:
            with _RUNS_LOCK:
                _RUNS[run_id]["error"] = str(e)
        finally:
            with _RUNS_LOCK:
                _RUNS[run_id]["done"] = True

    threading.Thread(target=_worker, daemon=True).start()
    return redirect(f"/runs/{run_id}")


@app.route("/runs/<run_id>/status", methods=["GET"])
def run_status(run_id: str) -> Any:
    with _RUNS_LOCK:
        state = _RUNS.get(run_id)
    if state is None:
        return {"error": "unknown run id"}, 404
    return {
        "lines": state["lines"],
        "done": state["done"],
        "error": state["error"],
        "run_dir": state["run_dir"],
    }


@app.route("/runs/<run_id>", methods=["GET"])
def view_run(run_id: str) -> Any:
    with _RUNS_LOCK:
        state = _RUNS.get(run_id)

    if state is None:
        # Not a run this server process triggered (e.g. after a restart) -- fall back to
        # the runs listing rather than guessing at a path from the id alone.
        return redirect("/runs")

    if state["error"]:
        return _page(
            "Run failed",
            f'<p class="err">{html.escape(state["error"])}</p><p><a href="/runs">back to runs</a></p>',
        )

    if not state["done"]:
        body = f"""
        <p>running... <span id="status"></span></p>
        <pre id="log"></pre>
        <script>
        async function poll() {{
          const r = await fetch('/runs/{run_id}/status');
          const j = await r.json();
          document.getElementById('log').textContent = j.lines.join('\\n');
          if (j.error) {{ document.getElementById('status').textContent = 'FAILED'; return; }}
          if (j.done) {{ location.reload(); return; }}
          setTimeout(poll, 1000);
        }}
        poll();
        </script>
        """
        return _page("Run in progress", body)

    report = Path(state["run_dir"]) / "out" / "report.html"
    if not report.exists():
        return _page(
            "Run finished, no report",
            '<p class="err">the pipeline finished but no report.html was written</p>'
            '<p><a href="/runs">back to runs</a></p>',
        )
    return send_file(report)


@app.route("/runs", methods=["GET"])
def runs_list() -> str:
    rows = list_runs()
    if not rows:
        body = "<p>no runs yet.</p>"
    else:
        items = "".join(
            f'<li>{html.escape(r["id"])} -- '
            + ('<a href="/runs/' + html.escape(r["id"]) + '">report</a>' if r["has_report"] else "no report")
            + "</li>"
            for r in rows
        )
        body = f"<ul>{items}</ul>"
    return _page("Past runs", body + '<p><a href="/">trigger a new run</a></p>')


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Local trigger-and-view webapp for the Hisaab pipeline. Binds to "
                     "127.0.0.1 only -- there is no --host flag, by design.",
    )
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    app.run(host="127.0.0.1", port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
