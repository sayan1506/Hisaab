"""Acceptance gate 5 — the truth file is unreachable from the matching path.

    python tools/check_isolation.py

Structural rule 1 of the architecture: ``truth.json`` feeds the scoring script and
nothing else. Nothing on the matching path may read it, ever.

That rule is worth enforcing mechanically rather than remembering, because the
failure is silent and total: a matcher that reads the answer key still produces a
match rate, an exception list and a confident-looking report. Nothing crashes. The
submission is simply void, and the only way to know is to have checked.

Five checks, all static -- no imports are executed, so this cannot be defeated by
a module that behaves differently when imported:

  1. Nothing on the matching path imports ``hisaab.scoring`` or ``truth_io``.
  2. Nothing on the matching path names the truth file in executable code.
  3. Only allowlisted modules read the truth file; only the generator writes it.
  4. ``data/`` holds no truth file, and ``truth/`` holds no CSV.
  5. No CSV under ``data/`` contains a path reference into ``truth/``.

In Phase 1 the matching path does not exist yet, so checks 1 and 2 scan an empty
set. That is the point of writing this now: the moment ``hisaab/matcher/`` appears
in Phase 3, it is already covered, with no one having to remember to add it.

**Why this inspects the AST rather than grepping for a string.** The first version
of this tool did a raw text search for ``truth.json``, and it failed on its own
codebase for two reasons, both instructive:

  * A docstring *explaining* the isolation rule counted as breaking it. The whole
    point of writing the rule down next to the code is that the next person reads
    it, so a check that punishes documentation is a check that will be deleted.
  * The generator legitimately *writes* the truth file. Reading and writing are
    opposite operations and the string is identical in both.

So the scan looks only at string literals in **executable** positions -- comments
never reach the AST at all, and docstrings are excluded explicitly -- and imports
are resolved structurally, relative imports included. A matcher docstring that
says "this module deliberately never reads truth.json" passes, which is correct:
that sentence is evidence of the discipline, not a violation of it.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Packages on the matching path. ``hisaab.matcher`` arrives in Phase 3; listing it
#: now means the guard is armed before there is anything to guard.
MATCHER_PACKAGES: tuple[str, ...] = (
    "hisaab/matcher",
    "hisaab/normalize",
    "hisaab/blocking",
    "hisaab/prove",
    "hisaab/classify",
    "hisaab/explain",
)

#: The one module allowed to read truth.json.
TRUTH_READER = "hisaab/scoring/truth_io.py"

#: The one module allowed to write it.
TRUTH_WRITER = "hisaab/generator/emit.py"

#: Modules allowed to **read** the answer key -- to import ``truth_io`` or call
#: ``load_truth``. Each is listed individually so this allowlist stays a decision
#: rather than a prefix that quietly widens. They score and audit; none of them
#: matches.
TRUTH_READERS: tuple[str, ...] = (
    TRUTH_READER,
    "hisaab/scoring/__init__.py",
    "tools/verify_output.py",   # gate 4: re-checks written files against truth
    "tools/repro_check.py",     # gates 1/2/6: hashes truth.json across runs
    "tools/check_isolation.py",  # this file
)

#: Trees allowed to **name** the truth path. Naming is not reading: the generator
#: writes ``truth.json`` and the CLI passes ``--truth`` through, which is the
#: opposite operation from reading it, spelled with the same string.
TRUTH_PATH_OWNERS: tuple[str, ...] = ("hisaab/generator/", "hisaab/scoring/", "tools/")

#: Names and literals that mean "this code touches the answer key". Matched against
#: executable code only -- never comments or docstrings.
TRUTH_TOKENS: tuple[str, ...] = ("truth.json", "load_truth", "truth_io")

#: Import targets that reach the scoring package however they are spelled.
SCORING_IMPORT_PREFIXES: tuple[str, ...] = ("hisaab.scoring", "scoring.truth_io")


class IsolationError(Exception):
    """The matching path can reach the answer key."""


def python_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*.py")
        if "__pycache__" not in p.parts and ".plan" not in p.parts
    )


def _parse(path: Path) -> ast.Module:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as e:
        raise IsolationError(f"{path}: cannot parse ({e})") from e


def _inert_strings(tree: ast.Module) -> set[int]:
    """Node ids of string literals that are evaluated and thrown away.

    Docstrings and bare string "comments". They cannot open a file, so a module
    that documents the isolation rule in prose is not violating it. Real comments
    never reach the AST at all, so they need no handling.
    """
    return {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }


def truth_references(path: Path) -> list[str]:
    """Truth-file references in *executable* code, as human-readable findings."""
    tree = _parse(path)
    inert = _inert_strings(tree)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in inert:
                continue
            if any(token in node.value for token in TRUTH_TOKENS):
                found.append(f"line {node.lineno}: string {node.value!r}")
        elif isinstance(node, ast.Name) and node.id in TRUTH_TOKENS:
            found.append(f"line {node.lineno}: name {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in TRUTH_TOKENS:
            found.append(f"line {node.lineno}: attribute .{node.attr}")
    return found


def reads_truth(path: Path) -> list[str]:
    """Evidence that ``path`` *reads* the answer key, as opposed to writing it.

    Reading means importing the scoring package or the loader, or calling
    ``load_truth``. Writing ``truth.json`` -- which the generator does by design --
    is deliberately not evidence of a read.
    """
    evidence: list[str] = []
    for mod in imports_of(path):
        if mod.startswith(SCORING_IMPORT_PREFIXES) or "truth_io" in mod:
            evidence.append(f"imports {mod}")
    tree = _parse(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name in ("load_truth", "load_manifest"):
                evidence.append(f"line {node.lineno}: calls {name}()")
    return evidence


def imports_of(path: Path) -> set[str]:
    """Every module named by an import in ``path``, resolved as dotted strings."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as e:
        raise IsolationError(f"{path}: cannot parse ({e})") from e
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            # Resolve relative imports against the package, so `from ..scoring
            # import x` inside hisaab/matcher/ is caught as hisaab.scoring.
            if node.level:
                parts = path.relative_to(ROOT).with_suffix("").parts[:-1]
                base = parts[: len(parts) - node.level + 1]
                prefix = ".".join(base)
                found.add(f"{prefix}.{node.module}" if node.module else prefix)
            elif node.module:
                found.add(node.module)
            found.update(
                f"{node.module}.{a.name}" for a in node.names if node.module and not node.level
            )
    return found


def check(verbose: bool = True) -> dict[str, object]:
    rel = lambda p: p.relative_to(ROOT).as_posix()  # noqa: E731
    all_py = python_files(ROOT / "hisaab") + python_files(ROOT / "tools")

    # --- 1 & 2: the matching path cannot reach truth ------------------------
    matcher_files = [
        p for p in all_py if any(rel(p).startswith(pkg + "/") for pkg in MATCHER_PACKAGES)
    ]
    for path in matcher_files:
        if evidence := reads_truth(path):
            raise IsolationError(
                f"{rel(path)} reaches the answer key ({'; '.join(evidence)}). The "
                f"matching path must not be able to read truth -- move whatever it "
                f"needs into a module that does not."
            )
        if refs := truth_references(path):
            raise IsolationError(
                f"{rel(path)} names the truth file in executable code "
                f"({'; '.join(refs[:3])}). The matching path must not reference it, "
                f"even by string literal. (A docstring saying it never reads truth is "
                f"fine -- this check ignores prose.)"
            )

    # --- 3: only allowlisted modules READ truth; only owners NAME its path ---
    # Reading and naming are separated deliberately. The generator names
    # truth.json because it writes it, and the CLI passes --truth through; neither
    # is a read, and conflating the two would either flag the writer or force the
    # allowlist so wide it stops meaning anything.
    unexpected_readers: list[str] = []
    for path in all_py:
        name = rel(path)
        if name in TRUTH_READERS:
            continue
        if evidence := reads_truth(path):
            unexpected_readers.append(f"{name} ({'; '.join(evidence)})")
    if unexpected_readers:
        raise IsolationError(
            "these modules read the answer key but are not on the TRUTH_READERS "
            "allowlist in tools/check_isolation.py:\n    "
            + "\n    ".join(unexpected_readers)
            + "\n  Either the read is wrong, or the allowlist needs a deliberate edit."
        )

    unexpected_names: list[str] = []
    for path in all_py:
        name = rel(path)
        if name.startswith(TRUTH_PATH_OWNERS):
            continue
        if refs := truth_references(path):
            unexpected_names.append(f"{name} ({refs[0]})")
    if unexpected_names:
        raise IsolationError(
            "these modules name the truth file outside the trees that own that "
            f"path: {unexpected_names}"
        )

    # --- 4: the two directories stay separate -------------------------------
    data_dir, truth_dir = ROOT / "data", ROOT / "truth"
    if data_dir.exists() and truth_dir.exists():
        if data_dir.resolve() == truth_dir.resolve():
            raise IsolationError("data/ and truth/ are the same directory")
        stray = [p.name for p in data_dir.iterdir() if p.name in ("truth.json", "run_manifest.json")]
        if stray:
            raise IsolationError(f"truth files found inside data/: {stray}")
        stray_csv = [p.name for p in truth_dir.iterdir() if p.suffix == ".csv"]
        if stray_csv:
            raise IsolationError(f"CSVs found inside truth/: {stray_csv}")

        # --- 5: no data file points at the truth directory ------------------
        for csv_path in sorted(data_dir.glob("*.csv")):
            text = csv_path.read_text(encoding="utf-8")
            for token in ("truth.json", "truth/", "run_manifest"):
                if token in text:
                    raise IsolationError(
                        f"{csv_path.name} contains {token!r} -- a data file must not "
                        f"reference anything under truth/"
                    )

    report: dict[str, object] = {
        "python_files_scanned": len(all_py),
        "matcher_files_scanned": len(matcher_files),
        "truth_readers": len(TRUTH_READERS),
    }
    if verbose:
        print(f"isolation: scanned {len(all_py)} python files")
        if matcher_files:
            print(f"  matching path: {len(matcher_files)} files, none can reach truth")
        else:
            print(
                f"  matching path: no files yet (Phase 3 creates hisaab/matcher/) -- "
                f"the guard is armed for {len(MATCHER_PACKAGES)} package paths"
            )
        print(f"  the only module that opens truth.json for reading: {TRUTH_READER}")
        print(f"  the only module that writes it:                    {TRUTH_WRITER}")
        print(f"  modules allowed to read through it: {len(TRUTH_READERS)}")
        for name in TRUTH_READERS:
            print(f"    {name}")
        if data_dir.exists() and truth_dir.exists():
            print("  data/ and truth/ are separate, and neither leaks into the other")
        print("\ngate 5 passes -- the answer key is unreachable from the matching path")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Verify truth.json isolation (gate 5).")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    try:
        check(verbose=not args.quiet)
    except IsolationError as e:
        print(f"ISOLATION FAILED\n  {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
