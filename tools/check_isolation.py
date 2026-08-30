"""Acceptance gate 5 — the truth file is unreachable from the matching path.

    python tools/check_isolation.py

Structural rule 1 of the architecture: ``truth.json`` feeds the scoring script and
nothing else. Nothing on the matching path may read it, ever.

That rule is worth enforcing mechanically rather than remembering, because the
failure is silent and total: a matcher that reads the answer key still produces a
match rate, an exception list and a confident-looking report. Nothing crashes. The
submission is simply void, and the only way to know is to have checked.

Six checks, all static -- no imports are executed, so this cannot be defeated by
a module that behaves differently when imported:

  1. Nothing on the matching path imports ``hisaab.scoring`` or ``truth_io``.
  2. Nothing on the matching path names the truth file in executable code.
  3. Only allowlisted modules read the truth file; only the generator writes it.
  4. ``data/`` holds no truth file, and ``truth/`` holds no CSV.
  5. No CSV under ``data/`` contains a path reference into ``truth/``.
  6. Nothing on the matching path imports ``hisaab.generator``.

In Phase 1 the matching path does not exist yet, so checks 1 and 2 scan an empty
set. That is the point of writing this now: the moment ``hisaab/matcher/`` appears
in Phase 3, it is already covered, with no one having to remember to add it.

**Check 6 closes a hole the first five left open.** Checks 1-5 are all about
``truth.json``, and they would pass a matcher that imported ``hisaab.generator`` --
which is a different leak with the same effect. That package knows the fee rates, the
T+n settlement cycle and the narration templates, so importing it is reading the
answer with extra steps: a matcher that asked the generator for the fee schedule
would "reconcile" perfectly in Phase 4 while modelling nothing.

The rule cuts both ways, and the direction that is *allowed* matters too. Shared
vocabulary lives in ``hisaab/common/`` precisely so neither side has to import the
other -- ``money.py`` moved there in Phase 2 and ``bizdays.py`` in Phase 3, both
because the generator and the matcher genuinely need the same logic. What the matcher
must duplicate instead is *schema*: the CSV header tuples in ``matcher/load.py``, like
``SUPPORTED_SCHEMA_VERSION`` in ``scoring/truth_io.py``, are copied so that drift
fails loudly. Logic is shared, schemas are duplicated, and the answer key is
unreachable.

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

#: Packages that must work from the inputs alone. ``hisaab.matcher`` arrives in Phase 3;
#: listing it now means the guard is armed before there is anything to guard.
#:
#: **Named for the matcher, but it gates four checks** -- 1, 2, 6 and 7 -- so adding an entry
#: bans four things at once: importing ``hisaab.scoring``, naming the truth path in code,
#: importing the generator, and importing the declared-fee report. Worth knowing before adding
#: one, because the four have different justifications and a new package inherits all of them.
#:
#: ``hisaab/triage`` (Phase 9) is here for check 6 above all. Checks 1 and 3 already covered
#: it -- check 3 walks every ``.py`` in the tree and is default-deny, so a new package is
#: guarded from the moment its first file exists -- but check 6 was scoped to *this tuple*, so
#: until triage joined it, the queue could import the generator's fee rates, T+n cycle and
#: narration templates with nothing to stop it. That is the worse leak of the two: triage's job
#: is explaining **why** a row failed, which is precisely the job that tempts an author toward
#: the generator's rate table, and unlike a truth import it yields a queue that *looks* right.
#: The summary line below reported "matching path: 11 files" while triage sat outside it, so
#: the number a reader trusts did not count the package with the most reason to cheat.
MATCHER_PACKAGES: tuple[str, ...] = (
    "hisaab/matcher",
    "hisaab/normalize",
    "hisaab/blocking",
    "hisaab/prove",
    "hisaab/classify",
    "hisaab/explain",
    "hisaab/triage",
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
    # --- Phase 2's scoring harness -------------------------------------------
    # Five entries, and the check below is coarser than it looks: importing
    # *anything* under hisaab.scoring counts as reaching truth, because a module
    # that can import the package can import the loader. So a module lands here
    # either because it genuinely opens the answer key, or because it imports a
    # sibling that does. The distinction is worth writing down per entry.
    "hisaab/scoring/cli.py",       # genuinely opens truth, then hands plain values down
    "hisaab/scoring/metrics.py",   # genuinely joins verdicts against the answer key
    "hisaab/scoring/report.py",    # transitive: imports Metrics for the type, never Truth
    "hisaab/scoring/__main__.py",  # transitive: imports .cli
    "tools/fixtures.py",           # gate 8: the oracle and saboteur read truth by design
    # Deliberately ABSENT: hisaab/scoring/verdict_io.py. It validates the matcher's
    # output -- the one scoring job with no business seeing the answers -- so it takes
    # the credit IDs, seed and month as plain arguments and imports no sibling that
    # could reach truth. A change that hands it a Truth object fails this gate, which
    # is the intended outcome rather than an inconvenience.
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

#: Import targets that reach the generator (check 6). The generator knows the fee
#: rates, the T+n cycle and the narration templates -- everything the matcher is
#: supposed to *infer*. Relative spellings resolve to these too; see ``imports_of``.
GENERATOR_IMPORT_PREFIXES: tuple[str, ...] = ("hisaab.generator", "generator.model")

#: The one matcher module allowed to read ``settlements.csv``'s declared ``fee_paise``,
#: ``gst_paise`` and ``tds_paise`` columns (check 7). It compares them against an
#: independently derived figure and **reports**; it returns no ``Verdict`` and decides
#: nothing. See its module docstring.
ADJUSTMENT_REPORT_MODULE = "hisaab/matcher/adjustments.py"

#: Modules allowed to import it. Only the CLI, which prints the report next to the
#: coverage number. Listed individually rather than by prefix so widening this stays a
#: decision -- the same reasoning as ``TRUTH_READERS``.
#:
#: **``engine.py`` is deliberately absent.** The engine produces verdicts, so handing it
#: the declared columns is exactly the move this check exists to refuse: subtracting a
#: declared fee closes every residual the instant ``--fees`` is on, and coverage reads
#: 100% with no fee model ever written. The comparison therefore happens in the CLI,
#: after ``run()`` has already committed to its answers and cannot be influenced.
ADJUSTMENT_REPORT_READERS: tuple[str, ...] = (
    "hisaab/matcher/cli.py",
)

#: Import targets that reach the adjustment report however they are spelled.
ADJUSTMENT_IMPORT_PREFIXES: tuple[str, ...] = (
    "hisaab.matcher.adjustments",
    "matcher.adjustments",
)


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


def imports_generator(path: Path) -> list[str]:
    """Evidence that ``path`` imports the generator -- check 6.

    Structurally identical to ``reads_truth``'s import half, and separate from it
    because the two failures need different explanations: reading truth is reading the
    answers, while importing the generator is reading *how the answers were made*.
    Both void the submission; only one of them is about a file.
    """
    return [
        f"imports {mod}"
        for mod in sorted(imports_of(path))
        if mod.startswith(GENERATOR_IMPORT_PREFIXES)
    ]


def imports_adjustments(path: Path) -> list[str]:
    """Evidence that ``path`` imports the declared-vs-derived report -- check 7.

    Third of the same shape, after ``reads_truth`` and ``imports_generator``, and separate
    for the same reason: this one is not about the answer key at all. It is about the
    matcher declining to *consume* a number the counterparty declared, so that its residual
    stays a test rather than a restatement.
    """
    return [
        f"imports {mod}"
        for mod in sorted(imports_of(path))
        if mod.startswith(ADJUSTMENT_IMPORT_PREFIXES)
    ]


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

        # --- 6: the matching path cannot import the generator ---------------
        if evidence := imports_generator(path):
            raise IsolationError(
                f"{rel(path)} imports the generator ({'; '.join(evidence)}). The "
                f"generator knows the fee rates, the T+n settlement cycle and the "
                f"narration templates -- everything the matcher is supposed to infer -- "
                f"so importing it is reading the answer with extra steps.\n"
                f"  If the two genuinely need the same logic, move it to "
                f"hisaab/common/ (as money.py and bizdays.py were). If it is a schema, "
                f"duplicate it with a comment saying why, the way matcher/load.py does "
                f"with the CSV headers, so drift fails loudly instead of hiding behind "
                f"a shared symbol."
            )

        # --- 7: the resolution path cannot import the adjustment report -----
        mod_name = rel(path)
        if (
            mod_name != ADJUSTMENT_REPORT_MODULE
            and mod_name not in ADJUSTMENT_REPORT_READERS
            and (evidence := imports_adjustments(path))
        ):
            raise IsolationError(
                f"{mod_name} imports the declared-vs-derived report "
                f"({'; '.join(evidence)}). That module reads settlements.csv's declared "
                f"fee_paise, gst_paise and tds_paise columns, and every other module on "
                f"the matching path re-derives those figures from an independently "
                f"declared rate table instead.\n"
                f"  The difference is the whole test: subtracting a declared fee closes "
                f"the residual the moment --fees populates it, so coverage would read "
                f"100% with no fee model ever written and no number moving to say one "
                f"was missing. The report is for a human to read next to the coverage "
                f"figure, never for the matcher to resolve with.\n"
                f"  If a verdict genuinely needs this, it does not -- it needs a rate. "
                f"Add one to fees.py, where being wrong about it shows up as a residual."
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
            print(
                f"  matching path: {len(matcher_files)} files, none can reach truth "
                f"and none imports the generator"
            )
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
        if matcher_files:
            print(
                "  check 6: the matcher may not import hisaab.generator -- shared logic "
                "goes to hisaab/common/, schemas get duplicated on purpose"
            )
            print(
                f"  check 7: the declared fee/gst/tds columns are read only by "
                f"{ADJUSTMENT_REPORT_MODULE}, which reports and never resolves"
            )
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
