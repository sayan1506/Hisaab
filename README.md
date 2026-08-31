# Hisaab

**Multi-source payment reconciliation with a measured, honest match rate.**
Razorpay AI Buildathon — Track 04, AI Finance Controller.

*Hisaab* is Hindi for **accounts**, and also for **a reckoning**. Both readings apply: the
tool reconciles payments against settlements against a bank statement, and it reports
what it could not resolve rather than quietly rounding it away.

---

## The claim this project is built to support

Most reconciliation demos report one number: a match rate. That number cannot be checked,
because the person reporting it also decided what counted as a match.

Hisaab reports **three** numbers, and they answer different questions:

| | |
|---|---|
| **Coverage** | how often the matcher committed to an answer |
| **Correctness** | how often the answer it committed to was *right* |
| **Arithmetic** | how often it could *prove* that answer, term by term |

90% coverage with zero wrong matches beats 100% coverage with 85% correctness, and it is
not close. A wrong match silently corrupts the books; an exception just gets a human to
look at it. Any system that averages the two into a single "accuracy" figure has hidden
the one distinction a finance team actually cares about.

The third number exists because linking a credit to the right settlement is not the same as
knowing *why* the amounts differ. A matcher can name the correct payments and still be
wrong about the fee, the GST, or which of the two absorbed a rounding paisa — so the
deduction is re-derived from independently declared rates and compared against the answer
key **one term at a time**. Kept as its own axis rather than folded into correctness,
because folding it in would silently change what every earlier number meant.

Reporting correctness requires knowing the true answer, so the data is **synthetic and
generated with an answer key** (`truth.json`) that the matcher is structurally forbidden
to read. Some cases in it are **planted as unresolvable** — genuinely ambiguous from the
inputs — so "my exception list is honest" becomes a measurement instead of a claim.

**A reported 100% match rate on messy data is a failing signal here, not a win.** It means
the matcher guessed on rows where the inputs contain no answer.

---

## Quick start

Python 3.12. **The core has no dependencies** — the generator, matcher, scorer, exception
queue and HTML report are standard library only, so everything below runs from a fresh
clone with no install step and no virtualenv. Phase 10's LLM layer (`hisaab/explain`) is
the one exception and needs one package, installed deliberately:

```bash
pip install -e ".[llm]"     # only for hisaab/explain; nothing else needs it
```

That split is enforced rather than promised: `check_isolation.py` check 8 fails the build
if anything outside `hisaab/explain/` imports `anthropic`, or any HTTP client at all. The
acceptance suite — every gate, including the LLM layer's — runs with nothing installed.

```bash
# 1. Generate a month of data plus the answer key (byte-reproducible from the seed)
python -m hisaab.generator --seed 42 --n 60

# 2. Match it, then score the result against the answer key
python -m hisaab.matcher --data data/ --out out/matches.json
python -m hisaab.scoring --matches out/matches.json --truth truth/
```

Step 1 is required first: `data/` and `truth/` are gitignored, because they regenerate
identically from the seed and versioning them would only add churn.

The scorer prints the metric block:

```
Records processed          60 bank rows (60 gateway, 0 non-gateway)
Run                        seed 42, 2026-08, clean mode, flags: none
Matcher                    tier1+2@0.5.0
Wall clock                 0.00s, unattended

Coverage                   60/60    (100.0%)   how often it committed
Correctness                60/60    (100.0%)   how often it was right
Arithmetic proved          60/60    (100.0%)   linked right and priced right
Wrong matches              0   (0 of them on planted-unresolvable rows)
Wrong ignores              0
Correct abstentions        n/a   (0 planted unresolvable in clean mode)
Missed                     0   resolvable, but it abstained

Exceptions                 0
Value in exceptions        ₹0.00
Est. human time to clear   0 min   (0 min exceptions + 0 min dismissals)
Same batch by hand         2 h 00 min   (60 on sight x 2 min + 0 chased x 15 min)
Time saved                 100.0%   (2 h 00 min less than by hand)
Break-even chased rate     n/a   (nothing chased, so no rate to cross)

Exception queue                 empty -- every row was resolved or ignored
```

Every rate prints as a **fraction before a percentage**, and the denominators deliberately
differ: correctness is scored over the rows the matcher committed to, and arithmetic over
the rows that were also linked correctly. A rate whose denominator is invisible is how "we
verified the arithmetic" comes to mean three rows out of two hundred.

100% is expected *here* and is not the failing signal above: this is clean mode, the
easiest rung of the difficulty dial, where one payment becomes one settlement becomes one
bank credit with no fees and no delay. Row 1 of the difficulty dial **requires** 100% —
it is the regression check, not an achievement.

What that number does **not** prove is worth stating in the same breath:

- **The date window does no work *in clean mode*.** Every credit lands on its settlement's
  own date, so ±1000 days scores exactly the same as ±0. Phase 4 fixed this by measurement
  rather than assertion: under `--settlement-delay`, `--window 0` scores **0%** coverage
  while `--window 1` scores 100%, and `--fees` alone still scores 100% at ±0 — which locates
  the requirement in the **1-business-day bank posting lag**, not in the T+2 settlement
  cycle. T+2 shifts `settled_on` and `value_date` together, so a credit-to-settlement join
  never sees it. Gate 10 holds that line. The window's *tie-break* was not merely untested:
  it was refuted and retired (see below).
- **The business-day calendar is exercised only by its own unit test**, including the edge
  where a weekend settlement is zero business days from Monday's credit.
- **The narration parser is not on the match path at all.** That is deliberate, and it is
  gated — see [what Tier 1 refuses to do](#what-tier-1-does-and-what-it-refuses-to-do).

Line 1 of stdout is the same block as JSON, for a caller that parses rather than reads.
`--quiet` prints that line alone.

### Then work the exceptions

A match rate tells you how well the matcher did. It does not tell anyone what to do on Monday
morning. That is the exception queue:

```bash
python -m hisaab.triage --matches out/matches.json --data data/
```

It groups what could not be reconciled **by cause**, ranks the groups **by money at risk**,
prices each one, and gives each a next action and the missing input that would stop it
recurring:

```
Exception queue: 14 row(s) in 6 group(s), ₹1,83,178.68 at risk, ~152 min to clear

1. PARTIAL_SETTLEMENT_PENDING  --  ₹1,03,274.66 across 5 row(s), ~25 min (5 min each)
     C0047        ₹77,334.29
     C0030        ₹12,610.00
     C0010        ₹12,519.83
     C0019           ₹641.26
     C0005           ₹169.28

   Do: This looks like a settlement with part of it held back, not a mismatch: the
   shortfall is within the reserve band. Confirm the rolling-reserve release schedule,
   then match the credit against the settlement net of the held amount.
   Would stop it recurring: the reserve held and its release date, which no input file
   states today

2. DISMISSED (not gateway money)  --  ₹33,979.00 across 1 row(s), ~3 min (3 min each)
     C0023        ₹33,979.00
   ...
```

Three things about that output are deliberate:

- **Ranked by money, not by effort.** A queue ordered by minutes puts forty three-minute
  dismissals above one ₹4-lakh unresolved credit. Value leads; effort breaks ties. Gate 16
  requires at least one cell in its sweep where the two keys *disagree*, because "ranked by
  value" is untestable on data where every ordering happens to agree.
- **Groups come from the codes that occurred**, never from the vocabulary. Thirteen codes are
  declared; a heading for each would show a person seven empty sections that read as results.
- **The hints are fixed text, written against the branch that raises each code.** Nothing here
  is generated. Advice a person acts on should be reviewable in a diff before it is read — and
  writing them this way caught four hints that misdescribed their own cause, including one
  that asked for a fuller bank reference when the matcher deliberately does not read it.

**It cannot see the answer key.** `hisaab/triage` reads `matches.json` and `data/` only; it is
on the same `MATCHER_PACKAGES` allowlist as the matcher, so the static check that keeps the
matcher off `truth.json` keeps the queue off it too. That is what makes this runnable on a real
month rather than a demo. It also refuses to build a queue from two different runs' files —
including the case where both runs have the same row count, so every credit id lines up and
only the matcher's own stated amounts reveal the mismatch.

### Compare against a known answer

The matcher is measured against four fixtures whose scores are known before they run — a
stub that abstains on everything, an oracle that copies the answer key, a saboteur that
corrupts exactly six matches, and a zip that matches by row position:

```bash
python tools/fixtures.py --fixture oracle --out out/oracle.json
python -m hisaab.scoring --matches out/oracle.json --truth truth/
```

The oracle exists to prove the target is *reachable* on this data, so a matcher shortfall
is unambiguously the matcher's fault and not the dataset's. Tier 1 currently matches it.

### Ask a model about the exceptions

Everything above is deterministic. This is the one part that isn't, and it is scoped
accordingly: `hisaab/explain` reads the same inputs a human reconciler would — the queue
and `data/` — never `truth.json`, never the generator, never a declared fee column. It has
no more information than the matcher does; it just talks about what the matcher already
decided.

```bash
python -m hisaab.explain --fixture --dry-run          # no key needed, no network call
python -m hisaab.explain --matches out/matches.json --data data/ --out out/explain.json
python -m hisaab.explain --fixture --ask "C0001:why is the credit less than the gross?"
```

`--dry-run` builds every request and prints it without sending anything, so the prompt and
the grouping can be reviewed with nothing installed. A live run needs `pip install -e
".[llm]"` and an API key; `--fixture` runs against a frozen, committed recording instead
(`fixtures/explain/fixture.json`), so the gate and this README example both run offline.

What gets checked, and what doesn't: every row id and every rupee figure the model writes
about an exception is checked against the statement it was shown, and anything cited that
appears in no row it saw is refused by default (`--permissive` downgrades that to a
warning). That check has less to verify than it sounds — an exception row carries **no**
computed decomposition, by definition, so there is no arithmetic there to check, only the
citations. `--ask` is where the arithmetic check lives: it only answers questions about
**RESOLVED** rows, which do carry a full decomposition, and it pulls a claimed sum out of
the model's prose into signed terms and requires them to close exactly, every term to be a
real figure in that row, and the total to equal the credit — refusing an invented term even
when the sum it sits in still adds up.

The isolation claim has an assertion behind it as of this phase. `check_isolation.py`
gained a check that bans any HTTP client or model SDK under all of `hisaab/`, with
`hisaab/explain` carved out as the only exempt leaf — exempt from that one ban and nothing
else, so the component allowed to talk to a model is also the one with no privileged read
access. Nothing that ships imports it, so the matcher cannot reach a model through the one
package allowed to hold one.

### Verify the whole thing

```bash
python -m hisaab.generator --seed 42 --n 60   # if you have not already
python tools/acceptance.py
```

Eighteen gates, one command, exit code is the verdict. Byte-identical output at a fixed
seed across two processes; invariants on three seeds in memory and again re-read from
disk; the leak audit; truth isolation; throughput; the assumptions file; the four
known-answer fixtures; the matcher at 100/100/0 across three seeds × two sizes, including
the check that blanking every bank narration changes no decision; gate 10, which turns
on the first two rows of the difficulty dial and proves the arithmetic per row; gate
11, which plants pairs that genuinely cannot be resolved and reads the abstention count;
gate 12, which requires both matcher tiers to carry rows so a Tier 1 regression cannot
hide behind a Tier 2 success; gate 13, which adds three deduction terms and pins the one
that must never be resolved; gate 14, which adds bank rows that are not gateway credits
and payments that never pay out; gate 15, the eleven-flag run, with foreign currency and a
bank statement missing its UTRs; gate 16, which requires the exception queue to be
complete, correctly valued, genuinely ranked, and honest about its own ROI claim; and gate
17, which runs the model layer offline against a frozen fixture and a recorded client —
proving the citation check, fabrication rejection, per-module resilience when the SDK is
absent, and (via `qa.py`) that an invented term inside an otherwise-closing sum is refused
by name; and gate 18, which renders the HTML report end to end, re-sums a rendered
decomposition off the page's own text rather than trusting the verdict that produced it,
requires two renders of the same input byte-identical outside one timestamp line, and
greps the page for truth vocabulary — scoped around the one caption `metric_block()`
itself ships that legitimately contains the word "resolvable" — so a real leak elsewhere
still fails it.

Gate 10 is the one worth reading the source of, because of what it declines to assert. The
plan called for a flat 100/100/0 under `--fees --settlement-delay`; measurement said the
plan was wrong, and the gate now permits coverage to fall **only** onto an honest
abstention, while correctness and the wrong-match count never bend.

Gate 11 is where `correct_abstention` stops being a promise. For three phases it printed
0/0, because every row in every scored run was resolvable; it now reads **4/4** on three
seeds × two sizes. The assertion that took the work is not the obvious one: before the
flag existed, a strategy reading **no date and no amount** — just the four digits in the
bank narration, joined onto the `utr` column — resolved 60/60, 200/200 and 1000/1000
credits *correctly*, because tails are drawn without replacement. So a pair colliding on
`(date, amount)` while keeping distinct UTRs is still separable by exhaustive string
matching, and calling it unresolvable would have been a false statement about the data.
Each planted pair now shares one UTR, and the gate re-runs that attack every time.

Gate 13 is the argument for never retiring an old gate. It is the first thing in the suite
to run `--netted-refunds` alongside `--batching` and `--settlement-report-late`, and it
found three defects sitting in code that thirteen module self-checks and twelve gates had
all called green — including **two wrong matches per run** at n=1000, the one number this
project says never moves. Tier 2's subset search priced each candidate payment at its
gross, net of fee, or net of tax, and never net of its *refund*; on a refunded settlement
whose membership was withheld, the true answer was therefore not in the search space at
all, and the search saw only coincidences. Usually none, and the row abstained. Twice per
run, one unrelated subset hit the shrunken target exactly and the row resolved wrongly.

Every earlier gate passed throughout, because all three defects live in the *interaction*
between flags rather than in any single one. Gate 13 also runs at n=1000 even under
`--skip-slow`, alone among the gates: the same two seeds that read 0.9962 at n=1000 read a
clean 1.0000 with zero wrong matches at n=200, since coincidental subsets scale with the
candidate pool. A fast run that dropped the large size would have reported the phase green
while blind to its only correctness failure.

---

## Current state — Phase 11 of 13

| | Phase | State |
|---|---|---|
| ✅ | **1. Generator, clean mode** | Done. Three CSVs plus an answer key, 1:1 exact matches |
| ✅ | **2. Scoring harness** | Done. Coverage/correctness, the exception queue, four known-answer fixtures |
| ✅ | **3. Matcher, Tier 1** | Done. Exact `(value_date, net_paise)` join, 100/100/0 on clean mode |
| ✅ | **4. Fees and the settlement delay** | Done. The residual **gates**, the decomposition is published per row and checked term by term, the window is proved load-bearing |
| ✅ | **4b. Planted unresolvables (`--dup-amounts`)** | Done. Each pair shares a date, an amount **and a UTR** — the last one because a tail-only join resolved 100% without either of the others |
| ✅ | **5. Batching and the Tier 2 subset search (`--batching`)** | Done. Many payments settle as one credit; membership is withheld partially, so the search is a **counted** subset-sum with a bound that refuses rather than guesses |
| ✅ | **6. Adjustments (`--tds`, `--netted-refunds`, `--reserve`)** | Done. Three more deduction terms. TDS and refunds net inside the settlement; the **reserve deliberately does not** — its magnitude appears in no input file, so the matcher diagnoses it and never resolves it |
| ✅ | **7. Orphans and noise rows** | Done. Bank rows that are not gateway credits, payments that never pay out; `IGNORED == plainly_foreign` checked both directions |
| ✅ | **8. Foreign currency and patchy UTRs** | Done. `--fx` costs ~19% coverage on purpose — Tier 2's uniqueness inference is voided, not widened, when a foreign payment sits in the pool |
| ✅ | **9. Exception ranking** | Done. The scored run becomes a triaged queue, grouped by cause, ranked by money, priced per group — and the ROI claim's sign error from Phase 2 is fixed and now withheld on any run with a wrong match |
| ✅ | **10. LLM layer** | Done. `hisaab/explain` — citation-checked explanations and arithmetic-checked Q&A over RESOLVED rows, isolated from every privileged input, gated offline against a frozen fixture |
| ✅ | **11. HTML report** | Done. `hisaab/report` — reads the (up to) five documents a run already wrote and renders one self-contained page, stdlib only; reproducible outside a single timestamp line, and re-checks every rendered decomposition's arithmetic against the text on the page rather than trusting it was correct in memory |
| ⬜ | 12–13. Holdout, write-up | Next |

The scorer was built **before** the matcher on purpose. Building it second means spending
Phase 3 eyeballing CSVs to decide whether a change helped; building it first turns every
later change into a number that moves. Phase 4 is where that paid: turning on `--fees`
alone would have sat at 100/100/0 while modelling nothing — the join keys on `net_paise`
and the bank credit is *derived* from the net, so fees wedge gross against net without
disturbing the match. No number would have moved to say the fee model was missing. Gating
on the residual first made `--fees` fail loudly instead.

### What Tier 1 does, and what it refuses to do

One strategy: index settlements by amount, then require the credit's `value_date` to be
within `--window` **business** days of the settlement's `settled_on` — ±0 in clean mode,
±1 once `--settlement-delay` introduces the bank posting lag. Exactly one candidate
resolves; two or more is `AMBIGUOUS_DUPLICATE_AMOUNT`; none is `NO_CANDIDATE`.
Candidates are **counted**, never taken first — "a candidate exists" and "exactly one
candidate exists" are different facts, and Phase 5's subset-sum depends on the
distinction.

Three shortcuts were available on this data and all three are deliberately declined:

- **The UTR tail resolves 60/60 on its own** — sixty distinct tails for sixty
  settlements. Joining on it would score 100% while the amount arithmetic was never
  exercised, and would *stay* at 100% through Phase 4 with no fee model ever written. The
  tail is recorded as corroboration in each verdict's `note` and nothing branches on it.
- **The nearest-date tie-break was refuted, not merely untested.** Phase 3 shipped it
  unit-tested at an artificially wide window, where it could not fire on real data. Phase 4's
  posting lag made it fire — and it was wrong in the direction that costs most. Every true
  (settlement, credit) pair sits at distance **+1**, measured across 5,040 pairs; the
  tie-break keeps the **minimum**, so whenever a same-day settlement shared an amount with
  the true one it preferred the impostor. At n=1000 that was 5–10 wrong matches per seed,
  while **coverage stayed at ~99.5% throughout — only correctness moved.** The generalisable
  part: at a constant non-zero lag, the closest candidate is the *least* likely one, so a
  proximity tie-break is wrong every time it fires rather than occasionally. Tier 1 now
  abstains on a multi-candidate pool instead, and a legitimate successor (infer the modal lag
  from rows that resolved unambiguously) is recorded as a Phase 5 candidate rather than
  smuggled in — it reads the lag off the inputs, so it would not leak, but it is a fitted
  parameter and Phase 4 is exact arithmetic.
- **A bare `net_paise` is unique at n=60** — and collides 1–2 times at n=200 and 42–64
  times at n=1000. So the key is the *pair*, the date does real work even at a ±0 window,
  and a wider window would be actively harmful rather than merely unnecessary.

The residual (`credit − Σ gross of matched payments`) is **computed** in Phase 3 and
**gates** from Phase 4 on. It is zero on every row in clean mode; computing it anyway meant
the fee model moved a number that already existed rather than introducing one.

Since Phase 4 a resolved row must also *publish* its arithmetic — gross, fee, GST and the
credit amount — so `residual == credit − expected` is re-runnable by a reader instead of
asserted. A match is refused outright when no declared rule closes the gap exactly, with no
tolerance band: `UNEXPLAINED_RESIDUAL` is an exception, not a match. The scorer then compares
those terms against the answer key's own decomposition **term by term, never on the total**,
because the total is forced to agree whenever the gross does — a fee 307p too high against a
GST 307p too low closes the identical gap and would otherwise score as correct. That third
axis (`decomposition_agreement`) is reported **separately** from coverage and correctness,
for the same reason those two are never averaged: 100% coverage over rows whose arithmetic
was never checked is the number this project exists to avoid printing.

**The rates the matcher subtracts are its own**, declared in `hisaab/matcher/fees.py` and
deliberately *not* imported from the generator — `tools/check_isolation.py` forbids that
import. Nothing reads `settlements.csv`'s `fee_paise` column either, though the loader parses
it: trusting a declared number is not explaining a gap, and subtracting the stated fee would
close every residual the instant `--fees` populated it. Two of those rates turned out to be
wrong when checked against Razorpay's published pricing, which is the failure mode the
separation exists to expose — see [ASSUMPTIONS.md](ASSUMPTIONS.md) #5–#9.

### The difficulty dial

The generator has thirteen mess flags, all defaulting to off. Clean mode is all of them
off, and it stays in the test set permanently as the regression check: *if it cannot hit
100% here, the code is broken.*

```bash
python -m hisaab.generator --help     # the flags, in difficulty order
```

**Eight are now live**, in difficulty order: `--fees`, `--settlement-delay`, `--dup-amounts`,
`--batching`, `--settlement-report-late`, `--tds`, `--netted-refunds` and `--reserve`. The
remaining five are declared and **refused by name** rather than silently ignored — passing one
exits non-zero instead of returning unchanged data labelled as having fees, which is how
Phase 1 originally behaved and is a quietly dangerous default for a tool whose whole output is
a claim about data.

Turning them all on at once is what gate 13 does, and it is the only thing in the suite that
does: three defects had been sitting in code that every individual flag's own tests called
green, because they lived in the interaction rather than in any one flag.

Each invariant also declares *which* flags it survives, rather than standing down for any of
the thirteen. The earlier design gated on "clean mode", meaning all thirteen off — so
`--fees`, which changes no cardinality and no date, switched off the cardinality, membership
and uniqueness checks along with everything else. The mislabelled run then lost exactly the
checks that would have caught the mislabelling. A suspended check is now named in
`run_manifest.json` rather than vanishing.

Here is the honest run — both flags on, at a size where the data is hard enough to produce a
genuine ambiguity:

```
Records processed          200 bank rows (200 gateway, 0 non-gateway)
Run                        seed 3, 2026-08, mess[fees,settlement_delay], flags: fees,settlement_delay
Matcher                    tier1+2@0.5.0

Coverage                   199/200  (99.5%)   how often it committed
Correctness                199/199  (100.0%)   how often it was right
Arithmetic proved          199/199  (100.0%)   linked right and priced right
Wrong matches              0   (0 of them on planted-unresolvable rows)
Missed                     1   resolvable, but it abstained

Exceptions                 1
Value in exceptions        ₹4,178.99
Est. human time to clear   8 min   (8 min exceptions + 0 min dismissals)
Same batch by hand         6 h 53 min   (199 on sight x 2 min + 1 chased x 15 min)
Time saved                 98.1%   (6 h 45 min less than by hand)
Break-even chased rate     none -- saving is unconditional

Exception queue (1, by value)

  C0005         ₹4,178.99  AMBIGUOUS_DUPLICATE_AMOUNT   ~8 min
```

**That 99.5% is the number this project is arguing for, and it is not a shortfall.** Two
settlements on that run genuinely share a net amount, and once the window opens far enough to
admit the bank posting lag, both are candidates for credit C0005. Nothing in the inputs
separates them. The matcher says so and hands a human one row worth ₹4,178.99.

The tempting fix is a tie-break on date proximity, and it is a trap — see below. The true
settlement sits at **+1** business day (the posting lag); the impostor sits at **+0**,
same-day. "Nearest wins" picks the impostor, and because the lag is *constant* it does so
every single time it fires. It would have turned this honest 99.5%/100% into a
100%/99.5% — the same headline, one silently corrupted book.

---

## Three design commitments

These are the parts a judge is most likely to probe, so they are stated rather than
buried.

### 1. The matcher cannot read the answer key

`truth.json` feeds the scorer and nothing else. That is enforced structurally, not
promised:

- It is written to a **separate directory** from the CSVs.
- It is read only through `hisaab/scoring/truth_io.py`, in a package the matching path
  does not import.
- `tools/check_isolation.py` **fails the build** if anything on the matching path imports
  that package or names the truth file in executable code. It scans `hisaab/matcher/`
  statically, so it cannot be defeated by a module that behaves differently when imported.

A sixth check closes a hole the first five left open: **nothing on the matching path may
import `hisaab.generator` either.** That package knows the fee rates, the T+n settlement
cycle and the narration templates — everything the matcher is supposed to infer — so
importing it is reading the answer with extra steps, and the truth-file checks would pass
it silently. Verified by a deliberate violation, not just asserted.

That rule fixes where shared code lives, and the distinction is deliberate: **logic is
shared through `hisaab/common/`, schemas are duplicated on purpose.** The business-day
calendar is shared, because two calendars disagreeing by one day produce a plausible wrong
answer that nothing detects. The five CSV header tuples are copied into the matcher's
loader, because a drifted schema must fail loudly instead of hiding behind a shared symbol.

The failure mode this prevents is silent and total: a matcher that reads the answer key
still produces a match rate, an exception list, and a confident-looking report. Nothing
crashes. The submission is simply void, and the only way to know is to have checked.

The same discipline applies inside the scorer. `verdict_io.py` — the module that validates
the matcher's output — is deliberately **not** on the truth allowlist: it receives the
credit IDs, seed and month as plain values, so the one scoring job that has no business
seeing the answers structurally cannot.

### 2. The matching engine is deterministic, on purpose

Integer paise throughout, no floats, **no LLM on the match path**. The model's job is
downstream of every decision: explaining a match, triaging exceptions, parsing bank
narration, answering questions.

This is a deliberate answer to "where's the AI?", not an omission. A reconciliation
figure has to be reproducible and defensible — the same seed must give the same number,
and every match must be explainable as arithmetic. A hallucinated journal entry is worse
than an unresolved one.

**Until Phase 10 that claim had no assertion behind it.** `check_isolation.py` had seven
checks and not one of them mentioned a network, so `matcher/tier1.py` could have called a
model mid-match and all sixteen gates would have passed green — the headline design claim,
enforced by nothing but the author's memory. Phase 10 added **check 8**: nothing under
`hisaab/` may import an HTTP client, a model SDK, or `subprocess`/`importlib`/`ctypes`
(29 banned names, resolved by AST, measured at **0 violations across all 42 files** before
it was armed, because a check that fails on arrival gets weakened rather than obeyed).

Why this matters more than it sounds: a model asked "does this credit match this
settlement?" returns a *plausible* verdict, so coverage and correctness still compute and
still look reasonable — while measuring a guess. It is the `truth.json` failure reached
from the opposite direction. Instead of leaking the answer in, it invents one.

The scope is deliberately wider than the matcher. `hisaab/common/` is imported *by* the
matching path, so a check scoped to `hisaab/matcher/` would have passed a network import
sitting one directory over — exactly the mistake check 6 made until Phase 9, when it was
scoped to a tuple that did not yet include `hisaab/triage/`. Once is a lesson; twice would
be a habit.

`hisaab/explain/` — the model layer — is exempt from the network half and from nothing
else. It stays on the same list as the matcher, so **the one component that talks to an LLM
is the one component with no privileged information at all**: it cannot read `truth.json`,
cannot import the generator, and cannot read the declared fee columns. It explains rows
from the same inputs a human reconciler would have. And nothing that ships may import it,
so the matcher cannot reach a model through the one tree allowed to hold one — without
that second half, the ban would be about spelling rather than behaviour.

Proven by deliberate violation, the way check 6 was: **seven planted mutants, each refused
by its own assertion** and named in the failure message, plus two controls that must stay
green (a `urllib` import inside `hisaab/explain/`, and that package importing its own
siblings). Two of the seven would have passed a narrower check — the `hisaab/common/`
import, and `from ..explain import ask` written inside the matcher.

### 3. Nothing is dropped, and the arithmetic says so

`matched + exceptions = total`, exactly. Enforced in two places: the scorer refuses a
verdict file that misses, duplicates or invents a bank row (naming the offending IDs),
and re-asserts the identity before computing anything. A matcher that omits rows cannot
score well by dropping the hard ones — it does not score at all.

Every assumption behind every number is in **[ASSUMPTIONS.md](ASSUMPTIONS.md)**, marked
🔴 unverified / 🟡 declared / 🟢 guaranteed. The status column is there to make a
correction cheap rather than embarrassing, and Phase 4 collected on that: the fee rates
were checked against Razorpay's published pricing and **two of the five were wrong**.
Netbanking is not priced below cards, and UPI is not free — zero *MDR* is not zero *fee*,
because the 2% platform fee still applies on the standard payment-gateway rail. Both are
now 🟢 with the source and date recorded; TDS stays 🔴, since it is tax withholding rather
than gateway pricing and the pricing page does not cover it.

The human-time estimates remain 🔴 and say so. **They also collected the largest correction
in this project so far, and it was the tool's own claim about itself.** Through Phase 8 the
metric block printed the tool's minutes beside a by-hand total and never subtracted them —
and on all six measured cells the by-hand figure was the *smaller* one, so the report claimed
a saving while the tool cost an operator 2–3× more time than ignoring it. Fifteen gates missed
it because no assertion put the two numbers on opposite sides of a comparison; each side was
individually right. The block now prints both totals, states the subtraction, and prints the
chased rate at which they cross — **13.34 min on the binding cell against an assumed 15**, so
a 1.66-minute margin, measured at 7.09–13.34 across three seeds × two sizes.

Two further versions of that line were wrong before one was checkable, and both were found by
regenerating this README's own output rather than by a test: a *negative* break-even printed
as a rate ("below −390 min, by hand is cheaper" — true of the arithmetic, meaningless as a
claim), and — the one worth reading — **a wrong match is invisible to a comparison built on
queue minutes.** It raises no exception, so it costs the tool side nothing: the `zip` fixture,
which matches by row position at 35% correctness with 39 wrong matches, printed `Time saved
100.0%`. The best possible figure, earned by getting rows silently wrong. The claim is now
withheld entirely on any run with a wrong match, rather than qualified — a percentage beside a
caveat is still the number a reader takes away — and no remediation rate was invented to
charge those rows with, because nobody was timed for finding a mis-booked entry months later.

---

## Layout

```
hisaab/
  common/         shared by both sides: money, IDs, reason codes, the verdict contract
  generator/      synthetic payments → settlements → bank statement, plus truth.json
  scoring/        reads truth + a matcher's verdicts, prints the metric block
  matcher/        load → normalize → block → fees → tier1 → engine; reads data/, never truth/
tools/
  acceptance.py       every gate, one command
  fixtures.py         four known-answer matchers (none of them a real matcher)
  check_isolation.py  gate 5: the answer key is unreachable from the matching path
  repro_check.py      byte-identical output at a fixed seed
  verify_output.py    re-reads the written files with an independent parser
```

`hisaab/common/verdict.py` is the interface between the two halves — the matcher writes
that format, the scorer reads it. It lives in `common/` because it touches no truth, so
being importable from the matching path costs nothing.

Every module has a `__main__` self-check (`python -m hisaab.common.money`), and gate 0
runs all of them in dependency order so the first failure is the deepest one.

---

## Reproducibility

The same `--seed` and `--month` produce byte-identical files, verified across two
processes with different hash seeds rather than assumed. Randomness is one named
substream per concern, so turning on a mess flag does not shift the data the other flags
produced — measured in Phase 4, not just intended: the payment gross and method vectors are
identical across clean, `--fees` and `--settlement-delay` at the same seed.

That property is load-bearing for more than tidiness. It is what makes a *comparison* across
two runs meaningful — "fees changed this many collisions" is only a statement about fees if
both runs contain the same payments. The isolation is per *flag*, though, not per config
value: changing the payment-method mix itself does move the stream, and did, which is why
`data/` and `truth/` are regenerated rather than edited.

**Seeds 1–5 are development seeds. Seed 99 is the holdout** and is not run until Phase 12.
Tuning a tolerance after seeing what it does to your number is the reconciliation
equivalent of test-set leakage, and it is detectable when a judge asks how you picked it.

Wall clock is the one non-deterministic value, and it is quarantined inside a `timing`
object in every document that carries it — so two runs of the same input differ only
there, and the metric block stays byte-comparable.
