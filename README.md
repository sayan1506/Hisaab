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

Python 3.12. **No dependencies** — standard library only, no virtualenv needed.

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
Matcher                    tier1@0.3.0
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
Est. human time to clear   0 min   (vs ~2 h 00 min for the batch by hand)

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

### Verify the whole thing

```bash
python -m hisaab.generator --seed 42 --n 60   # if you have not already
python tools/acceptance.py
```

Eleven gates, one command, exit code is the verdict. Byte-identical output at a fixed
seed across two processes; invariants on three seeds in memory and again re-read from
disk; the leak audit; truth isolation; throughput; the assumptions file; the four
known-answer fixtures; the matcher at 100/100/0 across three seeds × two sizes, including
the check that blanking every bank narration changes no decision; gate 10, which turns
on the first two rows of the difficulty dial and proves the arithmetic per row; and gate
11, which plants pairs that genuinely cannot be resolved and reads the abstention count.

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

---

## Current state — Phase 4b of 13

| | Phase | State |
|---|---|---|
| ✅ | **1. Generator, clean mode** | Done. Three CSVs plus an answer key, 1:1 exact matches |
| ✅ | **2. Scoring harness** | Done. Coverage/correctness, the exception queue, four known-answer fixtures |
| ✅ | **3. Matcher, Tier 1** | Done. Exact `(value_date, net_paise)` join, 100/100/0 on clean mode |
| ✅ | **4. Fees and the settlement delay** | Done. The residual **gates**, the decomposition is published per row and checked term by term, the window is proved load-bearing |
| ✅ | **4b. Planted unresolvables (`--dup-amounts`)** | Done. Each pair shares a date, an amount **and a UTR** — the last one because a tail-only join resolved 100% without either of the others |
| ⬜ | 5–8. Batching, adjustments, orphans, FX | Next. The rest of the difficulty dial, one flag at a time |
| ⬜ | 9–13. Exception ranking, LLM layer, HTML report, holdout, write-up | |

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

**Two are now live: `--fees` and `--settlement-delay`.** The remaining eleven are declared
and **refused by name** rather than silently ignored — passing one exits non-zero instead of
returning unchanged data labelled as having fees, which is how Phase 1 originally behaved
and is a quietly dangerous default for a tool whose whole output is a claim about data.

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
Matcher                    tier1@0.3.0

Coverage                   199/200  (99.5%)   how often it committed
Correctness                199/199  (100.0%)   how often it was right
Arithmetic proved          199/199  (100.0%)   linked right and priced right
Wrong matches              0   (0 of them on planted-unresolvable rows)
Missed                     1   resolvable, but it abstained

Exceptions                 1
Value in exceptions        ₹4,178.99
Est. human time to clear   8 min   (vs ~6 h 40 min for the batch by hand)

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

The human-time estimates remain 🔴 and say so, including the break-even ratio at which the
comparison stops flattering: at ~10 min per exception against a 2 min-per-row baseline, the
tool only saves time while exceptions stay under **20%** of bank rows.

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
