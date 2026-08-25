# Assumptions

Every number in this project that is not derived from the data is listed here, with
whether it is verified, assumed, or arbitrary-but-declared. Started in Phase 1,
finished in Phase 13; the write-up quotes this file rather than restating it.

The reason it exists this early: "state your assumptions, don't bluff them" is a
submission requirement, and the moment to write down a rate is when you pick it,
not when a judge asks. `tools/acceptance.py` gate 7 fails the build if a required
topic goes missing from this file.

**Status key** — 🔴 unverified assumption, needs checking before the final run ·
🟡 arbitrary but declared and frozen · 🟢 verified or structurally guaranteed

---

## Money and arithmetic

| # | Assumption | Status |
|---|---|---|
| 1 | **All money is integer paise.** No floats anywhere, at any point, including intermediate values. Enforced at the boundary by `money.paise()`, which rejects `float` and `bool`, and re-checked after the CSV round-trip by invariant I5. | 🟢 |
| 2 | **Rates are integer basis points**, never percentages as floats. 2% is `200`, 18% is `1800`. | 🟢 |
| 3 | **Rounding is half-up at the paisa**, computed as `(amount × bps + 5000) // 10000`. Exact integer arithmetic — no `Decimal` context to misconfigure. Half-up rather than banker's rounding because it is what invoice arithmetic conventionally uses and because it is the rule a human checking the sum by hand would apply. | 🟡 |
| 4 | Rounding is defined for **non-negative operands only**. Half-up on a negative is ambiguous, every rate in this system applies to a positive amount, and `mul_bps` asserts rather than guessing. | 🟢 |

The worked example from the track spec lands correctly under this rule: ₹1,111 at
200 bps is a ₹22.22 fee, and 1800 bps GST on that fee is 399.96 paise → 400.

## Fee model — the numbers most likely to be wrong

Declared in `config.FeeConfig` and **not used in Phase 1** (clean mode has zero
fees, not "fees applied consistently"). They exist now so the rates are a stated
decision rather than something invented under pressure in Phase 4.

| # | Assumption | Status |
|---|---|---|
| 5 | Card and wallet: **200 bps** (2%) | 🔴 |
| 6 | Netbanking: **190 bps** (1.9%) | 🔴 |
| 7 | UPI: **0 bps** | 🔴 |
| 8 | GST: **1800 bps** (18%), charged **on the fee**, not on the gross | 🔴 |
| 9 | TDS: **100 bps** (1%), only where `--tds` is on | 🔴 |

**All five need verifying against Razorpay's current published pricing before the
final run.** Real rates vary by payment method and by plan, and the numbers above
are plausible-looking rather than checked. Getting a fee rate subtly wrong in front
of a payments company is a bad failure mode, so the honest position is to state
them as assumptions and say so out loud — which is also what the track's own
checklist asks for.

## Dates and the calendar

| # | Assumption | Status |
|---|---|---|
| 10 | **Business days are Monday–Friday.** The holiday set is **empty**. | 🟡 |
| 11 | Payment capture happens on a business day, between **09:00 and 21:00 IST**. | 🟡 |
| 12 | **IST is a fixed UTC+05:30 offset.** Implemented with `timezone(timedelta(hours=5, minutes=30))`, deliberately not `zoneinfo.ZoneInfo("Asia/Kolkata")` — bare Windows Python ships no tzdata. IST has never observed DST, so the fixed offset is exact, not an approximation. | 🟢 |
| 13 | `settled_on` and `value_date` are **dates**; `captured_at` is an **ISO-8601 UTC timestamp** with a trailing `Z`. This mirrors how real exports differ from each other. | 🟢 |
| 14 | The month generated comes from `--month`, never from `date.today()`. | 🟢 |
| 15 | **T+0 in clean mode.** `--settlement-delay` introduces T+n in Phase 4. | 🟢 |
| 16 | T+n rolls forward off non-business days: money captured on a Saturday settles on the following business day, never on the Saturday. | 🟡 |

On #10: a real Indian bank calendar has gazetted holidays that vary by state, and
inventing a list would be bluffing. `BusinessCalendar` takes an injectable holiday
set, so a real one drops in without a code change if it becomes worth having.

On #11: the clamp is load-bearing, not cosmetic. `captured_at` is UTC and
`settled_on` is a date, so a 00:30 IST capture serialises to the *previous* UTC day
— and Phase 4 would then see a one-day gap and blame its own delay model for a
discrepancy the timezone created. Within 09:00–21:00 IST the UTC date and the IST
date are always equal. Phase 4 may relax this, but deliberately.

## Identifiers and file shapes

| # | Assumption | Status |
|---|---|---|
| 17 | **IDs are fixed-width 4**: `pay_0001`, `ord_0001`, `setl_0001`, `rfnd_0001`, `C0001`. Appendix A of the spec mixes `setl_01` and `C001` widths; those stop sorting lexicographically past 99 records, so one width is picked and never revisited. | 🟡 |
| 18 | **The bank statement has exactly four columns**: `row_id`, `value_date`, `amount_paise`, `narration`. No gateway identifier, no record count, no gross amount. Header pinned by invariant I9. | 🟢 |
| 19 | Bank narration comes in **four styles**, varying by channel and format. Real statements are inconsistent across banks; a single format would make the Phase 3 parser one regex and leave Phase 10's narration-parsing job with nothing to do. Style variance changes no amount and no date, so clean mode stays clean where it counts. | 🟡 |
| 20 | **UTR is present on every settlement row in Phase 1** and appears as a 4-digit tail in the narration. It is the one legitimate link between the gateway and the bank, and it is realistic — a UTR really does appear in both places. `--utr-patchy` makes it missing or truncated on some rows in Phase 8. | 🟡 |
| 21 | `settlement_items.csv` is a **fifth file**, declaring which payments each settlement contains. Appendix A gives `settlements.csv` no membership column, but Tier 2 needs the settlement report to declare its contents, and real gateway reports do have a line-item export. It links payments↔settlements only; the settlements↔bank link stays the hard part. This is what `--settlement-report-late` withholds. | 🟡 |
| 22 | `refunds.csv` is written **header-only** in clean mode. An empty file that exists is better than a missing one — the loader gets exercised from Phase 3 rather than blowing up in Phase 6. | 🟢 |

## Matching (Phase 3+) — declared now, frozen before the final run

Recorded here in Phase 1 so they are chosen on the merits rather than after seeing
a score. Tuning a tolerance once you know what it does to your number is the
reconciliation equivalent of test-set leakage, and it is detectable when a judge
asks how you picked it.

| # | Assumption | Status |
|---|---|---|
| 23 | **Tier 3 tolerance: ±50 paise and ±2 business days**, with a scoring margin over the runner-up of at least 2×. Both conditions required — "unique best" without a margin means choosing between two near-identical candidates on noise. | 🟡 pending Phase 3 |
| 24 | **Clean mode must be 100% resolvable** from date + amount. Invariant I3 asserts no two credits share a `(date, amount)` pair, because such a pair is genuinely indistinguishable and the honest verdict would be an abstention. That case is real and `--dup-amounts` plants it deliberately in Phase 8. | 🟢 |
| 25 | A match counts only if its **decomposition closes to zero paise**. Matched-but-unproven is an exception (`UNEXPLAINED_RESIDUAL`), not a match. | 🟡 pending Phase 4 |

## Reproducibility

| # | Assumption | Status |
|---|---|---|
| 26 | The same `--seed` and `--month` produce **byte-identical** files. Verified across two processes with different `PYTHONHASHSEED` values by `tools/repro_check.py`, not assumed. | 🟢 |
| 27 | Randomness is **one named substream per concern**, seeded `f"{seed}:{name}"`. `random.Random` hashes a string seed with SHA-512 internally, so it is stable across processes and platforms; the builtin `hash()` is not, and is never used for this. | 🟢 |
| 28 | **Seeds 1–5 are development seeds. Seed 99 is the holdout** and is not run until Phase 12. The reported numbers come from the holdout. | 🟡 |
| 29 | Files are written with `newline=""` and `lineterminator="\n"`, so output is byte-identical across Windows and POSIX. Asserted by a no-CR check on the written bytes. | 🟢 |

## Scoring (Phase 2) — what the reported numbers mean

The definitions behind every figure in the metric block. Judge question #1 in §21 of
the spec is *"what exactly counts as a match?"*, and these rows are the answer.

| # | Assumption | Status |
|---|---|---|
| 30 | **The unit of account is the bank credit**, not the payment. Coverage is resolved gateway credits ÷ total gateway credits. Phase 1 has 60 payments = 60 settlements = 60 credits, so every candidate denominator looks identical; Phase 5 makes many payments settle as one credit, and the choice then decides what the headline means. Payment-side coverage is a secondary metric, never the headline. | 🟡 |
| 31 | **Correctness is set equality on `payment_ids`, with no partial credit.** A match containing four of a batch's five payments is wrong, not 80% right. Partial credit would let a matcher that guesses broadly outscore one that abstains honestly, which is backwards for finance and backwards for this track. | 🟡 |
| 32 | **A right answer on a row planted as unresolvable still counts as a wrong match.** `--dup-amounts` (Phase 8) plants two credits sharing a date and an amount: the inputs cannot separate them, so a matcher that commits has even odds of being right by luck. Crediting that would reward guessing over abstaining. Counted in its own cell (`lucky_guess`), which also makes a non-zero count the cheapest available leak detector. | 🟡 |
| 33 | **Non-gateway rows are excluded from the headline** and reported as their own precision/recall pair. Correctly ignoring noise is the easiest row in the file; counting it as coverage would inflate the rate. | 🟡 |
| 34 | **Effort estimate: minutes per exception, by reason code** — 3 min (`NON_GATEWAY_CREDIT`), 5 (`PARTIAL_SETTLEMENT_PENDING`, `ROUNDING_DRIFT`), 8 (`AMBIGUOUS_DUPLICATE_AMOUNT`), 10 (`NO_CANDIDATE`, `REFUND_UNLINKED`), 12 (`UNEXPLAINED_RESIDUAL`), 15 (`AMBIGUOUS_MULTI_SUBSET`, `CREDIT_MISSING`, `SETTLEMENT_MISSING`), 20 (`FX_RATE_GAP`). Declared in `metrics.MINUTES_PER_EXCEPTION`, which asserts the table covers every reason code so a new code cannot silently price at zero. **These are estimates, not measurements** — nobody was timed. Phase 9 refines them into per-group figures. | 🔴 |
| 35 | **By-hand baseline: 2 minutes per bank row** — open the statement, find the settlement, tick it off. 60 rows ≈ 2 hours, which is the "vs ~2 h by hand" comparison in the metric block. Declared in `report.BY_HAND_MINUTES_PER_ROW`. Also an estimate; it is the denominator of any time-saved claim, so it is stated rather than implied. | 🔴 |
| 36 | **A low score exits 0.** The scorer's exit code reports whether a number could be produced, never whether the number is good: 0 scored, 1 the verdict file could not be trusted, 2 bad usage. A scorer that failed on a bad score could not measure a bad matcher, which is the matcher it will spend the most time measuring. | 🟢 |
| 37 | **Wall clock is excluded from the reproducibility comparison.** The metric JSON confines it to a `timing` object, as `run_manifest.json` does, so two runs of one matcher on one seed differ only inside that object. The human-readable block prints it freely. | 🟢 |

On #34 and #35: both feed the only figure in the submission that is neither measured
nor derived — "est. human time to clear". The honest position is that the shape of the
claim (exceptions are a small fraction of the batch) is robust, while the absolute
minutes are not. State them; do not defend them.

**The break-even the two assumptions imply, which the write-up must not walk past.** An
exception costs more per row than a routine tick-off, because the easy path already
failed. At 10 min per exception against a 2 min-per-row baseline, the tool saves time
only while exceptions stay **under 20% of bank rows** — below that the comparison
flatters, above it the honest reading is that a human would have been faster. The stub
fixture makes this visible on purpose: 60 exceptions score `10 h 00 min` against
`~2 h 00 min` by hand, which is the metric block correctly reporting that a matcher
which resolves nothing is worse than useless. Phase 13 should quote the ratio, not just
the minutes.

## Structural guarantees, not assumptions

Listed for completeness, because a judge is more likely to ask about these than
about the fee rates.

- `truth.json` is written to a **separate directory** from the CSVs, is read only
  through `hisaab/scoring/truth_io.py`, and `tools/check_isolation.py` fails the
  build if anything on the matching path can reach it.
- **Matched + exceptions = total, exactly.** No record is dropped. Enforced since
  Phase 2 in two places: `verdict_io.reconcile` refuses a verdict file that misses,
  duplicates or invents a bank row, naming the offending IDs, and `metrics.score`
  re-asserts `resolved + ignored + exceptions == total` before computing anything. A
  matcher that omits rows cannot score well by dropping the hard ones — it does not
  score at all.
- **The scorer reads the answer key; the validator does not.** `verdict_io.py` checks
  the matcher's output against plain values — the credit IDs to cover, the seed, the
  month — and imports nothing that could reach `truth.json`. It is the one scoring
  module deliberately absent from `check_isolation.py`'s allowlist.
- The matching engine is **deterministic by design**. No LLM sits on the match
  path; the model explains, triages, parses narration and answers Q&A, all
  downstream of every decision. Reproducibility and defensibility are the reason.

---

## Open — resolve before the final run

- [ ] Verify fee rates #5–#9 against Razorpay's current published pricing (Phase 4)
- [ ] Confirm the T+n cycle in #15/#16 is a defensible default and state that it varies (Phase 4)
- [ ] Freeze #23's tolerances in code and stop touching them (Phase 3)
- [ ] Decide whether a real holiday calendar is worth adding to #10 (Phase 4)
- [ ] Replace #34's per-code minutes with per-group estimates once exceptions are
      ranked (Phase 9), and decide whether #35's baseline is worth timing for real
