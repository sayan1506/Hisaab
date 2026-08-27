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

Declared in `config.FeeConfig`, re-declared independently in `matcher/fees.py`, and
**not used in Phase 1** (clean mode has zero fees, not "fees applied consistently").

**Checked against <https://razorpay.com/pricing/> on 2026-08-26, and two of the five
were wrong.** The whole point of the status column is to make this moment cheap, so
the corrections are recorded in place rather than quietly overwritten.

| # | Assumption | Status |
|---|---|---|
| 5 | Card and wallet: **200 bps** (2%) | 🟢 verified |
| 6 | Netbanking: **200 bps** (2%) — **corrected from 190.** The guess was that netbanking is priced below cards; the page prices every standard domestic instrument at one flat 2%. | 🟢 verified |
| 7 | UPI: **200 bps** (2%) — **corrected from 0.** UPI carries **zero MDR** by mandate, which is true and widely repeated, but Razorpay's own **2% platform fee still applies** on the standard payment-gateway rail. Zero MDR and zero fee are different claims and the original entry conflated them. | 🟢 verified |
| 7a | POS UPI and RuPay debit: **0 bps** (0.00%) — the zero-rated rail, on a POS terminal rather than the PG rail. | 🟢 verified |
| 7b | Corporate / business credit cards: **215 bps** (2.15%) | 🟢 verified |
| 7c | International cards: **300 bps** (up to 3%), by card **origin**, still settled in INR. Currency conversion is a separate concern — `--fx` is Phase 6. | 🟢 verified |
| 8 | GST: **1800 bps** (18%), charged **on the fee**, not on the gross | 🟢 verified |
| 9 | TDS: **100 bps** (1%), only where `--tds` is on | 🔴 |

**Which of these is a "verified fact" and which is still an assumption.** The rates
are sourced and dated, not permanently true: these are **list prices**, real rates are
negotiated at volume, and Indian MDR structures move with NPCI/RBI policy. That is why
every rate is overridable from the matcher's command line (`--fee-bps METHOD=BPS`)
rather than compiled in. #9 stays 🔴 on purpose — TDS is tax withholding under §194-O,
not gateway pricing, so the pricing page does not cover it and nothing about it was
confirmed. It is unused until Phase 6 turns `--tds` on (D7), so nothing depends on it yet.

**Why #7 was the expensive one, and what it was costing the evidence.** A wrong rate
that makes a fee *smaller* is not a bookkeeping slip; it removes rows from the test.
A zero-rated payment settles at its gross, so its residual is zero even under `--fees`
and it resolves with no fee model at all. UPI at 0 bps was **36% of rows** on a `--fees`
run — over a third of the sample was proving nothing about the fee model while counting
toward a coverage figure that looked like it did. With the free rail moved to `pos_upi`
that share is **~6%**, which makes `--fees` a materially sharper test than the number
alone suggests. This is also why the claim "the fee model moved the coverage number" has
to be measured **per method**: in aggregate, the zero-rated rows silently carry it.

The zero-rated branch was kept rather than deleted — `story.py` and `matcher/fees.py`
both take it, and a rule named `no deduction` is what distinguishes *nothing was
withheld* from *nothing was checked*. It is now exercised by something the page
actually confirms. Note that neither module names a method: the branch is reached by
**rate**, which is the only reason correcting a rate was a small change here.

## Dates and the calendar

| # | Assumption | Status |
|---|---|---|
| 10 | **Business days are Monday–Friday.** The holiday set is **empty**. | 🟡 |
| 11 | Payment capture happens on a business day, between **09:00 and 21:00 IST**. | 🟡 |
| 12 | **IST is a fixed UTC+05:30 offset.** Implemented with `timezone(timedelta(hours=5, minutes=30))`, deliberately not `zoneinfo.ZoneInfo("Asia/Kolkata")` — bare Windows Python ships no tzdata. IST has never observed DST, so the fixed offset is exact, not an approximation. | 🟢 |
| 13 | `settled_on` and `value_date` are **dates**; `captured_at` is an **ISO-8601 UTC timestamp** with a trailing `Z`. This mirrors how real exports differ from each other. | 🟢 |
| 14 | The month generated comes from `--month`, never from `date.today()`. | 🟢 |
| 15 | **T+0 in clean mode.** `--settlement-delay` introduces T+n, implemented in Phase 4. | 🟢 |
| 15a | **`--settlement-delay` turns on two different delays, and conflating them is the mistake to avoid.** `settlement_delay_days` (**T+2**) is capture → `settled_on`, the gateway's own settlement cycle. `posting_lag_days` (**1 business day**) is `settled_on` → the bank credit's `value_date`, the statement posting lag. Both are separately overridable and both default to 0 with the flag off. | 🟢 |
| 15b | **Only the posting lag is visible to the matcher's date window.** The bank credit is derived from the settlement, so a matcher joining credits to settlements sees `posting_lag_days` and never `settlement_delay_days` — which is why `--window 0` fails under a non-zero posting lag while T+2 alone costs nothing. Recorded because "T+2" is the number a reader expects the window to need, and it is the wrong one. | 🟢 |
| 16 | T+n rolls forward off non-business days: money captured on a Saturday settles on the following business day, never on the Saturday. Applied to **both** delays in 15a, so a Friday capture at T+2 with a 1-day lag lands the following Wednesday, not Sunday/Monday. | 🟡 |

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
| 23 | **Tier 3 tolerance: ±50 paise and ±2 business days**, with a scoring margin over the runner-up of at least 2×. Both conditions required — "unique best" without a margin means choosing between two near-identical candidates on noise. | 🟡 pending Tier 3 |
| 23a | **Tier 1 runs at ±0 paise and ±0 business days** — an exact join, not a tolerance. Phase 3 shipped the *interface* for #23 (`--window`, `--max-adjustment`, both defaulting to 0) and plumbed it to the CLI, so widening either is a parameter change rather than a rewrite. The #23 values themselves are still unused: no code reads them, and Tier 3 does not exist yet. Measured across seeds 1/2/3/42 × n=60/200/1000: every credit's `value_date` equals its settlement's `settled_on`, so ±0 costs nothing on clean mode. | 🟢 |
| 24 | **Clean mode must be 100% resolvable** from date + amount. Invariant I3 asserts no two credits share a `(date, amount)` pair, because such a pair is genuinely indistinguishable and the honest verdict would be an abstention. That case is real and `--dup-amounts` plants it deliberately in **Phase 4b**. | 🟢 |
| 24a | **The `(date, amount)` pair survives `--fees` and `--settlement-delay`, and the margin is measured rather than assumed.** **336 runs**: 12 seeds × {clean, fees, delay, fees+delay} × n=60/200/1000/2000/4000/8000/8968. Zero collisions at every size at or below **n=2000**; the first appears at n=4000 (1 pair), rising to 12 at n=8000 and 17 at the generator's ceiling. The project runs 60–1000, so the margin is **4×**. The guarantee has two different strengths, which is the part worth knowing: `story._unique_amount` nudges on `(capture_date, gross_paise)` while I3 checks `(value_date, amount_paise)`. Under **`--settlement-delay` it is structural** — the capture→value date map is injective on every delayed run, so the delay only *relabels* days, every within-day amount set carries over untouched, and delay-alone reproduces clean mode's count exactly at every size. Injectivity, not equality, is the property that matters: an earlier probe wrongly concluded the delay was unguaranteed from the fact that `capture_date != value_date` on every row, which tests the wrong thing. Under **`--fees` it is empirical**: fees break `amount == gross`, so the nudge no longer covers the checked key, and **every** collision found at n≥4000 is one `--fees` created (clean and delay stay at 0 throughout). Re-measured in step 7 after #6/#7 were corrected, because a rate change moves every number here — including the ones about dates, since `PAYMENT_METHODS` gained three entries and the RNG draw shifted. | 🟢 |
| 24b | **Fees both disperse and concentrate, and which dominates depends on the rate table** — so the original one-directional finding was too strong in the direction it was stated, though still the opposite of what `.plan/phase4.md` (e) predicted. Two channels: equal-gross payments on *different* rates net apart (dispersal), while a priced row's net can land on some other row's amount (concentration). Under the pre-correction table, with UPI at 0 bps against card at 200, dispersal dominated — 60–75% of equal-gross groups broke apart. With four methods now sharing 200 bps that falls to **20–30%**, so fees is a much weaker disperser than first recorded. Method note: the cross-run comparison this rests on is only valid because enabling a flag **does not perturb the payment stream** — gross and method vectors are byte-identical across clean/fees/delay at the same seed, verified rather than assumed, and confirmed a second way by re-deriving the same figures *within* a single run. | 🟢 |
| 24c | **The zero-rated rail is a collision magnet, and this is the channel that will widen.** `_unique_amount` guarantees distinct *gross* within a capture date. A zero-rated row settles **at** its gross, so its credit inherits that protection; a priced row's net is a derived value no invariant has ever compared to anything. So priced nets have many chances to land on a protected value: **47–58%** of collisions at n≥8000 involve a `pos_upi` row, against **11%** if method were irrelevant, and the priced member's gross sits **2.42%–2.60%** above the collision value — exactly the net-to-gross ratio at 200 and 215 bps, not a fitted range. Recorded because Phase 5's batching and Phase 6's TDS both add derived nets to the same channel. | 🟢 |
| 24d | **The generator cannot exceed n=8,968.** UTR tails are 4 digits drawn without replacement, so 9,000 exist and 32 are held back as spares for the credit fixup. Found the unpleasant way while measuring #24a: n=16000 raised `IndexError: list index out of range` from the line that formats a UTR — a message naming neither `n` nor the tail space, from a loop that was not at fault. `_draw_tails` capped its pool at 9,000 while the settlement loop kept indexing to `n`. Now refused at config time with a message that names the limit. **Not** fixed by widening the tail: `XXXX4471` is the shape Appendix A specifies, so the width is a spec constraint, and a unique tail is what makes `normalize.py`'s second independent join possible (`--utr-patchy` breaks it deliberately in Phase 8). | 🟢 |
| 25 | A match counts only if its **decomposition closes to zero paise**. Matched-but-unproven is an exception (`UNEXPLAINED_RESIDUAL`), not a match. Implemented in Phase 4: `tier1.py` refuses to resolve unless a declared rule closes the gap **exactly**, with no tolerance band. The decomposition is *published* per row (gross, fee, GST, and the credit amount), so `residual == credit − expected` is re-runnable by a reader rather than asserted — and the scorer compares the matcher's six terms against truth's own arithmetic. **Term by term, never on the total**, because the total is forced to agree whenever the gross does: a fee 307p too high against a GST 307p too low closes the identical gap and would score correct. Reported as a third axis (`decomposition_agreement`), kept separate from coverage and correctness rather than folded in. | 🟢 |

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
| 32 | **A right answer on a row planted as unresolvable still counts as a wrong match.** `--dup-amounts` (**Phase 4b**) plants two credits sharing a date and an amount: the inputs cannot separate them, so a matcher that commits has even odds of being right by luck. Crediting that would reward guessing over abstaining. Counted in its own cell (`lucky_guess`), which also makes a non-zero count the cheapest available leak detector. | 🟡 |
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
- [ ] Freeze #23's tolerances in code and stop touching them — **deferred to the phase
      that builds Tier 3**, not Phase 3. Phase 3 shipped the interface at ±0/±0 (#23a);
      the tolerance values still have no reader, so there is nothing yet to freeze. The
      discipline the original line was protecting still stands: pick them before seeing
      what they do to the number.
- [ ] Decide whether a real holiday calendar is worth adding to #10 (Phase 4)
- [ ] Replace #34's per-code minutes with per-group estimates once exceptions are
      ranked (Phase 9), and decide whether #35's baseline is worth timing for real
