# Pre-registration: fixing duplicate LLM proposals

Written 2026-11-18, **before** the fixes were implemented or run. It is committed on its own, so the git history
shows that the rules below came first. Any analysis not listed here is reported as *exploratory*.

Background: report Section 6 ("Where the budget goes"), `results/reports/duplicates.md`. GPT-4.1 rejects 52% of
its proposals (43% duplicates, mostly re-derived cells from 3-5 generations earlier that are no longer shown) and
evaluates about 145 architectures per search, against 270 for NSGA-II and regularised evolution.

## 1. Questions

- **Q1 (mechanism).** Do the fixes cut the waste and fill the budget?
- **Q2 (equal evaluations).** With 270 evaluations, the same as the baselines, does the LLM operator find better
  fronts than the non-LLM operators that use the same evaluator?
- **Q3 (low budget).** At 120 evaluations, where an LLM prior should help most, does it?
- **Q4 (the paper's budget).** With the paper's 120 LLM calls, does showing the history alone reduce waste and
  improve the front?

## 2. Arms (fixed now, not tuned later)

Common to all arms: GPT-4.1 (`gpt-4.1`), temperature 0.7, all other settings from `configs/base.yaml` (six
niches, 10 generations, 4 children per call, 6 parents shown, 13-proxy XGBoost predictor with n_train = 1,000,
seed 0), CIFAR-10, six devices (Edge GPU, Raspberry Pi 4, Edge TPU, Pixel 3, Eyeriss, FPGA). One search per
device.

| Arm | Fix 1: niche history in prompt | Fix 2: reject and re-prompt | Budget | LLM calls |
|---|---|---|---|---|
| **A** (main) | yes | yes, at most 5 retries per generate call | 270 evaluations | about 195-270 (reported) |
| **B** | yes | no | 120 calls, as the paper | 120 |
| Reference | no | no | 120 calls | 120 (existing runs, `llmnas-a3a92e57`, not re-run) |

- **Fix 1.** The generate prompt also lists every cell already evaluated in the niche (code only, no
  scores; at most 45 per niche), with the instruction not to propose any of them. The knowledge-base prompt is
  unchanged.
- **Fix 2.** After a generate reply, only the missing children are requested again. The retry message lists each
  rejected cell and the reason (`identical to parent <id>`, `already evaluated`, `repeated in this reply`,
  `breaks niche rule: <rule>`, `no latency on this device`, `not a valid code`). If 5 retries still leave a slot
  short, the slot stays short: nothing is padded with random cells. The knowledge base is updated once per
  niche and generation, as before, not on retries.
- Fix 2 can stop at 270 evaluations before generation 10 ends only if every slot is full; generations are not
  extended beyond 10.

Paid arms run only after explicit approval. Estimated cost: A about $7.50, B about $5.40 (6 searches each);
A revised to about $11, at most about $17 (amendment 1).
The same arms may also be run on the local models (DeepSeek-Coder-V2-16B, Qwen2.5-7B) at no API cost; those
results are secondary.

## 3. Controls

All with the same evaluator, CIFAR-10, six devices, **10 seeds (0-9)** each, so that the seed-to-seed spread is
estimated properly (these runs are free):

- **PEA with refill** (main comparator: the same framework and niches, only the operator differs). Rejected
  random children are regenerated, up to 50 attempts per slot, until each slot has 4 new children. If all 50
  draws are rejected, the slot takes a random unevaluated cell of the niche (amendment 1).
- Random sampling in niches (already never repeats).
- NSGA-II and regularised evolution (already refill to 270).

## 4. Measures

- **Primary:** hypervolume under the default convention (front normalisation, reference point 1.1, archived
  front, no accuracy floor, Edge TPU failed compilations excluded), read from the search trajectory at
  **N = 270** evaluations (Q2) and **N = 120** (Q3), by linear interpolation between generations. A run that
  ends before N is scored at its final value (amendment 1).
- **Secondary:** IGD at the same points; HV with the 85% accuracy floor; the 16-convention robustness table.
- **Process (Q1):** duplicate rate and its sources, out-of-niche rate, evaluations reached, LLM calls,
  input/output tokens, cost, wall time.

## 5. Decision rules

For each device d and control c: `margin(d, c) = max(2 x s.d. of c over its 10 seeds on d, 0.010)`, and
`diff(d, c) = HV(LLM, d) - mean HV(c, d)`. Significance is the device-clustered exact sign-flip test on the six
`diff` values (one-sided; its smallest possible p is 1/64 = 0.016), as in `llmnas.conventions`.

**Q2 and Q3, against each control separately** (PEA with refill is the primary comparator):

| Verdict | Rule |
|---|---|
| **Better** | `diff > +margin` on every device where `mean + margin < 1` (at least 3 such devices; amendment 2), **and** sign-flip p <= 0.05 over all six devices for "LLM better" |
| **Worse** | `diff < -margin` on at least 5 of 6 devices, **or** sign-flip p <= 0.05 for "LLM worse" |
| **Parity** | anything else |

If both "Better" conditions and a "Worse" condition held at once (impossible with these rules, but stated for
completeness), the verdict is **Parity**.

**Q1, the fixes work mechanically if** arm A reaches N >= 265 on all six devices **and** its duplicate rate is
at least 20 percentage points below the reference runs' 42.7%. Arm B: the duplicate rate falls by at least
10 points.

**Q4:** arm B against the reference runs, using the same rules with `margin = 0.010` (single runs on both sides).

## 6. What will be reported whatever the outcome

All four questions, each verdict, the full per-device tables, the process measures and the cost. If Q1 fails
(the fixes do not fill the budget), Q2 is reported but marked as not answered by this test. A "Parity" or
"Worse" verdict will be written up as the main result with the same prominence as a "Better" one would have had.

Expected result, stated in advance: Q1 passes; Q2 Parity against PEA and random sampling, Worse against NSGA-II
and regularised evolution; Q3 the same pattern with smaller differences; Q4 a smaller duplicate rate but Parity
with the reference runs.

## 7. Stopping rule and allowed changes

- One run per arm and device. A search that fails for an external reason (API error, time-out) is re-run with
  the same settings and the failure is reported.
- No changes to prompts, retry rules, temperature, model or caps after any paid result has been seen. A bug that
  makes a run not follow this document may be fixed and the affected runs repeated; the bug and the re-runs are
  reported.
- Implementation details not fixed here (exact prompt wording of the history list and retry message) are frozen
  in the commit that is checked with the mock backend before the first paid run; that commit hash is added
  below before the first paid call.

## 8. Record

- Pre-registration commit: this file's first commit.
- Implementation commit (frozen before paid runs): `754bc79` (2026-11-20; switches added in `67b262d`, amendment 1
  in `754bc79`). Checked with the mock backend and by replaying archived runs. Prompt wording and retry rules
  are fixed from this commit on.
- Analysis rule frozen with amendment 2 in the commit that adds it (see Section 9).
- Approval of paid arms: 2026-11-23, by the author, for arm A (about $11, at most $17) and arm B (about $5.40),
  GPT-4.1 via the OpenAI API, total cap $23. Given after the explainer of 2026-11-22 was read.

## 9. Amendments (made before any paid run)

**Amendment 1, 2026-11-20.** Found in an offline dry run on the real benchmark with the mock LLM (no paid call,
no GPT-4.1 result seen). With 50 random draws per slot and no fallback, PEA with refill filled only 217-265 of
its 270 evaluations (Edge GPU 260, Raspberry Pi 4 265, Edge TPU 260, Eyeriss 239, FPGA 233, Pixel 3 217): once
every in-niche neighbour of a niche's few parents has been evaluated, mutation and crossover can produce nothing
new. The main comparator could then not be read at N = 270. Changes:

1. PEA with refill: if all 50 draws for a slot are rejected, the slot takes a random unevaluated cell of the
   niche (`search.refill_fallback: random`), the rule NSGA-II and regularised evolution already use. With it the
   dry run reaches 270 on all six devices.
2. The LLM arms are unchanged: no padding with random cells, because padding would credit the LLM with random
   proposals.
3. Reading rule: a run that ends before N evaluations is scored at its final value (unused budget earns nothing).
4. Q1 is unchanged (arm A must reach N >= 265 on every device). The same dry run with the mock LLM, which ignores
   the retry feedback, reached 217-256; GPT-4.1 is not limited to single-edge changes, so this does not predict
   its result.
5. Cost: the mock needed 224-330 calls per arm-A search, more than the 195-270 assumed. Revised estimate for
   arm A: about $11 for six devices, at most about $17 if every retry is used. Arm B unchanged.

**Amendment 2, 2026-11-23.** Found when the 10-seed controls were analysed at N = 270 (no paid call made, no
arm-A or arm-B result exists). Seen at that point: the control results and the archived GPT-4.1 reference runs,
which this test already used as its starting point. HV is a ratio to the true front and cannot exceed 1. On Edge
GPU, Raspberry Pi 4 and Pixel 3, `mean + margin >= 1` for PEA with refill, NSGA-II and regularised evolution
(means 0.991-0.995, margin 0.010), so no arm could exceed the margin there and "at least 5 of 6 devices" could
never be met: the test could return "Worse" but never "Better". Changes:

1. "Better" requires `diff > +margin` on every device where `mean(c, d) + margin(d, c) < 1`, with at least 3 such
   devices, and the one-sided sign-flip p <= 0.05 over all six devices (unchanged; on 3 devices alone the
   smallest p would be 1/8). With fewer than 3 such devices, "Better" is not reachable and this is reported.
2. "Worse" and "Parity", the margins, Q1, Q4 (same rules), the arms, prompts and controls are unchanged.
3. At N = 270 the devices where a win is possible are Edge TPU, Eyeriss and FPGA for PEA with refill (margins
   0.029, 0.029, 0.041), NSGA-II and regularised evolution; the analysis report lists them per control.
4. Implemented in `llmnas.dupfix.verdict` with a test; the verdicts for the archived reference runs at N = 120
   are the same under the old and new rule.

## 10. Outcome (recorded 2026-11-23, after all paid runs; no rule above was changed)

Paid arms ran 2026-11-23 on commit `3e2dd02` (search code as frozen in `754bc79`), with no API failures and no
re-runs. Cost: arm A $14.07 (1,644 calls), arm B $5.53 (720 calls), total $19.60 of the $23 approved. Full
tables: `results/reports/dupfix.md` (`python -m llmnas dupfix`).

- **Q1: fails for both arms.** Arm A reached N = 231-251 (161 slots left empty after five retries; needed
  >= 265) and its duplicate rate rose to 48.3% (reference 42.7%; needed <= 22.7%). Arm B lowered it to 35.9%
  (needed <= 32.7%).
- **Q2 (N = 270):** arm A **Worse** than all four controls (p = 0.016 against PEA with refill, NSGA-II and
  regularised evolution, 0.031 against random sampling). Because Q1 failed, Section 6 applies: Q2 is reported
  but not answered by this test, since arm A never had the full budget.
- **Q3 (N = 120):** arm A **Parity** with PEA with refill, random sampling and NSGA-II, **Worse** than
  regularised evolution; the same pattern as the reference runs.
- **Q4:** arm B against the reference runs **Parity** (3 devices above, 1 below the 0.010 margin, mean
  +0.014, p (better) = 0.078).
- Against the result stated in advance (Section 6): Q2 and Q3 as expected, apart from Worse (not Parity)
  against PEA with refill and random sampling at N = 270; Q1 not as expected; Q4 as expected.
