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

Paid arms run only after explicit approval. Estimated cost: A about $7.50, B about $5.40 (6 searches each).
The same arms may also be run on the local models (DeepSeek-Coder-V2-16B, Qwen2.5-7B) at no API cost; those
results are secondary.

## 3. Controls

All with the same evaluator, CIFAR-10, six devices, **10 seeds (0-9)** each, so that the seed-to-seed spread is
estimated properly (these runs are free):

- **PEA with refill** (main comparator: the same framework and niches, only the operator differs). Rejected
  random children are regenerated, up to 50 attempts per slot, until each slot has 4 new children.
- Random sampling in niches (already never repeats).
- NSGA-II and regularised evolution (already refill to 270).

## 4. Measures

- **Primary:** hypervolume under the default convention (front normalisation, reference point 1.1, archived
  front, no accuracy floor, Edge TPU failed compilations excluded), read from the search trajectory at
  **N = 270** evaluations (Q2) and **N = 120** (Q3), by linear interpolation between generations.
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
| **Better** | `diff > +margin` on at least 5 of 6 devices, **and** sign-flip p <= 0.05 for "LLM better" |
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
- Implementation commit (frozen before paid runs): _to be filled in_.
- Approval of paid arms: _to be filled in (date, amount)_.
