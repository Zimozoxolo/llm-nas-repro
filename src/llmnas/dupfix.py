"""Pre-registered analysis of the duplicate-fix test: ``python -m llmnas dupfix``.

Implements Sections 4-5 of PREREGISTRATION_duplicate_fix.md (with amendments 1 and 2) and nothing else:

* HV under the default convention (front normalisation, reference point 1.1, archived front, no floor), read from
  each run's trajectory at N evaluations by linear interpolation between generations; a run that ends before N
  is scored at its final value.
* margin(d, c) = max(2 x s.d. of control c over its seeds on device d, 0.010)   (sample s.d., ddof = 1)
* diff(d, c)   = HV(arm, d) - mean HV(c, d)
* Better (amendment 2): diff > +margin on every device where mean + margin < 1 (at least 3 such devices) AND
  one-sided sign-flip p <= 0.05 over all six devices for "arm better"
  Worse:  diff < -margin on >= 5 of 6 devices OR  one-sided sign-flip p <= 0.05 for "arm worse"
  Parity: anything else (also if Better and Worse both held)
* Q1: arm A reaches N >= 265 on all six devices and its duplicate rate is >= 20 points below the reference's;
  arm B: duplicate rate >= 10 points below.
* Q4: arm B against the reference runs, same rules with margin = 0.010.

Arms are recognised from their configs (see ``conventions.method_of``); the controls come from
``dupfix_controls``. No search is run and no LLM is called.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .conventions import DEVICES, _signflip_p, load_runs
from .experiments import REPO

REF = "gpt-4.1"
ARM_A = "gpt-4.1 [history + re-prompt]"
ARM_B = "gpt-4.1 [history]"
CONTROLS = ["PEA with refill (no LLM)", "Random sampling (no LLM)", "NSGA-II (no LLM)", "Reg. evolution (no LLM)"]
PRIMARY = CONTROLS[0]
PRICE = (2.0, 8.0)  # GPT-4.1, $ per million input / output tokens
REF_DUP = 0.427  # duplicate rate of the reference runs, stated in the pre-registration


def hv_at(run: dict, n: int) -> float:
    """HV at n evaluations, interpolated between trajectory points; the final value if the run stopped earlier."""
    t = run["trajectory"]
    xs, ys = [p["n_evaluated"] for p in t], [p["hv"] for p in t]
    if n >= xs[-1]:
        return float(ys[-1])
    return float(np.interp(n, xs, ys))


def runs_table() -> pd.DataFrame:
    rows = []
    for it in load_runs(["cifar10"]):
        if it["method"] not in CONTROLS + [REF, ARM_A, ARM_B]:
            continue
        r, s = it["run"], it["run"]["proposal_stats"]
        u = r.get("llm_usage") or {}
        rows.append(dict(method=it["method"], device=it["device"], seed=it["seed"], n_evaluated=r["n_evaluated"],
                         hv270=hv_at(r, 270), hv120=hv_at(r, 120), hv_final=r["hv"],
                         dup_rate=s["duplicate"] / s["proposed"] if s.get("proposed") else np.nan,
                         out_of_niche=s["out_of_niche"] / s["proposed"] if s.get("proposed") else np.nan,
                         slots_unfilled=s.get("slots_unfilled", np.nan), calls=u.get("api_calls", 0),
                         in_tok=u.get("input_tokens", 0), out_tok=u.get("output_tokens", 0),
                         wall_s=u.get("llm_wall_time_s", r.get("search_time_s"))))  # fmt: skip
    df = pd.DataFrame(rows)
    if len(df):
        df["cost_usd"] = (df.in_tok * PRICE[0] + df.out_tok * PRICE[1]) / 1e6
    return df


def control_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per control and device: seeds, mean, s.d. and the pre-registered margin."""
    g = df[df.method.isin(CONTROLS)].groupby(["method", "device"])[col]
    out = pd.DataFrame(dict(seeds=g.count(), mean=g.mean(), sd=g.std(ddof=1)))
    out["margin"] = np.maximum(2 * out.sd, 0.010)
    return out


def verdict(arm: pd.Series, ctl_mean: pd.Series, margin: pd.Series) -> dict:
    """arm, ctl_mean, margin: indexed by device. Returns the per-device diffs, counts, p-values and the verdict."""
    devs = [d for d in DEVICES if d in arm.index and d in ctl_mean.index]
    diff = np.array([arm[d] - ctl_mean[d] for d in devs])
    m = np.array([margin[d] for d in devs])
    # amendment 2: HV is capped at 1, so "Better" counts only devices where mean + margin < 1 (at least 3 needed)
    room = np.array([ctl_mean[d] + margin[d] < 1.0 for d in devs], dtype=bool)
    above, below = int((diff > m).sum()), int((diff < -m).sum())
    above_room = int(((diff > m) & room).sum())
    p_better, p_worse = _signflip_p(-diff), _signflip_p(diff)
    better = room.sum() >= 3 and above_room == room.sum() and p_better <= 0.05
    worse = below >= 5 or p_worse <= 0.05
    v = "Parity" if better == worse else ("Better" if better else "Worse")
    if len(devs) < 6:
        v += f" (only {len(devs)} of 6 devices)"
    return dict(devices=len(devs), above=above, below=below, room=int(room.sum()), above_room=above_room,
                room_devices=[d for d, r in zip(devs, room) if r], mean_diff=float(diff.mean()) if len(diff) else np.nan,
                p_better=p_better, p_worse=p_worse, verdict=v, diffs=dict(zip(devs, diff.round(4))))  # fmt: skip


def report(out: Path | None = None) -> str:
    df = runs_table()
    out = out or REPO / "results" / "reports" / "dupfix.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)
    L = ["# Duplicate-fix test (pre-registered analysis)\n",
         "Rules: PREREGISTRATION_duplicate_fix.md, Sections 4-5 and amendments 1-2. HV = default convention, read at N "
         "evaluations from the trajectory (final value if the run ended earlier). CIFAR-10.\n"]  # fmt: skip
    arms = {m: df[df.method == m] for m in (ARM_A, ARM_B, REF)}
    have = {m: g for m, g in arms.items() if len(g)}
    L.append("Runs found: " + ", ".join(f"{m}: {len(df[df.method == m])}" for m in CONTROLS + [REF, ARM_A, ARM_B]) + "\n")

    for col, label in (("hv270", "N = 270 (Q2)"), ("hv120", "N = 120 (Q3)")):
        cs = control_stats(df, col)
        L.append(f"## Controls at {label}: mean HV, s.d. over seeds, margin\n")
        L.append("| Control | " + " | ".join(DEVICES) + " |")
        L.append("|---|" + "---|" * len(DEVICES))
        for c in CONTROLS:
            cells = []
            for d in DEVICES:
                if (c, d) in cs.index:
                    r = cs.loc[(c, d)]
                    cells.append(f"{r['mean']:.3f} ± {r['sd']:.3f} (m {r['margin']:.3f}, n={int(r['seeds'])})")
                else:
                    cells.append("–")
            L.append(f"| {c} | " + " | ".join(cells) + " |")
        L.append("")
        for m, g in have.items():
            if m == REF and col == "hv270":
                continue  # the reference runs stop at about 145 evaluations; Q2 concerns arm A
            a = g.set_index("device")[col]
            L.append(f"**{m} at {label}** " + ("(primary)" if m == ARM_A else "(secondary)"))
            L.append("")
            L.append("| Control | verdict | devices above / below margin | above where a win is possible "
                     "(amendment 2) | mean diff | p (better) | p (worse) |")  # fmt: skip
            L.append("|---|---|---|---|---|---|---|")
            for c in CONTROLS:
                sub = cs.xs(c, level="method")
                v = verdict(a, sub["mean"], sub["margin"])
                tag = " (primary comparator)" if c == PRIMARY else ""
                L.append(f"| {c}{tag} | **{v['verdict']}** | {v['above']} / {v['below']} of {v['devices']} | "
                         f"{v['above_room']} of {v['room']} ({', '.join(v['room_devices']) or 'none'}) | "
                         f"{v['mean_diff']:+.4f} | {v['p_better']:.3f} | {v['p_worse']:.3f} |")  # fmt: skip
            L.append("")

    L.append("## Q1: mechanism\n")
    L.append("| Method | min N | mean N | duplicate rate | out of niche | slots unfilled | calls | cost ($) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for m in [REF, ARM_A, ARM_B, PRIMARY]:
        g = df[df.method == m]
        if not len(g):
            L.append(f"| {m} | not run yet | | | | | | |")
            continue
        L.append(f"| {m} | {g.n_evaluated.min()} | {g.n_evaluated.mean():.0f} | {g.dup_rate.mean():.1%} | "
                 f"{g.out_of_niche.mean():.1%} | {g.slots_unfilled.sum():.0f} | {g.calls.sum()} | "
                 f"{g.cost_usd.sum():.2f} |")  # fmt: skip
    L.append("")
    if ARM_A in have:
        g = have[ARM_A]
        ok = len(g) == 6 and g.n_evaluated.min() >= 265 and g.dup_rate.mean() <= REF_DUP - 0.20
        L.append(f"Arm A passes Q1: **{'yes' if ok else 'no'}** (needs N >= 265 on all six devices and duplicate "
                 f"rate <= {REF_DUP - 0.20:.1%}).")  # fmt: skip
    if ARM_B in have:
        g = have[ARM_B]
        ok = g.dup_rate.mean() <= REF_DUP - 0.10
        L.append(f"Arm B passes Q1: **{'yes' if ok else 'no'}** (needs duplicate rate <= {REF_DUP - 0.10:.1%}).")
    L.append("")

    if ARM_B in have and REF in have:
        L.append("## Q4: arm B against the reference runs (120 calls each, margin 0.010)\n")
        b, r = have[ARM_B].set_index("device").hv_final, have[REF].set_index("device").hv_final
        v = verdict(b, r, pd.Series(0.010, index=r.index))
        L.append(f"**{v['verdict']}**: {v['above']} / {v['below']} devices above / below, mean diff "
                 f"{v['mean_diff']:+.4f}, p (better) {v['p_better']:.3f}, p (worse) {v['p_worse']:.3f}.\n")  # fmt: skip

    text = "\n".join(L) + "\n"
    out.write_text(text, encoding="utf-8")
    return text
