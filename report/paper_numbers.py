"""Every number in the results tables of reproduction_only.tex, from the archived runs: ``python -I report/paper_numbers.py``.

Writes results/reports/paper_numbers.md. No search is run and no LLM is called. Conventions:
* HV / IGD: the default convention (llmnas.metrics, as stored in run.json).
* "(>=80%)": re-scored with architectures below 80% test accuracy removed from the true front and the found front
  (llmnas.compare, as ``compare-llms --min-acc 80``).
* s.d. is the sample s.d. over seeds (ddof = 1).
* Matched budget: a control's HV read from its trajectory at the LLM run's number of evaluated architectures N
  (linear interpolation; llmnas.dupfix.hv_at), averaged over the control's seeds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llmnas.compare import _cost, _floored_evaluator  # noqa: E402
from llmnas.conventions import DEVICES, load_runs  # noqa: E402
from llmnas.dupfix import hv_at  # noqa: E402
from llmnas.experiments import REPO  # noqa: E402
from llmnas.metrics import objectives  # noqa: E402

LLMS = ["gpt-4.1", "claude-opus-5-5", "claude-haiku-4-5", "deepseek-coder-v2-16b-8k", "qwen2.5-7b-8k"]
CTL = ["PEA (no LLM)", "PEA with refill (no LLM)", "Random sampling (no LLM)", "NSGA-II (no LLM)", "Reg. evolution (no LLM)"]
SHORT = {"gpt-4.1": "GPT-4.1", "claude-opus-5-5": "Opus 5.5", "claude-haiku-4-5": "Haiku 4.5",
         "deepseek-coder-v2-16b-8k": "DeepSeek", "qwen2.5-7b-8k": "Qwen", "PEA (no LLM)": "PEA",
         "PEA with refill (no LLM)": "PEA+refill", "Random sampling (no LLM)": "RS", "NSGA-II (no LLM)": "NSGA-II",
         "Reg. evolution (no LLM)": "RegEvo"}  # fmt: skip
_FLOOR = {}


def floored(run, dev):
    if dev not in _FLOOR:
        _FLOOR[dev] = _floored_evaluator("cifar10", dev, 80.0)
    p = np.array([[q["lat"], q["true_acc"]] for q in run["final_front"]], float).reshape(-1, 2)
    p = p[~np.isnan(p).any(axis=1)]
    k = p[p[:, 1] >= 80.0]
    s = _FLOOR[dev].score(objectives(k[:, 0], k[:, 1])) if len(k) else {"hv": 0.0, "igd": float("nan")}
    return s["hv"], s["igd"], bool((p[:, 1] < 20).any())


def table() -> pd.DataFrame:
    rows = []
    for it in load_runs(["cifar10"]):
        if it["method"] not in LLMS + CTL:
            continue
        r, st, u = it["run"], it["run"]["proposal_stats"], it["run"].get("llm_usage") or {}
        hf, igf, cell = floored(r, it["device"])
        c = r["config"]["llm"]
        rows.append(dict(method=it["method"], device=it["device"], seed=it["seed"], hv=r["hv"], igd=r["igd"],
                         hv80=hf, igd80=igf, cell=cell, n=r["n_evaluated"],
                         usable=st["accepted"] / max(st["proposed"], 1), proposed=st["proposed"], accepted=st["accepted"],
                         minutes=r["search_time_s"] / 60, tin=u.get("input_tokens", 0), tout=u.get("output_tokens", 0),
                         cost=_cost(c["model"], u.get("input_tokens", 0), u.get("output_tokens", 0),
                                    "localhost" in str(c.get("base_url") or "")) if it["method"] in LLMS else 0.0,
                         run=r))  # fmt: skip
    return pd.DataFrame(rows)


def pm(x, d=3):
    x = pd.Series(x).dropna()
    return f"{x.mean():.{d}f}" + (f" ± {x.std(ddof=1):.{d}f}" if len(x) > 1 else "")


def main():
    df = table()
    L = ["# Numbers for the paper (CIFAR-10)\n", __doc__.split("\n\n", 1)[1], ""]
    meths = LLMS + CTL

    L.append("## 1. Edge GPU, per method\n")
    L.append("| Method | seeds | HV | IGD | HV>=80 | IGD>=80 | cell | usable | N | minutes | cost $ | tokens in/out (k) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    e = df[df.device == "edgegpu"]
    for m in meths:
        g = e[e.method == m]
        if g.empty:
            continue
        L.append(f"| {m} | {len(g)} | {pm(g.hv)} | {pm(g.igd)} | {pm(g.hv80)} | {pm(g.igd80)} | {g.cell.sum()}/{len(g)} | "
                 f"{g.accepted.mean():.0f}/{g.proposed.mean():.0f} ({g.usable.mean():.0%}) | {g.n.mean():.0f} | "
                 f"{g.minutes.mean():.1f} | {g.cost.mean():.2f} | {g.tin.mean() / 1e3:.0f}/{g.tout.mean() / 1e3:.0f} |")  # fmt: skip
    L.append("\nPer-seed LLM runs on Edge GPU:\n")
    for _, r in e[e.method.isin(LLMS)].sort_values(["method", "seed"]).iterrows():
        L.append(f"- {r.method} seed {r.seed}: HV {r.hv:.3f}, IGD {r.igd:.3f}, HV80 {r.hv80:.3f}, IGD80 {r.igd80:.3f}, "
                 f"N {r.n}, usable {r.accepted}/{r.proposed}, {r.minutes:.1f} min, cell {r.cell}")  # fmt: skip
    for m in CTL:
        g = e[e.method == m]
        L.append(f"- {m}: HV80 range {g.hv80.min():.3f}-{g.hv80.max():.3f}, IGD80 range {g.igd80.min():.3f}-{g.igd80.max():.3f}")

    for col, lab in (("hv", "default"), ("hv80", ">= 80% accuracy")):
        L.append(f"\n## 2. HV ({lab}) on six devices: mean ± s.d. over seeds (seeds in brackets)\n")
        L.append("| Method | " + " | ".join(DEVICES) + " | mean over devices |")
        L.append("|---|" + "---|" * (len(DEVICES) + 1))
        for m in meths:
            g = df[df.method == m]
            if g.device.nunique() < 6:
                continue
            cells = [f"{pm(g[g.device == d][col])} ({(g.device == d).sum()})" for d in DEVICES]
            L.append(f"| {m} | " + " | ".join(cells) + f" | {g.groupby('device')[col].mean().mean():.3f} |")

    L.append("\n## 3. LLM runs above each control's device mean (every LLM run = one device x seed)\n")
    L.append("| Convention | LLM runs | " + " | ".join(SHORT[c] for c in CTL) + " |")
    L.append("|---|---|" + "---|" * len(CTL))
    six = df[df.method.isin([m for m in LLMS if df[df.method == m].device.nunique() == 6])]
    for col, lab in (("hv", "default"), ("hv80", ">=80%")):
        cm = df[df.method.isin(CTL)].groupby(["method", "device"])[col].mean()
        cells = []
        for c in CTL:
            wins = [f"{SHORT[r.method]}/{r.device}/s{r.seed}" for _, r in six.iterrows() if r[col] > cm[(c, r.device)]]
            cells.append(f"{len(wins)}" + (f" ({', '.join(wins)})" if 0 < len(wins) <= 6 else ""))
        L.append(f"| {lab} | {len(six)} | " + " | ".join(cells) + " |")
    L.append("\nRange of the LLM runs per device (all seeds) and of the control means:\n")
    for col in ("hv", "hv80"):
        for d in DEVICES:
            g, cm = six[six.device == d][col], df[df.method.isin(CTL) & (df.device == d)].groupby("method")[col].mean()
            L.append(f"- {col} {d}: LLM runs {g.min():.3f}-{g.max():.3f}; control means {cm.min():.3f}-{cm.max():.3f}")

    L.append("\n## 4. Matched budget: controls read at each LLM run's N (mean over LLM seeds and control seeds)\n")
    L.append("| LLM | device | seeds | N | LLM HV | " + " | ".join(f"{SHORT[c]}@N" for c in CTL) + " |")
    L.append("|---|---|---|---|---|" + "---|" * len(CTL))
    summ = []
    ctl_runs = {(c, d): [r for r in df[(df.method == c) & (df.device == d)].run] for c in CTL for d in DEVICES}
    for m in [m for m in LLMS if df[df.method == m].device.nunique() == 6]:
        for d in DEVICES:
            g = df[(df.method == m) & (df.device == d)]
            at = {c: np.mean([np.mean([hv_at(cr, n) for cr in ctl_runs[(c, d)]]) for n in g.n]) for c in CTL}
            summ.append(dict(method=m, device=d, n=g.n.mean(), hv=g.hv.mean(), **at))
            L.append(f"| {SHORT[m]} | {d} | {len(g)} | {g.n.mean():.0f} | {g.hv.mean():.3f} | "
                     + " | ".join(("**" if g.hv.mean() > at[c] else "") + f"{at[c]:.3f}" + ("**" if g.hv.mean() > at[c] else "")
                                  for c in CTL) + " |")  # fmt: skip
    s = pd.DataFrame(summ)
    L.append("\n| LLM | mean N | LLM HV | " + " | ".join(f"{SHORT[c]}@N" for c in CTL) + " | devices LLM above (of 6) |")
    L.append("|---|---|---|" + "---|" * len(CTL) + "---|")
    for m, g in s.groupby("method", sort=False):
        above = " / ".join(f"{SHORT[c]} {int((g.hv > g[c]).sum())}" for c in CTL)
        L.append(f"| {SHORT[m]} | {g.n.mean():.0f} | {g.hv.mean():.3f} | " + " | ".join(f"{g[c].mean():.3f}" for c in CTL)
                 + f" | {above} |")  # fmt: skip

    L.append("\n## 5. Controls over devices at N = 145 and N = 270 (default HV, mean over seeds then devices)\n")
    for c in CTL:
        v = {n: np.mean([np.mean([hv_at(r, n) for r in ctl_runs[(c, d)]]) for d in DEVICES]) for n in (86, 145, 270)}
        sd = np.mean([np.std([hv_at(r, 145) for r in ctl_runs[(c, d)]], ddof=1) for d in DEVICES])
        L.append(f"- {c}: N=86 {v[86]:.3f}, N=145 {v[145]:.3f}, N=270 {v[270]:.3f} (gain 145->270 {v[270] - v[145]:+.3f}); "
                 f"mean seed s.d. at 145: {sd:.3f}; mean N evaluated {df[df.method == c].n.mean():.0f}")  # fmt: skip

    L.append("\n## 6. Seed-to-seed spread of the LLM runs with three seeds (default HV and HV>=80, s.d. per device)\n")
    for m in LLMS:
        g = df[df.method == m]
        if g.groupby("device").size().max() < 2:
            continue
        for col in ("hv", "hv80"):
            sds = g.groupby("device")[col].std(ddof=1)
            L.append(f"- {m} {col}: s.d. " + ", ".join(f"{d} {sds[d]:.3f}" for d in DEVICES)
                     + f"; range over devices {sds.min():.3f}-{sds.max():.3f}; N {g.n.min()}-{g.n.max()} (mean {g.n.mean():.0f})")
    L.append("\nSearch cost per model (all runs): "
             + "; ".join(f"{SHORT[m]} {len(g)} runs, {g.minutes.min():.0f}-{g.minutes.max():.0f} min, ${g.cost.min():.2f}-{g.cost.max():.2f}"
                         for m, g in df[df.method.isin(LLMS)].groupby("method")))  # fmt: skip

    out = REPO / "results" / "reports" / "paper_numbers.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
