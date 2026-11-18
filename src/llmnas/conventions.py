"""Re-score every archived run under several hypervolume conventions: ``python -m llmnas conventions``.

The LLM-NAS paper does not state how it normalises the objectives, which reference point it uses, which set of
architectures it scores, or whether degenerate (chance-level) cells count. This module re-scores the archived
final fronts of all runs under a grid of plausible conventions, so a conclusion can be checked for robustness:

    normalisation   front (ideal/nadir of the true front)   | space (min/max over the whole search space)
    reference point 1.1 (default)                             | 1.0
    scored set      archive (final front of the search)       | all (every evaluated architecture)
    accuracy floor  none                                      | 85% of the best accuracy in the benchmark

The floor removes architectures below 85% of the best test accuracy from the true front and from every
discovered set (CIFAR-10: < 80.2%, CIFAR-100: < 62.5%, ImageNet16-120: < 40.2%); it generalises the 80%
CIFAR-10 floor used in the paper to the other datasets.
No search is re-run and no LLM is called.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import load_benchmark
from .experiments import REPO
from .metrics import FrontEvaluator, objectives

DEVICES = ["edgegpu", "raspi4", "edgetpu", "pixel3", "eyeriss", "fpga"]
CONVENTIONS = [dict(norm=n, ref=r, source=s, floor=f)
               for n, r, s, f in itertools.product(["front", "space"], [1.1, 1.0], ["archive", "all"], [None, 0.85])]
FLOOR_FRAC = 0.85


def conv_name(c: dict) -> str:
    return f"{c['norm']}/r={c['ref']}/{c['source']}/" + ("no floor" if c["floor"] is None else f"floor {int(c['floor'] * 100)}%")


def method_of(r: dict) -> str | None:
    c = r["config"]
    s = c["search"]
    if c["llm"].get("backend") == "mock":
        return None
    op = s["operator"]
    if op == "nsga2":
        return "NSGA-II (no LLM)"
    if not s["partitioning"] or c["predictor"]["kind"] != "zc_ensemble":
        return None  # ablated variants are not compared here
    if op == "llm":
        if not s.get("knowledge_base", True):
            return None
        m = c["llm"]["model"]
        return m + (f" [effort={c['llm']['effort']}]" if c["llm"].get("effort") else "")
    return {"random": "PEA (no LLM)", "random_sample": "Random sampling (no LLM)"}.get(op)


def load_runs(datasets, root: Path | None = None) -> list[dict]:
    root = root or REPO / "results"
    out, seen = [], set()
    for ds in datasets:
        for f in sorted((root / "runs").glob(f"*/{ds}/*/seed*/run.json")):
            r = json.loads(f.read_text())
            if r["config"].get("type", "search") not in ("search", "nsga2"):
                continue
            m = method_of(r)
            if m is None:
                continue
            key = (m, ds, r["device"], r["seed"])
            if key in seen:  # identical configuration archived twice: keep the first
                continue
            seen.add(key)
            out.append(dict(method=m, dataset=ds, device=r["device"], seed=r["seed"], run=r, path=str(f)))
    return out


_EV: dict = {}


def _evaluator(ds: str, dev: str, conv: dict) -> tuple[FrontEvaluator, float | None]:
    key = (ds, dev, conv["norm"], conv["ref"], conv["floor"])
    if key not in _EV:
        b = load_benchmark(ds)
        lat, acc = b.latencies(dev), b.test_acc
        thr = None
        keep = ~np.isnan(lat)
        if conv["floor"] is not None:
            thr = conv["floor"] * float(np.nanmax(acc))
            keep &= acc >= thr
        _EV[key] = (FrontEvaluator(objectives(lat[keep], acc[keep]), normalization=conv["norm"],
                                   ref_point=(conv["ref"], conv["ref"])), thr)  # fmt: skip
    return _EV[key]


def score_run(item: dict, conv: dict) -> dict:
    r = item["run"]
    ev, thr = _evaluator(item["dataset"], item["device"], conv)
    pts = r["final_front"] if conv["source"] == "archive" else r["evaluated"]
    p = np.array([[q["lat"], q["true_acc"]] for q in pts], float).reshape(-1, 2)
    p = p[~np.isnan(p).any(axis=1)]
    if thr is not None:
        p = p[p[:, 1] >= thr]
    s = ev.score(objectives(p[:, 0], p[:, 1])) if len(p) else {"hv": 0.0, "igd": float("nan")}
    return dict(hv=s["hv"], igd=s["igd"])


def table(datasets=("cifar10", "cifar100")) -> pd.DataFrame:
    rows = []
    for item in load_runs(datasets):
        for c in CONVENTIONS:
            rows.append(dict(method=item["method"], dataset=item["dataset"], device=item["device"],
                             seed=item["seed"], convention=conv_name(c), **score_run(item, c)))  # fmt: skip
    return pd.DataFrame(rows)


def report(datasets=("cifar10", "cifar100"), out: Path | None = None) -> str:
    df = table(datasets)
    out = out or REPO / "results" / "reports" / "conventions.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)
    controls = ["PEA (no LLM)", "Random sampling (no LLM)", "NSGA-II (no LLM)"]
    lines = ["# Robustness of the results to the hypervolume convention\n",
             "Every archived run re-scored under 16 conventions (normalisation x reference point x scored set x "
             f"accuracy floor at {int(FLOOR_FRAC * 100)}% of the best accuracy). Device-level values are means over "
             "seeds; the table gives the mean over devices that the method was run on. LLM rows are single runs "
             "per device. No search was re-run.\n"]
    for ds in datasets:
        d = df[df.dataset == ds]
        if d.empty:
            continue
        per_dev = d.groupby(["method", "device", "convention"]).hv.mean().reset_index()
        piv = per_dev.groupby(["method", "convention"]).hv.mean().unstack("convention")
        ndev = per_dev.groupby("method").device.nunique()
        lines.append(f"\n## {ds}: mean HV over devices\n")
        convs = [conv_name(c) for c in CONVENTIONS]
        lines.append("| Method | Devices | " + " | ".join(c.replace("/", " / ") for c in convs) + " |")
        lines.append("|---|---|" + "---|" * len(convs))
        for m in sorted(piv.index, key=lambda m: (m in controls, m)):
            lines.append(f"| {m} | {ndev[m]} | " + " | ".join(f"{piv.loc[m, c]:.3f}" for c in convs) + " |")
        # head-to-head: each LLM run against each control's seed mean, same device and convention
        llms = [m for m in piv.index if m not in controls]
        lines.append(f"\n### {ds}: LLM runs that beat each control (same device and convention)\n")
        lines.append("| Convention | " + " | ".join(f"beats {c}" for c in controls if c in piv.index) + " |")
        lines.append("|---|" + "---|" * sum(c in piv.index for c in controls))
        for c in convs:
            pc = per_dev[per_dev.convention == c].set_index(["method", "device"]).hv
            cells = []
            for ctl in controls:
                if ctl not in piv.index:
                    continue
                wins = tot = 0
                for m in llms:
                    for dev in per_dev[(per_dev.method == m)].device.unique():
                        if (ctl, dev) in pc.index:
                            tot += 1
                            wins += pc[(m, dev)] > pc[(ctl, dev)]
                cells.append(f"{wins}/{tot}")
            lines.append(f"| {c.replace('/', ' / ')} | " + " | ".join(cells) + " |")
        # ranking stability
        ranks = piv.rank(ascending=False)
        lines.append(f"\n### {ds}: rank of each method under each convention (1 = best)\n")
        lines.append("| Method | best rank | worst rank |")
        lines.append("|---|---|---|")
        for m in ranks.index:
            lines.append(f"| {m} | {int(ranks.loc[m].min())} | {int(ranks.loc[m].max())} |")
    text = "\n".join(lines) + "\n"
    out.write_text(text, encoding="utf-8")
    return text
