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
    if c["predictor"].get("n_train", 1000) != 1000:
        return None  # predictor-size sensitivity runs are reported by llmnas.sensitivity
    op = s["operator"]
    if op == "nsga2":
        return "NSGA-II (no LLM)"
    if op == "regevo":
        return "Reg. evolution (no LLM)"
    if not s["partitioning"] or c["predictor"]["kind"] != "zc_ensemble":
        return None  # ablated variants are not compared here
    if op == "llm":
        if not s.get("knowledge_base", True):
            return None
        m = c["llm"]["model"]
        m += f" [effort={c['llm']['effort']}]" if c["llm"].get("effort") else ""
        fix = [t for t, on in (("history", s.get("show_history")), ("re-prompt", s.get("reprompt_retries")))
               if on]  # fmt: skip
        return m + (f" [{' + '.join(fix)}]" if fix else "")  # duplicate-fix arms are separate methods
    if op == "random" and s.get("refill_attempts"):
        return "PEA with refill (no LLM)"
    return {"random": "PEA (no LLM)", "random_sample": "Random sampling (no LLM)"}.get(op)


def load_runs(datasets, root: Path | None = None) -> list[dict]:
    root = root or REPO / "results"
    out, seen = [], set()
    for ds in datasets:
        for f in sorted((root / "runs").glob(f"*/{ds}/*/seed*/run.json")):
            r = json.loads(f.read_text())
            if r["config"].get("type", "search") not in ("search", "nsga2", "regevo"):
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


def table(datasets=("cifar10", "cifar100", "ImageNet16-120")) -> pd.DataFrame:
    rows = []
    for item in load_runs(datasets):
        for c in CONVENTIONS:
            rows.append(dict(method=item["method"], dataset=item["dataset"], device=item["device"],
                             seed=item["seed"], convention=conv_name(c), **score_run(item, c)))  # fmt: skip
    return pd.DataFrame(rows)


def _signflip_p(diffs) -> float:
    """Exact one-sided sign-flip permutation p-value: share of sign flips whose mean is <= the observed mean."""
    diffs = np.asarray(diffs, float)
    obs = diffs.mean()
    signs = np.array(list(itertools.product([1, -1], repeat=len(diffs))))
    return float(np.mean((signs * np.abs(diffs)).mean(axis=1) <= obs + 1e-12))


def head_to_head(per_dev: pd.DataFrame, llms, ctl: str, conv: str, metric: str) -> dict:
    """LLM runs against one control on the same device and convention; higher HV / lower IGD is better.

    Returns wins/total, a pooled one-sided binomial sign test (treats the pairs as independent, which they are
    not: every LLM on a device faces the same control mean), a Wilcoxon signed-rank test on the same pairs, and
    a device-clustered exact sign-flip test on the per-device mean difference (the conservative one: six devices
    give a smallest possible p of 1/64 = 0.016).
    """
    from scipy.stats import binom, wilcoxon

    pc = per_dev[per_dev.convention == conv].set_index(["method", "device"])[metric]
    sign = 1.0 if metric == "hv" else -1.0
    pairs = []  # (device, LLM advantage)
    for m in llms:
        for dev in per_dev[per_dev.method == m].device.unique():
            if (ctl, dev) in pc.index and np.isfinite(pc[(m, dev)]) and np.isfinite(pc[(ctl, dev)]):
                pairs.append((dev, sign * (pc[(m, dev)] - pc[(ctl, dev)])))
    if not pairs:
        return {}
    d = np.array([x for _, x in pairs])
    wins, n = int((d > 0).sum()), len(d)
    p_binom = float(binom.cdf(wins, n, 0.5))
    try:
        p_wil = float(wilcoxon(d, alternative="less").pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p_wil = float("nan")
    by_dev = pd.Series(d, index=[k for k, _ in pairs]).groupby(level=0).mean()
    return dict(wins=wins, n=n, p_binom=p_binom, p_wilcoxon=p_wil, n_dev=len(by_dev),
                dev_wins=int((by_dev > 0).sum()), p_cluster=_signflip_p(by_dev.values), mean_diff=float(d.mean()))


def _fmt_p(p: float) -> str:
    return "n/a" if not np.isfinite(p) else (f"{p:.1e}" if p < 1e-3 else f"{p:.3f}")


CONTROLS = ["PEA (no LLM)", "Random sampling (no LLM)", "NSGA-II (no LLM)", "Reg. evolution (no LLM)",
            "PEA with refill (no LLM)"]


def report(datasets=("cifar10", "cifar100", "ImageNet16-120"), out: Path | None = None) -> str:
    df = table(datasets)
    out = out or REPO / "results" / "reports" / "conventions.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)
    controls = CONTROLS
    lines = ["# Robustness of the results to the hypervolume convention\n",
             "Every archived run re-scored under 16 conventions (normalisation x reference point x scored set x "
             f"accuracy floor at {int(FLOOR_FRAC * 100)}% of the best accuracy). Device-level values are means over "
             "seeds; the tables give the mean over the devices that the method was run on. Higher HV and lower IGD "
             "are better. No search was re-run and no LLM was called.\n",
             "Significance (one-sided, H1: the LLM runs are worse than the control): *pooled sign* = binomial test "
             "on the wins, treating every LLM-device pair as independent; *Wilcoxon* = signed-rank test on the same "
             "pairs; *device-clustered* = exact sign-flip permutation test on the mean LLM-minus-control difference "
             "per device (pairs on one device share the control, so this is the conservative test; with six devices "
             "its smallest possible p is 1/64 = 0.016).\n"]
    convs = [conv_name(c) for c in CONVENTIONS]
    sig_rows = []
    for ds in datasets:
        d = df[df.dataset == ds]
        if d.empty:
            continue
        per_dev = d.groupby(["method", "device", "convention"])[["hv", "igd"]].mean().reset_index()
        ndev = per_dev.groupby("method").device.nunique()
        present = [c for c in controls if c in set(per_dev.method)]
        # head-to-head counts cover the LLM-NAS method as published; the duplicate-fix arms ("[history...]") are
        # scored in the tables but judged only by their pre-registered analysis (llmnas.dupfix)
        llms = sorted(m for m in set(per_dev.method) if m not in controls and "[" not in m)
        for metric, label in (("hv", "HV"), ("igd", "IGD")):
            piv = per_dev.groupby(["method", "convention"])[metric].mean().unstack("convention")
            lines.append(f"\n## {ds}: mean {label} over devices\n")
            lines.append("| Method | Devices | " + " | ".join(c.replace("/", " / ") for c in convs) + " |")
            lines.append("|---|---|" + "---|" * len(convs))
            for m in sorted(piv.index, key=lambda m: (m in controls, m)):
                lines.append(f"| {m} | {ndev[m]} | " + " | ".join(f"{piv.loc[m, c]:.3f}" for c in convs) + " |")
            ranks = piv.rank(ascending=(metric == "igd"))
            lines.append(f"\n### {ds}: rank by {label} under each convention (1 = best)\n")
            lines.append("| Method | best rank | worst rank |")
            lines.append("|---|---|---|")
            for m in ranks.index:
                lines.append(f"| {m} | {int(ranks.loc[m].min())} | {int(ranks.loc[m].max())} |")
            if not llms:
                continue
            lines.append(f"\n### {ds}: LLM runs that beat each control by {label} (same device and convention)\n")
            lines.append("| Convention | " + " | ".join(f"beats {c}" for c in present) + " |")
            lines.append("|---|" + "---|" * len(present))
            for c in convs:
                cells = []
                for ctl in present:
                    h = head_to_head(per_dev, llms, ctl, c, metric)
                    cells.append(f"{h['wins']}/{h['n']}" if h else "-")
                    if h:
                        sig_rows.append(dict(dataset=ds, metric=label, convention=c, control=ctl, **h))
                lines.append(f"| {c.replace('/', ' / ')} | " + " | ".join(cells) + " |")
    if sig_rows:
        sig = pd.DataFrame(sig_rows)
        sig.to_csv(out.with_name("conventions_significance.csv"), index=False)
        lines.append("\n## Significance of the head-to-head results\n")
        lines.append("Default convention (front / r=1.1 / archive / no floor), then the range over all 16 "
                     "conventions.\n")
        lines.append("| Dataset | Metric | Control | LLM wins (default) | pooled sign p | Wilcoxon p | devices won "
                     "| device-clustered p | wins over 16 conv. | max device-clustered p over 16 conv. |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        default = conv_name(CONVENTIONS[0])
        for (ds, met, ctl), g in sig.groupby(["dataset", "metric", "control"], sort=False):
            r0 = g[g.convention == default].iloc[0]
            lines.append(f"| {ds} | {met} | {ctl} | {r0.wins}/{r0.n} | {_fmt_p(r0.p_binom)} | "
                         f"{_fmt_p(r0.p_wilcoxon)} | {r0.dev_wins}/{r0.n_dev} | {_fmt_p(r0.p_cluster)} | "
                         f"{g.wins.min()}-{g.wins.max()} of {r0.n} | {_fmt_p(g.p_cluster.max())} |")
    text = "\n".join(lines) + "\n"
    out.write_text(text, encoding="utf-8")
    return text
