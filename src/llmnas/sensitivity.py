"""Predictor-size sensitivity of the no-LLM controls: ``python -m llmnas sensitivity``.

The paper does not say how many architectures the XGBoost predictor is trained on (this repo assumes 1,000).
The controls are cheap, so they are re-run with ``predictor.n_train`` = 200, 1,000 and 5,000 on all three
datasets and six devices (three seeds each) to check that the comparison between methods does not depend on
that choice. LLM runs are not repeated (that would cost API money); only the controls are.

    python -m llmnas run baselines --set predictor.n_train=200
    python -m llmnas run table5_ablation --variants no_llm_pea random_sampling \
        --datasets cifar10 cifar100 ImageNet16-120 --set predictor.n_train=200
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .experiments import REPO

LABELS = {"random": "PEA (no LLM)", "random_sample": "Random sampling (no LLM)", "nsga2": "NSGA-II (no LLM)",
          "regevo": "Reg. evolution (no LLM)"}  # fmt: skip
DATASETS = ["cifar10", "cifar100", "ImageNet16-120"]


def table(root: Path | None = None) -> pd.DataFrame:
    root = root or REPO / "results"
    rows = []
    for f in sorted((root / "runs").glob("*/*/*/seed*/run.json")):
        r = json.loads(f.read_text())
        c = r["config"]
        if c["llm"].get("backend") == "mock" or c.get("type") == "slm" or r["dataset"] not in DATASETS:
            continue
        op = c["search"]["operator"]
        if op not in LABELS or c["predictor"]["kind"] != "zc_ensemble":
            continue
        if op in ("random", "random_sample") and not c["search"]["partitioning"]:
            continue
        p = r.get("predictor") or {}
        rows.append(dict(method=LABELS[op], dataset=r["dataset"], device=r["device"], seed=r["seed"],
                         n_train=c["predictor"]["n_train"], hv=r["hv"], igd=r["igd"],
                         spearman=p.get("spearman_test", p.get("spearman", float("nan")))))  # fmt: skip
    return pd.DataFrame(rows).drop_duplicates(["method", "dataset", "device", "seed", "n_train"])


def report(out: Path | None = None) -> str:
    df = table()
    out = out or REPO / "results" / "reports" / "sensitivity_ntrain.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)
    sizes = sorted(df.n_train.unique())
    lines = ["# Predictor training-set size: sensitivity of the no-LLM controls\n",
             "HV and IGD under the default convention (front normalisation, reference point 1.1, archived "
             "front), mean over 6 devices x 3 seeds. `rank` orders the four controls by HV within each "
             "(dataset, n_train). The LLM runs all use n_train = 1,000 and are not repeated here.\n"]
    for ds in DATASETS:
        d = df[df.dataset == ds]
        if d.empty:
            continue
        g = d.groupby(["method", "n_train"]).agg(hv=("hv", "mean"), igd=("igd", "mean"), sp=("spearman", "mean"),
                                                 n=("hv", "size")).reset_index()  # fmt: skip
        g["rank"] = g.groupby("n_train").hv.rank(ascending=False).astype(int)
        lines.append(f"\n## {ds}\n")
        lines.append("| Method | " + " | ".join(f"HV (n={s})" for s in sizes) + " | "
                     + " | ".join(f"IGD (n={s})" for s in sizes) + " | " + " | ".join(f"rank (n={s})" for s in sizes)
                     + " |")  # fmt: skip
        lines.append("|---|" + "---|" * (3 * len(sizes)))
        for m in sorted(g.method.unique()):
            row = g[g.method == m].set_index("n_train")
            cell = lambda col, fmt: [fmt.format(row.loc[s, col]) if s in row.index else "-" for s in sizes]  # noqa: E731
            lines.append(f"| {m} | " + " | ".join(cell("hv", "{:.3f}")) + " | " + " | ".join(cell("igd", "{:.4f}"))
                         + " | " + " | ".join(cell("rank", "{:d}")) + " |")  # fmt: skip
        sp = g.groupby("n_train").sp.mean()
        if sp.notna().any():
            lines.append("\nPredictor Spearman correlation on held-out architectures: "
                         + ", ".join(f"n={s}: {sp[s]:.3f}" for s in sizes if s in sp.index) + "\n")  # fmt: skip
    text = "\n".join(lines) + "\n"
    out.write_text(text, encoding="utf-8")
    return text
