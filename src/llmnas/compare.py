"""Side-by-side comparison of LLMs on the same (dataset, device): ``python -m llmnas compare-llms``.

Scans every finished full-LLM-NAS run (LLM operator, six niches, knowledge base on) under
results/runs, groups them by LLM model and reports quality, how many proposals were usable,
tokens, wall time and an estimated list-price cost. Non-LLM controls are added for reference.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import numpy as np

from .benchmark import load_benchmark
from .experiments import REPO
from .metrics import FrontEvaluator, objectives

# USD per million tokens (input, output), published list prices as of 2026-10 - check before quoting.
# Local models cost nothing in API fees. Reasoning/thinking tokens are billed as output.
PRICES = {
    "gpt-4.1": (2.0, 8.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5-1": (10.0, 50.0),
}


def _cost(model: str, tin: int, tout: int, local: bool = False):
    if local:
        return 0.0  # served from localhost (Ollama etc.): no API fee
    if model in PRICES:
        pi, po = PRICES[model]
        return (tin * pi + tout * po) / 1e6
    return 0.0 if model.startswith(("qwen", "llama", "gemma", "mistral", "gpt-oss")) else float("nan")


def _floored_evaluator(dataset: str, device: str, min_acc: float) -> FrontEvaluator:
    """True front restricted to architectures with test accuracy >= min_acc (drops degenerate chance-level cells)."""
    b = load_benchmark(dataset)
    lat = b.latencies(device)
    keep = (~np.isnan(lat)) & (b.test_acc >= min_acc)
    return FrontEvaluator(objectives(lat[keep], b.test_acc[keep]))


def collect(dataset: str, device: str, results_dir: str | Path | None = None, min_acc: float = 80.0) -> pd.DataFrame:
    root = Path(results_dir) if results_dir else REPO / "results"
    rows = []
    floored = _floored_evaluator(dataset, device, min_acc)
    for f in sorted((root / "runs").glob(f"*/{dataset}/{device}/seed*/run.json")):
        r = json.loads(f.read_text())
        c = r["config"]
        if c.get("type", "search") != "search":
            continue
        s, u = c["search"], r.get("llm_usage") or {}
        st = r["proposal_stats"]
        llm = s["operator"] == "llm"
        if llm and not (s["partitioning"] and s.get("knowledge_base", True)):
            continue  # ablated LLM variants are not the full method
        if c["llm"].get("backend") == "mock":
            continue
        if not llm and not s["partitioning"]:
            continue
        model = c["llm"]["model"] if llm else {"random": "PEA (no LLM)", "random_sample": "random sampling (no LLM)"}[s["operator"]]
        if llm and c["llm"].get("effort"):
            model += f" [effort={c['llm']['effort']}]"
        pts = np.array([[q["lat"], q["true_acc"]] for q in r["final_front"]], float).reshape(-1, 2)
        kept = pts[pts[:, 1] >= min_acc]
        fs = floored.score(objectives(kept[:, 0], kept[:, 1])) if len(kept) else {"hv": 0.0, "igd": float("nan")}
        rows.append(dict(
            method=model, seed=r["seed"], hv=r["hv"], igd=r["igd"], hv_floor=fs["hv"], igd_floor=fs["igd"],
            degenerate=bool((pts[:, 1] < 20).any()), n_front=r["n_front"], n_eval=r["n_evaluated"],
            usable=st["accepted"] / max(st["proposed"], 1), out_of_niche=st["out_of_niche"], duplicate=st["duplicate"],
            invalid=st["invalid"], calls=u.get("api_calls", 0), tokens_in=u.get("input_tokens", 0),
            tokens_out=u.get("output_tokens", 0), minutes=r["search_time_s"] / 60,
            cost_usd=_cost(c["llm"]["model"], u.get("input_tokens", 0), u.get("output_tokens", 0),
                           "localhost" in str(c["llm"].get("base_url") or "")) if llm else 0.0,
            llm=llm,
        ))  # fmt: skip
    return pd.DataFrame(rows)


def compare_llms(dataset: str, device: str, results_dir: str | Path | None = None, min_acc: float = 80.0) -> str:
    df = collect(dataset, device, results_dir, min_acc)
    if df.empty:
        return f"No finished runs for {dataset}/{device}."
    order = df.groupby("method").llm.first().sort_values(ascending=False).index  # LLMs first
    lines = [f"LLM comparison on {dataset} / {device} (full LLM-NAS: 6 niches, knowledge base on)\n",
             f"HV/IGD: default convention (all architectures) and re-scored on architectures with test accuracy >= "
             f"{min_acc:g}% only (drops the degenerate chance-level cell on the true front). 'Finds 10% cell' = the "
             "final front contains an architecture below 20% accuracy.\n",
             f"| Method | Seeds | HV↑ | IGD↓ | HV (acc≥{min_acc:g}%)↑ | IGD (acc≥{min_acc:g}%)↓ | Finds 10% cell | Usable | "
             "Out-of-niche | Duplicates | Tokens in/out | Minutes | Est. cost / search |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for m in order:
        g = df[df.method == m]
        pm = lambda x: f"{x.mean():.3f}" + (f" ± {x.std(ddof=0):.3f}" if len(x) > 1 else "")  # noqa: E731
        cost = "n/a (no API)" if not g.llm.iloc[0] or g.cost_usd.iloc[0] == 0 else f"${g.cost_usd.mean():.2f}"
        if g.llm.iloc[0] and pd.isna(g.cost_usd.iloc[0]):
            cost = "unknown price"
        tok = f"{g.tokens_in.mean() / 1e3:.0f}K / {g.tokens_out.mean() / 1e3:.0f}K" if g.llm.iloc[0] else "-"
        lines.append(f"| {m} | {len(g)} | {pm(g.hv)} | {pm(g.igd)} | {pm(g.hv_floor)} | {pm(g.igd_floor)} | "
                     f"{int(g.degenerate.sum())}/{len(g)} | {g.usable.mean():.0%} | {g.out_of_niche.mean():.0f} | "
                     f"{g.duplicate.mean():.0f} | {tok} | {g.minutes.mean():.1f} | {cost} |")  # fmt: skip
    lines.append("\nNote: LLM rows are single runs unless 'Seeds' says otherwise; cost uses list prices in "
                 "llmnas/compare.py (thinking tokens are billed as output).")
    return "\n".join(lines)
