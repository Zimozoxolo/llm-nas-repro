"""Tables and figures from finished runs, side by side with the paper's numbers.

Everything is written to ``results/reports/<experiment>/`` as Markdown + CSV (+ PNG).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import paper_results as P  # noqa: E402
from .benchmark import load_benchmark  # noqa: E402
from .data import DEVICE_LABELS  # noqa: E402
from .experiments import REPO, load_results  # noqa: E402
from .metrics import objectives, pareto_mask  # noqa: E402
from .search_space import CONV3, str_to_code  # noqa: E402

DS_LABEL = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "ImageNet16-120": "ImageNet16-120"}


def _out_dir(exp: dict) -> Path:
    root = Path(exp["base"].get("results_dir") or "results")
    d = (root if root.is_absolute() else REPO / root) / "reports" / exp["name"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pm(x: pd.Series, digits: int = 3) -> str:
    if len(x) == 0:
        return "-"
    return f"{x.mean():.{digits}f}" + (f" ± {x.std(ddof=0):.{digits}f}" if len(x) > 1 else "")


def summary_frame(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        u = r.get("llm_usage") or {}
        rows.append(dict(variant=r["variant"], dataset=r["dataset"], device=r["device"], seed=r["seed"],
                         hv=r["hv"], igd=r["igd"], n_front=r["n_front"], n_evaluated=r["n_evaluated"],
                         api_calls=u.get("api_calls", 0), input_tokens=u.get("input_tokens", 0),
                         output_tokens=u.get("output_tokens", 0), search_time_s=r["search_time_s"],
                         predictor_spearman=(r.get("predictor") or {}).get("spearman_test")))  # fmt: skip
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- tables


def table3(df: pd.DataFrame, variant: str = "llmnas") -> str:
    lines = ["| Dataset | Method | " + " | ".join(f"{DEVICE_LABELS[d]} HV↑ | IGD↓" for d in P.DEVICE_ORDER) + " |",
             "|---|---|" + "---|---|" * len(P.DEVICE_ORDER)]  # fmt: skip
    for ds in P.TABLE3:
        sub = df[(df.dataset == ds) & (df.variant == variant)]
        if sub.empty:
            continue
        cells = []
        for d in P.DEVICE_ORDER:
            s = sub[sub.device == d]
            cells += [_pm(s.hv), _pm(s.igd)]
        lines.append(f"| {DS_LABEL[ds]} | **Ours (reproduced)** | " + " | ".join(cells) + " |")
        for m, vals in P.TABLE3[ds].items():
            cells = [f"{vals[d][0]:.3f} | {vals[d][1]:.3f}" for d in P.DEVICE_ORDER]
            lines.append(f"| | {m} (paper) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table5(df: pd.DataFrame, dataset: str = "cifar100") -> str:
    sub = df[df.dataset == dataset]
    lines = ["| Variant | Avg HV↑ (ours) | Avg IGD↓ (ours) | Avg HV (paper) | Avg IGD (paper) | #runs |",
             "|---|---|---|---|---|---|"]  # fmt: skip
    for v, g in sub.groupby("variant", sort=False):
        per_dev = g.groupby("device")[["hv", "igd"]].mean()  # average seeds, then ± across devices
        paper = P.TABLE5.get(v)
        ph = f"{paper[0]:.3f} ± {paper[1]:.3f}" if paper else "-"
        pi = f"{paper[2]:.4f} ± {paper[3]:.4f}" if paper else "-"
        lines.append(f"| {v} | {_pm(per_dev.hv)} | {_pm(per_dev.igd, 4)} | {ph} | {pi} | {len(g)} |")
    return "\n".join(lines)


def table4(df: pd.DataFrame) -> str:
    g = df.groupby("variant").agg(api_calls=("api_calls", "mean"), in_tok=("input_tokens", "mean"),
                                  out_tok=("output_tokens", "mean"), time_s=("search_time_s", "mean"),
                                  runs=("seed", "count"))  # fmt: skip
    lines = ["| Variant | API calls / search | input tokens | output tokens | wall time / search | runs |",
             "|---|---|---|---|---|---|"]  # fmt: skip
    for v, r in g.iterrows():
        lines.append(f"| {v} | {r.api_calls:.0f} | {r.in_tok:,.0f} | {r.out_tok:,.0f} | {r.time_s / 60:.1f} min | {int(r.runs)} |")
    lines.append("\nPaper: " + "; ".join(f"{k}: {v}" for k, v in P.TABLE4.items()))
    return "\n".join(lines)


def _knee(lat: np.ndarray, acc: np.ndarray) -> int:
    x = (lat - lat.min()) / (np.ptp(lat) or 1)
    y = (acc.max() - acc) / (np.ptp(acc) or 1)
    return int(np.argmax(np.abs(x + y - 1)))  # distance from the line joining the two extremes


def table2(results: list[dict], min_acc: float = 89.0) -> str:
    """Pooled over seeds: highest accuracy, knee point, fastest with acc >= min_acc."""
    lines = ["| Device | Row | Ours: acc (%) | Ours: lat (ms) | arch | Paper: acc | Paper: lat |",
             "|---|---|---|---|---|---|---|"]  # fmt: skip
    for dev, paper_rows in P.TABLE2_LLMNAS.items():
        pts = {}
        for r in results:
            if r["dataset"] == "cifar10" and r["device"] == dev and r["variant"] == "llmnas":
                for f in r["final_front"]:
                    pts[f["code"]] = (f["lat"], f["true_acc"])
        if not pts:
            continue
        codes = list(pts)
        lat = np.array([pts[c][0] for c in codes])
        acc = np.array([pts[c][1] for c in codes])
        m = pareto_mask(objectives(lat, acc))
        codes, lat, acc = [c for c, k in zip(codes, m) if k], lat[m], acc[m]
        ok = np.where(acc >= min_acc)[0]
        picks = {"top accuracy": int(np.argmax(acc)), "knee": _knee(lat, acc),
                 f"fastest ≥{min_acc:g}%": int(ok[np.argmin(lat[ok])]) if len(ok) else None}  # fmt: skip
        for (name, i), pr in zip(picks.items(), paper_rows):
            ours = (f"{acc[i]:.2f} | {lat[i]:.2f} | {codes[i]}") if i is not None else "- | - | -"
            lines.append(f"| {DEVICE_LABELS[dev]} | {name} | {ours} | {pr[0]:.2f}±{pr[1]:.2f} | {pr[2]:.2f} |")
    return "\n".join(lines)


def paper_point_check(data_dir=None, tol_acc: float = 0.01, tol_lat: float = 0.01) -> str:
    """Do the paper's Table-2 LLM-NAS points exist in our benchmark tables? (data-alignment check)"""
    bench = load_benchmark("cifar10", data_dir)
    lines = ["| Device | Paper acc | Paper lat | #matching archs in HW-NAS-Bench/NB201 | example |", "|---|---|---|---|---|"]
    for dev, rows in P.TABLE2_LLMNAS.items():
        lat = bench.latencies(dev)
        for acc, _, l in rows:
            idx = np.where((np.abs(bench.test_acc - acc) <= tol_acc + 0.005) & (np.abs(lat - l) <= tol_lat + 0.005))[0]
            ex = bench.df.code.iloc[idx[0]] if len(idx) else "-"
            lines.append(f"| {DEVICE_LABELS[dev]} | {acc:.2f} | {l:.2f} | {len(idx)} | {ex} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- figures


def plot_fronts(results: list[dict], dataset: str, out: Path, seed: int = 0, title: str = "") -> None:
    """Figures 6-8 style: true front vs each variant's final front, one panel per device."""
    bench = load_benchmark(dataset)
    devices = [d for d in P.DEVICE_ORDER if any(r["device"] == d and r["dataset"] == dataset for r in results)]
    if not devices:
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, dev in zip(axes.flat, devices):
        obj = objectives(bench.latencies(dev), bench.test_acc)
        tf = obj[pareto_mask(obj)]
        tf = tf[np.argsort(tf[:, 0])]
        ax.scatter(obj[:, 0], -obj[:, 1], s=2, c="0.85", label="Search space", rasterized=True)
        ax.step(tf[:, 0], -tf[:, 1], where="post", c="k", lw=1.2, ls="--", label="True Pareto front")
        for r in results:
            if r["dataset"] != dataset or r["device"] != dev or r["seed"] != seed or "final_front" not in r:
                continue
            f = np.array([[p["lat"], p["true_acc"]] for p in r["final_front"]])
            if len(f) == 0:
                continue
            f = f[np.argsort(f[:, 0])]
            ax.step(f[:, 0], f[:, 1], where="post", marker="o", ms=3, lw=1, label=f"{r['variant']} (HV {r['hv']:.3f})")
        ax.set_title(DEVICE_LABELS[dev])
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(np.percentile(bench.test_acc, 5), bench.test_acc.max() + 1)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(title or f"Pareto fronts on {DS_LABEL[dataset]} (seed {seed})")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_convergence(df_traj: pd.DataFrame, out: Path) -> None:
    if df_traj.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for v, g in df_traj.groupby("variant"):
        m = g.groupby("gen")[["hv", "igd"]].mean()
        axes[0].plot(m.index, m.hv, marker="o", label=v)
        axes[1].plot(m.index, m.igd, marker="o", label=v)
    axes[0].set(xlabel="Generation", ylabel="HV (mean over runs)")
    axes[1].set(xlabel="Generation", ylabel="IGD (mean over runs)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_bias(results: list[dict], device: str, out: Path) -> None:
    """Figure 1: generated architectures and their front for each strategy."""
    runs = [r for r in results if r["device"] == device and r["seed"] == min(x["seed"] for x in results)]
    if not runs:
        return
    bench = load_benchmark(runs[0]["dataset"])
    obj = objectives(bench.latencies(device), bench.test_acc)
    tf = obj[pareto_mask(obj)]
    tf = tf[np.argsort(tf[:, 0])]
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 4.2), sharey=True)
    for ax, r in zip(np.atleast_1d(axes), runs):
        pts = np.array([[e["lat"], e["true_acc"]] for e in r["evaluated"]])
        ax.scatter(obj[:, 0], -obj[:, 1], s=2, c="0.85", rasterized=True, label="Search space")
        ax.step(tf[:, 0], -tf[:, 1], where="post", c="k", ls="--", lw=1, label="Theoretical frontier")
        ax.scatter(pts[:, 0], pts[:, 1], s=12, label="Generated")
        f = pts[pareto_mask(objectives(pts[:, 0], pts[:, 1]))]
        f = f[np.argsort(f[:, 0])]
        ax.step(f[:, 0], f[:, 1], where="post", c="C3", lw=1.5, label="Method frontier")
        ax.set_title(f"{r['variant']}  (HV {r['hv']:.3f}, {len(pts)} unique)")
        ax.set_xlabel("Latency (ms)")
        ax.set_ylim(70, 100) if r["dataset"] == "cifar10" else None
    np.atleast_1d(axes)[0].set_ylabel("Accuracy (%)")
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Exploration bias - {DEVICE_LABELS[device]}, {DS_LABEL[runs[0]['dataset']]}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_fig3(out: Path, dataset: str = "cifar10") -> None:
    """Figure 3: parameter count distribution grouped by number of nor_conv_3x3."""
    bench = load_benchmark(dataset)
    n3 = np.array([sum(c == CONV3 for c in str_to_code(s)) for s in bench.df.arch_str])
    params = bench.df.zc_params.to_numpy()
    fig, ax = plt.subplots(figsize=(6, 4))
    for k in range(int(n3.max()) + 1):
        ax.hist(params[n3 == k], bins=40, alpha=0.7, label=f"{k} × conv_3×3")
    ax.set(xlabel="Parameters (M)", ylabel="Count", title="HW-NAS-Bench search space")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- entry point


def make_report(exp: dict) -> Path:
    out = _out_dir(exp)
    results = load_results(exp)
    if not results:
        raise SystemExit(f"No finished runs for experiment {exp['name']!r} yet.")
    df = summary_frame(results)
    df.to_csv(out / "runs.csv", index=False)
    traj = pd.DataFrame([dict(variant=r["variant"], dataset=r["dataset"], device=r["device"], seed=r["seed"], **t)
                         for r in results for t in r.get("trajectory", [])])  # fmt: skip
    traj.to_csv(out / "trajectories.csv", index=False)

    md = [f"# Report: {exp['name']}\n", f"{len(results)} finished runs.\n"]
    if exp.get("type", "search") == "search":
        if "llmnas" in df.variant.values:
            md += ["## Table 3 - HV / IGD (mean ± std over seeds)\n", table3(df), ""]
            md += ["## Table 2 - selected CIFAR-10 architectures (pooled over seeds)\n", table2(results), ""]
        if len(exp["variants"]) > 1:
            for ds in sorted(df.dataset.unique()):
                md += [f"## Table 5 - variants on {DS_LABEL[ds]} (seed-mean, ± across devices)\n", table5(df, ds), ""]
    md += ["## Table 4 - search cost\n", table4(df), ""]
    if exp.get("type") == "bias":
        md += ["## Exploration bias - HV/IGD per strategy\n",
               df.groupby(["variant", "device"])[["hv", "igd", "n_evaluated"]].mean().round(4).to_markdown(), ""]  # fmt: skip
    md += ["## Per-run results\n", df.round(4).to_markdown(index=False)]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")

    for ds in df.dataset.unique():
        plot_fronts(results, ds, out / f"fronts_{ds}.png")
    plot_convergence(traj, out / "convergence.png")
    if exp.get("type") == "bias":
        for dev in df.device.unique():
            plot_bias(results, dev, out / f"bias_{dev}.png")
    print(f"Report written to {out / 'report.md'}")
    return out
