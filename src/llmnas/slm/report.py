"""Report for SLM experiments.

There is no exhaustive ground truth for this space (unlike HW-NAS-Bench), so HV and
IGD are computed against the **pooled reference front**: the non-dominated set of
every feasible architecture evaluated by any run of the experiment for the same
(space, device). Objectives are min-max normalised over that pool, reference point
1.1 per objective, HV reported as HV(run) / HV(pooled front). This is the standard
practice when the true front is unknown; numbers are therefore only comparable
*within* one experiment.
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..experiments import REPO, load_results  # noqa: E402
from ..metrics import hypervolume_2d, igd  # noqa: E402
from .costs import Workload, device_from_config, evaluate_costs  # noqa: E402
from .prompts import OBJ_LABELS  # noqa: E402
from .reference import REFERENCE_SLMS  # noqa: E402
from .search import _load_device, nondominated  # noqa: E402
from .space import Sampler, SpaceSpec, niches_from_config  # noqa: E402


def hypervolume(points: np.ndarray, ref: np.ndarray) -> float:
    """Exact HV for 2 or 3 minimised objectives (3-D by slicing along the last objective)."""
    pts = np.asarray(points, float)
    pts = pts[np.all(pts < ref, axis=1)]
    if len(pts) == 0:
        return 0.0
    if pts.shape[1] == 2:
        return hypervolume_2d(pts, ref)
    pts = pts[np.argsort(pts[:, 2])]
    zs = list(pts[:, 2]) + [ref[2]]
    return float(sum(hypervolume_2d(pts[: i + 1, :2], ref[:2]) * (zs[i + 1] - zs[i]) for i in range(len(pts))))


def _feasible_obj(run: dict, max_gen: int | None = None, objectives: list[str] | None = None) -> np.ndarray:
    """Objective rows of feasible evaluated architectures; ``objectives`` re-scores saved runs
    on a different objective set (e.g. quality+latency) without re-running anything."""
    objs = objectives or run["objectives"]
    rows = [[e["costs"][OBJ_LABELS[o]] for o in objs] for e in run["evaluated"]
            if e["costs"]["feasible"] and (max_gen is None or e["gen"] <= max_gen)]  # fmt: skip
    return np.array(rows, float).reshape(-1, len(objs))


def score_runs(results: list[dict], objectives: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, traj = [], []
    groups: dict[tuple, list] = {}
    for r in results:
        groups.setdefault((r["dataset"], r["device"]), []).append(r)
    for (space, dev), group in groups.items():
        pool = np.vstack([_feasible_obj(r, objectives=objectives) for r in group])
        lo, hi = pool.min(0), pool.max(0)
        norm = lambda x: (x - lo) / np.where(hi > lo, hi - lo, 1)  # noqa: E731
        ref_front = norm(pool[nondominated(pool)])
        r_pt = np.full(pool.shape[1], 1.1)
        hv_ref = hypervolume(ref_front, r_pt)
        for r in group:
            for g in range(r["config"]["search"]["generations"] + 1):
                o = _feasible_obj(r, g, objectives)
                f = norm(o[nondominated(o)]) if len(o) else o
                s = dict(variant=r["variant"], space=space, device=dev, seed=r["seed"], gen=g,
                         hv=hypervolume(f, r_pt) / hv_ref if len(f) else 0.0,
                         igd=igd(f, ref_front) if len(f) else np.nan, n_front=len(f))  # fmt: skip
                traj.append(s)
            st = r["proposal_stats"]
            u = r.get("llm_usage") or {}
            rows.append({**traj[-1], "n_evaluated": r["n_evaluated"],
                         "valid_rate": st["accepted"] / max(st["proposed"], 1),
                         "out_of_niche": st["out_of_niche"], "duplicate": st["duplicate"], "repaired": st["repaired"],
                         "api_calls": u.get("api_calls", 0), "time_s": r["search_time_s"]})  # fmt: skip
    df = pd.DataFrame(rows).drop(columns=["gen"])
    return df, pd.DataFrame(traj)


def _size_latency(run: dict) -> np.ndarray:
    """(decoder params M, latency ms) of every feasible evaluated architecture."""
    return np.array([[e["costs"]["decoder_params_m"], e["costs"]["latency_ms"]]
                     for e in run["evaluated"] if e["costs"]["feasible"]], float).reshape(-1, 2)  # fmt: skip


def _best_latency_at_least(pts: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """For each size level N: the lowest latency among points with >= N decoder params (NaN if none)."""
    order = np.argsort(-pts[:, 0])  # largest first, running minimum of latency
    sizes, run_min = pts[order, 0], np.minimum.accumulate(pts[order, 1])
    idx = np.searchsorted(-sizes, -levels, side="right") - 1  # last point with size >= level
    return np.where(idx >= 0, run_min[np.clip(idx, 0, None)], np.nan)


def latency_overhead(results: list[dict], n_levels: int = 40) -> pd.DataFrame:
    """Latency overhead at matched quality.

    Quality is the decoder-parameter proxy, so "at least as good" = ">= N decoder params".
    For log-spaced N across the pooled front of each (space, device): t*(N) is the fastest
    such model found by *any* run, t_m(N) the fastest found by this run. Reported:
    geometric mean of t_m/t* - 1 over covered levels (0 = matches the best everywhere),
    coverage = share of levels where the run has any qualifying model, and the same
    overhead restricted to each niche's size band.
    """
    rows = []
    groups: dict[tuple, list] = {}
    for r in results:
        groups.setdefault((r["dataset"], r["device"]), []).append(r)
    for (space, dev), group in groups.items():
        pool = np.vstack([_size_latency(r) for r in group])
        front = pool[nondominated(np.column_stack([-pool[:, 0], pool[:, 1]]))]
        levels = np.geomspace(front[:, 0].min(), front[:, 0].max(), n_levels)
        best = _best_latency_at_least(pool, levels)
        bounds = list(group[0]["config"]["spaces"][space]["niches"]) + [np.inf]
        for r in group:
            ratio = _best_latency_at_least(_size_latency(r), levels) / best
            ok = ~np.isnan(ratio)
            row = dict(variant=r["variant"], space=space, device=dev, seed=r["seed"],
                       lat_overhead=float(np.exp(np.mean(np.log(ratio[ok]))) - 1) if ok.any() else np.nan,
                       coverage=float(ok.mean()))  # fmt: skip
            for i in range(len(bounds) - 1):
                band = ok & (levels >= bounds[i]) & (levels < bounds[i + 1])
                row[f"S{i}"] = float(np.exp(np.mean(np.log(ratio[band]))) - 1) if band.any() else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def reference_comparison(results: list[dict], cfg: dict, space: str, device: str, variant: str = "llmnas") -> pd.DataFrame:
    """LTS-style claim: for each hand-designed SLM, the fastest searched model with >= its decoder params."""
    spec = SpaceSpec.from_config({**cfg["spaces"][space]["spec"], "name": space})
    dev = device_from_config(_load_device(cfg, device)[0])
    wl = Workload(**cfg["workload"])
    found = [e for r in results if r["dataset"] == space and r["device"] == device and r["variant"] == variant
             for e in r["evaluated"] if e["costs"]["feasible"]]  # fmt: skip
    rows = []
    for name, ref in REFERENCE_SLMS.items():
        c = evaluate_costs(ref["arch"], spec, dev, wl)  # same vocab as the space: architecture-only comparison
        better = [e for e in found if e["costs"]["decoder_params_m"] >= c["decoder_params_m"]]
        row = dict(model=name, decoder_params_m=round(c["decoder_params_m"], 1), latency_ms=round(c["latency_ms"], 1),
                   memory_mb=round(c["memory_mb"]), feasible=c["feasible"])  # fmt: skip
        if better:
            b = min(better, key=lambda e: e["costs"]["latency_ms"])
            row.update(found_decoder_params_m=round(b["costs"]["decoder_params_m"], 1),
                       found_latency_ms=round(b["costs"]["latency_ms"], 1), found_memory_mb=round(b["costs"]["memory_mb"]),
                       speedup=round(c["latency_ms"] / b["costs"]["latency_ms"], 2),
                       found_arch=f"L{b['arch']['n_layer']} d{b['arch']['d_model']} " + str(b["arch"]["layers"][0]))  # fmt: skip
        rows.append(row)
    return pd.DataFrame(rows)


def plot_fronts(results, cfg, space, device, out: Path, seed=0):
    spec = SpaceSpec.from_config({**cfg["spaces"][space]["spec"], "name": space})
    dev = device_from_config(_load_device(cfg, device)[0])
    wl = Workload(**cfg["workload"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for r in results:
        if r["dataset"] != space or r["device"] != device or r["seed"] != seed:
            continue
        e = [x for x in r["evaluated"] if x["costs"]["feasible"]]
        o = np.array([[x["costs"]["est_loss"], x["costs"]["latency_ms"], x["costs"]["memory_mb"]] for x in e])
        for ax, j, lab in ((axes[0], 1, "Latency (ms)"), (axes[1], 2, "Memory (MB)")):
            m = nondominated(o[:, [0, j]])
            f = o[m][np.argsort(o[m][:, j])]
            ax.step(f[:, j], f[:, 0], where="post", marker="o", ms=3, label=r["variant"])
            ax.set_xlabel(lab)
    for name, ref in REFERENCE_SLMS.items():
        c = evaluate_costs(ref["arch"], spec, dev, wl)
        for ax, key in ((axes[0], "latency_ms"), (axes[1], "memory_mb")):
            ax.scatter(c[key], c["est_loss"], marker="*", s=80, c="k")
            ax.annotate(name, (c[key], c["est_loss"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    for ax in axes:
        ax.set_ylabel("Estimated loss (decoder-param proxy)")
        ax.set_xscale("log")
        ax.legend(fontsize=7)
    fig.suptitle(f"{space} on {device} (seed {seed}); stars = hand-designed SLMs (same vocab as the space)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_niche_clusters(cfg, space, device, out: Path, n=3000):
    """Figure-3 analogue: latency distribution of random architectures, coloured by niche."""
    spec = SpaceSpec.from_config({**cfg["spaces"][space]["spec"], "name": space})
    dev = device_from_config(_load_device(cfg, device)[0])
    wl = Workload(**cfg["workload"])
    niches = niches_from_config(cfg["spaces"][space]["niches"])
    rng = random.Random(0)
    samp = Sampler(spec, max_tries=2000)
    fig, ax = plt.subplots(figsize=(7, 4))
    for nc in niches:
        lats = [evaluate_costs(a, spec, dev, wl)["latency_ms"] for a in
                (samp.sample(nc, rng, set()) for _ in range(n // len(niches))) if a]  # fmt: skip
        if lats:
            ax.hist(lats, bins=40, alpha=0.6, label=f"{nc.name}: {nc.lo:g}-{nc.hi:g}M")
    ax.set(xscale="log", xlabel="Latency (ms)", ylabel="Count", title=f"Latency by niche - {space} on {device}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _pct(x):
    x = x.dropna()
    if len(x) == 0:
        return "-"
    return f"{x.mean():+.1%}" + (f" ± {x.std(ddof=0):.1%}" if len(x) > 1 else "")


def _pm(x):
    return f"{x.mean():.3f} ± {x.std(ddof=0):.3f}" if len(x) > 1 else f"{x.mean():.3f}"


def make_slm_report(exp: dict) -> Path:
    root = Path(exp["base"].get("results_dir") or "results")
    out = (root if root.is_absolute() else REPO / root) / "reports" / exp["name"]
    out.mkdir(parents=True, exist_ok=True)
    results = load_results(exp)
    if not results:
        raise SystemExit(f"No finished runs for experiment {exp['name']!r} yet.")
    df, traj = score_runs(results)
    df.to_csv(out / "runs.csv", index=False)
    traj.to_csv(out / "trajectories.csv", index=False)
    md = [f"# Report: {exp['name']}\n", f"{len(results)} finished runs. HV/IGD are relative to the pooled "
          "reference front of this experiment (see `slm/report.py`).\n"]  # fmt: skip
    configured = results[0]["objectives"]
    df2 = score_runs(results, ["quality", "latency"])[0] if configured != ["quality", "latency"] else None
    lo = latency_overhead(results)
    lo.to_csv(out / "latency_overhead.csv", index=False)
    md += ["**Primary metric - latency overhead at matched quality:** for 40 log-spaced decoder-parameter "
           "levels N, the fastest model with >= N decoder params found by the run, relative to the fastest "
           "found by any run (geometric mean; 0% = best everywhere). Coverage = share of levels the run "
           "reaches at all.\n"]  # fmt: skip
    for (space, dev), g in df.groupby(["space", "device"]):
        md += [f"## {space} on {dev}\n",
               f"HV/IGD use the configured objectives {configured}"
               + ("; HV₂/IGD₂ re-score the same runs on (quality, latency) only." if df2 is not None else "") + "\n",
               "| Variant | latency overhead↓ | coverage | HV↑ | IGD↓ | HV₂↑ | IGD₂↓ | front | valid proposals "
               "| API calls | runs |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        lg = lo[(lo.space == space) & (lo.device == dev)]
        for v, gv in g.groupby("variant", sort=False):
            g2 = df2[(df2.space == space) & (df2.device == dev) & (df2.variant == v)] if df2 is not None else None
            two = f"{_pm(g2.hv)} | {_pm(g2.igd)}" if g2 is not None else "- | -"
            lv = lg[lg.variant == v]
            md.append(f"| {v} | {_pct(lv.lat_overhead)} | {lv.coverage.mean():.0%} | {_pm(gv.hv)} | {_pm(gv.igd)} | "
                      f"{two} | {gv.n_front.mean():.1f} | {gv.valid_rate.mean():.0%} | {gv.api_calls.mean():.0f} | "
                      f"{len(gv)} |")  # fmt: skip
        labels = next(r["config"] for r in results if r["dataset"] == space)["spaces"][space].get("niche_labels") or []
        bands = [c for c in lg.columns if c.startswith("S") and c[1:].isdigit()]
        per = lg.groupby("variant", sort=False)[bands].mean()
        md += ["", "Latency overhead per niche (mean over seeds; blank = level not reached):\n",
               "| Variant | " + " | ".join(f"{b} {labels[int(b[1:])] if int(b[1:]) < len(labels) else ''}" for b in bands) + " |",
               "|---|" + "---|" * len(bands)]  # fmt: skip
        for v, row in per.iterrows():
            md.append(f"| {v} | " + " | ".join("" if np.isnan(x) else f"{x:+.1%}" for x in row) + " |")
        cfg = next(r["config"] for r in results if r["dataset"] == space and r["device"] == dev)
        if "llmnas" in set(g.variant):
            md += ["", "**Hand-designed SLMs vs the fastest LLM-NAS model with at least as many decoder parameters** "
                   "(same cost model and vocabulary):\n", reference_comparison(results, cfg, space, dev).to_markdown(index=False)]  # fmt: skip
        plot_fronts(results, cfg, space, dev, out / f"fronts_{space}_{dev}.png")
        plot_niche_clusters(cfg, space, dev, out / f"niches_{space}_{dev}.png")
        md.append("")
    fig, ax = plt.subplots(figsize=(6, 4))
    for v, g in traj.groupby("variant"):
        m = g.groupby("gen").hv.mean()
        ax.plot(m.index, m.values, marker="o", label=v)
    ax.set(xlabel="Generation", ylabel="HV (vs pooled front)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "convergence.png", dpi=150)
    plt.close(fig)
    md += ["## Per-run results\n", df.round(4).to_markdown(index=False)]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Report written to {out / 'report.md'}")
    return out
