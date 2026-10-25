"""Fronts figure for the CIFAR-10 / Edge GPU comparison: every finished full-LLM-NAS run (seed 0) plus the
non-LLM controls, against the true Pareto front. Re-run after adding LLMs:  python report/make_cnn_figure.py"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from llmnas.benchmark import load_benchmark
from llmnas.metrics import objectives, pareto_mask

ROOT = os.path.join(os.path.dirname(__file__), "..")
b = load_benchmark("cifar10")
obj = objectives(b.latencies("edgegpu"), b.test_acc)
tf = obj[pareto_mask(obj)]
tf = tf[np.argsort(tf[:, 0])]

runs = []
for f in sorted(glob.glob(os.path.join(ROOT, "results/runs/*/cifar10/edgegpu/seed0/run.json"))):
    r = json.load(open(f))
    c = r["config"]
    if c.get("type", "search") != "search" or c["llm"].get("backend") == "mock":
        continue
    s = c["search"]
    if s["operator"] == "llm" and s["partitioning"] and s.get("knowledge_base", True):
        label = c["llm"]["model"].replace("-8k", "").replace("-16k", "")
    elif s["operator"] == "random" and s["partitioning"] and s["generations"] == 10:
        label = "PEA (no LLM)"
    elif s["operator"] == "random_sample" and s["partitioning"]:
        label = "Random sampling in niches"
    else:
        continue
    runs.append((label, r))

fig, ax = plt.subplots(figsize=(7.2, 4.8))
ax.scatter(obj[:, 0], -obj[:, 1], s=2, c="0.85", rasterized=True, label="Search space (15,625)")
ax.step(tf[:, 0], -tf[:, 1], where="post", c="k", ls="--", lw=1.2, label="True Pareto front")
for label, r in runs:
    f = np.array([[p["lat"], p["true_acc"]] for p in r["final_front"]])
    f = f[np.argsort(f[:, 0])]
    ls = ":" if "no LLM" in label or "Random" in label else "-"
    ax.step(f[:, 0], f[:, 1], where="post", marker="o", ms=3, lw=1, ls=ls, label=f"{label} (HV {r['hv']:.3f})")
ax.set(xlabel="Latency (ms)", ylabel="Test accuracy (%)", ylim=(80, 95.5), title="CIFAR-10, Edge GPU, seed 0")
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "figures", "cnn_fronts_edgegpu.png")
fig.savefig(out, dpi=160)
print("wrote", out, [lab for lab, _ in runs])
