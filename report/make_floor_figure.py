"""HV (default convention) vs HV with architectures below 80% accuracy removed, per method.
Run after the experiments:  python report/make_floor_figure.py"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.insert(0, os.path.dirname(__file__))
from paper_numbers import table  # noqa: E402  (same scoring as the paper's tables)

df = table()
df = df[df.device == "edgegpu"].rename(columns={"hv80": "hv_floor", "cell": "degenerate"})
order = ["claude-opus-5-5", "gpt-4.1", "qwen2.5-7b-8k", "deepseek-coder-v2-16b-8k", "PEA (no LLM)", "Random sampling (no LLM)",
         "NSGA-II (no LLM)", "Reg. evolution (no LLM)"]
order = [m for m in order if m in set(df.method)]
labels = {"claude-opus-5-5": "Claude\nOpus 5.5", "gpt-4.1": "GPT-4.1", "qwen2.5-7b-8k": "Qwen2.5\n7B", "deepseek-coder-v2-16b-8k": "DeepSeek\nCoder-V2",
          "PEA (no LLM)": "PEA\n(no LLM)", "Random sampling (no LLM)": "Random\nsampling",
          "NSGA-II (no LLM)": "NSGA-II", "Reg. evolution (no LLM)": "Reg.\nevolution"}
x = np.arange(len(order))
fig, ax = plt.subplots(figsize=(8.4, 3.8))
for off, col, lab, key in ((-0.19, "C0", "All architectures (default)", "hv"), (0.19, "C1", "Architectures with accuracy ≥ 80% only", "hv_floor")):
    m = [df[df.method == k][key].mean() for k in order]
    s = [df[df.method == k][key].std(ddof=1) if (df.method == k).sum() > 1 else 0 for k in order]
    ax.bar(x + off, m, 0.36, yerr=s, capsize=3, color=col, label=lab)
    for xi, mi in zip(x + off, m):
        ax.text(xi, mi + 0.012, f"{mi:.2f}", ha="center", fontsize=7)
for xi, k in zip(x, order):  # mark methods whose final front contains the degenerate 10% cell
    g = df[df.method == k]
    if g.degenerate.any():
        ax.text(xi, 0.45, f"cell found\n{int(g.degenerate.sum())}/{len(g)}", ha="center", fontsize=7, color="white")
ax.set_xticks(x, [labels[k] for k in order], fontsize=8)
ax.set(ylabel="Hypervolume (fraction of true front)", ylim=(0.4, 1.08), title="CIFAR-10, Edge GPU: effect of the degenerate 10%-accuracy cell")
ax.legend(fontsize=7, loc="upper left", ncol=2)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "figures", "hv_floor.png")
fig.savefig(out, dpi=160)
print("wrote", out)
