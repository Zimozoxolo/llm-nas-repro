"""Sanity checks that the local benchmark tables match what the paper used."""

from __future__ import annotations

import numpy as np

from .benchmark import load_benchmark
from .data import DATASETS, DEVICES
from .predictor import build_predictor
from .report import paper_point_check
from .search_space import PAPER_NICHES

# NB201 200-epoch test-accuracy optima (Dong & Yang, 2020)
EXPECTED_BEST = {"cifar10": 94.37, "cifar100": 73.51, "ImageNet16-120": 47.31}
# Example rows printed in Figure 2 of the paper: code -> (latency ms, accuracy %)
FIG2_EXAMPLES = {(2, 2, 3, 1, 0, 3): (5.42, 71.77), (1, 2, 3, 1, 0, 3): (4.39, 70.91),
                 (2, 3, 3, 2, 0, 0): (1.77, 61.73), (3, 1, 3, 1, 2, 0): (3.34, 69.59),
                 (1, 3, 3, 1, 2, 0): (2.56, 64.17)}  # fmt: skip


def check_data(data_dir=None) -> None:
    print("== Benchmark tables")
    for ds in DATASETS:
        b = load_benchmark(ds, data_dir)
        best = b.test_acc.max()
        flag = "OK" if abs(best - EXPECTED_BEST[ds]) < 0.02 else "MISMATCH"
        print(f"  {ds:15s} rows={len(b.df)}  best test acc={best:.2f} (expected {EXPECTED_BEST[ds]}) {flag}")

    print("\n== Niche sizes (Table 1)")
    for n in PAPER_NICHES:
        print(f"  {n.name}: {len(n.members()):5d} architectures - {n.rationale}")

    print("\n== Paper Table-2 LLM-NAS points found in the benchmark (CIFAR-10)")
    print(paper_point_check(data_dir))

    print("\n== Figure-2 example rows: best-matching (dataset, device)")
    for code, (lat, acc) in FIG2_EXAMPLES.items():
        hits = []
        for ds in DATASETS:
            b = load_benchmark(ds, data_dir)
            a = b.accuracy(code)
            for dev in DEVICES:
                l = b.latency(code, dev)
                hits.append((abs(a - acc) / 10 + abs(l - lat) / max(lat, 1e-9), ds, dev, a, l))
        err, ds, dev, a, l = min(hits)
        print(f"  {code}: paper lat={lat} acc={acc} -> closest {ds}/{dev}: lat={l:.2f} acc={a:.2f}")

    print("\n== ZC-ensemble predictor (paper: Spearman ~0.90)")
    for ds in DATASETS:
        b = load_benchmark(ds, data_dir)
        reps = [build_predictor(b, "zc_ensemble", seed=s).report for s in range(3)]
        syn = build_predictor(b, "synflow", seed=0).report
        sp = np.array([r.spearman_test for r in reps])
        print(f"  {ds:15s} XGB(13 ZC) Spearman vs test acc = {sp.mean():.3f} ± {sp.std():.3f}   "
              f"(synflow alone: {syn.spearman_test:.3f})")  # fmt: skip
