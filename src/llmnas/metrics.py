"""Pareto front, Hypervolume (HV) and Inverted Generational Distance (IGD) - Sec. 4.1.

Both objectives are *minimised* internally: ``(latency, -accuracy)``.

The paper does not state its normalisation or reference point. HV values close to
1.0 ("near-perfect") imply a ratio to the true front's HV, so by default we report

    HV  = HV(S, r) / HV(P*, r)
    IGD = mean_{p in P*} min_{s in S} ||p - s||_2

after min-max normalising both objectives with the ideal/nadir points of the true
front P* (``normalization="front"``) and using reference point r = (1.1, 1.1).
``normalization="space"`` uses the min/max over the whole search space instead.
Points worse than r contribute zero HV. Change these in the config if you settle
on a different convention, and say so in your write-up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def pareto_mask(points: np.ndarray) -> np.ndarray:
    """Non-dominated mask for minimisation. ``points``: (n, 2)."""
    pts = np.asarray(points, float)
    order = np.lexsort((pts[:, 1], pts[:, 0]))  # by f1, then f2
    mask = np.zeros(len(pts), bool)
    best_f2 = np.inf
    for i in order:
        if np.isnan(pts[i]).any():  # unmeasured: never on a front
            continue
        if pts[i, 1] < best_f2:
            mask[i] = True
            best_f2 = pts[i, 1]
    return mask  # exact duplicates: only the first occurrence is kept


def hypervolume_2d(points: np.ndarray, ref: np.ndarray) -> float:
    pts = np.asarray(points, float)
    pts = pts[(pts[:, 0] < ref[0]) & (pts[:, 1] < ref[1])]
    if len(pts) == 0:
        return 0.0
    pts = pts[pareto_mask(pts)]
    pts = pts[np.argsort(pts[:, 0])]
    hv, prev_f2 = 0.0, ref[1]
    for f1, f2 in pts:
        hv += (ref[0] - f1) * (prev_f2 - f2)
        prev_f2 = f2
    return float(hv)


def igd(front: np.ndarray, true_front: np.ndarray) -> float:
    d = np.linalg.norm(true_front[:, None, :] - front[None, :, :], axis=-1)
    return float(d.min(axis=1).mean())


def objectives(lat: np.ndarray, acc: np.ndarray) -> np.ndarray:
    return np.stack([np.asarray(lat, float), -np.asarray(acc, float)], axis=1)


@dataclass
class FrontEvaluator:
    """Holds the true front P* for one (dataset, device) and scores discovered sets."""

    all_obj: np.ndarray
    normalization: str = "front"
    ref_point: tuple[float, float] = (1.1, 1.1)

    def __post_init__(self):
        self.all_obj = self.all_obj[~np.isnan(self.all_obj).any(axis=1)]
        self.true_mask = pareto_mask(self.all_obj)
        self.true_front = self.all_obj[self.true_mask]
        src = self.true_front if self.normalization == "front" else self.all_obj
        self.lo, self.hi = src.min(axis=0), src.max(axis=0)
        self.ref = np.asarray(self.ref_point, float)
        self.true_front_n = self.normalize(self.true_front)
        self.true_hv = hypervolume_2d(self.true_front_n, self.ref)

    @classmethod
    def from_benchmark(cls, bench, device: str, acc_kind: str = "test", **kw) -> "FrontEvaluator":
        acc = bench.test_acc if acc_kind == "test" else bench.val_acc
        return cls(objectives(bench.latencies(device), acc), **kw)

    def normalize(self, obj: np.ndarray) -> np.ndarray:
        return (obj - self.lo) / np.where(self.hi > self.lo, self.hi - self.lo, 1.0)

    def score(self, obj: np.ndarray) -> dict:
        obj = np.asarray(obj, float).reshape(-1, 2)
        obj = obj[~np.isnan(obj).any(axis=1)]
        if len(obj) == 0:
            return {"hv": 0.0, "igd": float("nan"), "n_front": 0}
        front = obj[pareto_mask(obj)]
        fn = self.normalize(front)
        return {
            "hv": hypervolume_2d(fn, self.ref) / self.true_hv,
            "igd": igd(fn, self.true_front_n),
            "n_front": int(len(front)),
        }
