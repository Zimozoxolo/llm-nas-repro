"""Training-free accuracy predictors (Sec. 3.3).

* ``zc_ensemble`` - XGBoost on the 13 NAS-Bench-Suite-Zero proxies (the paper's predictor).
* ``synflow``     - single-proxy ablation ("without ZC Ensemble (uses Synflow)").
                    Synflow is a score, not an accuracy, so it is mapped to the accuracy
                    scale with an isotonic fit on the same training sample; rankings are
                    exactly Synflow's.
* ``oracle``      - ground-truth lookup, for debugging / upper bounds only.

All predictors are fit on ``n_train`` randomly sampled architectures whose labels are
NB201 *validation* accuracy (what one would have without touching the test set), and
then predict for the whole space once so the search does O(1) lookups.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import kendalltau, spearmanr

from .benchmark import Benchmark
from .data import ZC_PROXIES


@dataclass
class PredictorReport:
    kind: str
    n_train: int
    spearman_val: float
    spearman_test: float
    kendall_test: float


class Predictor:
    def __init__(self, bench: Benchmark, preds: np.ndarray, report: PredictorReport):
        self.bench, self.preds, self.report = bench, preds, report

    def __call__(self, code) -> float:
        return float(self.preds[self.bench.row_of[tuple(code)]])


def _features(bench: Benchmark) -> np.ndarray:
    x = np.nan_to_num(bench.zc.astype(float), nan=0.0, posinf=1e30, neginf=-1e30)
    return np.sign(x) * np.log1p(np.abs(x))  # tree models don't need this; keeps values sane


def build_predictor(
    bench: Benchmark,
    kind: str = "zc_ensemble",
    n_train: int = 1000,
    seed: int = 0,
    target: str = "val",
    xgb_params: dict | None = None,
) -> Predictor:
    rng = np.random.default_rng(seed)
    n = len(bench.codes)
    train_idx = rng.choice(n, size=min(n_train, n), replace=False)
    y_all = bench.val_acc if target == "val" else bench.test_acc
    y = y_all[train_idx]

    if kind == "zc_ensemble":
        from xgboost import XGBRegressor

        params = dict(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, random_state=seed, n_jobs=-1)  # fmt: skip
        params.update(xgb_params or {})
        model = XGBRegressor(**params)
        x = _features(bench)
        model.fit(x[train_idx], y)
        preds = model.predict(x)
    elif kind == "synflow":
        from sklearn.isotonic import IsotonicRegression

        s = _features(bench)[:, ZC_PROXIES.index("synflow")]
        iso = IsotonicRegression(out_of_bounds="clip").fit(s[train_idx], y)
        # tiny monotone tie-breaker keeps Synflow's full ranking inside isotonic plateaus
        preds = iso.predict(s) + 1e-6 * (s - s.min()) / (np.ptp(s) + 1e-12)
    elif kind == "oracle":
        preds = bench.test_acc.copy()
    else:
        raise ValueError(f"unknown predictor kind {kind!r}")

    mask = np.ones(n, bool)
    mask[train_idx] = False
    report = PredictorReport(
        kind=kind,
        n_train=len(train_idx),
        spearman_val=float(spearmanr(preds[mask], bench.val_acc[mask]).statistic),
        spearman_test=float(spearmanr(preds[mask], bench.test_acc[mask]).statistic),
        kendall_test=float(kendalltau(preds[mask], bench.test_acc[mask]).statistic),
    )
    return Predictor(bench, np.asarray(preds, float), report)
