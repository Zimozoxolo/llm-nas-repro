"""Lookup-table view of one dataset: latency per device, accuracy and ZC proxies."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DEVICES, ZC_PROXIES, default_data_dir
from .search_space import Code, str_to_code


class Benchmark:
    def __init__(self, dataset: str, data_dir: Path | None = None):
        path = Path(data_dir or default_data_dir()) / "processed" / f"nb201_{dataset}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run `python -m llmnas prepare-data` first.")
        self.dataset = dataset
        self.df = pd.read_csv(path)
        self.codes: list[Code] = [str_to_code(s) for s in self.df.arch_str]
        self.row_of = {c: i for i, c in enumerate(self.codes)}
        self.test_acc = self.df.test_acc.to_numpy()
        self.val_acc = self.df.val_acc.to_numpy()
        self.zc = self.df[[f"zc_{p}" for p in ZC_PROXIES]].to_numpy()

    def latencies(self, device: str) -> np.ndarray:
        """Latency per row; NaN where HW-NAS-Bench stores -1. For Edge TPU that marks 4,699 architectures that failed
        the Edge TPU compiler (GATECH-EIC/HW-NAS-Bench issue #11), so they are infeasible on that device and are
        excluded from its search space and true front."""
        if device not in DEVICES:
            raise KeyError(f"unknown device {device!r}; choose from {list(DEVICES)}")
        lat = self.df[f"lat_{device}"].to_numpy(dtype=float)
        return np.where(lat > 0, lat, np.nan)

    def unmeasured(self, device: str) -> set:
        return {self.codes[i] for i in np.where(np.isnan(self.latencies(device)))[0]}

    def latency(self, code: Code, device: str) -> float:
        return float(self.latencies(device)[self.row_of[tuple(code)]])

    def accuracy(self, code: Code, kind: str = "test") -> float:
        arr = self.test_acc if kind == "test" else self.val_acc
        return float(arr[self.row_of[tuple(code)]])

    def rows(self, codes) -> np.ndarray:
        return np.array([self.row_of[tuple(c)] for c in codes], dtype=int)


@lru_cache(maxsize=None)
def load_benchmark(dataset: str, data_dir: str | None = None) -> Benchmark:
    return Benchmark(dataset, Path(data_dir) if data_dir else None)
