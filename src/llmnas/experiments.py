"""Experiment grids: base config + experiment overrides + per-variant overrides.

Runs are stored by *content hash* of their effective config:

    results/runs/<variant>-<hash8>/<dataset>/<device>/seed<k>/{run.json, llm_calls.jsonl}

so identical settings requested by two experiments (e.g. the full LLM-NAS model in
Table 3 and Table 5) are computed once, and a finished run is never recomputed.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
_HASH_EXCLUDE = {("llm", "mode"), ("llm", "timeout"), ("results_dir",), ("data_dir",), ("variant",),
                 ("code_version",)}


def code_version() -> str:
    """git commit of this repo (+ '-dirty' if there are uncommitted changes), recorded in every run."""
    import subprocess

    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True,
                             check=True).stdout.strip()  # fmt: skip
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "configs"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()  # fmt: skip
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def deep_merge(base: dict, over: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def parse_sets(sets: list[str] | None) -> dict:
    """['llm.model=gpt-4o', 'search.generations=5'] -> nested dict."""
    out: dict = {}
    for s in sets or []:
        key, _, val = s.partition("=")
        node = out
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    return out


def load_experiment(path: str | Path, cli_overrides: dict | None = None) -> dict:
    path = Path(path)
    for folder in (REPO / "configs" / "experiments", REPO / "configs" / "slm" / "experiments"):
        if not path.exists() and (folder / f"{path}.yaml").exists():
            path = folder / f"{path}.yaml"
    exp = yaml.safe_load(open(path))
    base_file = {"slm": REPO / "configs" / "slm" / "base.yaml"}.get(exp.get("base_config"), REPO / "configs" / "base.yaml")
    base = yaml.safe_load(open(base_file))
    exp["base"] = deep_merge(deep_merge(base, exp.get("overrides")), cli_overrides)
    return exp


def variant_config(exp: dict, variant: str) -> dict:
    cfg = deep_merge(exp["base"], exp["variants"][variant])
    cfg["variant"] = variant
    cfg.setdefault("type", exp.get("type", "search"))
    return cfg


def config_hash(cfg: dict) -> str:
    def strip(d, prefix=()):
        return {k: strip(v, prefix + (k,)) if isinstance(v, dict) else v
                for k, v in d.items() if prefix + (k,) not in _HASH_EXCLUDE}  # fmt: skip

    return hashlib.sha1(json.dumps(strip(cfg), sort_keys=True).encode()).hexdigest()[:8]


def run_dir(cfg: dict, dataset: str, device: str, seed: int) -> Path:
    root = Path(cfg.get("results_dir") or "results")
    if not root.is_absolute():
        root = REPO / root
    return root / "runs" / f"{cfg['variant']}-{config_hash(cfg)}" / dataset / device / f"seed{seed}"


def iter_runs(exp: dict, variants=None, datasets=None, devices=None, seeds=None):
    for v in variants or exp["variants"]:
        cfg = variant_config(exp, v)
        for ds in datasets or exp["datasets"]:
            for dev in devices or exp["devices"]:
                for seed in seeds if seeds is not None else exp["seeds"]:
                    yield cfg, ds, dev, seed, run_dir(cfg, ds, dev, seed)


def run_experiment(exp: dict, force: bool = False, **filters) -> None:
    from .bias import run_bias
    from .search import run_search
    from .slm.search import run_slm_search

    from .baselines import run_nsga2

    runners = {"bias": run_bias, "search": run_search, "slm": run_slm_search, "nsga2": run_nsga2}

    todo = list(iter_runs(exp, **filters))
    version = code_version()
    failed: list[str] = []
    for i, (cfg, ds, dev, seed, rdir) in enumerate(todo, 1):
        tag = f"[{i}/{len(todo)}] {cfg['variant']} {ds} {dev} seed{seed}"
        if (rdir / "run.json").exists() and not force and cfg["llm"].get("mode") != "replay":
            print(f"{tag}: done, skipping ({rdir})")
            continue
        print(f"{tag}: running -> {rdir}")
        rdir.mkdir(parents=True, exist_ok=True)
        cfg = {**cfg, "code_version": version}
        with open(rdir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        try:
            res = runners[cfg["type"]](cfg, ds, dev, seed, rdir)
        except Exception as e:  # one failed run (e.g. LLM timeouts) must not abort the rest of the grid
            failed.append(tag)
            print(f"{tag}: FAILED ({type(e).__name__}: {e}); continuing - rerun the command to retry it")
            continue
        print(f"{tag}: HV={res['hv']:.4f} IGD={res['igd']:.4f} front={res['n_front']} "
              f"evaluated={res['n_evaluated']} time={res['search_time_s']}s")  # fmt: skip
    if failed:
        print(f"\n{len(failed)} run(s) failed: " + "; ".join(failed))


def load_results(exp: dict, **filters) -> list[dict]:
    out = []
    for cfg, ds, dev, seed, rdir in iter_runs(exp, **filters):
        p = rdir / "run.json"
        if p.exists():
            out.append(json.load(open(p)))
    return out
