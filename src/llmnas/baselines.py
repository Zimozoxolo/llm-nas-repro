"""Conventional multi-objective baseline: NSGA-II on the same evaluator and budget as LLM-NAS.

``python -m llmnas run baselines`` runs it. It uses exactly the evaluator of the LLM-NAS runs (HW-NAS-Bench
latency lookup + the zero-cost-proxy XGBoost predictor, fitted with the same seed) and the same number of
evaluated architectures as a partitioned run: an initial population of ``init_pop_per_niche x 6`` (30) and
``n_child x 6`` (24) offspring in each of ``generations`` (10) generations, i.e. 270 evaluations. It does not
use the six niches: NSGA-II keeps diversity with crowding distance instead.

Selection, survival and the final front use (latency, predicted accuracy); the final front is re-scored with
true accuracy, as for every other method, so results are directly comparable with run.json of run_search.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from .benchmark import load_benchmark
from .metrics import objectives, pareto_mask
from .predictor import build_predictor
from .search import _cached_evaluator
from .search_space import FULL_SPACE, PAPER_NICHES, format_code, random_crossover, random_mutation


def _fronts(F: np.ndarray) -> list[list[int]]:
    """Fast non-dominated sorting for minimisation; F is (n, 2)."""
    n = len(F)
    dominated_by = [[] for _ in range(n)]
    count = np.zeros(n, int)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                dominated_by[i].append(j)
            elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                count[i] += 1
    fronts, cur = [], [i for i in range(n) if count[i] == 0]
    while cur:
        fronts.append(cur)
        nxt = []
        for i in cur:
            for j in dominated_by[i]:
                count[j] -= 1
                if count[j] == 0:
                    nxt.append(j)
        cur = nxt
    return fronts


def _crowding(F: np.ndarray, idx: list[int]) -> dict[int, float]:
    d = {i: 0.0 for i in idx}
    if len(idx) <= 2:
        return {i: float("inf") for i in idx}
    for m in range(F.shape[1]):
        order = sorted(idx, key=lambda i: F[i, m])
        lo, hi = F[order[0], m], F[order[-1], m]
        d[order[0]] = d[order[-1]] = float("inf")
        if hi == lo:
            continue
        for k in range(1, len(order) - 1):
            d[order[k]] += (F[order[k + 1], m] - F[order[k - 1], m]) / (hi - lo)
    return d


def _rank_and_crowd(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rank, crowd = np.zeros(len(F), int), np.zeros(len(F))
    for r, fr in enumerate(_fronts(F)):
        cd = _crowding(F, fr)
        for i in fr:
            rank[i], crowd[i] = r, cd[i]
    return rank, crowd


def run_nsga2(cfg: dict, dataset: str, device: str, seed: int, run_dir: Path) -> dict:
    t0 = time.perf_counter()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    s_cfg, p_cfg, e_cfg = cfg["search"], cfg["predictor"], cfg["evaluation"]
    rng = random.Random(seed)
    bench = load_benchmark(dataset, cfg.get("data_dir"))
    lat_all = bench.latencies(device)
    predictor = build_predictor(bench, kind=p_cfg["kind"], n_train=p_cfg["n_train"], seed=seed,
                                target=p_cfg.get("target", "val"), xgb_params=p_cfg.get("xgb_params"))  # fmt: skip
    pop_size = s_cfg["init_pop_per_niche"] * len(PAPER_NICHES)
    n_off = s_cfg["n_child"] * len(PAPER_NICHES)
    unmeasured = bench.unmeasured(device)
    ev: dict[tuple, dict] = {}
    stats = dict(proposed=0, invalid=0, out_of_niche=0, duplicate=0, no_latency=0, accepted=0, parse_failures=0)

    def evaluate(code, gen, origin):
        ev[code] = dict(pred=predictor(code), lat=float(lat_all[bench.row_of[code]]), niche="FULL", gen=gen,
                        origin=origin)

    def F_of(codes):  # minimise (latency, -predicted accuracy)
        return objectives([ev[c]["lat"] for c in codes], [ev[c]["pred"] for c in codes])

    pop = FULL_SPACE.sample(rng, pop_size, exclude=unmeasured)
    for c in pop:
        evaluate(c, 0, "init")
    trajectory = [_traj(0, ev, bench, device, e_cfg)]

    for gen in range(1, s_cfg["generations"] + 1):
        rank, crowd = _rank_and_crowd(F_of(pop))

        def tournament():
            i, j = rng.randrange(len(pop)), rng.randrange(len(pop))
            better = i if (rank[i], -crowd[i]) < (rank[j], -crowd[j]) else j
            return pop[better]

        children = []
        tries = 0
        while len(children) < n_off and tries < 50 * n_off:
            tries += 1
            if rng.random() < s_cfg["crossover_prob"]:
                child, op = random_crossover(tournament(), tournament(), FULL_SPACE, rng), "crossover"
            else:
                child, op = random_mutation(tournament(), FULL_SPACE, rng), "mutation"
            stats["proposed"] += 1
            if child is None:
                stats["invalid"] += 1
            elif child in ev or child in children:
                stats["duplicate"] += 1
            elif child in unmeasured:
                stats["no_latency"] += 1
            else:
                stats["accepted"] += 1
                children.append(child)
        while len(children) < n_off:  # rare: fill with fresh random samples to keep the budget exact
            extra = FULL_SPACE.sample(rng, 1, exclude=set(ev) | unmeasured | set(children))
            if not extra:
                break
            children.append(extra[0])
        for c in children:
            evaluate(c, gen, "nsga2")
        union = pop + children
        rank, crowd = _rank_and_crowd(F_of(union))
        order = sorted(range(len(union)), key=lambda i: (rank[i], -crowd[i]))
        pop = [union[i] for i in order[:pop_size]]
        trajectory.append(_traj(gen, ev, bench, device, e_cfg))
        print(f"  [{dataset}/{device}/seed{seed}] nsga2 gen {gen:2d}: evaluated={len(ev):4d} "
              f"HV={trajectory[-1]['hv']:.3f}")

    codes = list(ev)
    final = [c for c, keep in zip(codes, pareto_mask(F_of(codes))) if keep]
    kind = e_cfg.get("acc", "test")
    score = _cached_evaluator(bench, device, e_cfg).score(
        objectives([ev[c]["lat"] for c in final], [bench.accuracy(c, kind) for c in final]))
    result = dict(
        dataset=dataset, device=device, seed=seed, variant=cfg.get("variant", "nsga2"),
        hv=score["hv"], igd=score["igd"], n_front=score["n_front"], n_evaluated=len(ev),
        search_time_s=round(time.perf_counter() - t0, 2), llm_usage=None, proposal_stats=stats,
        predictor=predictor.report.__dict__,
        final_front=sorted([dict(code=format_code(c), pred_acc=round(ev[c]["pred"], 3),
                                 true_acc=bench.accuracy(c, kind), lat=ev[c]["lat"], niche="FULL") for c in final],
                           key=lambda d: d["lat"]),
        evaluated=[dict(code=format_code(c), **e, true_acc=bench.accuracy(c, kind)) for c, e in ev.items()],
        knowledge_base={}, trajectory=trajectory, config=cfg,
    )  # fmt: skip
    with open(run_dir / "run.json", "w") as f:
        json.dump(result, f, indent=1, default=str)
    return result


def _traj(gen, ev, bench, device, e_cfg) -> dict:
    codes = list(ev)
    F = objectives([ev[c]["lat"] for c in codes], [ev[c]["pred"] for c in codes])
    front = [c for c, keep in zip(codes, pareto_mask(F)) if keep]
    kind = e_cfg.get("acc", "test")
    s = _cached_evaluator(bench, device, e_cfg).score(
        objectives([ev[c]["lat"] for c in front], [bench.accuracy(c, kind) for c in front]))
    return dict(gen=gen, n_evaluated=len(ev), **s)
