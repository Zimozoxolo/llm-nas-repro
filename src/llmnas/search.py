"""LLM-NAS search loop (Algorithm 1) and its ablations.

One call to ``run_search`` = one (dataset, device, seed) search. Variants are pure
config switches so every ablation shares exactly the same code path:

    search.partitioning   true  -> six niches of Table 1   | false -> one FULL niche
    search.operator       llm   -> LLM co-evolution         | random -> PEA (no LLM)
                                                            | random_sample -> uniform sampling
    search.knowledge_base true  -> Stage 1 KB update        | false -> static prompt
    predictor.kind        zc_ensemble | synflow | oracle
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from .benchmark import load_benchmark
from .llm import LLMSession, make_backend
from .metrics import FrontEvaluator, objectives, pareto_mask
from .predictor import build_predictor
from .prompts import (GEN_SYSTEM, KB_SYSTEM, generation_prompt, kb_update_prompt, parse_children,
                      parse_knowledge_base)  # fmt: skip
from .search_space import PAPER_NICHES, Niche, format_code, get_niches, random_crossover, random_mutation


def _dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """a, b = (latency, pred_acc). Lower latency and higher accuracy are better."""
    return a[0] <= b[0] and a[1] >= b[1] and (a[0] < b[0] or a[1] > b[1])


class NicheState:
    def __init__(self, niche: Niche):
        self.niche = niche
        self.archive: list[tuple] = []  # Pareto archive P_k (codes)
        self.knowledge_base: list[str] = []
        self.kb_history: list[list[str]] = []
        self.last_cycle: list[tuple] = []

    def update_archive(self, code: tuple, ev: dict) -> bool:
        new = (ev[code]["lat"], ev[code]["pred"])
        if any(_dominates((ev[c]["lat"], ev[c]["pred"]), new) or c == code for c in self.archive):
            return False
        self.archive = [c for c in self.archive if not _dominates(new, (ev[c]["lat"], ev[c]["pred"]))]
        self.archive.append(code)
        return True


def run_search(cfg: dict, dataset: str, device: str, seed: int, run_dir: Path) -> dict:
    t_start = time.perf_counter()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    s_cfg, p_cfg, e_cfg, l_cfg = cfg["search"], cfg["predictor"], cfg["evaluation"], cfg.get("llm", {})
    rng = random.Random(seed)

    bench = load_benchmark(dataset, cfg.get("data_dir"))
    lat_all = bench.latencies(device)
    predictor = build_predictor(bench, kind=p_cfg["kind"], n_train=p_cfg["n_train"], seed=seed,
                                target=p_cfg.get("target", "val"), xgb_params=p_cfg.get("xgb_params"))  # fmt: skip

    niches = get_niches(s_cfg["partitioning"])
    n_child = s_cfg["n_child"]
    if not s_cfg["partitioning"] and s_cfg.get("match_budget", True):
        n_child *= len(PAPER_NICHES)  # same number of candidates as the partitioned run
    states = [NicheState(n) for n in niches]
    use_llm = s_cfg["operator"] == "llm"
    session = None
    if use_llm:
        mode = l_cfg.get("mode", "record")
        backend = None if mode == "replay" else make_backend({**l_cfg, "seed": l_cfg.get("seed", seed)})
        session = LLMSession(backend, run_dir / "llm_calls.jsonl", mode=mode)

    ev: dict[tuple, dict] = {}  # every evaluated architecture
    unmeasured = bench.unmeasured(device)  # no latency on this device -> not in the search space
    stats = dict(proposed=0, invalid=0, out_of_niche=0, duplicate=0, no_latency=0, accepted=0, parse_failures=0)

    def evaluate(code, niche_name, gen, origin, rationale=""):
        ev[code] = dict(pred=predictor(code), lat=float(lat_all[bench.row_of[code]]), niche=niche_name,
                        gen=gen, origin=origin, rationale=rationale)  # fmt: skip

    def entry(i, code):
        e = ev[code]
        return dict(id=i, code=code, acc=e["pred"], lat=e["lat"], rationale=e["rationale"])

    # ---- Phase 1: initial population per niche
    for st in states:
        for code in st.niche.sample(rng, s_cfg["init_pop_per_niche"], exclude=unmeasured):
            evaluate(code, st.niche.name, 0, "init")
            st.update_archive(code, ev)
            st.last_cycle.append(code)

    trajectory = [_trajectory_point(0, states, ev, bench, device, e_cfg)]

    # ---- Phase 2: partitioned co-evolution
    for gen in range(1, s_cfg["generations"] + 1):
        for st in states:
            parents = _select_parents(st, ev, rng, s_cfg["n_parents_shown"])
            p_entries = [entry(i, c) for i, c in enumerate(parents)]
            lc_entries = [entry(f"L{i}", c) for i, c in enumerate(st.last_cycle)]
            ops = ["crossover" if len(parents) >= 2 and rng.random() < s_cfg["crossover_prob"] else "mutation"
                   for _ in range(n_child)]  # fmt: skip

            if use_llm:
                meta = dict(generation=gen, niche_name=st.niche.name, niche=st.niche, parents=parents,
                            n_child=n_child, operations=ops)  # fmt: skip
                if s_cfg.get("knowledge_base", True):
                    text = session(KB_SYSTEM, kb_update_prompt(device, dataset, st.niche, p_entries, lc_entries,
                                                               st.knowledge_base, gen), {**meta, "stage": "kb_update"})  # fmt: skip
                    kb = parse_knowledge_base(text)
                    if kb is not None:
                        st.knowledge_base = kb
                    st.kb_history.append(list(st.knowledge_base))
                limit = _latency_limit(s_cfg, st.niche, lat_all, bench)
                text = session(GEN_SYSTEM, generation_prompt(device, dataset, st.niche, p_entries, lc_entries,
                                                            st.knowledge_base, ops, limit), {**meta, "stage": "generate"})  # fmt: skip
                children = parse_children(text)
                if not children:
                    stats["parse_failures"] += 1
            else:
                sample_only = s_cfg["operator"] == "random_sample"
                children = _random_children(parents, ops, st.niche, rng, set(ev) | unmeasured, sample_only)

            st.last_cycle = []
            for ch in children[:n_child]:
                stats["proposed"] += 1
                code = ch["code"]
                if code is None:
                    stats["invalid"] += 1
                elif not st.niche.contains(code):
                    stats["out_of_niche"] += 1
                elif code in ev:
                    stats["duplicate"] += 1
                elif code in unmeasured:
                    stats["no_latency"] += 1
                else:
                    stats["accepted"] += 1
                    evaluate(code, st.niche.name, gen, ch.get("operation") or "llm", ch.get("rationale", ""))
                    st.update_archive(code, ev)
                    st.last_cycle.append(code)
        trajectory.append(_trajectory_point(gen, states, ev, bench, device, e_cfg))
        print(f"  [{dataset}/{device}/seed{seed}] gen {gen:2d}: evaluated={len(ev):4d} "
              f"HV={trajectory[-1]['hv']:.3f} IGD={trajectory[-1]['igd']:.4f}")  # fmt: skip

    # ---- Phase 3: aggregation
    final_codes = _final_front(states, ev)
    score_codes = final_codes if e_cfg.get("front_source", "archive") == "archive" else list(ev)
    acc_kind = e_cfg.get("acc", "test")
    score = _cached_evaluator(bench, device, e_cfg).score(objectives([ev[c]["lat"] for c in score_codes],
                                       [bench.accuracy(c, acc_kind) for c in score_codes]))  # fmt: skip

    result = dict(
        dataset=dataset, device=device, seed=seed, variant=cfg.get("variant", "llmnas"),
        hv=score["hv"], igd=score["igd"], n_front=score["n_front"], n_evaluated=len(ev),
        search_time_s=round(time.perf_counter() - t_start, 2),
        llm_usage=session.usage() if session else None,
        proposal_stats=stats,
        predictor=predictor.report.__dict__,
        final_front=sorted(
            [dict(code=format_code(c), pred_acc=round(ev[c]["pred"], 3), true_acc=bench.accuracy(c, acc_kind),
                  lat=ev[c]["lat"], niche=ev[c]["niche"]) for c in final_codes],
            key=lambda d: d["lat"]),  # fmt: skip
        evaluated=[dict(code=format_code(c), **{k: v for k, v in e.items() if k != "rationale"},
                        true_acc=bench.accuracy(c, acc_kind)) for c, e in ev.items()],  # fmt: skip
        knowledge_base={st.niche.name: st.kb_history for st in states},
        trajectory=trajectory,
        config=cfg,
    )
    out_name = "run_replay.json" if l_cfg.get("mode") == "replay" else "run.json"
    with open(run_dir / out_name, "w") as f:
        json.dump(result, f, indent=1, default=str)
    return result


# --------------------------------------------------------------------------- helpers


def _select_parents(st: NicheState, ev: dict, rng: random.Random, k: int) -> list[tuple]:
    """Pareto archive members; if the archive is larger than k, the k-1 newest plus random others."""
    arch = list(st.archive)
    if len(arch) <= k:
        chosen = arch
    else:
        recent = [c for c in arch if c in st.last_cycle][: k // 2]
        rest = [c for c in arch if c not in recent]
        chosen = recent + rng.sample(rest, k - len(recent))
    return sorted(chosen, key=lambda c: ev[c]["lat"])


def _random_children(parents, ops, niche, rng, exclude, sample_only=False) -> list[dict]:
    """PEA operators (random mutation / uniform crossover), or uniform sampling."""
    out = []
    for op in ops:
        child = None
        if parents and not sample_only:
            if op == "crossover" and len(parents) >= 2:
                child = random_crossover(*rng.sample(parents, 2), niche, rng)
            if child is None:
                child = random_mutation(rng.choice(parents), niche, rng)
        if child is None:
            pool = niche.sample(rng, 1, exclude=exclude)
            child = pool[0] if pool else None
        out.append({"code": child, "operation": op, "rationale": ""})
    return out


def _latency_limit(s_cfg: dict, niche: Niche, lat_all: np.ndarray, bench) -> float | None:
    mode = s_cfg.get("latency_limit")
    if mode in (None, "none"):
        return None
    if mode == "niche_max":  # the slowest architecture in the niche (device constraint, not a target)
        return float(np.nanmax([lat_all[bench.row_of[c]] for c in niche.members()]))
    return float(mode)


def _final_front(states: list[NicheState], ev: dict) -> list[tuple]:
    union = sorted({c for st in states for c in st.archive})
    if not union:
        return []
    obj = objectives([ev[c]["lat"] for c in union], [ev[c]["pred"] for c in union])
    return [c for c, keep in zip(union, pareto_mask(obj)) if keep]


def _trajectory_point(gen, states, ev, bench, device, e_cfg) -> dict:
    codes = _final_front(states, ev) if e_cfg.get("front_source", "archive") == "archive" else list(ev)
    evaluator = _cached_evaluator(bench, device, e_cfg)
    kind = e_cfg.get("acc", "test")
    s = evaluator.score(objectives([ev[c]["lat"] for c in codes], [bench.accuracy(c, kind) for c in codes]))
    return dict(gen=gen, n_evaluated=len(ev), **s)


_EVAL_CACHE: dict = {}


def _cached_evaluator(bench, device, e_cfg) -> FrontEvaluator:
    key = (bench.dataset, device, e_cfg.get("acc", "test"), e_cfg.get("normalization", "front"),
           tuple(e_cfg.get("ref_point", (1.1, 1.1))))  # fmt: skip
    if key not in _EVAL_CACHE:
        _EVAL_CACHE[key] = FrontEvaluator.from_benchmark(bench, device, key[2], normalization=key[3],
                                                         ref_point=key[4])  # fmt: skip
    return _EVAL_CACHE[key]
