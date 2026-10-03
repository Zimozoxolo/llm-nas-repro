"""LLM-NAS (Algorithm 1) over the decoder-only transformer space.

Same switches as the CNN search, so the ablations carry over unchanged:

    search.partitioning   true -> decoder-parameter niches | false -> one FULL niche
    search.operator       llm | random (mutation/crossover, LTS-style p=0.3) | random_sample
    search.knowledge_base Stage-1 KB updates on/off
    search.kb_seed        none | survey  (start the KB from the SLM survey's insights)

``operator: random`` + ``partitioning: false`` is LTS's own search (evolution over
the Pareto front, decoder-param proxy, hardware in the loop) at the same budget.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from ..llm import LLMSession, make_backend
from ..metrics import pareto_mask
from .costs import Workload, device_from_config, evaluate_costs
from .prompts import (GEN_SYSTEM, KB_SYSTEM, OBJ_LABELS, SEED_RULES, generation_prompt, kb_update_prompt,
                      parse_knowledge_base, parse_slm_children)  # fmt: skip
from .space import (FULL_SLM_SPACE, Sampler, SpaceSpec, arch_key, crossover, mutate, niches_from_config,
                    repair)  # fmt: skip


def nd_dominates(a, b) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def nondominated(points: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows (minimisation, any number of objectives)."""
    pts = np.asarray(points, float)
    if pts.shape[1] == 2:
        return pareto_mask(pts)
    mask = np.array([not np.any(np.all(pts <= p, axis=1) & np.any(pts < p, axis=1)) for p in pts])
    seen = set()
    for i in np.where(mask)[0]:  # exact duplicates: keep the first
        t = tuple(pts[i])
        mask[i] = t not in seen
        seen.add(t)
    return mask


class Niche:
    def __init__(self, niche):
        self.niche = niche
        self.archive: list[str] = []
        self.kb: list[str] = []
        self.kb_history: list[list[str]] = []
        self.last_cycle: list[str] = []

    def update(self, key: str, ev: dict) -> None:
        new = ev[key]["obj"]
        if any(nd_dominates(ev[k]["obj"], new) or ev[k]["obj"] == new for k in self.archive):
            return
        self.archive = [k for k in self.archive if not nd_dominates(new, ev[k]["obj"])] + [key]


def run_slm_search(cfg: dict, space_name: str, device_name: str, seed: int, run_dir: Path) -> dict:
    t_start = time.perf_counter()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    s_cfg, l_cfg = cfg["search"], cfg.get("llm", {})
    sp_cfg = cfg["spaces"][space_name]
    spec = SpaceSpec.from_config({**sp_cfg["spec"], "name": space_name})
    dev_cfg, cal = _load_device(cfg, device_name)
    dev = device_from_config(dev_cfg)
    wl = Workload(**cfg["workload"])
    if cal and cal.get("weight_dtype") != wl.weight_dtype:
        print(f"  WARNING: {device_name} was calibrated with {cal['weight_dtype']} weights but the workload uses "
              f"{wl.weight_dtype}; set workload.weight_dtype={cal['weight_dtype']} for calibrated latencies.")
    wl_text = (f"{wl.prompt_tokens}-token prompt, {wl.gen_tokens} generated tokens, {wl.context}-token KV cache, "
               f"{wl.weight_dtype} weights")  # fmt: skip
    objectives = cfg["objectives"]
    rng = random.Random(seed)
    mock_rng = random.Random(seed + 7919)  # the mock backend must not consume the search's RNG (replay)

    measurer = None
    if cfg["hardware"]["source"] == "measured":
        from .measure import MeasurementCache

        root = Path(cfg["hardware"].get("cache_dir") or Path(run_dir).parents[4] / "slm_measurements")
        measurer = MeasurementCache(root, cfg["hardware"].get("measure_device", "cpu"), wl)

    niches = niches_from_config(sp_cfg["niches"], sp_cfg.get("niche_labels")) if s_cfg["partitioning"] \
        else [FULL_SLM_SPACE]  # fmt: skip
    n_child = s_cfg["n_child"] * (len(sp_cfg["niches"]) if not s_cfg["partitioning"] and s_cfg.get("match_budget", True) else 1)
    states = [Niche(n) for n in niches]
    init_pop = s_cfg["init_pop_per_niche"] * (len(sp_cfg["niches"]) if not s_cfg["partitioning"]
                                              and s_cfg.get("match_budget", True) else 1)  # fmt: skip
    seed_kb = list(SEED_RULES) if s_cfg.get("kb_seed") == "survey" else []
    for st in states:
        st.kb = list(seed_kb)

    use_llm = s_cfg["operator"] == "llm"
    session = None
    if use_llm:
        mode = l_cfg.get("mode", "record")
        backend = None if mode == "replay" else make_backend({**l_cfg, "seed": l_cfg.get("seed", seed)})
        session = LLMSession(backend, run_dir / "llm_calls.jsonl", mode=mode)

    sampler = Sampler(spec)
    ev: dict[str, dict] = {}
    stats = dict(proposed=0, invalid=0, repaired=0, out_of_niche=0, duplicate=0, infeasible=0, accepted=0,
                 parse_failures=0)  # fmt: skip

    def evaluate(arch, niche_name, gen, origin, rationale=""):
        key = arch_key(arch)
        costs = evaluate_costs(arch, spec, dev, wl)
        if measurer is not None:
            m = measurer.get(arch, spec)
            costs["measured"] = m
            if m.get("ok"):
                costs.update(prefill_ms=m["prefill_ms"], decode_ms_per_token=m["decode_ms_per_token"],
                             latency_ms=m["latency_ms"])  # fmt: skip
                if m.get("peak_memory_mb"):
                    costs["memory_mb"] = m["peak_memory_mb"]
            else:
                costs["feasible"] = False
        if cfg["hardware"].get("latency_limit_ms") and costs["latency_ms"] > cfg["hardware"]["latency_limit_ms"]:
            costs["feasible"] = False
        ev[key] = dict(arch=arch, niche=niche_name, gen=gen, origin=origin, rationale=rationale, costs=costs,
                       obj=[float(costs[OBJ_LABELS[o]]) for o in objectives])  # fmt: skip
        return key

    def entry(i, key):
        e = ev[key]
        return dict(id=i, arch=e["arch"], costs=e["costs"], rationale=e["rationale"])

    # ---- Phase 1: initial population
    for st in states:
        for _ in range(init_pop):
            a = sampler.sample(st.niche, rng, set(ev))
            if a is None:
                break
            k = evaluate(a, st.niche.name, 0, "init")
            if ev[k]["costs"]["feasible"]:
                st.update(k, ev)
            st.last_cycle.append(k)

    # ---- Phase 2: partitioned co-evolution
    for gen in range(1, s_cfg["generations"] + 1):
        for st in states:
            parents = _select_parents(st, ev, rng, s_cfg["n_parents_shown"])
            p_entries = [entry(i, k) for i, k in enumerate(parents)]
            lc_entries = [entry(f"L{i}", k) for i, k in enumerate(st.last_cycle)]
            ops = ["crossover" if len(parents) >= 2 and rng.random() < s_cfg["crossover_prob"] else "mutation"
                   for _ in range(n_child)]  # fmt: skip

            if use_llm:
                meta = dict(generation=gen, niche_name=st.niche.name, n_child=n_child, operations=ops,
                            mock_generate=lambda: _random_children([ev[k]["arch"] for k in parents], ops, st.niche,
                                                                   spec, mock_rng, set(ev), s_cfg, as_json=True))  # fmt: skip
                if s_cfg.get("knowledge_base", True):
                    text = session(KB_SYSTEM, kb_update_prompt(dev.name, spec, st.niche, wl_text, objectives,
                                                               p_entries, lc_entries, st.kb, gen),
                                   {**meta, "stage": "kb_update"})  # fmt: skip
                    kb = parse_knowledge_base(text)
                    if kb is not None:
                        st.kb = kb
                    st.kb_history.append(list(st.kb))
                text = session(GEN_SYSTEM, generation_prompt(dev.name, spec, st.niche, wl_text, objectives, p_entries,
                                                             lc_entries, st.kb, ops), {**meta, "stage": "generate"})  # fmt: skip
                children = parse_slm_children(text)
                if not children:
                    stats["parse_failures"] += 1
            else:
                children = _random_children([ev[k]["arch"] for k in parents], ops, st.niche, spec, rng, set(ev),
                                            s_cfg)  # fmt: skip

            st.last_cycle = []
            for ch in children[:n_child]:
                stats["proposed"] += 1
                arch, notes = repair(ch["arch"], spec) if ch["arch"] is not None else (None, [])
                if arch is None:
                    stats["invalid"] += 1
                    continue
                stats["repaired"] += bool(notes)
                if not st.niche.contains(arch):
                    stats["out_of_niche"] += 1
                elif arch_key(arch) in ev:
                    stats["duplicate"] += 1
                else:
                    k = evaluate(arch, st.niche.name, gen, ch.get("operation") or "llm", ch.get("rationale", ""))
                    st.last_cycle.append(k)
                    if ev[k]["costs"]["feasible"]:
                        stats["accepted"] += 1
                        st.update(k, ev)
                    else:
                        stats["infeasible"] += 1
        n_front = len(_final_front(states, ev))
        print(f"  [{space_name}/{device_name}/seed{seed}] gen {gen:2d}: evaluated={len(ev):4d} front={n_front}")

    final = _final_front(states, ev)
    out_name = "run_replay.json" if l_cfg.get("mode") == "replay" else "run.json"
    result = dict(
        kind="slm", dataset=space_name, device=device_name, seed=seed, variant=cfg.get("variant"),
        objectives=objectives, n_front=len(final), n_evaluated=len(ev),
        search_time_s=round(time.perf_counter() - t_start, 2),
        llm_usage=session.usage() if session else None, proposal_stats=stats,
        final_front=final, evaluated=[{"key": k, **v} for k, v in ev.items()],
        knowledge_base={st.niche.name: st.kb_history for st in states},
        hv=float("nan"), igd=float("nan"),  # computed against the pooled reference front by the report
        config=cfg,
    )  # fmt: skip
    with open(run_dir / out_name, "w") as f:
        json.dump(result, f, indent=1, default=str)
    return result


# --------------------------------------------------------------------------- helpers


def _load_device(cfg: dict, name: str) -> dict:
    from ..experiments import REPO

    d = dict(cfg["devices"][name])
    cal = d.pop("calibration", None)
    cal = cal and (Path(cal) if Path(cal).is_absolute() else REPO / cal)
    meta = None
    if cal and cal.exists():
        meta = json.load(open(cal))
        d["fit"] = meta["fit"]
    return d, meta


def _select_parents(st: Niche, ev: dict, rng: random.Random, k: int) -> list[str]:
    arch = list(st.archive)
    if len(arch) > k:
        recent = [c for c in arch if c in st.last_cycle][: k // 2]
        arch = recent + rng.sample([c for c in arch if c not in recent], k - len(recent))
    return sorted(arch, key=lambda c: ev[c]["obj"][1] if len(ev[c]["obj"]) > 1 else ev[c]["obj"][0])


def _random_children(parents, ops, niche, spec, rng, seen, s_cfg, as_json=False):
    out = []
    for op in ops:
        child = None
        for _ in range(50):
            if s_cfg["operator"] == "random_sample" or not parents:
                child = None
                break
            if op == "crossover" and len(parents) >= 2:
                c = crossover(*rng.sample(parents, 2), spec, rng)
            else:
                c = mutate(rng.choice(parents), spec, rng, s_cfg.get("mutation_prob", 0.3))
            if c is not None and niche.contains(c) and arch_key(c) not in seen:
                child = c
                break
        if child is None:
            child = Sampler(spec).sample(niche, rng, seen)
        out.append({"arch": child, "operation": op, "rationale": ""} if not as_json else
                   {"child_id": str(len(out)), "operation": op, "architecture": child, "rationale": "mock"})  # fmt: skip
    return out


def _final_front(states: list[Niche], ev: dict) -> list[str]:
    union = sorted({k for st in states for k in st.archive})
    if not union:
        return []
    mask = nondominated(np.array([ev[k]["obj"] for k in union]))
    return [k for k, m in zip(union, mask) if m]
