"""Exploration-bias study (Figure 1 and Appendix D / Figure 9).

Two un-partitioned prompting strategies, compared against a full LLM-NAS run:

* ``normal``            - plain task description, no feedback between rounds (Fig. 1a)
* ``latency_optimized`` - explicit "aim for diverse latencies" hint + previous round's
                          accuracy/latency fed back (Fig. 1b)

Accuracy fed back to the LLM is the predictor's (training-free, as in LLM-NAS);
plots and HV/IGD use benchmark ground truth.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .benchmark import load_benchmark
from .llm import LLMSession, make_backend
from .metrics import objectives
from .predictor import build_predictor
from .prompts import BIAS_SYSTEM, bias_prompt, parse_children
from .search import _cached_evaluator
from .search_space import format_code


def run_bias(cfg: dict, dataset: str, device: str, seed: int, run_dir: Path) -> dict:
    t0 = time.perf_counter()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    b_cfg, p_cfg, e_cfg, l_cfg = cfg["bias"], cfg["predictor"], cfg["evaluation"], cfg.get("llm", {})
    variant = b_cfg["prompt"]

    bench = load_benchmark(dataset, cfg.get("data_dir"))
    lat_all = bench.latencies(device)
    predictor = build_predictor(bench, kind=p_cfg["kind"], n_train=p_cfg["n_train"], seed=seed,
                                target=p_cfg.get("target", "val"))  # fmt: skip
    mode = l_cfg.get("mode", "record")
    backend = None if mode == "replay" else make_backend({**l_cfg, "seed": l_cfg.get("seed", seed)})
    session = LLMSession(backend, run_dir / "llm_calls.jsonl", mode=mode)

    unmeasured = bench.unmeasured(device)
    seen: dict[tuple, dict] = {}
    history: list[dict] = []
    proposed = 0
    for rnd in range(b_cfg["rounds"]):
        text = session(BIAS_SYSTEM, bias_prompt(device, dataset, b_cfg["per_round"], variant, history),
                       {"stage": f"bias_{variant}", "generation": rnd})  # fmt: skip
        history = []
        for ch in parse_children(text)[: b_cfg["per_round"]]:
            proposed += 1
            code = ch["code"]
            if code is None or code in unmeasured:
                continue
            rec = dict(pred=predictor(code), lat=float(lat_all[bench.row_of[code]]),
                       true_acc=bench.accuracy(code, e_cfg.get("acc", "test")), round=rnd)  # fmt: skip
            seen.setdefault(code, rec)
            history.append(dict(id=len(history), code=code, acc=rec["pred"], lat=rec["lat"]))

    codes = list(seen)
    score = _cached_evaluator(bench, device, e_cfg).score(
        objectives([seen[c]["lat"] for c in codes], [seen[c]["true_acc"] for c in codes]))  # fmt: skip
    result = dict(dataset=dataset, device=device, seed=seed, variant=cfg.get("variant"), hv=score["hv"],
                  igd=score["igd"], n_front=score["n_front"], n_evaluated=len(codes), n_proposed=proposed,
                  unique_ratio=len(codes) / max(proposed, 1), search_time_s=round(time.perf_counter() - t0, 2),
                  llm_usage=session.usage(),
                  evaluated=[dict(code=format_code(c), **r) for c, r in seen.items()], config=cfg)  # fmt: skip
    out_name = "run_replay.json" if l_cfg.get("mode") == "replay" else "run.json"
    with open(run_dir / out_name, "w") as f:
        json.dump(result, f, indent=1, default=str)
    return result
