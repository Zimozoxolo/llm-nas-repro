"""Helpers behind the slm-* CLI commands."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import yaml

from ..experiments import REPO
from .costs import Workload, device_from_config, evaluate_costs
from .reference import REFERENCE_SLMS, reference_spec
from .space import SpaceSpec, total_params


def _base() -> dict:
    return yaml.safe_load(open(REPO / "configs" / "slm" / "base.yaml"))


def reference(device: str) -> None:
    cfg = _base()
    spec = SpaceSpec.from_config({**cfg["spaces"]["slm"]["spec"], "name": "slm"})
    dev = device_from_config(cfg["devices"].get(device, {"profile": device}))
    wl = Workload(**cfg["workload"])
    rows = []
    for name, ref in REFERENCE_SLMS.items():
        own = reference_spec(ref, spec)
        c = evaluate_costs(ref["arch"], own, dev, wl)
        tp = total_params(ref["arch"], own) / 1e6
        rows.append(dict(model=name, ours_total_m=round(tp), official_m=ref["official_params_m"],
                         error=f"{(tp - ref['official_params_m']) / ref['official_params_m']:+.1%}",
                         decoder_m=round(c["decoder_params_m"]), est_loss=round(c["est_loss"], 3),
                         latency_ms=round(c["latency_ms"], 1), memory_mb=round(c["memory_mb"]),
                         vocab=ref["vocab_size"]))  # fmt: skip
    print(f"Reference SLMs on {dev.name} (own vocabulary; {wl.weight_dtype} weights, "
          f"{wl.prompt_tokens}+{wl.gen_tokens} tokens)\n")
    print(pd.DataFrame(rows).to_markdown(index=False))


def calibrate_cmd(a) -> None:
    from .measure import calibrate

    cfg = _base()
    spec = SpaceSpec.from_config({**cfg["spaces"][a.space]["spec"], "name": a.space})
    wl = Workload(**cfg["workload"])
    res = calibrate(spec, wl, a.device, a.n, a.seed, REPO / "results" / "slm_measurements", a.max_decoder_m)
    res.update(space=a.space, prompt_tokens=wl.prompt_tokens, gen_tokens=wl.gen_tokens,
               date=time.strftime("%Y-%m-%d"))  # fmt: skip
    out = Path(a.out) if Path(a.out).is_absolute() else REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"\nFit on {res['n']} architectures ({res['device_tag']}): Spearman(model, measured latency) = "
          f"{res['spearman_latency']:.3f}, mean abs. error = {res['mape_latency']:.1%}\nSaved {out}")
