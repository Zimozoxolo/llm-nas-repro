"""On-device measurement (LTS-style): build the candidate with random weights in
PyTorch, time prefill and per-token decode with a KV cache, record peak memory.

No training is involved - only hardware cost is measured. Results are cached per
device in ``results/slm_measurements/<device_tag>.jsonl`` so the same architecture
is never measured twice and measured runs stay reproducible.

``calibrate`` measures a spread of architectures and fits the analytical model
(t = a*work + b*n_layer + c, separately for prefill and decode) by least squares.
"""

from __future__ import annotations

import json
import platform
import random
import time
from pathlib import Path

import numpy as np

from .costs import Workload, bytes_decode_token, evaluate_costs, flops_prefill
from .space import SpaceSpec, arch_key, kv_heads, layer_list, random_arch


def _torch():
    import torch

    return torch


def build_model(arch: dict, spec: SpaceSpec, device: str, dtype):
    torch = _torch()
    nn, F = torch.nn, torch.nn.functional
    d = arch["d_model"]

    class Block(nn.Module):
        def __init__(self, ly):
            super().__init__()
            self.nh, self.nkv = ly["n_head"], kv_heads(ly)
            self.hd = d // self.nh
            self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
            self.q = nn.Linear(d, self.nh * self.hd, bias=False)
            self.kv = nn.Linear(d, 2 * self.nkv * self.hd, bias=False)
            self.o = nn.Linear(self.nh * self.hd, d, bias=False)
            self.gated = arch["ffn"] == "gated"
            self.up = nn.Linear(d, ly["d_inner"] * (2 if self.gated else 1), bias=False)
            self.down = nn.Linear(ly["d_inner"], d, bias=False)

        def forward(self, x, cache):
            B, T, _ = x.shape
            h = self.n1(x)
            q = self.q(h).view(B, T, self.nh, self.hd).transpose(1, 2)
            k, v = self.kv(h).view(B, T, 2, self.nkv, self.hd).unbind(2)
            k, v = k.transpose(1, 2), v.transpose(1, 2)
            if cache is not None:
                k, v = torch.cat([cache[0], k], 2), torch.cat([cache[1], v], 2)
            new_cache = (k, v)
            if self.nkv != self.nh:
                rep = self.nh // self.nkv
                k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=cache is None and T > 1)
            x = x + self.o(a.transpose(1, 2).reshape(B, T, -1))
            u = self.up(self.n2(x))
            if self.gated:
                g, u = u.chunk(2, -1)
                u = F.silu(g) * u
            else:
                u = F.gelu(u)
            return x + self.down(u), new_cache

    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(spec.vocab_size, d)
            self.blocks = nn.ModuleList(Block(ly) for ly in layer_list(arch))
            self.norm = nn.RMSNorm(d)
            self.head = nn.Linear(d, spec.vocab_size, bias=False)
            if spec.tie_embeddings:
                self.head.weight = self.emb.weight

        def forward(self, ids, caches=None):
            x = self.emb(ids)
            caches = caches or [None] * len(self.blocks)
            out = []
            for blk, c in zip(self.blocks, caches):
                x, nc = blk(x, c)
                out.append(nc)
            return self.head(self.norm(x[:, -1:])), out

    return LM().to(device=device, dtype=dtype).eval()


def device_tag(device: str) -> str:
    torch = _torch()
    if device.startswith("cuda"):
        return "cuda_" + torch.cuda.get_device_name(0).replace(" ", "_")
    return "cpu_" + (platform.processor() or platform.machine()).replace(" ", "_").replace(",", "")[:40]


def measure(arch: dict, spec: SpaceSpec, wl: Workload, device: str = "cpu", repeats: int = 3) -> dict:
    torch = _torch()
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)
    try:
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model = build_model(arch, spec, device, dtype)
        ids = torch.randint(0, spec.vocab_size, (1, wl.prompt_tokens), device=device)
        pre, dec = [], []
        with torch.inference_mode():
            for r in range(repeats + 1):  # first run is warm-up
                sync()
                t0 = time.perf_counter()
                logits, caches = model(ids)
                sync()
                t1 = time.perf_counter()
                tok = logits.argmax(-1)
                for _ in range(wl.gen_tokens):
                    logits, caches = model(tok, caches)
                    tok = logits.argmax(-1)
                sync()
                t2 = time.perf_counter()
                if r:
                    pre.append(t1 - t0)
                    dec.append((t2 - t1) / wl.gen_tokens)
        peak = torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else None
        del model
        return dict(ok=True, prefill_ms=1e3 * float(np.median(pre)), decode_ms_per_token=1e3 * float(np.median(dec)),
                    latency_ms=1e3 * (float(np.median(pre)) + wl.gen_tokens * float(np.median(dec))),
                    peak_memory_mb=peak, dtype=str(dtype).replace("torch.", ""))  # fmt: skip
    except (RuntimeError, MemoryError) as e:  # includes torch.cuda.OutOfMemoryError
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return dict(ok=False, error=f"{type(e).__name__}: {str(e)[:120]}")


class MeasurementCache:
    def __init__(self, root: Path, device: str, wl: Workload):
        self.device, self.wl = device, wl
        self.path = Path(root) / f"{device_tag(device)}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = {}
        if self.path.exists():
            for line in open(self.path):
                r = json.loads(line)
                self.data[(r["key"], r["workload"])] = r["result"]

    def _wkey(self) -> str:
        return f"P{self.wl.prompt_tokens}-G{self.wl.gen_tokens}"

    def get(self, arch: dict, spec: SpaceSpec) -> dict:
        k = (spec.name + ":" + arch_key(arch), self._wkey())
        if k not in self.data:
            res = measure(arch, spec, self.wl, self.device)
            self.data[k] = res
            with open(self.path, "a") as f:
                f.write(json.dumps({"key": k[0], "workload": k[1], "result": res}) + "\n")
        return self.data[k]


def calibrate(spec: SpaceSpec, wl: Workload, device: str, n: int, seed: int, cache_root: Path,
              max_decoder_m: float | None = None) -> dict:  # fmt: skip
    """Measure n random architectures and fit t = a*work + b*n_layer + c for prefill and decode."""
    rng = random.Random(seed)
    cache = MeasurementCache(cache_root, device, wl)
    rows = []
    f32 = Workload(**{**wl.__dict__, "weight_dtype": "f16" if device.startswith("cuda") else "f32"})
    while len(rows) < n:
        a = random_arch(spec, rng)
        if a is None or (max_decoder_m and evaluate_costs(a, spec, _dummy(), f32)["decoder_params_m"] > max_decoder_m):
            continue
        m = cache.get(a, spec)
        if not m.get("ok"):
            continue
        rows.append((a, m))
        print(f"  measured {len(rows)}/{n}: L={a['n_layer']} d={a['d_model']} "
              f"prefill={m['prefill_ms']:.1f}ms decode={m['decode_ms_per_token']:.2f}ms/tok")  # fmt: skip

    def fit(work, t):
        X = np.column_stack([work, [a["n_layer"] for a, _ in rows], np.ones(len(rows))])
        coef, *_ = np.linalg.lstsq(X, np.asarray(t) / 1e3, rcond=None)
        return [float(max(c, 0.0)) for c in coef]

    pf = fit([flops_prefill(a, spec, wl.prompt_tokens) for a, _ in rows], [m["prefill_ms"] for _, m in rows])
    df = fit([np.mean([bytes_decode_token(a, spec, f32, wl.prompt_tokens + t) for t in (0, wl.gen_tokens // 2)])
              for a, _ in rows], [m["decode_ms_per_token"] for _, m in rows])  # fmt: skip

    from scipy.stats import spearmanr

    from .costs import DeviceProfile

    cal = DeviceProfile("calibrated", 1, 1, 1, fit={"prefill": pf, "decode": df})
    pred = [evaluate_costs(a, spec, cal, f32)["latency_ms"] for a, _ in rows]
    meas = [m["latency_ms"] for _, m in rows]
    return dict(device_tag=device_tag(device), n=len(rows), fit={"prefill": pf, "decode": df},
                weight_dtype=f32.weight_dtype, spearman_latency=float(spearmanr(pred, meas).statistic),
                mape_latency=float(np.mean(np.abs(np.array(pred) - meas) / np.array(meas))))  # fmt: skip


def _dummy():
    from .costs import DeviceProfile

    return DeviceProfile("dummy", 1, 1, 1e9)

