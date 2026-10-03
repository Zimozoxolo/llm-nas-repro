"""Training-free objectives for decoder-only language models.

Quality (LTS, Javaheripi et al. 2022)
    Decoder (non-embedding) parameter count ranks fully trained autoregressive
    transformers with Spearman 0.97-0.99 regardless of shape. To give the LLM an
    interpretable number, it is mapped through the Kaplan et al. (2020) scaling law
    L(N) = (N_c / N)^0.076 with N_c = 8.8e13; this is monotone, so the ranking is
    exactly LTS's proxy. It is an *estimate*, not a measured loss.

Hardware (SLM survey, Lu et al. 2024, Sec. 4)
    * prefill is compute-bound      -> FLOPs / effective throughput
    * decode is memory-bound        -> bytes read per token / effective bandwidth
    * every layer adds fixed overhead (depth hurts at equal size)
    * memory = weights (quantised) + KV cache (context) + compute buffer
      (~ n_batch x vocab x 4 bytes, why big vocabularies cost memory)
    Device numbers are nominal datasheet values with guessed efficiencies.
    ``python -m llmnas slm-calibrate`` replaces them with a least-squares fit to
    real measurements on your hardware (``measure.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .space import SpaceSpec, decoder_params, kv_heads, layer_list, total_params

KAPLAN_NC, KAPLAN_ALPHA = 8.8e13, 0.076
BYTES_PER_PARAM = {"f32": 4.0, "f16": 2.0, "q8_0": 1.0625, "q6_k": 0.8203, "q5_k": 0.6875, "q4_k_m": 0.5625}


def estimated_loss(arch: dict) -> float:
    return (KAPLAN_NC / max(decoder_params(arch), 1)) ** KAPLAN_ALPHA


@dataclass
class DeviceProfile:
    name: str
    peak_tflops: float  # dense throughput at the compute dtype
    mem_bw_gbs: float  # memory bandwidth
    memory_gb: float  # usable memory (feasibility limit)
    prefill_eff: float = 0.35  # fraction of peak reached in prefill
    decode_eff: float = 0.6  # fraction of bandwidth reached in decode
    layer_overhead_us: float = 30.0  # kernel-launch / scheduling cost per layer per forward pass
    fixed_overhead_ms: float = 1.0  # per forward pass
    note: str = ""
    fit: dict = field(default_factory=dict)  # set by calibration: {"prefill": [a,b,c], "decode": [a,b,c]}


# The two survey devices use datasheet peak/bandwidth with efficiencies fitted by eye to the
# survey's own llama.cpp measurements (Fig. 8: ~25-150 ms first token, ~20-80 ms/token decode
# for 0.1-3B models at Q4 on Jetson; Fig. 11: Qwen1.5-1.8B on phones). Treat them as rough.
DEVICES = {
    "jetson_orin_nx_16gb": DeviceProfile("jetson_orin_nx_16gb", 3.8, 102.4, 16, prefill_eff=0.47, decode_eff=0.21,
                                         layer_overhead_us=150, fixed_overhead_ms=3.0,
                                         note="SLM-survey Jetson Orin NX (GPU, llama.cpp); fitted by eye to survey Fig. 8."),
    "snapdragon_8gen1_cpu": DeviceProfile("snapdragon_8gen1_cpu", 0.25, 51.2, 12, prefill_eff=0.4, decode_eff=0.3,
                                          layer_overhead_us=100, fixed_overhead_ms=2.0,
                                          note="SLM-survey phones (Xiaomi 12S CPU, llama.cpp); fitted by eye to survey Fig. 11."),
    "rtx3050ti_laptop": DeviceProfile("rtx3050ti_laptop", 10.0, 192.0, 4,
                                      note="Your laptop GPU. Calibrate with slm-calibrate --device cuda."),
    "generic_cpu": DeviceProfile("generic_cpu", 0.5, 40.0, 16, prefill_eff=0.5, decode_eff=0.5,
                                 layer_overhead_us=50, note="Placeholder x86 laptop CPU - calibrate."),
}  # fmt: skip


@dataclass
class Workload:
    prompt_tokens: int = 50  # SLM survey's standard setting
    gen_tokens: int = 50
    context: int = 2048  # KV cache allocation (survey fixes 2048)
    weight_dtype: str = "q4_k_m"  # survey: 4-bit, Q4_K_M best
    kv_dtype: str = "f16"
    compute_buffer_tokens: int = 512  # llama.cpp n_batch


def flops_prefill(arch: dict, spec: SpaceSpec, P: int) -> float:
    d = arch["d_model"]
    dense = 2.0 * decoder_params(arch) * P
    attn = sum(2.0 * P * P * d for _ in layer_list(arch))  # QK^T and AV, causal ~ half of 4*P^2*d
    return dense + attn + 2.0 * d * spec.vocab_size  # logits for the last position


def bytes_decode_token(arch: dict, spec: SpaceSpec, wl: Workload, position: int) -> float:
    bpp = BYTES_PER_PARAM[wl.weight_dtype]
    weights = (decoder_params(arch) + spec.vocab_size * arch["d_model"]) * bpp  # all layers + LM head
    return weights + kv_cache_bytes(arch, wl, position)


def kv_cache_bytes(arch: dict, wl: Workload, tokens: int) -> float:
    d = arch["d_model"]
    kvb = BYTES_PER_PARAM[wl.kv_dtype]
    return sum(2.0 * kv_heads(ly) * (d // ly["n_head"]) * tokens * kvb for ly in layer_list(arch))


def evaluate_costs(arch: dict, spec: SpaceSpec, dev: DeviceProfile, wl: Workload) -> dict:
    """Analytical latency (ms) and memory (MB) for one architecture on one device."""
    L = arch["n_layer"]
    P, G = wl.prompt_tokens, wl.gen_tokens
    f = flops_prefill(arch, spec, P)
    weights_bytes = total_params(arch, spec) * BYTES_PER_PARAM[wl.weight_dtype]
    b_mean = sum(bytes_decode_token(arch, spec, wl, P + t) for t in range(0, G, max(1, G // 5))) / len(
        range(0, G, max(1, G // 5)))  # fmt: skip

    if dev.fit:  # calibrated: t = a*work + b*n_layer + c  (seconds -> ms)
        a, b, c = dev.fit["prefill"]
        prefill_ms = 1e3 * (a * f + b * L + c)
        a, b, c = dev.fit["decode"]
        decode_ms = 1e3 * (a * b_mean + b * L + c)
    else:
        compute_s = f / (dev.peak_tflops * 1e12 * dev.prefill_eff)
        load_s = weights_bytes / (dev.mem_bw_gbs * 1e9 * dev.decode_eff)
        prefill_ms = 1e3 * max(compute_s, load_s) + L * dev.layer_overhead_us / 1e3 + dev.fixed_overhead_ms
        decode_ms = 1e3 * b_mean / (dev.mem_bw_gbs * 1e9 * dev.decode_eff) + L * dev.layer_overhead_us / 1e3 \
            + dev.fixed_overhead_ms

    kv_mb = kv_cache_bytes(arch, wl, wl.context) / 2**20
    buf_mb = wl.compute_buffer_tokens * spec.vocab_size * 4 / 2**20
    mem_mb = weights_bytes / 2**20 + kv_mb + buf_mb
    return dict(
        decoder_params_m=decoder_params(arch) / 1e6,
        total_params_m=total_params(arch, spec) / 1e6,
        est_loss=estimated_loss(arch),
        prefill_ms=prefill_ms,
        decode_ms_per_token=decode_ms,
        latency_ms=prefill_ms + G * decode_ms,  # end-to-end for the survey's 50-in / 50-out setting
        memory_mb=mem_mb,
        kv_cache_mb=kv_mb,
        feasible=mem_mb <= dev.memory_gb * 1024,
    )


def device_from_config(cfg: dict) -> DeviceProfile:
    """cfg = {"profile": "<name>", optional overrides incl. "fit"}."""
    base = DEVICES[cfg["profile"]] if cfg.get("profile") in DEVICES else DeviceProfile(cfg.get("profile", "custom"), 1, 1, 1)
    over = {k: v for k, v in cfg.items() if k != "profile" and v is not None}
    return DeviceProfile(**{**base.__dict__, **over})
