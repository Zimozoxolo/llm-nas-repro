"""Hand-designed SLMs from the survey (Lu et al. 2024, Table 1), in this repo's encoding.

Layer count, hidden size and head count match the survey's Table 1. FFN size, KV heads,
vocabulary and embedding tying come from each model's public Hugging Face
``config.json`` (verify before quoting: ``python -m llmnas slm-reference``).
Models whose blocks don't fit the encoding (Gemma-2's head_dim != d_model/n_head,
OpenELM's per-layer scaling, MiniCPM's MLA) are left out.

They are scored with the same proxy and cost model as searched architectures, with
their own vocabulary, so they appear as reference points on the Pareto plots (the
role OPT-350M plays in LTS).
"""

from __future__ import annotations

from .space import SpaceSpec


def _m(L, d, nh, kv, di, vocab, tied, official_m, ffn="gated", source=""):
    return dict(arch={"n_layer": L, "d_model": d, "ffn": ffn,
                      "layers": [{"d_inner": di, "n_head": nh, "n_kv_head": "n_head" if kv == nh else kv}]},
                vocab_size=vocab, tie_embeddings=tied, official_params_m=official_m, source=source)  # fmt: skip


REFERENCE_SLMS = {
    "SmolLM-135M": _m(30, 576, 9, 3, 1536, 49152, True, 135, source="HuggingFaceTB/SmolLM-135M"),
    "SmolLM-360M": _m(32, 960, 15, 5, 2560, 49152, True, 362, source="HuggingFaceTB/SmolLM-360M"),
    "Qwen2.5-0.5B": _m(24, 896, 14, 2, 4864, 151936, True, 494, source="Qwen/Qwen2.5-0.5B"),
    "TinyLlama-1.1B": _m(22, 2048, 32, 4, 5632, 32000, False, 1100, source="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"),
    "Llama-3.2-1B": _m(16, 2048, 32, 8, 8192, 128256, True, 1236, source="meta-llama/Llama-3.2-1B"),
    "Pythia-1.4B": _m(24, 2048, 16, 16, 8192, 50304, False, 1415, ffn="standard", source="EleutherAI/pythia-1.4b"),
    "Qwen2.5-1.5B": _m(28, 1536, 12, 2, 8960, 151936, True, 1544, source="Qwen/Qwen2.5-1.5B"),
    "SmolLM-1.7B": _m(24, 2048, 32, 32, 8192, 49152, True, 1711, source="HuggingFaceTB/SmolLM-1.7B"),
    "Phi-2": _m(32, 2560, 32, 32, 10240, 51200, False, 2780, ffn="standard", source="microsoft/phi-2"),
    "Qwen2.5-3B": _m(36, 2048, 16, 2, 11008, 151936, True, 3086, source="Qwen/Qwen2.5-3B"),
    "Llama-3.2-3B": _m(28, 3072, 24, 8, 8192, 128256, True, 3213, source="meta-llama/Llama-3.2-3B"),
}  # fmt: skip


def reference_spec(ref: dict, base: SpaceSpec) -> SpaceSpec:
    """The search space's spec with the reference model's own vocabulary and tying."""
    return SpaceSpec(**{**base.__dict__, "vocab_size": ref["vocab_size"], "tie_embeddings": ref["tie_embeddings"]})
