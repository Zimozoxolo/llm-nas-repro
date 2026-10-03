"""Stage-1 / Stage-2 prompts for the SLM search, mirroring LLM-NAS Appendix E.

``SEED_RULES`` turns the SLM survey's "Insights" boxes and LTS's main finding into
initial knowledge-base rules (``search.kb_seed: survey``); with ``kb_seed: none`` the
LLM starts from an empty knowledge base, exactly like the CNN reproduction.
"""

from __future__ import annotations

import json
import re

from ..prompts import _extract_json, parse_knowledge_base  # noqa: F401  (re-exported)
from .space import SLMNiche, SpaceSpec

SEED_RULES = [
    # LTS, Sec. 4.3
    "Quality depends almost only on decoder parameter count, not on shape (width vs depth), "
    "so at a fixed parameter budget choose the shape that is cheapest on the device.",
    # SLM survey, Sec. 4.1 insights
    "Use fewer, wider layers when prefill latency matters, because prefill is compute-bound and every "
    "extra layer adds fixed per-layer overhead.",
    "Decode latency grows with the bytes of weights read per token, so it scales almost linearly with "
    "parameter count; shape matters less there than in prefill.",
    # SLM survey, Sec. 4.3.2
    "Use grouped-query attention (n_kv_head < n_head) because it shrinks the KV cache, which dominates "
    "memory at long context.",
    "Avoid spending parameters on the vocabulary/LM head on memory-limited devices, because the output "
    "layer's compute buffer scales with vocabulary size.",
    # SLM survey, Sec. 2.2
    "A gated FFN (SwiGLU) with d_inner/d_model between 2 and 8 is the standard choice in current SLMs.",
]

OBJ_LABELS = {"quality": "est_loss", "latency": "latency_ms", "memory": "memory_mb"}


def _fmt(e: dict) -> str:
    c = e["costs"]
    s = (f"  ID {e['id']}: {json.dumps(e['arch'], separators=(',', ':'))}\n"
         f"     decoder_params={c['decoder_params_m']:.1f}M est_loss={c['est_loss']:.4f} "
         f"latency={c['latency_ms']:.1f}ms (prefill {c['prefill_ms']:.1f} + decode {c['decode_ms_per_token']:.2f}/tok) "
         f"memory={c['memory_mb']:.0f}MB")  # fmt: skip
    if e.get("rationale"):
        s += f"\n     rationale: {e['rationale']}"
    return s


def _table(entries: list[dict]) -> str:
    return "\n".join(_fmt(e) for e in entries) if entries else "  (none)"


def _context(device: str, spec: SpaceSpec, niche: SLMNiche, wl_text: str, objectives: list[str]) -> str:
    goals = ", ".join({"quality": "low estimated loss (more decoder parameters)", "latency": "low latency",
                       "memory": "low memory"}[o] for o in objectives)  # fmt: skip
    return f"""- Target device: {device}
- Workload: {wl_text}
- Objectives (Pareto, all to minimise): {goals}
- Niche constraint:
{niche.constraint_text()}
- Search space (decoder-only transformer):
{spec.describe()}"""


KB_SYSTEM = """You are a NAS analyst for on-device small language models. Summarize design
heuristics for the given hardware-aware transformer search space."""

GEN_SYSTEM = """You are an expert designer of small language models that performs evolutionary
search inside a given niche under hardware constraints."""


def kb_update_prompt(device, spec, niche, wl_text, objectives, parents, last_cycle, kb, generation) -> str:
    return f"""[Context]
{_context(device, spec, niche, wl_text, objectives)}
- Top Pareto parents from generation {generation}:
{_table(parents)}
- Crossover & mutation results from the last cycle:
{_table(last_cycle)}
- Current knowledge base:
{json.dumps(kb, indent=1) if kb else "  (empty)"}

[Instruction]
1. Identify architectural choices (depth, width, FFN size, heads, KV heads) that consistently
reduce latency or memory at a given decoder parameter count.
2. Identify choices that consistently hurt either metric.
3. Write explicit, concise rules of the form "Use/avoid ... because ...".
4. Remove or revise outdated rules that conflict with new evidence.

[Output format]
Return a JSON list called Updated_Knowledge_Base:
["rule 1", "rule 2", ...]"""


def generation_prompt(device, spec, niche, wl_text, objectives, parents, last_cycle, kb, operations) -> str:
    plan = "\n".join(f"  child {i}: {op}" for i, op in enumerate(operations))
    example = json.dumps(parents[0]["arch"] if parents else {"n_layer": 12, "d_model": 768, "ffn": spec.ffn[0],
                         "layers": [{"d_inner": 2048, "n_head": 12, "n_kv_head": spec.n_kv_head[0]}]})  # fmt: skip
    return f"""[Context]
{_context(device, spec, niche, wl_text, objectives)}
- Current Pareto parents with metrics:
{_table(parents)}
- Crossover & mutation results from the last cycle:
{_table(last_cycle)}

[Knowledge Base]
{json.dumps(kb, indent=1) if kb else "(empty)"}

[Evolution Operation]
Perform {len(operations)} new candidate generations.
Operation for each child (crossover combines two parents, mutation modifies one parent):
{plan}
For each child:
* Say exactly which fields you combine or change.
* Justify the change with its expected effect on the objectives.
* Satisfy the niche constraint and the search-space rules.
* Do not repeat any architecture listed above.

[Output format]
Return a list of JSON objects:
[
{{"child_id": "...", "operation": "crossover/mutation",
  "architecture": {example},
  "rationale": "..."}},
...
]"""


def parse_slm_children(text: str) -> list[dict]:
    """-> [{'arch': dict|None, 'operation': str, 'rationale': str}]"""
    data = _extract_json(text)
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), [data])
    out = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            arch = item.get("architecture", item.get("architecture_code"))
            if isinstance(arch, str):
                try:
                    arch = json.loads(arch)
                except json.JSONDecodeError:
                    arch = None
            if arch is None and {"n_layer", "d_model"} <= set(item):
                arch = item
            out.append({"arch": arch if isinstance(arch, dict) else None,
                        "operation": str(item.get("operation", "")),
                        "rationale": re.sub(r"\s+", " ", str(item.get("rationale", "")))[:400]})  # fmt: skip
    return out
