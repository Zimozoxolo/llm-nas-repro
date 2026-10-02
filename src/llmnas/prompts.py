"""Prompt templates. Stage 1 / Stage 2 follow Appendix E of the paper verbatim where
the paper gives text; the bits it leaves as placeholders (search-space legend, how
parents are tabulated) are filled in minimally and marked ``# filled-in``.
"""

from __future__ import annotations

import json
import re

from .data import DEVICE_LABELS
from .search_space import NUM_EDGES, OPS, Niche, format_code, is_valid_code

DATASET_LABELS = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "ImageNet16-120": "ImageNet16-120"}

# filled-in: the paper's Figure 2 encoding, so the LLM can read/write architecture codes
SPACE_LEGEND = """Search space (NAS-Bench-201 cell, HW-NAS-Bench): an architecture is a 6-tuple
(a, b, c, d, e, f) giving the operator on each edge of a 4-node cell, in the order
node1<-node0, node2<-node0, node2<-node1, node3<-node0, node3<-node1, node3<-node2.
Operator codes: 0: none (zeroize), 1: skip_connect, 2: nor_conv_1x1, 3: nor_conv_3x3, 4: avg_pool_3x3."""


def _target(device: str, dataset: str) -> str:
    return f"{DEVICE_LABELS.get(device, device)}, {DATASET_LABELS.get(dataset, dataset)}"


def _table(entries: list[dict], with_rationale: bool) -> str:
    if not entries:
        return "  (none)"
    lines = []
    for e in entries:
        s = (f"  ID {e['id']}: {format_code(e['code'])}  pred_accuracy={e['acc']:.2f}%  "
             f"latency={e['lat']:.3f} ms")  # fmt: skip
        if with_rationale and e.get("rationale"):
            s += f"  rationale: {e['rationale']}"
        lines.append(s)
    return "\n".join(lines)


# --------------------------------------------------------------------------- Stage 1

KB_SYSTEM = """You are a NAS analyst. Summarize design heuristics
for the given hardware-aware search space."""


def kb_update_prompt(device: str, dataset: str, niche: Niche, parents: list[dict],
                     last_cycle: list[dict], knowledge_base: list[str], generation: int) -> str:  # fmt: skip
    return f"""[Context]
- Target device and dataset: {_target(device, dataset)}
- Niche definition:
{niche.constraint_text()}
{SPACE_LEGEND}
- Top Pareto parents from generation {generation}:
{_table(parents, with_rationale=True)}
- Crossover & mutation results from the last cycle:
{_table(last_cycle, with_rationale=True)}
- Current knowledge base:
{json.dumps(knowledge_base, indent=1) if knowledge_base else "  (empty)"}

[Instruction]
1. Identify operator or connection patterns that
consistently improve accuracy at acceptable latency.
2. Identify patterns that consistently hurt either metric.
3. Write explicit, concise rules of the form
"Use/avoid ... because ...".
4. Remove or revise outdated rules that conflict with new evidence.

[Output format]
Return a JSON-like list called Updated_Knowledge_Base:
[
{{rule_1}},
{{rule_2}},
...
]"""


# --------------------------------------------------------------------------- Stage 2

GEN_SYSTEM = """You are an expert NAS designer that performs evolutionary
search inside a given niche under hardware constraints."""


def generation_prompt(device: str, dataset: str, niche: Niche, parents: list[dict], last_cycle: list[dict],
                      knowledge_base: list[str], operations: list[str], latency_limit: float | None) -> str:  # fmt: skip
    limit = f"\\le {latency_limit:.3f} ms" if latency_limit else "as low as possible"
    ops_plan = "\n".join(f"  child {i}: {op}" for i, op in enumerate(operations))
    constraints = niche.constraint_text()
    if latency_limit:
        constraints += f"\n- Hardware latency must remain below {latency_limit:.3f} ms"
    return f"""[Context]
- Target device and dataset: {_target(device, dataset)}
- Niche constraints:
{constraints}
{SPACE_LEGEND}
- Current Pareto parents with metrics:
{_table(parents, with_rationale=False)}
- Crossover & mutation results from the last cycle:
{_table(last_cycle, with_rationale=True)}

[Knowledge Base]
{json.dumps(knowledge_base, indent=1) if knowledge_base else "(empty)"}

[Evolution Operation]
Perform {len(operations)} new candidate generations.
Operation for each child (crossover uses two parents, mutation modifies one parent):
{ops_plan}
For each child:
* Describe exactly which blocks/edges you combine or modify.
* Justify each change with expected effect on
accuracy and latency ({limit}).
* Ensure all constraints are satisfied.
* Do not repeat any architecture listed above.

[Output format]
Return a list of JSON objects:
[
{{
"child_id": "...",
"operation": "crossover/mutation",
"architecture_code": "(a, b, c, d, e, f)",
"rationale": "..."
}},
...
]"""


# --------------------------------------------------------------------------- exploration-bias study (Fig. 1)

BIAS_SYSTEM = "You are an expert NAS assistant."


def bias_prompt(device: str, dataset: str, n: int, variant: str, history: list[dict]) -> str:
    base = f"""Target device and dataset: {_target(device, dataset)}.
{SPACE_LEGEND}
Propose {n} architectures for this device and dataset."""
    if variant == "latency_optimized":
        base += """
Aim for DIVERSE latencies: cover the full range from very fast to very accurate models,
so that the set forms a good accuracy-latency Pareto front."""
        if history:
            base += "\nResults of the previous round:\n" + _table(history, with_rationale=False)
    return base + """
Return a JSON list: [{"architecture_code": "(a, b, c, d, e, f)", "rationale": "..."}, ...]"""


# --------------------------------------------------------------------------- parsing

_CODE_RE = re.compile(r"[\(\[]\s*([0-4])\s*,\s*([0-4])\s*,\s*([0-4])\s*,\s*([0-4])\s*,\s*([0-4])\s*,\s*([0-4])\s*[\)\]]")


def _extract_json(text: str):
    text = re.sub(r"```(?:json)?", "", text)
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def _to_code(value) -> tuple | None:
    if isinstance(value, (list, tuple)):
        try:
            code = tuple(int(v) for v in value)
        except (TypeError, ValueError):
            return None
        return code if is_valid_code(code) else None
    if isinstance(value, str):
        m = _CODE_RE.search(value)
        if m:
            return tuple(int(g) for g in m.groups())
        names = re.findall("|".join(OPS), value)
        if len(names) == NUM_EDGES:
            return tuple(OPS.index(n) for n in names)
    return None


def parse_children(text: str) -> list[dict]:
    """-> [{'code': tuple|None, 'operation': str, 'rationale': str}]"""
    data = _extract_json(text)
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), [data])
    out = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                out.append({"code": _to_code(item.get("architecture_code", item.get("architecture"))),
                            "operation": str(item.get("operation", "")),
                            "rationale": str(item.get("rationale", ""))[:500]})  # fmt: skip
    if not out:  # fall back to any tuples in free text
        out = [{"code": tuple(int(g) for g in m.groups()), "operation": "", "rationale": ""}
               for m in _CODE_RE.finditer(text)]  # fmt: skip
    return out


def parse_knowledge_base(text: str, max_rules: int = 20) -> list[str] | None:
    data = _extract_json(text)
    if isinstance(data, dict):
        data = data.get("Updated_Knowledge_Base", next((v for v in data.values() if isinstance(v, list)), None))
    if not isinstance(data, list):
        return None
    rules = []
    for r in data:
        if isinstance(r, dict):
            r = next(iter(r.values()), "") if len(r) == 1 else json.dumps(r)
        r = str(r).strip()
        if r:
            rules.append(r[:300])
    return rules[:max_rules]
