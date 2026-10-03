"""Decoder-only transformer search space (LTS + SLM-survey), parameter counting, niches,
and the random operators used by the non-LLM baselines.

An architecture is a plain dict, so it can go straight into/out of an LLM prompt:

    {"n_layer": 24, "d_model": 1536, "ffn": "gated",
     "layers": [{"d_inner": 4096, "n_head": 12, "n_kv_head": 4}]}

``layers`` has one entry when the space is homogeneous (applies to every layer) and
``n_layer`` entries when it is heterogeneous (LTS's per-layer search space).
Vocabulary size and embedding tying are fixed per space (see ``SpaceSpec``).
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- spec


@dataclass(frozen=True)
class SpaceSpec:
    name: str
    heterogeneous: bool
    n_layer: tuple[int, int, int]  # (min, max, step)
    d_model: tuple[int, int, int]
    d_inner: tuple[int, int, int]
    ffn_ratio: tuple[float, float]  # d_inner / d_model bounds
    n_head: tuple[int, ...]
    n_kv_head: tuple  # ints, or "n_head" for plain multi-head attention
    ffn: tuple[str, ...]  # "gated" (SwiGLU, 3 matrices) and/or "standard" (2 matrices)
    vocab_size: int
    tie_embeddings: bool
    head_dims: tuple[int, ...] = ()  # allowed d_model / n_head; empty = any integer

    @classmethod
    def from_config(cls, cfg: dict) -> "SpaceSpec":
        c = dict(cfg)
        for k in ("n_layer", "d_model", "d_inner", "ffn_ratio", "n_head", "n_kv_head", "ffn", "head_dims"):
            if k in c and c[k] is not None:
                c[k] = tuple(c[k])
        return cls(**c)

    def grid(self, key: str) -> list[int]:
        lo, hi, step = getattr(self, key)
        return list(range(lo, hi + 1, step))

    def describe(self) -> str:
        kv = ", ".join(str(k) for k in self.n_kv_head)
        lines = [
            f"- n_layer: {self.n_layer[0]}..{self.n_layer[1]} (step {self.n_layer[2]})",
            f"- d_model: {self.d_model[0]}..{self.d_model[1]} (step {self.d_model[2]})",
            f"- d_inner (FFN hidden size): {self.d_inner[0]}..{self.d_inner[1]} (step {self.d_inner[2]}), "
            f"and {self.ffn_ratio[0]:g} <= d_inner/d_model <= {self.ffn_ratio[1]:g}",
            f"- n_head: one of {list(self.n_head)}; d_model must be divisible by n_head"
            + (f" with head_dim = d_model/n_head in {list(self.head_dims)}" if self.head_dims else ""),
            f"- n_kv_head: one of [{kv}] (n_kv_head < n_head = grouped-query attention, 1 = multi-query, "
            "'n_head' = standard multi-head); n_head must be divisible by n_kv_head",
            f"- ffn: one of {list(self.ffn)} (gated = SwiGLU with 3 matrices, standard = 2 matrices)",
            f"- fixed: vocab_size={self.vocab_size}, tied input/output embeddings={self.tie_embeddings}",
            "- layers: " + ("a list with exactly n_layer entries (each layer may differ)" if self.heterogeneous
                            else "a list with exactly ONE entry, used for every layer"),
        ]  # fmt: skip
        return "\n".join(lines)


# --------------------------------------------------------------------------- canonical form


def arch_key(arch: dict) -> str:
    return json.dumps(arch, sort_keys=True, separators=(",", ":"))


def layer_list(arch: dict) -> list[dict]:
    """Expanded per-layer configs (length n_layer)."""
    ls = arch["layers"]
    return ls * arch["n_layer"] if len(ls) == 1 else ls


def kv_heads(layer: dict) -> int:
    kv = layer["n_kv_head"]
    return layer["n_head"] if kv == "n_head" else int(kv)


def _nearest(value, choices):
    return min(choices, key=lambda c: (abs(c - value), c))


def _repair_kv(kv, n_head: int, spec: SpaceSpec):
    """Nearest allowed KV-head count that divides n_head; canonical 'n_head' for plain MHA."""
    allowed = sorted({n_head if k == "n_head" else int(k) for k in spec.n_kv_head
                      if k == "n_head" or (int(k) <= n_head and n_head % int(k) == 0)})  # fmt: skip
    if not allowed:
        return None
    try:
        target = n_head if kv == "n_head" else int(kv)
    except (TypeError, ValueError):
        target = n_head
    val = _nearest(target, allowed)
    return "n_head" if val == n_head and "n_head" in spec.n_kv_head else val


def repair(arch: dict, spec: SpaceSpec) -> tuple[dict | None, list[str]]:
    """Snap an (LLM- or operator-produced) architecture onto the grid and fix the
    divisibility constraints. Returns (arch or None, list of notes about what changed)."""
    notes: list[str] = []
    try:
        a = {"n_layer": int(arch["n_layer"]), "d_model": int(arch["d_model"]),
             "ffn": str(arch.get("ffn", spec.ffn[0])), "layers": [dict(x) for x in arch["layers"]]}  # fmt: skip
    except (KeyError, TypeError, ValueError):
        return None, ["missing or malformed fields"]
    if not a["layers"]:
        return None, ["empty layers"]

    for key in ("n_layer", "d_model"):
        g = spec.grid(key)
        v = _nearest(a[key], g)
        if v != a[key]:
            notes.append(f"{key} {a[key]}->{v}")
            a[key] = v
    if a["ffn"] not in spec.ffn:
        notes.append(f"ffn {a['ffn']}->{spec.ffn[0]}")
        a["ffn"] = spec.ffn[0]

    if spec.heterogeneous:
        if len(a["layers"]) == 1:
            a["layers"] = a["layers"] * a["n_layer"]
        elif len(a["layers"]) != a["n_layer"]:
            notes.append(f"{len(a['layers'])} layer entries for n_layer={a['n_layer']}")
            a["layers"] = (a["layers"] + [a["layers"][-1]] * a["n_layer"])[: a["n_layer"]]
    elif len(a["layers"]) != 1:
        if len({arch_key(x) for x in a["layers"]}) > 1:
            notes.append("heterogeneous layers collapsed to the first")
        a["layers"] = a["layers"][:1]

    d = a["d_model"]
    heads_ok = [h for h in spec.n_head if d % h == 0 and (not spec.head_dims or d // h in spec.head_dims)]
    if not heads_ok:
        return None, notes + [f"no valid n_head for d_model={d}"]
    inner_grid = [x for x in spec.grid("d_inner") if spec.ffn_ratio[0] * d <= x <= spec.ffn_ratio[1] * d]
    if not inner_grid:
        return None, notes + [f"no valid d_inner for d_model={d}"]
    fixed = []
    for i, layer in enumerate(a["layers"]):
        try:
            di, nh, kv = int(layer["d_inner"]), int(layer["n_head"]), layer.get("n_kv_head", "n_head")
        except (KeyError, TypeError, ValueError):
            return None, notes + [f"layer {i} malformed"]
        di2, nh2 = _nearest(di, inner_grid), _nearest(nh, heads_ok)
        kv2 = _repair_kv(kv, nh2, spec)
        if kv2 is None:
            return None, notes + [f"layer {i}: no valid n_kv_head for n_head={nh2}"]
        if (di2, nh2) != (di, nh) or str(kv2) != str(kv):
            notes.append(f"layer {i}: ({di},{nh},{kv})->({di2},{nh2},{kv2})")
        fixed.append({"d_inner": di2, "n_head": nh2, "n_kv_head": kv2})
    a["layers"] = fixed
    return a, notes


# --------------------------------------------------------------------------- parameter counts


def layer_params(d: int, layer: dict, ffn: str) -> int:
    nh = layer["n_head"]
    hd = d // nh
    attn = 2 * d * d + 2 * d * kv_heads(layer) * hd  # q, o + k, v
    mlp = (3 if ffn == "gated" else 2) * d * layer["d_inner"]
    return attn + mlp + 2 * d  # two RMSNorms


def decoder_params(arch: dict) -> int:
    """Non-embedding parameters - LTS's zero-cost proxy for perplexity."""
    d = arch["d_model"]
    return sum(layer_params(d, ly, arch["ffn"]) for ly in layer_list(arch)) + d  # + final norm


def total_params(arch: dict, spec: SpaceSpec) -> int:
    emb = spec.vocab_size * arch["d_model"]
    return decoder_params(arch) + emb * (1 if spec.tie_embeddings else 2)


# --------------------------------------------------------------------------- niches


@dataclass(frozen=True)
class SLMNiche:
    """Disjoint band of decoder-parameter count, the transformer analogue of Table 1."""

    name: str
    lo: float  # millions of decoder params, inclusive
    hi: float  # exclusive
    rationale: str = ""

    def contains(self, arch: dict) -> bool:
        return self.lo <= decoder_params(arch) / 1e6 < self.hi

    def constraint_text(self) -> str:
        hi = "no upper limit" if self.hi == float("inf") else f"below {self.hi:g}M"
        return f"- MUST have decoder (non-embedding) parameters of at least {self.lo:g}M and {hi}"


def niches_from_config(bounds: list[float], labels: list[str] | None = None) -> list[SLMNiche]:
    edges = list(bounds) + [float("inf")]
    labels = labels or [""] * (len(edges) - 1)
    return [SLMNiche(f"S{i}", edges[i], edges[i + 1], labels[i]) for i in range(len(edges) - 1)]


FULL_SLM_SPACE = SLMNiche("FULL", 0.0, float("inf"), "entire search space (no partitioning)")


# --------------------------------------------------------------------------- random operators


def random_arch(spec: SpaceSpec, rng: random.Random) -> dict:
    d = rng.choice(spec.grid("d_model"))
    n = rng.choice(spec.grid("n_layer"))
    ffn = rng.choice(spec.ffn)

    def layer():
        return {"d_inner": rng.choice(spec.grid("d_inner")), "n_head": rng.choice(spec.n_head),
                "n_kv_head": rng.choice(spec.n_kv_head)}  # fmt: skip

    arch = {"n_layer": n, "d_model": d, "ffn": ffn,
            "layers": [layer() for _ in range(n if spec.heterogeneous else 1)]}  # fmt: skip
    return repair(arch, spec)[0]


@dataclass
class Sampler:
    """Rejection sampler for niches, with a memo of how hard each niche is to hit."""

    spec: SpaceSpec
    max_tries: int = 20000
    _stats: dict = field(default_factory=dict)

    def sample(self, niche: SLMNiche, rng: random.Random, exclude: set[str]) -> dict | None:
        for _ in range(self.max_tries):
            a = random_arch(self.spec, rng)
            if a and niche.contains(a) and arch_key(a) not in exclude:
                return a
        return None


def mutate(arch: dict, spec: SpaceSpec, rng: random.Random, p: float = 0.3) -> dict | None:
    """LTS-style mutation: each gene re-drawn with probability p (at least one changes)."""
    a = copy.deepcopy(arch)
    a["layers"] = layer_list(a) if spec.heterogeneous else a["layers"]
    genes = ["n_layer", "d_model", "ffn"] + [(i, k) for i in range(len(a["layers"]))
                                            for k in ("d_inner", "n_head", "n_kv_head")]  # fmt: skip
    chosen = [g for g in genes if rng.random() < p] or [rng.choice(genes)]
    for g in chosen:
        if g == "ffn":
            a["ffn"] = rng.choice(spec.ffn)
        elif g in ("n_layer", "d_model"):
            grid = spec.grid(g)
            i = grid.index(a[g]) if a[g] in grid else 0
            a[g] = grid[max(0, min(len(grid) - 1, i + rng.choice([-3, -2, -1, 1, 2, 3])))]
        else:
            i, k = g
            if k == "d_inner":
                grid = spec.grid("d_inner")
                j = grid.index(a["layers"][i][k]) if a["layers"][i][k] in grid else 0
                a["layers"][i][k] = grid[max(0, min(len(grid) - 1, j + rng.choice([-4, -2, -1, 1, 2, 4])))]
            else:
                a["layers"][i][k] = rng.choice(spec.n_head if k == "n_head" else spec.n_kv_head)
    out, _ = repair(a, spec)
    return out


def crossover(a: dict, b: dict, spec: SpaceSpec, rng: random.Random) -> dict | None:
    """Uniform crossover over global genes and (aligned) per-layer configs."""
    child = {k: (a if rng.random() < 0.5 else b)[k] for k in ("n_layer", "d_model", "ffn")}
    la, lb = layer_list(a), layer_list(b)
    if spec.heterogeneous:
        n = child["n_layer"]
        pad = lambda ls: (ls + [ls[-1]] * n)[:n]  # noqa: E731
        la, lb = pad(la), pad(lb)
        child["layers"] = [dict(x if rng.random() < 0.5 else y) for x, y in zip(la, lb)]
    else:
        x, y = a["layers"][0], b["layers"][0]
        child["layers"] = [{k: (x if rng.random() < 0.5 else y)[k] for k in ("d_inner", "n_head", "n_kv_head")}]
    out, _ = repair(child, spec)
    return out
