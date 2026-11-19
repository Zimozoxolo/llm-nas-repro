"""NAS-Bench-201 / HW-NAS-Bench cell search space and the paper's niches.

Encoding used throughout this repo is the one from Figure 2 of the paper:
an architecture is a 6-tuple ``(a, b, c, d, e, f)`` of operator codes

    0: none (zeroize)   1: skip_connect   2: nor_conv_1x1
    3: nor_conv_3x3     4: avg_pool_3x3

and the six positions follow the NB201 arch-string order

    |a~0|+|b~0|c~1|+|d~0|e~1|f~2|
     n1<-0  n2<-0 n2<-1  n3<-0 n3<-1 n3<-2

NASLib (and therefore NAS-Bench-Suite-Zero's ``zc_nasbench201.json``) uses a
different op order and edge order; ``naslib_key_to_code`` converts. That
mapping was verified against all 15,625 architectures (validation accuracies
match exactly), see ``tests/test_search_space.py``.
"""

from __future__ import annotations

import ast
import itertools
from functools import lru_cache
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
OP_INDEX = {op: i for i, op in enumerate(OPS)}
NUM_EDGES = 6
CONV3, CONV1 = OP_INDEX["nor_conv_3x3"], OP_INDEX["nor_conv_1x1"]

# NASLib: OP_NAMES = [Identity, Zero, ReLUConvBN3x3, ReLUConvBN1x1, AvgPool1x1]
_NASLIB_OPS = ["skip_connect", "none", "nor_conv_3x3", "nor_conv_1x1", "avg_pool_3x3"]
# NASLib edge list (1,2),(1,3),(1,4),(2,3),(2,4),(3,4) -> NB201 string positions
_NASLIB_TO_STR_POS = [0, 1, 3, 2, 4, 5]

Code = tuple[int, int, int, int, int, int]


def code_to_str(code: Sequence[int]) -> str:
    o = [OPS[i] for i in code]
    return f"|{o[0]}~0|+|{o[1]}~0|{o[2]}~1|+|{o[3]}~0|{o[4]}~1|{o[5]}~2|"


def str_to_code(arch_str: str) -> Code:
    ops = [tok.split("~")[0] for tok in arch_str.replace("+", "").split("|") if tok]
    return tuple(OP_INDEX[o] for o in ops)  # type: ignore[return-value]


def naslib_key_to_code(key: str | Sequence[int]) -> Code:
    t = ast.literal_eval(key) if isinstance(key, str) else key
    return tuple(OP_INDEX[_NASLIB_OPS[t[i]]] for i in _NASLIB_TO_STR_POS)  # type: ignore[return-value]


def is_valid_code(code) -> bool:
    try:
        return len(code) == NUM_EDGES and all(int(c) == c and 0 <= c < len(OPS) for c in code)
    except TypeError:
        return False


def all_codes() -> list[Code]:
    return list(itertools.product(range(len(OPS)), repeat=NUM_EDGES))  # type: ignore[arg-type]


def format_code(code: Sequence[int]) -> str:
    return "(" + ", ".join(str(int(c)) for c in code) + ")"


# --------------------------------------------------------------------------- niches


@dataclass(frozen=True)
class Niche:
    """A disjoint subspace defined by counts of nor_conv_3x3 / nor_conv_1x1 (Table 1)."""

    name: str
    conv3_min: int
    conv3_max: int
    conv1_min: int = 0
    conv1_max: int = NUM_EDGES
    rationale: str = ""

    def contains(self, code: Sequence[int]) -> bool:
        n3 = sum(1 for c in code if c == CONV3)
        n1 = sum(1 for c in code if c == CONV1)
        return self.conv3_min <= n3 <= self.conv3_max and self.conv1_min <= n1 <= self.conv1_max

    def violation(self, code: Sequence[int]) -> str | None:
        """Why ``code`` breaks this niche's operator counts (None if it does not)."""

        def bound(lo: int, hi: int) -> str:
            return f"exactly {lo}" if lo == hi else f"at least {lo}" if hi >= NUM_EDGES else f"{lo}-{hi}"

        n3 = sum(1 for c in code if c == CONV3)
        n1 = sum(1 for c in code if c == CONV1)
        if not self.conv3_min <= n3 <= self.conv3_max:
            return f"has {n3} x nor_conv_3x3, needs {bound(self.conv3_min, self.conv3_max)}"
        if not self.conv1_min <= n1 <= self.conv1_max:
            return f"has {n1} x nor_conv_1x1, needs {bound(self.conv1_min, self.conv1_max)}"
        return None

    def members(self) -> list[Code]:
        return _members(self)

    def sample(self, rng: random.Random, k: int, exclude: Iterable[Code] = ()) -> list[Code]:
        ex = set(exclude)
        pool = [c for c in self.members() if c not in ex]
        return rng.sample(pool, min(k, len(pool)))

    def constraint_text(self) -> str:
        """Niche constraint block injected into prompts (Appendix E format)."""

        def count(lo: int, hi: int, op: str) -> str:
            if lo == hi:
                return f"exactly {lo} x {op}"
            if hi >= NUM_EDGES:
                return f"at least {lo} x {op}" if lo > 0 else f"any number (0-{NUM_EDGES}) of {op}"
            return f"{lo}-{hi} x {op}"

        n3 = count(self.conv3_min, self.conv3_max, "nor_conv_3x3")
        lines = [f"- MUST use {n3}"]
        if self.conv1_min > 0 or self.conv1_max < NUM_EDGES:
            lines.append(f"- MUST use {count(self.conv1_min, self.conv1_max, 'nor_conv_1x1')}")
        else:
            free = NUM_EDGES - self.conv3_min
            lines.append(f"- CAN use 0-{free} x nor_conv_1x1")
        lines.append("- ALLOWED operators: none, skip_connect, avg_pool_3x3")
        return "\n".join(lines)


@lru_cache(maxsize=None)
def _members(niche: Niche) -> list[Code]:
    return [c for c in all_codes() if niche.contains(c)]


# Table 1 of the paper.
PAPER_NICHES: list[Niche] = [
    Niche("S0", 0, 0, 0, 0, "Explores non-convolutional architectures"),
    Niche("S1", 0, 0, 1, NUM_EDGES, "Focuses on simple, low-latency models"),
    Niche("S2", 1, 1, rationale="Entry-level complex architectures"),
    Niche("S3", 2, 2, rationale="Mid-level complexity"),
    Niche("S4", 3, 3, rationale="High-level complexity"),
    Niche("S5", 4, NUM_EDGES, rationale="Explores the most complex designs"),
]

# Used by the "without partitioning" ablation and the exploration-bias study.
FULL_SPACE = Niche("FULL", 0, NUM_EDGES, rationale="Entire search space (no partitioning)")


def get_niches(partitioning: bool) -> list[Niche]:
    return PAPER_NICHES if partitioning else [FULL_SPACE]


# --------------------------------------------------------------------------- random operators (PEA)


def random_mutation(code: Code, niche: Niche, rng: random.Random, max_tries: int = 200) -> Code | None:
    """Change one edge's op, keeping the child inside ``niche``."""
    for _ in range(max_tries):
        child = list(code)
        pos = rng.randrange(NUM_EDGES)
        child[pos] = rng.choice([o for o in range(len(OPS)) if o != code[pos]])
        if niche.contains(child):
            return tuple(child)  # type: ignore[return-value]
    return None


def random_crossover(a: Code, b: Code, niche: Niche, rng: random.Random, max_tries: int = 200) -> Code | None:
    """Uniform crossover, keeping the child inside ``niche``."""
    for _ in range(max_tries):
        child = tuple(x if rng.random() < 0.5 else y for x, y in zip(a, b))
        if niche.contains(child) and child not in (a, b):
            return child  # type: ignore[return-value]
    return None
