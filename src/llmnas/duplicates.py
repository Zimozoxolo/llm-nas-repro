"""Where do duplicate LLM proposals come from?  ``python -m llmnas duplicates``

Replays every archived LLM run from its llm_calls.jsonl (no LLM is called) and classifies each proposal that
the search rejected as a duplicate, by where the repeated cell had been seen before:

    parent shown     one of the Pareto parents listed in this very prompt
    last cycle shown one of the previous cycle's children listed in this prompt
    sibling          an earlier accepted child in the same reply
    hidden history   evaluated earlier in this niche but not shown in this prompt

and also records, for each generate call, how full the niche is (evaluated / niche size) and which operation
(mutation / crossover) the child was asked for.
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import pandas as pd

from .benchmark import load_benchmark
from .conventions import load_runs
from .experiments import REPO
from .prompts import parse_children
from .search_space import PAPER_NICHES

_TUPLE = re.compile(r"ID (L?\d+): \(([0-4](?:, [0-4]){5})\)")
SIZE = {n.name: len(n.members()) for n in PAPER_NICHES}
NICHE = {n.name: n for n in PAPER_NICHES}


def _shown(prompt: str) -> tuple[set, set]:
    parents, last = set(), set()
    for ident, t in _TUPLE.findall(prompt):
        code = tuple(int(x) for x in t.split(", "))
        (last if ident.startswith("L") else parents).add(code)
    return parents, last


def classify_run(item: dict) -> list[dict]:
    r = item["run"]
    calls_f = Path(item["path"]).with_name("llm_calls.jsonl")
    if not calls_f.exists():
        return []
    ev = {tuple(int(x) for x in e["code"].strip("()").split(", ")): e for e in r["evaluated"]}
    seen = {c for c, e in ev.items() if e["gen"] == 0}  # initial population
    unmeasured = load_benchmark(item["dataset"]).unmeasured(item["device"])
    n_child = r["config"]["search"]["n_child"]
    rows = []
    for line in calls_f.read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        if c["stage"] != "generate":
            continue
        niche = NICHE[c["niche_name"]]
        parents, last = _shown(c["user"])
        in_niche_seen = sum(1 for x in seen if niche.contains(x))
        batch = []
        for k, ch in enumerate((parse_children(c["response"]) or [])[:n_child]):
            code = ch["code"]
            row = dict(method=item["method"], dataset=item["dataset"], device=item["device"], seed=item["seed"],
                       gen=c["generation"], niche=niche.name, op=ch.get("operation") or "?",
                       fill=in_niche_seen / SIZE[niche.name])  # fmt: skip
            if code is None:
                row["outcome"] = "invalid"
            elif not niche.contains(code):
                row["outcome"] = "out_of_niche"
            elif code in seen or code in batch:
                row["outcome"] = "duplicate"
                row["source"] = ("parent shown" if code in parents else "last cycle shown" if code in last
                                 else "sibling" if code in batch else "hidden history")  # fmt: skip
            elif code in unmeasured:
                row["outcome"] = "no_latency"
            else:
                row["outcome"] = "new"
                batch.append(code)
            rows.append(row)
        seen |= set(batch)
    return rows


def table() -> pd.DataFrame:
    rows = []
    for item in load_runs(["cifar10", "cifar100", "ImageNet16-120"]):
        if item["run"]["config"]["search"]["operator"] == "llm":
            rows += classify_run(item)
    return pd.DataFrame(rows)


def report(out: Path | None = None) -> str:
    df = table()
    out = out or REPO / "results" / "reports" / "duplicates.md"
    df.to_csv(out.with_suffix(".csv"), index=False)
    lines = ["# Where duplicate LLM proposals come from\n",
             "Replayed from the archived prompts and replies (no LLM called). Shares are of all proposals.\n"]
    outs = ["new", "duplicate", "out_of_niche", "no_latency", "invalid"]
    lines.append("| Model | proposals | " + " | ".join(outs) + " |")
    lines.append("|---|---|" + "---|" * len(outs))
    for m, g in df.groupby("method"):
        lines.append(f"| {m} | {len(g)} | " + " | ".join(f"{(g.outcome == o).mean():.1%}" for o in outs) + " |")
    lines.append("")
    srcs = ["parent shown", "last cycle shown", "sibling", "hidden history"]
    lines.append("| Model | proposals | duplicates | " + " | ".join(srcs) + " |")
    lines.append("|---|---|---|" + "---|" * len(srcs))
    for m, g in df.groupby("method"):
        dup = g[g.outcome == "duplicate"]
        lines.append(f"| {m} | {len(g)} | {len(dup) / len(g):.1%} | "
                     + " | ".join(f"{(dup.source == s).sum() / len(g):.1%}" for s in srcs) + " |")  # fmt: skip
    lines.append("\n## Duplicate rate by niche (all models)\n")
    lines.append("| Niche | size | proposals | duplicate rate | mean share of niche already evaluated |")
    lines.append("|---|---|---|---|---|")
    for n, g in df.groupby("niche"):
        lines.append(f"| {n} | {SIZE[n]} | {len(g)} | {(g.outcome == 'duplicate').mean():.1%} | {g.fill.mean():.1%} |")
    lines.append("\n## Duplicate rate by requested operation\n")
    lines.append("| Model | mutation | crossover |")
    lines.append("|---|---|---|")
    for m, g in df.groupby("method"):
        rate = g.groupby(g.op.str.lower()).outcome.apply(lambda s: (s == "duplicate").mean())
        lines.append(f"| {m} | {rate.get('mutation', float('nan')):.1%} | {rate.get('crossover', float('nan')):.1%} |")
    lines.append("\n## Duplicate rate by generation (all models)\n")
    lines.append("| Generation | " + " | ".join(str(g) for g in sorted(df.gen.unique())) + " |")
    lines.append("|---|" + "---|" * df.gen.nunique())
    lines.append("| duplicate rate | " + " | ".join(f"{(g.outcome == 'duplicate').mean():.0%}"
                                                  for _, g in df.groupby("gen")) + " |")  # fmt: skip
    text = "\n".join(lines) + "\n"
    out.write_text(text, encoding="utf-8")
    return text
