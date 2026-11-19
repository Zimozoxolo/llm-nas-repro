"""Duplicate fixes (PREREGISTRATION_duplicate_fix.md): niche history in the prompt, re-prompting, PEA refill.

All offline: the mock backend or scripted fake LLMs, synthetic data.
"""

import json

import pytest
import yaml

import llmnas.search as search_mod
from llmnas.experiments import REPO, deep_merge
from llmnas.llm import LLMBackend, LLMResponse, MockBackend
from llmnas.prompts import HISTORY_CAP, generation_prompt, retry_prompt
from llmnas.search import run_search
from llmnas.search_space import PAPER_NICHES, format_code

BASE = yaml.safe_load(open(REPO / "configs" / "base.yaml"))
ARM_A = {"show_history": True, "reprompt_retries": 5, "budget_evaluations": 270}


def _cfg(data_dir, gens=2, **search):
    return deep_merge(BASE, {"data_dir": str(data_dir), "llm": {"backend": "mock"}, "predictor": {"n_train": 300},
                             "search": {"generations": gens, **search}})


def _calls(run_dir):
    return [json.loads(x) for x in (run_dir / "llm_calls.jsonl").read_text().splitlines()]


class Scripted(LLMBackend):
    """Fake LLM: ``reply(meta) -> list of codes`` for generate calls; the mock's rules for the knowledge base."""

    model = "scripted"

    def __init__(self, reply):
        self.reply, self.mock = reply, MockBackend(0)

    def complete(self, system, user, meta):
        if meta.get("stage") == "kb_update":
            return self.mock.complete(system, user, meta)
        codes = self.reply(meta)
        return LLMResponse(json.dumps([{"child_id": i, "operation": "mutation", "rationale": "",
                                        "architecture_code": c if isinstance(c, str) else format_code(c)}
                                       for i, c in enumerate(codes)]))  # fmt: skip


def _use(monkeypatch, backend):
    monkeypatch.setattr(search_mod, "make_backend", lambda cfg: backend)


# --------------------------------------------------------------------------- the switches are off by default


def test_defaults_leave_prompts_unchanged():
    n = PAPER_NICHES[2]
    args = ("edgegpu", "cifar10", n, [], [], [], ["mutation"], None)
    assert generation_prompt(*args) == generation_prompt(*args, history=None)
    assert "already evaluated in this niche" not in generation_prompt(*args)
    for key in ("show_history", "reprompt_retries", "refill_attempts", "budget_evaluations"):
        assert key not in BASE["search"]  # adding them to base.yaml would change every existing run hash


def test_default_run_has_no_retry_calls(synthetic_data_dir, tmp_path):
    res = run_search(_cfg(synthetic_data_dir), "cifar10", "edgegpu", 0, tmp_path)
    assert res["llm_usage"]["api_calls"] == 6 * 2 * 2
    assert "retry_calls" not in res["proposal_stats"]
    assert {c["stage"] for c in _calls(tmp_path)} == {"kb_update", "generate"}


# --------------------------------------------------------------------------- fix 1: history


def test_history_lists_niche_cells_and_is_capped():
    n = PAPER_NICHES[3]
    cells = n.members()[:60]
    p = generation_prompt("edgegpu", "cifar10", n, [], [], [], ["mutation"], None, history=cells[:3])
    assert "already evaluated in this niche (3); do not propose any of them" in p
    assert all(format_code(c) in p for c in cells[:3])
    p = generation_prompt("edgegpu", "cifar10", n, [], [], [], ["mutation"], None, history=cells)
    assert f"({HISTORY_CAP})" in p and format_code(cells[-1]) in p and format_code(cells[0]) not in p


def test_history_in_search_prompts(synthetic_data_dir, tmp_path):
    res = run_search(_cfg(synthetic_data_dir, show_history=True), "cifar10", "edgegpu", 0, tmp_path)
    gen = [c for c in _calls(tmp_path) if c["stage"] == "generate"]
    assert len(gen) == 12 and all("already evaluated in this niche" in c["user"] for c in gen)
    # the generation-2 prompt for S0 lists every S0 cell evaluated in generations 0-1
    s0 = [e["code"] for e in res["evaluated"] if e["niche"] == "S0" and e["gen"] <= 1]
    prompt = next(c["user"] for c in gen if c["generation"] == 2 and c["niche_name"] == "S0")
    assert f"in this niche ({len(s0)})" in prompt and all(code in prompt for code in s0)


# --------------------------------------------------------------------------- fix 2: re-prompting


def test_retry_reasons_and_no_padding(synthetic_data_dir, tmp_path, monkeypatch):
    """A model that copies a parent, breaks the niche, repeats itself and writes garbage gets told why, 5 times."""

    def reply(meta):
        par, niche = meta["parents"], meta["niche"]
        outside = (3, 3, 3, 3, 3, 3) if niche.contains((0,) * 6) else (0,) * 6
        return [par[0], outside, "(9, 9, 9, 9, 9, 9)", par[0]][: meta["n_child"]]

    _use(monkeypatch, Scripted(reply))
    res = run_search(_cfg(synthetic_data_dir, gens=1, reprompt_retries=5), "cifar10", "edgegpu", 0, tmp_path)
    st = res["proposal_stats"]
    assert st["accepted"] == 0 and res["n_evaluated"] == 30  # nothing padded with random cells
    assert st["retry_calls"] == 6 * 5 and st["slots_unfilled"] == 6 * 4
    calls = _calls(tmp_path)
    assert res["llm_usage"]["api_calls"] == 6 * (1 + 1 + 5)
    retry = [c for c in calls if c["stage"] == "generate_retry"]
    assert len(retry) == 30
    first = retry[0]["user"]
    assert "[Rejected proposals]" in first
    assert ": identical to parent 0" in first and "breaks niche rule: has " in first
    assert "(unreadable): not a valid code" in first
    assert first.split("[Rejected proposals]")[0].rstrip() == calls[1]["user"].rstrip()  # original prompt kept


def test_retry_asks_only_for_missing_children(synthetic_data_dir, tmp_path, monkeypatch):
    """First reply: one new child and three copies; the retry asks for 3 and those 3 are new."""
    asked = []

    def reply(meta):
        asked.append((meta["stage"], meta["n_child"]))
        par, niche = meta["parents"], meta["niche"]
        fresh = [c for c in niche.members() if c not in reply.used][: meta["n_child"]]
        reply.used.update(fresh)
        return fresh[:1] + [par[0]] * 3 if meta["stage"] == "generate" else fresh

    reply.used = set()
    _use(monkeypatch, Scripted(reply))
    cfg = _cfg(synthetic_data_dir, gens=1, reprompt_retries=5)
    res = run_search(cfg, "cifar10", "edgegpu", 0, tmp_path)
    # fresh cells may collide with the initial population, so allow a second retry, but never more children
    assert all(n <= 3 for s, n in asked if s == "generate_retry")
    assert res["proposal_stats"]["slots_unfilled"] == 0 and res["n_evaluated"] == 30 + 24
    assert ("generate_retry", 3) in asked


def test_arm_a_fills_budget_and_replays(synthetic_data_dir, tmp_path):
    cfg = _cfg(synthetic_data_dir, gens=10, **ARM_A)
    a = run_search(cfg, "cifar10", "edgegpu", 0, tmp_path)
    st = a["proposal_stats"]
    # the mock is a random operator that ignores the feedback, so a few slots may stay empty (never padded)
    assert a["n_evaluated"] == 270 - st["slots_unfilled"] and a["n_evaluated"] >= 260
    assert st["retry_calls"] > 0 and a["trajectory"][-1]["n_evaluated"] == a["n_evaluated"]
    assert a["llm_usage"]["api_calls"] == 6 * 10 * 2 + st["retry_calls"]
    assert {c.get("retry") for c in _calls(tmp_path) if c["stage"] == "generate_retry"} <= {1, 2, 3, 4, 5}
    b = run_search(deep_merge(cfg, {"llm": {"mode": "replay"}}), "cifar10", "edgegpu", 0, tmp_path)
    assert b["llm_usage"]["replay_mismatches"] == 0 and b["hv"] == a["hv"]
    assert b["proposal_stats"] == a["proposal_stats"]


def test_budget_cap_stops_early(synthetic_data_dir, tmp_path):
    res = run_search(_cfg(synthetic_data_dir, gens=10, budget_evaluations=50), "cifar10", "edgegpu", 0, tmp_path)
    assert res["n_evaluated"] == 50


# --------------------------------------------------------------------------- PEA with refill


def test_pea_refill_fills_budget(synthetic_data_dir, tmp_path):
    cfg = _cfg(synthetic_data_dir, gens=10, operator="random", refill_attempts=50, budget_evaluations=270)
    res = run_search(cfg, "cifar10", "edgegpu", 0, tmp_path)
    st = res["proposal_stats"]
    # a parent's in-niche mutation neighbourhood can be used up, so a slot may stay empty after 50 draws (no padding)
    assert res["n_evaluated"] == 270 - st["slots_unfilled"] and res["n_evaluated"] >= 255
    assert st["accepted"] == 240 - st["slots_unfilled"]
    assert st["proposed"] == 240 + st["refill_extra_draws"]  # one first draw per slot, plus redraws
    plain = run_search(_cfg(synthetic_data_dir, gens=10, operator="random"), "cifar10", "edgegpu", 0, tmp_path / "p")
    assert plain["n_evaluated"] < 270  # without refill PEA loses its duplicates, as in the archived runs


# --------------------------------------------------------------------------- reasons


def test_niche_violation_text():
    s2, s5 = PAPER_NICHES[2], PAPER_NICHES[5]
    assert s2.violation((3, 3, 0, 0, 0, 0)) == "has 2 x nor_conv_3x3, needs exactly 1"
    assert s5.violation((3, 0, 0, 0, 0, 0)) == "has 1 x nor_conv_3x3, needs at least 4"
    assert PAPER_NICHES[0].violation((2, 0, 0, 0, 0, 0)) == "has 1 x nor_conv_1x1, needs exactly 0"
    assert all(n.violation(c) is None for n in PAPER_NICHES for c in n.members()[:50])


def test_retry_prompt_format():
    p = retry_prompt("BASE", [("(0, 0, 0, 0, 0, 0)", "already evaluated")], ["crossover", "mutation"])
    assert p.startswith("BASE\n\n[Rejected proposals]")
    assert "(0, 0, 0, 0, 0, 0): already evaluated" in p and "Propose 2 replacement children" in p
    assert "child 0: crossover\n  child 1: mutation" in p


@pytest.mark.parametrize("name", ["dupfix_controls", "dupfix_arms"])
def test_experiment_files_load(name):
    from llmnas.experiments import config_hash, load_experiment, variant_config

    e = load_experiment(name)
    hashes = {v: config_hash(variant_config(e, v)) for v in e["variants"]}
    if name == "dupfix_controls":  # identical configs to the archived controls, so seeds 0-2 are reused
        assert (hashes["random_sampling"], hashes["nsga2"], hashes["regevo"]) == ("c18f639e", "a77208c7", "7830069e")
