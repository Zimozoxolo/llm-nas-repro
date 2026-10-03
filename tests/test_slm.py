"""SLM extension: search space, proxy/costs, reference models, metrics, offline end-to-end."""

import random

import numpy as np
import pytest
import yaml

from llmnas.experiments import REPO, deep_merge
from llmnas.slm.costs import DEVICES, Workload, evaluate_costs
from llmnas.slm.prompts import parse_slm_children
from llmnas.slm.reference import REFERENCE_SLMS, reference_spec
from llmnas.slm.report import hypervolume
from llmnas.slm.search import nondominated, run_slm_search
from llmnas.slm.space import (SpaceSpec, arch_key, crossover, decoder_params, mutate, niches_from_config,
                              random_arch, repair, total_params)

BASE = yaml.safe_load(open(REPO / "configs" / "slm" / "base.yaml"))
SPECS = {n: SpaceSpec.from_config({**BASE["spaces"][n]["spec"], "name": n}) for n in BASE["spaces"]}


@pytest.mark.parametrize("name", list(REFERENCE_SLMS))
def test_reference_param_counts_match_official(name):
    ref = REFERENCE_SLMS[name]
    tp = total_params(ref["arch"], reference_spec(ref, SPECS["slm"])) / 1e6
    assert abs(tp - ref["official_params_m"]) / ref["official_params_m"] < 0.01


@pytest.mark.parametrize("space", ["slm", "lts_gpt2"])
def test_random_archs_valid_and_operators_stay_valid(space):
    spec, rng = SPECS[space], random.Random(0)
    archs = [a for a in (random_arch(spec, rng) for _ in range(200)) if a]
    assert len(archs) > 150
    for a in archs[:50]:
        assert repair(a, spec) == (a, [])  # already canonical
        for child in (mutate(a, spec, rng), crossover(a, rng.choice(archs), spec, rng)):
            assert child is None or repair(child, spec)[0] == child
    if spec.heterogeneous:
        assert any(len({arch_key(x) for x in a["layers"]}) > 1 for a in archs)


def test_repair_snaps_llm_output():
    spec = SPECS["slm"]
    a, notes = repair({"n_layer": 21, "d_model": 1000, "layers": [{"d_inner": 3000, "n_head": 15, "n_kv_head": 3}]}, spec)
    assert a["d_model"] % 128 == 0 and a["d_model"] % a["layers"][0]["n_head"] == 0 and notes
    assert repair({"d_model": 512}, spec)[0] is None


def test_niches_partition_by_decoder_params():
    niches = niches_from_config(BASE["spaces"]["slm"]["niches"])
    rng = random.Random(1)
    for _ in range(300):
        a = random_arch(SPECS["slm"], rng)
        if a:
            assert sum(n.contains(a) for n in niches) == 1


def test_costs_monotone_in_size():
    spec, dev, wl = SPECS["slm"], DEVICES["jetson_orin_nx_16gb"], Workload()
    small = evaluate_costs(REFERENCE_SLMS["SmolLM-360M"]["arch"], spec, dev, wl)
    big = evaluate_costs(REFERENCE_SLMS["Qwen2.5-3B"]["arch"], spec, dev, wl)
    assert small["est_loss"] > big["est_loss"]
    assert small["latency_ms"] < big["latency_ms"] and small["memory_mb"] < big["memory_mb"]


def test_gqa_reduces_kv_cache():
    spec, dev, wl = SPECS["slm"], DEVICES["jetson_orin_nx_16gb"], Workload()
    a = {"n_layer": 24, "d_model": 2048, "ffn": "gated", "layers": [{"d_inner": 8192, "n_head": 16, "n_kv_head": "n_head"}]}
    g = deep_merge(a, {})
    g["layers"] = [{"d_inner": 8192, "n_head": 16, "n_kv_head": 2}]
    assert evaluate_costs(g, spec, dev, wl)["kv_cache_mb"] * 8 == pytest.approx(evaluate_costs(a, spec, dev, wl)["kv_cache_mb"])
    assert decoder_params(g) < decoder_params(a)


def test_hypervolume_3d_known_value():
    assert hypervolume(np.array([[0.0, 0.0, 0.0]]), np.ones(3)) == pytest.approx(1.0)
    pts = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.0]])
    assert hypervolume(pts, np.ones(3)) == pytest.approx(0.25 + 0.5 - 0.125)


def test_nondominated_3d():
    pts = np.array([[1, 1, 1], [0, 2, 2], [2, 2, 2], [1, 1, 1], [2, 0, 3]])
    assert nondominated(pts).tolist() == [True, True, False, False, True]


def test_parse_slm_children():
    txt = '[{"child_id":"0","operation":"mutation","architecture":{"n_layer":12,"d_model":768,"ffn":"gated",' \
          '"layers":[{"d_inner":2048,"n_head":12,"n_kv_head":4}]},"rationale":"x"}]'
    out = parse_slm_children(txt)
    assert out[0]["arch"]["d_model"] == 768 and out[0]["operation"] == "mutation"


def _cfg(**over):
    cfg = deep_merge(BASE, {"search": {"generations": 2}, "llm": {"backend": "mock"}, "variant": "t"})
    return deep_merge(cfg, over)


@pytest.mark.parametrize("over", [{}, {"search": {"partitioning": False, "operator": "random"}},
                                  {"search": {"operator": "random_sample"}}, {"search": {"kb_seed": "survey"}}])
def test_slm_search_end_to_end(tmp_path, over):
    res = run_slm_search(_cfg(**over), "slm", "jetson_orin_nx_16gb", 0, tmp_path)
    assert res["n_front"] > 0 and res["n_evaluated"] > 20
    if over.get("search", {}).get("operator", "llm") == "llm":
        assert res["llm_usage"]["api_calls"] == 6 * 2 * 2


def test_slm_replay(tmp_path):
    a = run_slm_search(_cfg(), "lts_gpt2", "snapdragon_8gen1_cpu", 3, tmp_path)
    b = run_slm_search(_cfg(llm={"mode": "replay"}), "lts_gpt2", "snapdragon_8gen1_cpu", 3, tmp_path)
    assert a["final_front"] == b["final_front"] and b["llm_usage"]["replay_mismatches"] == 0


def test_measurement_tiny_model_cpu():
    torch = pytest.importorskip("torch")  # noqa: F841
    from llmnas.slm.measure import measure

    spec = SPECS["lts_gpt2"]
    a = {"n_layer": 2, "d_model": 128, "ffn": "standard", "layers": [{"d_inner": 256, "n_head": 2, "n_kv_head": "n_head"}] * 2}
    m = measure(a, spec, Workload(prompt_tokens=8, gen_tokens=4), "cpu", repeats=1)
    assert m["ok"] and m["prefill_ms"] > 0 and m["decode_ms_per_token"] > 0
