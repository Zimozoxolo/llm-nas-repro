"""End-to-end runs on synthetic data with the mock backend (no network)."""

import copy
import json

import yaml

from llmnas.experiments import REPO, config_hash, deep_merge
from llmnas.search import run_search

BASE = yaml.safe_load(open(REPO / "configs" / "base.yaml"))


def _cfg(data_dir, **over):
    cfg = deep_merge(BASE, {"data_dir": str(data_dir), "search": {"generations": 2}, "llm": {"backend": "mock"},
                            "predictor": {"n_train": 300}})
    return deep_merge(cfg, over)


def test_llmnas_mock_run(synthetic_data_dir, tmp_path):
    res = run_search(_cfg(synthetic_data_dir), "cifar10", "edgegpu", 0, tmp_path)
    assert 0 < res["hv"] <= 1.0 + 1e-9
    assert res["llm_usage"]["api_calls"] == 6 * 2 * 2  # niches x generations x (KB + generate)
    assert all(f["niche"] for f in res["final_front"])
    assert (tmp_path / "llm_calls.jsonl").exists() and (tmp_path / "run.json").exists()


def test_replay_reproduces_run(synthetic_data_dir, tmp_path):
    cfg = _cfg(synthetic_data_dir)
    a = run_search(cfg, "cifar10", "edgegpu", 1, tmp_path)
    b = run_search(deep_merge(cfg, {"llm": {"mode": "replay"}}), "cifar10", "edgegpu", 1, tmp_path)
    assert a["hv"] == b["hv"] and a["final_front"] == b["final_front"]
    assert b["llm_usage"]["replay_mismatches"] == 0
    assert json.load(open(tmp_path / "run_replay.json"))["hv"] == a["hv"]


def test_ablation_variants(synthetic_data_dir, tmp_path):
    for i, over in enumerate([{"search": {"partitioning": False}}, {"search": {"operator": "random"}},
                              {"predictor": {"kind": "synflow"}}, {"search": {"operator": "random_sample"}}]):
        res = run_search(_cfg(synthetic_data_dir, **over), "cifar100", "fpga", 0, tmp_path / str(i))
        assert res["n_evaluated"] > 0


def test_config_hash_ignores_mode():
    c = copy.deepcopy(BASE)
    c["variant"] = "x"
    d = deep_merge(c, {"llm": {"mode": "replay"}})
    assert config_hash(c) == config_hash(d)
    assert config_hash(c) != config_hash(deep_merge(c, {"search": {"n_child": 9}}))
