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


def test_anthropic_backend_extracts_text_and_handles_refusal(monkeypatch):
    """Offline check of the Claude backend: skips thinking blocks, sends no sampling params, handles refusals."""
    from types import SimpleNamespace as NS

    from llmnas.llm_anthropic import AnthropicBackend

    calls = []

    class FakeMessages:
        def __init__(self, resp):
            self.resp = resp

        def create(self, **kw):
            calls.append(kw)
            return self.resp

    usage = NS(input_tokens=11, output_tokens=7)
    ok = NS(stop_reason="end_turn", usage=usage, content=[NS(type="thinking", thinking=""), NS(type="text", text="[1]"),
                                                         NS(type="text", text="[2]")])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    b = AnthropicBackend(model="claude-opus-5-5", max_tokens=16000, effort="high")
    b.client = NS(messages=FakeMessages(ok))
    r = b.complete("sys", "user", {})
    assert r.text == "[1][2]" and (r.input_tokens, r.output_tokens) == (11, 7)
    kw = calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["system"] == "sys" and kw["output_config"] == {"effort": "high"}
    assert "temperature" not in kw and kw["max_tokens"] == 16000
    b.client = NS(messages=FakeMessages(NS(stop_reason="refusal", usage=usage, content=[])))
    assert b.complete("s", "u", {}).text == ""


def test_anthropic_backend_requires_key(monkeypatch, tmp_path):
    import pytest as _pytest

    from llmnas.llm_anthropic import AnthropicBackend

    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    with _pytest.raises(SystemExit):
        AnthropicBackend()
