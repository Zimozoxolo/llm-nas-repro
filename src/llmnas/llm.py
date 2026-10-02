"""LLM backends + a call log that makes every run replayable.

Backends
--------
* ``openai``    - any OpenAI-compatible Chat Completions endpoint. Default model is
                  ``gpt-4.1`` (the paper's engine). Set ``base_url`` to use OpenRouter,
                  Ollama (``http://localhost:11434/v1``), vLLM, Gemini's OpenAI endpoint...
* ``anthropic`` - Claude via the official SDK (see ``llm_anthropic.py``).
* ``mock``      - offline stand-in that answers with random valid mutations/crossovers.
                  It is NOT an LLM; use it only for smoke tests and CI.

Replay
------
Every call is appended to ``<run_dir>/llm_calls.jsonl`` (prompt, response, token usage,
wall time). Running again with ``llm.mode: replay`` feeds the logged responses back in
order, so a published run can be re-executed bit-for-bit without API access or cost.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .search_space import NUM_EDGES, OPS, format_code, random_crossover, random_mutation


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMBackend:
    model: str = "unknown"

    def complete(self, system: str, user: str, meta: dict[str, Any]) -> LLMResponse:
        raise NotImplementedError


class OpenAIBackend(LLMBackend):
    def __init__(self, model: str = "gpt-4.1", temperature: float = 0.7, base_url: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY", max_tokens: int = 4096, seed: int | None = None):  # fmt: skip
        from openai import OpenAI

        key = os.environ.get(api_key_env) or ("ollama" if base_url and "localhost" in base_url else None)
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model, self.temperature, self.max_tokens, self.seed = model, temperature, max_tokens, seed

    def complete(self, system: str, user: str, meta: dict[str, Any]) -> LLMResponse:
        kwargs = dict(model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
                      messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])  # fmt: skip
        if self.seed is not None:
            kwargs["seed"] = self.seed
        r = self.client.chat.completions.create(**kwargs)
        u = r.usage
        return LLMResponse(r.choices[0].message.content or "", getattr(u, "prompt_tokens", 0),
                           getattr(u, "completion_tokens", 0))  # fmt: skip


class MockBackend(LLMBackend):
    """Random operator dressed up as an LLM response. For tests only."""

    model = "mock"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def complete(self, system: str, user: str, meta: dict[str, Any]) -> LLMResponse:
        if meta.get("stage") == "kb_update":
            rules = ["Use skip_connect next to heavy convs because it is cheap.",
                     "Avoid many avg_pool_3x3 because they add latency for little accuracy."]  # fmt: skip
            return LLMResponse(json.dumps({"Updated_Knowledge_Base": rules}))
        niche, parents = meta.get("niche"), meta.get("parents") or []
        children = []
        for i, op in enumerate(meta.get("operations", ["mutation"] * meta.get("n_child", 1))):
            child = None
            if niche is not None and parents:
                if op == "crossover" and len(parents) >= 2:
                    a, b = self.rng.sample(parents, 2)
                    child = random_crossover(a, b, niche, self.rng)
                if child is None:
                    child = random_mutation(self.rng.choice(parents), niche, self.rng)
            if child is None:
                child = tuple(self.rng.randrange(len(OPS)) for _ in range(NUM_EDGES))
            children.append({"child_id": f"c{i}", "operation": op, "architecture_code": format_code(child),
                             "rationale": "mock: random operator"})  # fmt: skip
        return LLMResponse(json.dumps(children))


def make_backend(cfg: dict) -> LLMBackend:
    kind = cfg.get("backend", "openai")
    if kind == "openai":
        return OpenAIBackend(model=cfg.get("model", "gpt-4.1"), temperature=cfg.get("temperature", 0.7),
                             base_url=cfg.get("base_url"), api_key_env=cfg.get("api_key_env", "OPENAI_API_KEY"),
                             max_tokens=cfg.get("max_tokens", 4096), seed=cfg.get("seed"))  # fmt: skip
    if kind == "anthropic":
        from .llm_anthropic import AnthropicBackend

        return AnthropicBackend(model=cfg.get("model", "claude-opus-5-5"), max_tokens=cfg.get("max_tokens", 16000),
                                effort=cfg.get("effort"))  # fmt: skip
    if kind == "mock":
        return MockBackend(seed=cfg.get("seed", 0) or 0)
    raise ValueError(f"unknown llm backend {kind!r}")


# --------------------------------------------------------------------------- logging / replay


def _hash(system: str, user: str) -> str:
    return hashlib.sha256((system + "\x00" + user).encode()).hexdigest()[:16]


@dataclass
class LLMSession:
    """Wraps a backend; records or replays calls and tallies cost."""

    backend: LLMBackend | None
    log_path: Path
    mode: str = "record"  # record | replay
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_time_s: float = 0.0
    replay_mismatches: int = 0
    _replay: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.log_path = Path(self.log_path)
        if self.mode == "replay":
            with open(self.log_path) as f:
                self._replay = [json.loads(line) for line in f]
        else:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("")

    def __call__(self, system: str, user: str, meta: dict[str, Any]) -> str:
        idx = self.calls
        self.calls += 1
        h = _hash(system, user)
        if self.mode == "replay":
            if idx >= len(self._replay):
                raise RuntimeError(f"replay log {self.log_path} has only {len(self._replay)} calls")
            rec = self._replay[idx]
            if rec["prompt_hash"] != h:
                self.replay_mismatches += 1
            self.input_tokens += rec.get("input_tokens", 0)
            self.output_tokens += rec.get("output_tokens", 0)
            self.wall_time_s += rec.get("wall_time_s", 0.0)
            return rec["response"]

        t0 = time.perf_counter()
        resp = self._call_with_retry(system, user, meta)
        dt = time.perf_counter() - t0
        self.input_tokens += resp.input_tokens
        self.output_tokens += resp.output_tokens
        self.wall_time_s += dt
        public_meta = {k: v for k, v in meta.items() if k in ("stage", "generation", "niche_name")}
        rec = dict(call=idx, model=self.backend.model, prompt_hash=h, **public_meta, system=system, user=user,
                   response=resp.text, input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                   wall_time_s=round(dt, 3))  # fmt: skip
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return resp.text

    def _call_with_retry(self, system: str, user: str, meta: dict, tries: int = 5) -> LLMResponse:
        for attempt in range(tries):
            try:
                return self.backend.complete(system, user, meta)
            except Exception as e:  # network / rate-limit; SDKs already retry transient errors
                if attempt == tries - 1:
                    raise
                wait = 2**attempt * 5
                print(f"  LLM call failed ({type(e).__name__}: {e}); retrying in {wait}s")
                time.sleep(wait)
        raise AssertionError("unreachable")

    def usage(self) -> dict:
        return dict(api_calls=self.calls, input_tokens=self.input_tokens, output_tokens=self.output_tokens,
                    llm_wall_time_s=round(self.wall_time_s, 2), replay_mismatches=self.replay_mismatches)  # fmt: skip
