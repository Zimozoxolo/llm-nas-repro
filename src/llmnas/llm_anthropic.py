"""Optional Claude backend (``llm.backend: anthropic``).

Notes for current Claude models (Opus 5.5 / Sonnet 5.5 / Fable 5.1):
* sampling parameters (``temperature`` etc.) are not accepted, so none are sent;
  run-to-run variation is handled by repeating seeds, as with any LLM backend.
* thinking is adaptive by default and its tokens count towards ``max_tokens`` - keep that
  at 16000 or more (the experiment file does) or visible answers can be truncated;
  ``effort`` (low..max) is the depth control.
* no server-side model fallback is configured on purpose: a NAS run must use one
  fixed model to be reproducible. A refusal is logged as an empty answer, which the
  search treats as an invalid proposal.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import anthropic

from .llm import LLMBackend, LLMResponse


class AnthropicBackend(LLMBackend):
    def __init__(self, model: str = "claude-opus-5-5", max_tokens: int = 16000, effort: str | None = None):
        have_profile = os.environ.get("ANTHROPIC_PROFILE") or (Path.home() / ".config" / "anthropic").exists()
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") or have_profile):
            raise SystemExit("ANTHROPIC_API_KEY is not set. Put `ANTHROPIC_API_KEY=sk-ant-...` in llm-nas-repro/.env "
                             "(saved, no quotes) or set it as an environment variable.")
        self.client = anthropic.Anthropic()
        self.model, self.max_tokens, self.effort = model, max_tokens, effort

    def complete(self, system: str, user: str, meta: dict[str, Any]) -> LLMResponse:
        kwargs: dict[str, Any] = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                                      messages=[{"role": "user", "content": user}])  # fmt: skip
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        r = self.client.messages.create(**kwargs)
        if r.stop_reason == "refusal":
            return LLMResponse("", r.usage.input_tokens, r.usage.output_tokens)
        text = "".join(b.text for b in r.content if b.type == "text")
        return LLMResponse(text, r.usage.input_tokens, r.usage.output_tokens)
