"""Ollama 适配器：stdlib only / /api/generate 直拨 prompt（不走 chat template）.

为什么 /api/generate 不是 /api/chat（与 [`play/agent_engine/ollama_client.py`](play/agent_engine/ollama_client.py) 反向）：
  lm-eval 哲学要求 task 完全拥有 prompt 字面字符串。/api/chat 会按模型 chat template 包裹
  user/assistant role + system prompt，破坏 prompt 字面可复现；/api/generate 直拨 raw prompt.

只实现 generate_until（phase 3 唯一用到的 request type）；loglikelihood 走 ABC default
抛 NotImplementedError——phase 9 calibration 再开（届时 ollama 端用 /api/embeddings 或
HF transformers tokenizer 直算）.

为什么不复用 play/agent_engine/ollama_client.py：
  play/ 子项目互不 import（grep 验证零交叉），保持 evals 自洽。这里 stdlib /api/generate
  封装本身极薄（< 60 行），重复实现成本远低于跨子项目耦合.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import ClassVar

from ..api import Request, Response, Usage
from .base import LM


class OllamaLM(LM):
    """`OllamaLM(model="qwen2.5:32b")` → 走本地 ollama HTTP，名字落 `ollama:<model>`.

    `base_url` 优先级：构造参数 > env `EVALS_OLLAMA_BASE_URL` > 默认 `localhost:11434`.
    `temperature=0.0` + `seed=0` 默认让测试更确定（ollama options.seed 透传）.
    """

    DEFAULT_BASE_URL: ClassVar[str] = "http://localhost:11434"

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        temperature: float = 0.0,
        seed: int | None = 0,
        request_timeout: float = 120.0,
    ) -> None:
        self.model = model
        env_url = os.environ.get("EVALS_OLLAMA_BASE_URL")
        self.base_url = (base_url or env_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.temperature = temperature
        self.seed = seed
        self.request_timeout = request_timeout
        self.name = f"ollama:{model}"

    def generate_until(self, requests: list[Request]) -> list[Response]:
        """串行调用 /api/generate；phase 1+ 并发优化在 runner 层做（统一对所有 LM）.

        phase 6 起填 Response.usage / latency_ms：
          - `prompt_eval_count` → Usage.tokens_in（缺字段 → None）
          - `eval_count` → Usage.tokens_out（缺字段 → None）
          - `total_duration`（ns）→ latency_ms（ns / 1e6）；ollama 报的端到端时间，
            比 perf_counter 更准（不含 Python 调用栈 / urllib socket 排队）.
        老版本 ollama 服务可能不返回这些字段——getattr 风格 .get(...) 返 None，
        与 efficiency_aggregated"非 None 收集"协议天然兼容.
        """
        out: list[Response] = []
        for req in requests:
            options: dict = {
                "temperature": self.temperature,
                "num_predict": req.max_tokens,
            }
            if self.seed is not None:
                options["seed"] = self.seed
            if req.until:
                options["stop"] = list(req.until)

            body = {
                "model": self.model,
                "prompt": req.prompt,
                "stream": False,
                "options": options,
            }
            payload = json.dumps(body).encode("utf-8")
            http_req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(http_req, timeout=self.request_timeout) as resp:
                data = json.loads(resp.read())
            text = data.get("response", "") or ""
            tokens_in = data.get("prompt_eval_count")
            tokens_out = data.get("eval_count")
            usage: Usage | None = None
            if tokens_in is not None or tokens_out is not None:
                usage = Usage(tokens_in=tokens_in, tokens_out=tokens_out)
            total_duration_ns = data.get("total_duration")
            latency_ms: float | None = None
            if total_duration_ns is not None:
                latency_ms = float(total_duration_ns) / 1_000_000.0
            out.append(
                Response(
                    doc_id=req.doc_id,
                    text=text,
                    latency_ms=latency_ms,
                    usage=usage,
                )
            )
        return out
