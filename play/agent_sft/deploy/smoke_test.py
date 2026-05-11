"""Phase 4 smoke test — verify `agent-sft-qwen` tag emits `<tool_call>` blocks.

不走 agent_engine（agent_engine 1-round 烟测在 Phase 4 验收清单作为加项另跑），
只直接调 Ollama HTTP API 验证：

  1. tag 注册成功（GET /api/tags 含 agent-sft-qwen）
  2. 给一个明显需要工具调用的 prompt + tools schema，输出
     parsed `tool_calls`（Ollama 解析 chat template 渲染的 <tool_call> 块）
     或原始 content 含 `<tool_call>` 字面块。

Phase 4 验收口径：bytes-to-bytes 跑通；效果数字留 Phase 5。

用法：
    python smoke_test.py
    python smoke_test.py --tag agent-sft-qwen --host http://localhost:11434
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_TAG = "agent-sft-qwen"
DEFAULT_HOST = "http://localhost:11434"

RETRIEVE_DOCS_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve_docs",
        "description": "Retrieve historical documents matching a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return.",
                },
            },
            "required": ["query"],
        },
    },
}


def http_get(url: str, timeout: int = 10) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post(url: str, payload: dict, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def assert_tag_exists(host: str, tag: str) -> None:
    data = http_get(f"{host}/api/tags")
    names = [m["name"] for m in data.get("models", [])]
    if not any(n == tag or n.startswith(f"{tag}:") for n in names):
        sys.exit(f"[smoke] FAIL: tag '{tag}' not found in ollama. Listed: {names}")
    print(f"[smoke] OK  tag '{tag}' is registered.")


def has_tool_call(resp: dict) -> tuple[bool, str]:
    """Return (ok, reason). Looks at both parsed tool_calls and raw content."""
    msg = resp.get("message") or {}
    if msg.get("tool_calls"):
        return True, f"parsed tool_calls field: {msg['tool_calls']}"
    content = msg.get("content") or ""
    if "<tool_call>" in content:
        return True, f"raw <tool_call> in content (Ollama parser miss; manual extract works)"
    return False, f"no tool_calls; content={content[:200]!r}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--tag", default=DEFAULT_TAG)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--print-response", action="store_true",
                   help="dump full Ollama /api/chat response on success")
    args = p.parse_args(argv)

    print(f"[smoke] host={args.host}  tag={args.tag}")

    try:
        assert_tag_exists(args.host, args.tag)
    except urllib.error.URLError as e:
        sys.exit(f"[smoke] FAIL: cannot reach ollama at {args.host}: {e}")

    payload = {
        "model": args.tag,
        "messages": [
            {
                "role": "user",
                "content": (
                    "请用 retrieve_docs 工具查询主题为 '项目代号 X 历史 commit' 的"
                    "文档，返回前 5 条。请直接发起工具调用。"
                ),
            }
        ],
        "tools": [RETRIEVE_DOCS_TOOL],
        "stream": False,
        "options": {"temperature": 0.0, "seed": 0},
    }

    print("[smoke] POST /api/chat ...")
    try:
        resp = http_post(f"{args.host}/api/chat", payload)
    except urllib.error.URLError as e:
        sys.exit(f"[smoke] FAIL: /api/chat error: {e}")

    ok, reason = has_tool_call(resp)
    if args.print_response:
        print("[smoke] full response:")
        print(json.dumps(resp, ensure_ascii=False, indent=2))

    if not ok:
        print(f"[smoke] FAIL: {reason}", file=sys.stderr)
        return 1

    print(f"[smoke] OK  tool_call detected — {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
