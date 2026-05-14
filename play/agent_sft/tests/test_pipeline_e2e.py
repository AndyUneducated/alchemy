"""End-to-end pipeline smoke：envelope → extract + synthesize → split → format.

补全 5 个 per-module 单测的盲区：**上游改 schema 忘改下游不会挂**。
单测各自用 typed fixture / inline YAML，模块间契约（Triple dict ↔ format_triple
入参 ↔ Scenario meta）只在这一条贯通跑里被联调.

挑 `runs_1k_fast_7b_r0_124/tool_chain-r0.json` 作 fixture：
  - 已 git tracked，不脏 working dir
  - synthesize 路径产 2 triples（extractor 在 fast scenario `max_retries=0` 下
    永 0；e2e 走 synthesize 完全 cover Triple schema 流）
  - 6.7 KB，全 Python 流水线 ~50 ms

不依赖 Ollama / 网络 / LLM；sys.path 注入由 [`conftest.py`](conftest.py) 处理.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from extractor import extract_triples  # type: ignore[import-not-found]
from formatter import format_triple  # type: ignore[import-not-found]
from split import split_train_val  # type: ignore[import-not-found]
from synthesize import (  # type: ignore[import-not-found]
    envelope_to_synthetic_triples,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
AGENT_SFT = REPO_ROOT / "play" / "agent_sft"
ENVELOPE = (
    AGENT_SFT / "data" / "triples" / "runs_1k_fast_7b_r0_124" / "tool_chain-r0.json"
)
SCENARIO = AGENT_SFT / "data" / "scenarios" / "tool_chain_fast.md"


@pytest.fixture(scope="module")
def envelope() -> dict:
    if not ENVELOPE.exists():
        pytest.skip(f"fixture envelope not found: {ENVELOPE}")
    if not SCENARIO.exists():
        pytest.skip(f"fixture scenario not found: {SCENARIO}")
    return json.loads(ENVELOPE.read_text(encoding="utf-8"))


def test_pipeline_envelope_to_chat_samples(envelope):
    """单 envelope 走完 4 步：extract / synthesize / split / format，输出 schema 完整."""
    # 1. extract + synthesize → Triple dicts
    extracted = extract_triples(envelope, SCENARIO, run_id=0, scenario_name="tool_chain")
    synthesized = envelope_to_synthetic_triples(
        envelope, SCENARIO, run_id=0, scenario_name="tool_chain"
    )
    triples = [dataclasses.asdict(t) for t in extracted + synthesized]
    assert triples, "smoke fixture envelope should produce ≥1 triple"

    # 2. split（单 run_id < 阈值 → fallback 全 train，覆盖 small-batch 路径）
    train, val = split_train_val(triples, val_ratio=0.2)
    assert len(train) + len(val) == len(triples)
    assert val == []  # 仅 1 unique run_id → fallback
    assert train == triples

    # 3. format Triple → MLX-LM chat sample
    samples = [format_triple(t, SCENARIO) for t in train]
    samples = [s for s in samples if s is not None]
    assert samples, "format step dropped all triples — drop rule regression?"

    # 4. assert SFT schema 完整
    s = samples[0]
    assert set(s.keys()) == {"messages", "tools"}, f"unexpected top keys: {s.keys()}"
    roles = [m["role"] for m in s["messages"]]
    assert roles == ["system", "user", "assistant"], f"unexpected role order: {roles}"

    asst = s["messages"][-1]
    assert asst["content"] == ""
    assert len(asst["tool_calls"]) == 1
    tc = asst["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["function"]["name"]  # required_tool not empty
    args = json.loads(tc["function"]["arguments"])  # 必须是 valid JSON
    assert isinstance(args, dict)

    assert s["tools"], "tools list must not be empty (agent visibility)"
    tool_names = {t["function"]["name"] for t in s["tools"]}
    assert tc["function"]["name"] in tool_names, (
        "tool_call.function.name must appear in tools[] — 防 schema 漂移"
    )


def test_pipeline_split_isolates_by_run_id(envelope):
    """伪造 5 个 run_id 走过 split 阈值，验证 train/val 按 run_id disjoint."""
    base = envelope_to_synthetic_triples(
        envelope, SCENARIO, run_id=0, scenario_name="tool_chain"
    )
    assert base, "fixture should yield ≥1 synthesized triple"

    # 复制 5 份不同 run_id，保留 scenario / turn_idx 不变
    triples: list[dict] = []
    for rid in range(5):
        for t in base:
            d = dataclasses.asdict(t)
            d["run_id"] = rid
            triples.append(d)

    train, val = split_train_val(triples, val_ratio=0.2)
    assert train and val, "5 run_ids 应触发真切分而非 fallback"

    train_rids = {s["run_id"] for s in train}
    val_rids = {s["run_id"] for s in val}
    assert train_rids.isdisjoint(val_rids), (
        f"train/val run_id leak: train={train_rids} val={val_rids}"
    )
    # 末 20% (= floor(5*0.2)=1) → 1 个 run_id 进 val
    assert len(val_rids) == 1
    assert max(train_rids) < min(val_rids), "val 应取末位 run_id"
