"""End-to-end pipeline smoke: envelope → extract + synthesize → split → format.

Complete 5 blind spots in per-module single testing: **If the upstream changes the schema and forgets to change the schema, the downstream will not fail**.
Each single test uses typed fixture / inline YAML, inter-module contract (Triple dict ↔ format_triple
(Enter ↔ Scenario meta) is only jointly adjusted in this run-through.

Select `runs_1k_fast_7b_r0_124/tool_chain-r0.json` as fixture:
  - git tracked, not dirty working dir
  - synthesize path produces 2 triples (extractor in fast scenario `max_retries=0`
    Yong 0; e2e goes synthesize completely cover Triple schema flow)
  - 6.7 KB, full Python pipeline ~50 ms

No dependency on Ollama/Network/LLM; sys.path injection is handled by [`conftest.py`](conftest.py)."""

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
    """A single envelope completes 4 steps: extract / synthesize / split / format, and the output schema is complete."""
    # 1. extract + synthesize → Triple dicts
    extracted = extract_triples(envelope, SCENARIO, run_id=0, scenario_name="tool_chain")
    synthesized = envelope_to_synthetic_triples(
        envelope, SCENARIO, run_id=0, scenario_name="tool_chain"
    )
    triples = [dataclasses.asdict(t) for t in extracted + synthesized]
    assert triples, "smoke fixture envelope should produce ≥1 triple"

# 2. split (single run_id < threshold → fallback full train, covering small-batch path)
# 2. split (single run_id < threshold → fallback full train, covering small-batch path)
    train, val = split_train_val(triples, val_ratio=0.2)
    assert len(train) + len(val) == len(triples)
    assert val == []  # Only 1 unique run_id → fallback
    assert train == triples

    # 3. format Triple → MLX-LM chat sample
    samples = [format_triple(t, SCENARIO) for t in train]
    samples = [s for s in samples if s is not None]
    assert samples, "format step dropped all triples — drop rule regression?"

# 4. assert SFT schema complete
# 4. assert SFT schema complete
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
    args = tc["function"]["arguments"]  #v1.5+ is dict (Qwen3.5 chat_template strict items)
    assert isinstance(args, dict)

    assert s["tools"], "tools list must not be empty (agent visibility)"
    tool_names = {t["function"]["name"] for t in s["tools"]}
    assert tc["function"]["name"] in tool_names, (
        "tool_call.function.name must appear in tools[] — 防 schema 漂移"
    )


def test_pipeline_split_isolates_by_run_id(envelope):
    """Forge 5 run_ids to pass the split threshold, and verify train/val disjoint according to run_id."""
    base = envelope_to_synthetic_triples(
        envelope, SCENARIO, run_id=0, scenario_name="tool_chain"
    )
    assert base, "fixture should yield ≥1 synthesized triple"

# Copy 5 copies of different run_id, leaving scenario / turn_idx unchanged
# Copy 5 copies of different run_id, leaving scenario / turn_idx unchanged
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
# Last 20% (= floor(5*0.2)=1) → 1 run_id into val
# Last 20% (= floor(5*0.2)=1) → 1 run_id into val
    assert len(val_rids) == 1
    assert max(train_rids) < min(val_rids), "val 应取末位 run_id"
