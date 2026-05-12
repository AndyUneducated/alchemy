"""Phase 1 baseline 主指标：require_tool 服从性.

测 `play/agent_engine/scenarios/*.md` 上模型在 require_tool step 的"第一次到位率"
（= 1 - nudge_fire_rate）. 与 `agent_traj` 同走 envelope subprocess 模式 + 同期望
gold.jsonl schema（`scenario_path` 是唯一必填 metadata），仅度量函数不同：

  | task               | 度量轴        | 信号 |
  |---|---|---|
  | agent_traj         | trajectory 整体（task_success / tool F1 / coverage / ...） | 终态对错 |
  | **nudge_fire_rate** | require_tool step 的第一次响应行为                          | 过程服从性 |

设计要点：
  - **output_type='none'**：与 agent_traj 同；runner 跳 LM 调用；agent_engine
    subprocess 跑全链路 LLM.
  - **expected_require_tool_turns 自动派生**：从 scenario YAML frontmatter 解析，
    避免 gold.jsonl 手维护与 scenario 漂移. 见 [`metrics/nudge.derive_expected_turns`].
  - **失败模式 taxonomy**（DECISIONS Phase 1 ADR §6）：missed / wrong_tool / wrong_args
    三桶；wrong_args 当前 deferred（artifact handler 在 error 路径不发 event）.
  - **聚合**：top-level 按 require_tool_total 加权平均，by_scenario / by_tool /
    by_failure_mode 三个 breakdown 字典写入 aggregated.

向后兼容：`run_fn=None` 默认构造可用于 score 路径（trajectory 从 predictions 读）.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, ClassVar

from ..api import Doc, Response, SampleResult
from ..metrics.nudge import (
    FAILURE_MODES,
    compute_nudge_fire_rate,
    derive_expected_turns,
)
from ..registry import register_task
from .base import Task

DATA_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "nudge_fire_rate" / "gold.jsonl"
)

# scenarios 路径解析根（与 agent_engine_run.make_run_fn 同源）
PLAY_DIR = Path(__file__).resolve().parents[2]

RunFn = Callable[[str], dict[str, Any]]


@register_task("nudge_fire_rate")
class NudgeFireRate(Task):
    """Agent require_tool 服从性 task.

    构造：
      - `run_fn=None`     → 仅 score 路径可用（envelope 从 predictions JSONL 读）
      - `run_fn=callable` → run 路径 process_docs hook 自动 subprocess 跑 agent_engine
    """

    name: ClassVar[str] = "nudge_fire_rate"
    output_type: ClassVar[str] = "none"

    def __init__(self, run_fn: RunFn | None = None) -> None:
        self.data_path = DATA_PATH
        self._run_fn = run_fn

    # ---- ABC implementations -------------------------------------------------

    def docs(self) -> Iterable[Doc]:
        with self.data_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                yield Doc(
                    id=row["id"],
                    input=row.get("input", ""),
                    target=row.get("target"),
                    metadata=dict(row.get("metadata", {})),
                )

    def doc_to_text(self, doc: Doc) -> str:
        return ""

    def doc_to_target(self, doc: Doc) -> str:
        return doc.target or ""

    def process_docs(self, docs: list[Doc]) -> list[Doc]:
        """run 路径：subprocess 跑 envelope，再派生 expected_turns，pin 到 metadata."""
        if self._run_fn is None:
            return docs
        out: list[Doc] = []
        for d in docs:
            scenario_path = d.metadata.get("scenario_path")
            if not scenario_path:
                raise ValueError(
                    f"nudge_fire_rate doc id={d.id!r} missing 'scenario_path' in metadata"
                )
            envelope = self._run_fn(scenario_path)
            out.append(_pin_envelope(d, envelope))
        return out

    def load_prediction(self, doc: Doc, row: dict) -> tuple[Doc, Response]:
        """score 路径：predictions JSONL row 内的 envelope → metadata['trajectory']
        + 派生 expected_turns；Response 占位.

        Predictions JSONL 由 run 路径写出，含 §16 envelope 全 5 字段（含 typed
        transcript entry + usage list）.
        """
        envelope = {
            "transcript": row["transcript"],
            "artifact": row["artifact"],
            "warnings": row["warnings"],
            "success": row["success"],
            "usage": row["usage"],
        }
        enriched = _pin_envelope(doc, envelope)
        return enriched, Response(doc_id=doc.id)

    def process_results(self, doc: Doc, response: Response) -> SampleResult:
        traj = doc.metadata.get("trajectory", {}) or {}
        expected = doc.metadata.get("expected_require_tool_turns") or []

        result = compute_nudge_fire_rate(traj, expected)

        # SampleResult.metrics 仅放标量（一级嵌套约束 + 直观聚合）；breakdown 详情
        # 进 artifacts 供 aggregation drill-down.
        metrics: dict[str, float | None | dict[str, float | None]] = {
            "nudge_fire_rate": result["nudge_fire_rate"],
            "nudge_fire_count": float(result["nudge_fire_count"]),
            "require_tool_total": float(result["require_tool_total"]),
        }

        artifacts: dict[str, Any] = {
            "scenario_path": doc.metadata.get("scenario_path"),
            "by_tool": result["by_tool"],
            "by_failure_mode": result["by_failure_mode"],
            "per_turn": result["per_turn"],
        }

        return SampleResult(
            doc_id=doc.id,
            prediction="",
            target=doc.target or "",
            metrics=metrics,
            artifacts=artifacts,
        )

    def aggregation(self) -> dict[str, Callable[[list[SampleResult]], Any]]:
        return {
            "nudge_fire_rate": _weighted_rate,
            "nudge_fire_count": _sum_metric("nudge_fire_count"),
            "require_tool_total": _sum_metric("require_tool_total"),
            "by_scenario": _by_scenario,
            "by_tool": _by_tool,
            "by_failure_mode": _by_failure_mode,
        }

    def higher_is_better(self) -> dict[str, bool]:
        return {
            # nudge_fire_rate 越低越好 —— 反向 metric，与项目其它指标的"高=好"约定相反.
            # 标 False 让 show / 排序 UI 正确处理；breakdown dicts 不进 higher_is_better.
            "nudge_fire_rate": False,
            "nudge_fire_count": False,
            "require_tool_total": True,  # 量纲——多 turn 算分母，不是越多越好但
                                          # 没有"越少越好"的反向语义；中性放 True 不误导.
        }


# ---------- module-level helpers --------------------------------------------

def _pin_envelope(doc: Doc, envelope: dict[str, Any]) -> Doc:
    """envelope + scenario_path → metadata['trajectory'] + ['expected_require_tool_turns'].

    envelope schema 与 agent_engine.result.Result 同形（§16，5 字段）：
      {transcript, artifact, warnings, success, usage}
    严格透传——缺字段直接 KeyError，与 `Result.from_dict` 对齐.
    """
    trajectory = {
        "transcript": list(envelope["transcript"]),
        "artifact": dict(envelope["artifact"]),
        "warnings": list(envelope["warnings"]),
        "success": bool(envelope["success"]),
        "usage": list(envelope["usage"]),
    }
    scenario_path = doc.metadata.get("scenario_path")
    expected: list[dict[str, Any]] = []
    if scenario_path:
        sp = Path(scenario_path)
        if not sp.is_absolute():
            sp = (PLAY_DIR / sp).resolve()
        if sp.exists():
            expected = derive_expected_turns(sp)
        # 文件不存在不抛——score 路径的 stub fixture 可能用虚构路径；该 doc 走"no
        # require_tool turns" 路径（rate=None），与 brainstorm 等无 require_tool
        # scenario 行为一致.

    new_meta = {
        **doc.metadata,
        "trajectory": trajectory,
        "expected_require_tool_turns": expected,
    }
    return replace(doc, metadata=new_meta)


# ---------- aggregation closures --------------------------------------------

def _weighted_rate(srs: list[SampleResult]) -> float | None:
    """跨 doc 的"全局" nudge_fire_rate = Σ fires / Σ totals.

    显式按 require_tool_total 加权——比简单平均 doc-level rate 更对：每个 require_tool
    turn 是一个独立 Bernoulli 试验，加权后 SE 收紧 √N 倍.
    """
    total = sum(int(s.metrics.get("require_tool_total") or 0) for s in srs)
    fires = sum(int(s.metrics.get("nudge_fire_count") or 0) for s in srs)
    if total == 0:
        return None
    return fires / total


def _sum_metric(key: str) -> Callable[[list[SampleResult]], float]:
    """把 per-sample metric key 求和——给 nudge_fire_count / require_tool_total 用."""
    def _agg(srs: list[SampleResult]) -> float:
        return float(sum(float(s.metrics.get(key) or 0.0) for s in srs))
    _agg.__name__ = f"sum_{key}"
    return _agg


def _by_scenario(srs: list[SampleResult]) -> dict[str, float | None]:
    """{doc.id: nudge_fire_rate of that doc}.

    doc.id 即 scenario id（gold.jsonl 行序约定，与 agent_traj 同源）. 无 require_tool
    turn 的 scenario 在这里值为 None——breakdown 表格里渲染 <n/a>.
    """
    out: dict[str, float | None] = {}
    for s in srs:
        rate = s.metrics.get("nudge_fire_rate")
        out[s.doc_id] = float(rate) if isinstance(rate, (int, float)) else None
    return out


def _by_tool(srs: list[SampleResult]) -> dict[str, float | None]:
    """{tool_name: 跨 doc 的加权平均 fire rate}.

    same weighting strategy as _weighted_rate（按 turn 数加权，不按 doc 数）.
    """
    bucket: dict[str, dict[str, int]] = {}
    for s in srs:
        per_tool = s.artifacts.get("by_tool", {}) or {}
        for tool, counts in per_tool.items():
            b = bucket.setdefault(tool, {"fired": 0, "total": 0})
            b["fired"] += int(counts.get("fired", 0))
            b["total"] += int(counts.get("total", 0))
    out: dict[str, float | None] = {}
    for tool, b in bucket.items():
        out[tool] = (b["fired"] / b["total"]) if b["total"] > 0 else None
    return out


def _by_failure_mode(srs: list[SampleResult]) -> dict[str, int]:
    """{mode: 跨 doc 的累计计数}. 三桶都显式列出（含 wrong_args=0）."""
    counter = {m: 0 for m in FAILURE_MODES}
    for s in srs:
        per_mode = s.artifacts.get("by_failure_mode", {}) or {}
        for m, n in per_mode.items():
            if m in counter:
                counter[m] += int(n)
    return counter
