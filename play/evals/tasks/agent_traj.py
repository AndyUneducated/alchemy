"""Phase 5 vertical slice：族 5 agent trajectory task.

3 个针对 `play/agent_engine/scenarios/*.md` 的 trajectory eval doc + 4 份 stub
predictions（perfect / partial / wrong_decision / garbage）。教学叙事核心：
"在 process metric 与 outcome metric 上看 agent 行为质量阶梯"——

  | 预测            | task_success | tool_call_set_f1 | trajectory_match | coverage | 故事 |
  |---|---|---|---|---|---|
  | perfect         | 1.0          | ~1.0             | ~1.0             | ~1.0     | 上界 sanity |
  | partial         | 0.0          | ~0.6             | ~0.6             | ~0.6     | tools 部分 / 未 finalize → 失败（**核心叙事**：process > 0 但 outcome=0） |
  | wrong_decision  | 0.0          | ~1.0             | ~1.0             | ~1.0     | tools 全调到位但决策错（**反向叙事**：tool 调用对 ≠ 任务对） |
  | garbage         | 0.0          | 0.0              | 0                | 0.0      | 下界 sanity |

设计要点：
  - **output_type='none'**（phase 4 引入的 literal）：runner 跳过 LM 调用. agent_engine
    内部完整 LLM 链路在 subprocess 中跑，evals 这一侧无 LM-side 输出.
  - **gold.jsonl 行序按"smoke → 中等 → 重"排（DECISIONS §7.1.3）**：
    brainstorm（2 步 ~10-30s）→ example（kitchen sink ~1-2 分钟）→ panel（5 角色 × 11 步 数分钟）.
    `--limit 1` 自然命中 brainstorm 单跑做 CI 友好烟雾测试，与 tests/conftest.py 显式
    选 brainstorm 的策略对齐. 重排数据本身字节不变.
  - **process_docs 注入 trajectory**（run 路径）：通过 `models.agent_engine_run.make_run_fn`
    fork agent_engine subprocess + 解析 JSON envelope（cli.py --save-result-json 写出），
    把 `{transcript, artifact, warnings, success}` 注入 `doc.metadata['trajectory']`.
  - **load_prediction 注入 trajectory**（score 路径）：从 row 里抽同一份 envelope 注入
    `doc.metadata['trajectory']`，path B+C 双路径同址.
  - **judge_lm 可选**：None → 仅 4 个非 judge metric；给 judge_lm 时挂 plan_quality
    （直接复用 judge_core.g_eval，不在 metrics/trajectory.py 里重复一份）.
  - **task_success predicate** 由 task 在 process_results 内基于 doc.metadata 装：
    panel-style → `predicate_decision_in_options`（finalize + decision ∈ 白名单）；
    free-form → `predicate_speakers_covered`（所有 expected_speakers 都发言 + success=True）.

向后兼容：本 task 通过 `run_fn=None` 默认构造也能在 score 路径正常工作；run 路径才必
须注入 run_fn（默认 cli.py 注入）.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, ClassVar

from .._ae_bridge import Result
from ..api import Doc, Response, SampleResult
from ..metrics.judge_core import g_eval
from ..metrics.trajectory import (
    argument_correctness,
    predicate_decision_in_options,
    predicate_speakers_covered,
    task_success,
    tool_call_set_f1,
    trajectory_coverage,
    trajectory_match,
)
from ..models.base import LM
from ..registry import register_task
from .base import Task

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "agent_traj" / "gold.jsonl"

PLAN_QUALITY_TEMPLATE = (
    "Rate the agent run on the dimension '{dimension}' from 1-5 "
    "(1=poor, 5=excellent).\n"
    "Topic: {input}\n"
    "Reference outcome (if any): {reference}\n"
    "Final agent transcript summary + tools used:\n{response}\n"
    "Score (1-5):"
)

PLAN_QUALITY_DIMENSIONS = ("plan_structure", "tool_choice", "completeness")

RunFn = Callable[[str], dict[str, Any]]


@register_task("agent_traj")
class AgentTraj(Task):
    """Agent trajectory eval：5 metric + optional plan_quality（judge）.

    构造：
      - `run_fn=None`              → 仅 score 路径可用（trajectory 从 predictions 读）
      - `run_fn=callable`          → run 路径 process_docs hook 自动 subprocess 跑 agent_engine
      - `judge_lm=None`            → 仅 4 个非 judge metric
      - `judge_lm=lm`              → 加 plan_quality（多维 G-Eval 取 mean）
    """

    name: ClassVar[str] = "agent_traj"
    output_type: ClassVar[str] = "none"  # phase 4 literal：runner 跳 lm.generate_until

    def __init__(
        self,
        run_fn: RunFn | None = None,
        judge_lm: LM | None = None,
    ) -> None:
        self.data_path = DATA_PATH
        self._run_fn = run_fn
        self._judge_lm = judge_lm
        if judge_lm is not None:
            self._judge_plan = g_eval(
                judge_lm,
                dimensions=PLAN_QUALITY_DIMENSIONS,
                prompt_template=PLAN_QUALITY_TEMPLATE,
            )
        else:
            self._judge_plan = None

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
        """output_type='none' 时 runner 不调；保留方法仅为 ABC 满足."""
        return ""

    def doc_to_target(self, doc: Doc) -> str:
        return doc.target or ""

    def process_docs(self, docs: list[Doc]) -> list[Doc]:
        """run 路径：在 LM 步前一次性 subprocess 跑完所有 scenario，trajectory 注入 metadata."""
        if self._run_fn is None:
            return docs
        out: list[Doc] = []
        for d in docs:
            scenario_path = d.metadata.get("scenario_path")
            if not scenario_path:
                raise ValueError(
                    f"agent_traj doc id={d.id!r} missing 'scenario_path' in metadata"
                )
            envelope = self._run_fn(scenario_path)
            out.append(_pin_trajectory(d, envelope))
        return out

    def load_prediction(self, doc: Doc, row: dict) -> tuple[Doc, Response]:
        """score 路径：row 内的 envelope 字段 → doc.metadata['trajectory']；Response 占位.

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
        enriched = _pin_trajectory(doc, envelope)
        return enriched, Response(doc_id=doc.id)

    def process_results(self, doc: Doc, response: Response) -> SampleResult:
        predicate = self._select_predicate(doc)

        ts = task_success(predicate)
        f1 = tool_call_set_f1()
        ac = argument_correctness()
        tm = trajectory_match()
        coverage_kind = doc.metadata.get("coverage_kind", "callers")
        cov = trajectory_coverage(kind=coverage_kind)

        metrics: dict[str, float | None] = {
            "task_success": float(ts(doc, response)),
            "tool_call_set_f1": float(f1(doc, response)),
            "argument_correctness": float(ac(doc, response)),
            "trajectory_match": float(tm(doc, response)),
            "trajectory_coverage": float(cov(doc, response)),
        }

        if self._judge_plan is not None:
            judge_resp = _trajectory_summary_response(doc)
            dim_scores = self._judge_plan(doc, judge_resp)
            # DECISIONS §X wave 4：g_eval 现在返 dict[str, float | None]——单维 parse 全失败
            # 该维 None；plan_quality mean 走 valid 子集；全部 None → plan_quality 不写键，
            # aggregator (_mean_metric) 自然过滤；与 phase 7 P2 体例一致.
            valid = [v for v in dim_scores.values() if v is not None]
            if valid:
                metrics["plan_quality"] = sum(valid) / len(valid)
            for dim, score in dim_scores.items():
                # 子维度走私有键（'_' 前缀），不上聚合面板，仅供 drill-down；
                # None 直接落 None（落盘 JSON null）保留 drill-down 价值.
                metrics[f"_plan_{dim}"] = float(score) if score is not None else None

        traj = doc.metadata.get("trajectory", {}) or {}
        artifacts: dict[str, Any] = {
            "scenario_path": doc.metadata.get("scenario_path"),
            "tool_seq": list(traj.get("tool_seq", [])),
            "tool_calls": list(traj.get("tool_calls", [])),
            "decision": traj.get("decision"),
            "warnings": list(traj.get("warnings", [])),
        }

        return SampleResult(
            doc_id=doc.id,
            prediction="",  # output_type='none'，无 LM-side 输出
            target=doc.target or "",
            metrics=metrics,
            artifacts=artifacts,
        )

    def aggregation(self) -> dict[str, Callable[[list[SampleResult]], float | None]]:
        agg: dict[str, Callable[[list[SampleResult]], float | None]] = {
            "task_success": _mean_metric("task_success"),
            "tool_call_set_f1": _mean_metric("tool_call_set_f1"),
            "argument_correctness": _mean_metric("argument_correctness"),
            "trajectory_match": _mean_metric("trajectory_match"),
            "trajectory_coverage": _mean_metric("trajectory_coverage"),
        }
        if self._judge_lm is not None:
            agg["plan_quality"] = _mean_metric("plan_quality")
        return agg

    def collect_judge_responses(self) -> tuple[list[Response], str | None]:
        """DECISIONS §7.3：从 g_eval closure 的 _recorder 拉 LM 调用记录."""
        if self._judge_plan is None:
            return [], None
        rec = getattr(self._judge_plan, "_recorder", None)
        if rec is None:
            return [], None
        return list(rec.responses), rec.model_label

    def higher_is_better(self) -> dict[str, bool]:
        out = {
            "task_success": True,
            "tool_call_set_f1": True,
            "argument_correctness": True,
            "trajectory_match": True,
            "trajectory_coverage": True,
        }
        if self._judge_lm is not None:
            out["plan_quality"] = True
        return out

    # ---- predicate 选择 ------------------------------------------------------

    @staticmethod
    def _select_predicate(doc: Doc) -> Callable[[Doc], bool]:
        """按 metadata 选 predicate；显式优于隐式：作者可在 gold.jsonl 直接声明
        `success_predicate: "decision_in_options" | "speakers_covered"`，否则按是否
        声明 `expected_decision_options` 自动 fallback：
          - 有 expected_decision_options → decision_in_options
          - 无 / 仅声明 expected_speakers → speakers_covered
        """
        kind = doc.metadata.get("success_predicate")
        if kind == "decision_in_options":
            return predicate_decision_in_options
        if kind == "speakers_covered":
            return predicate_speakers_covered
        # 自动 fallback
        if doc.metadata.get("expected_decision_options"):
            return predicate_decision_in_options
        return predicate_speakers_covered


# ---------- module-level helpers --------------------------------------------

def _pin_trajectory(doc: Doc, envelope: dict[str, Any]) -> Doc:
    """从 envelope 派生 tool_calls / tool_seq / decision，写回 doc.metadata['trajectory'].

    envelope 必须形似 `Result.asdict()`（§16，5 字段）：`{transcript, artifact,
    warnings, success, usage}`. `Result.from_dict` 严格反序列化（缺字段 KeyError）.

    `trajectory` 字典内的 `transcript` / `usage` 都 reserialize 成 list[dict] 形态
    供 evals 度量层（[`metrics/trajectory.py`]）按 dict 消费——metadata 经过 predictions
    JSONL 落盘 + 读回时也保持同型.
    """
    result = Result.from_dict(envelope)
    tool_calls = [
        {"tool": c.tool, "caller": c.caller, "arguments": dict(c.arguments)}
        for c in result.tool_calls()
    ]
    trajectory = {
        "transcript": [dataclasses.asdict(e) for e in result.transcript],
        "artifact": dict(result.artifact),
        "warnings": list(result.warnings),
        "success": bool(result.success),
        "usage": [dataclasses.asdict(u) for u in result.usage],
        "tool_calls": tool_calls,
        "tool_seq": [c["tool"] for c in tool_calls],
        "decision": result.find_finalize_decision(),
    }
    return replace(doc, metadata={**doc.metadata, "trajectory": trajectory})


def _trajectory_summary_response(doc: Doc) -> Response:
    """把 trajectory 拍扁成短文本，喂给 g_eval 的 {response} 占位.

    G-Eval 的 prompt 期待一段可读的 response，此处把 tool 序列 + 关键 artifact 摘要
    拍成一段 Chinese-friendly text，让 judge 能基于这段做评分.
    """
    traj = doc.metadata.get("trajectory", {}) or {}
    tool_seq = traj.get("tool_seq", []) or []
    decision = traj.get("decision")
    warnings = traj.get("warnings", []) or []
    artifact = traj.get("artifact") or {}
    parts = [
        f"Tools called (in order): {', '.join(tool_seq) if tool_seq else '(none)'}.",
        f"Final decision: {decision if decision else '(no finalize)'}.",
    ]
    if warnings:
        parts.append(f"Warnings: {' | '.join(warnings)}.")
    if artifact:
        sections = "; ".join(f"{k}: {v[:60]}" for k, v in artifact.items() if v)
        if sections:
            parts.append(f"Artifact sections: {sections}.")
    return Response(doc_id=doc.id, text=" ".join(parts))


def _mean_metric(key: str) -> Callable[[list[SampleResult]], float | None]:
    """工厂：对 SampleResult.metrics[key] 求均值的 aggregation 闭包.

    DECISIONS §X wave 4：None 占位"未测得"——key 缺 / value=None 都过滤；
    全集为空 → None（与 safety / qa_open 同形）.
    """

    def _agg(srs: list[SampleResult]) -> float | None:
        if not srs:
            return None
        vals = [
            s.metrics[key]
            for s in srs
            if key in s.metrics and s.metrics[key] is not None
        ]
        if not vals:
            return None
        return sum(vals) / len(vals)

    _agg.__name__ = f"mean_{key}"
    return _agg
