"""Phase 8 IAA tasks 的工程健壮性回归——锁住 sklearn / krippendorff / scipy 直调
在退化输入下不再 raise / 不再 emit NaN.

驱动两个真实的失败路径（不需要真 LM）：

1. **score 路径 + `--limit` 缩到只剩单类**：predictions/constant_majority.jsonl 头 5 行
   全 ham → sklearn binary metrics (`precision/recall/f1/fbeta` with `pos_label='spam'`)
   原本 `ValueError: pos_label=spam is not a valid label`，krippendorff 原本
   `ValueError: There has to be more than one value in the domain`；
2. **run 路径 (`output_type='none'`)**：runner 给占位 Response(doc_id=..) → process_results
   看到 pred="" → 触发 sklearn multiclass-vs-binary 报错 + scipy correlation NaN
   传染下游 (json.dumps 写 NaN 是非合法 JSON, 跨 run JSON_EXTRACT 必坏).

修复策略（DECISIONS §8 工程兜底）：
  - per-class metrics 加 `_pos_label_present(yt, yp)` 短路；
  - krippendorff 加 「<2 unique value」 短路；
  - 所有可能 NaN 的 metrics (cohens_kappa / weighted_kappa* / pearsonr / spearmanr /
    kendalltau / fleiss_kappa / krippendorff / icc_1_1) 包 `_nan_to_zero`.

任一保护失效 → JSON 序列化 emit NaN 或调用 sklearn raise → 测试 fail-loud.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import ClassVar

import pytest

from evals.api import Doc, Request, Response
from evals.models.base import LM
from evals.runner import evaluate_run, evaluate_score
from evals.tasks import iaa_nominal as _iaa_nominal_mod  # noqa: F401  (registry side effects)
from evals.tasks import iaa_ordinal as _iaa_ordinal_mod  # noqa: F401
from evals.tasks.iaa_nominal import IaaNominal
from evals.tasks.iaa_ordinal import IaaOrdinal


class _UnusedLM(LM):
    """output_type='none' 任务下 runner 不应触发 generate_until；任何调用 = 工程契约破."""

    def __init__(self) -> None:
        self.name = "unused"

    def generate_until(self, requests: list[Request]) -> list[Response]:
        raise AssertionError(
            f"output_type='none' branch must not invoke LM (got {len(requests)} reqs)"
        )


def _assert_aggregated_is_finite_json(aggregated: dict) -> None:
    """所有 leaf 数值必须是有限 float（无 NaN / Inf），且 json.dumps 不依赖 allow_nan."""
    flat: list[tuple[str, float]] = []

    def _walk(prefix: str, obj) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(obj, (int, float)):
            flat.append((prefix, float(obj)))

    _walk("", aggregated)
    for path, val in flat:
        assert math.isfinite(val), f"non-finite metric leaked: {path} = {val!r}"
    json.dumps(aggregated, allow_nan=False)


# ---------- iaa_nominal: score path with --limit small (single-class slice) ----------


def test_iaa_nominal_score_limit_single_class_does_not_raise():
    """`--limit 5` 后 constant_majority preds 只剩 5 个 ham → sklearn binary metrics
    + krippendorff 原本两路都炸. 修复后必须给出有限值."""
    task = IaaNominal()
    preds = (
        Path(__file__).resolve().parent.parent
        / "data" / "iaa_nominal" / "predictions" / "constant_majority.jsonl"
    )
    r = evaluate_score(task, preds, limit=5)
    _assert_aggregated_is_finite_json(r.aggregated)
    # 单类 ham + 全 ham 预测：accuracy=1, spam-面 metrics=0 (pos_label 缺席短路)
    assert r.aggregated["accuracy"] == 1.0
    assert r.aggregated["precision_spam"] == 0.0
    assert r.aggregated["recall_spam"] == 0.0
    assert r.aggregated["f1_spam"] == 0.0
    assert r.aggregated["f_beta_2"] == 0.0
    # 全单类 → 退化 1.0 (scott_pi/gwet_ac1 单类约定) / 0.0 (kappa NaN→0)
    assert r.aggregated["scott_pi"] == 1.0
    assert r.aggregated["gwet_ac1"] == 1.0
    assert r.aggregated["cohens_kappa"] == 0.0  # NaN→0 (Pe=1 退化)
    assert r.aggregated["krippendorff_alpha"] == 0.0  # 单值域短路


def test_iaa_nominal_score_limit_single_class_metrics_keys_complete():
    """退化路径必须仍给齐全部 14 个 first-class scalar + 1 个 _confusion_matrix dict."""
    task = IaaNominal()
    preds = (
        Path(__file__).resolve().parent.parent
        / "data" / "iaa_nominal" / "predictions" / "constant_majority.jsonl"
    )
    r = evaluate_score(task, preds, limit=5)
    expected = {
        "accuracy", "balanced_accuracy", "mcc", "f1_micro", "f1_macro", "f_beta_2",
        "precision_spam", "recall_spam", "f1_spam",
        "cohens_kappa", "scott_pi", "gwet_ac1", "fleiss_kappa", "krippendorff_alpha",
        "_confusion_matrix",
    }
    assert expected.issubset(set(r.aggregated))


# ---------- iaa_nominal: run path (output_type='none') ----------


def test_iaa_nominal_run_path_does_not_raise_or_call_lm():
    """run path: pred='' 在所有 30 doc 上触发 sklearn multiclass + binary 冲突；
    必须被 _pos_label_present 短路 + NaN 兜底 + LM 一次没被调."""
    task = IaaNominal()
    lm = _UnusedLM()
    r = evaluate_run(task, lm)
    _assert_aggregated_is_finite_json(r.aggregated)
    assert r.n == 30
    # 占位 pred 与 gold 全部不一致：accuracy=0, sanity 系列 0
    assert r.aggregated["accuracy"] == 0.0
    assert r.aggregated["f_beta_2"] == 0.0
    assert r.aggregated["precision_spam"] == 0.0
    # cohens_kappa 在 pred 全 "" 时 sklearn Pe 退化 → NaN → 我们 →0
    assert r.aggregated["cohens_kappa"] == 0.0
    assert r.aggregated["fleiss_kappa"] == 0.0  # raters 缺席 → guard 短路 0


def test_iaa_nominal_run_path_small_limit():
    """--limit 5 + run path: 双重退化（单类 yt + 占位 yp）"""
    task = IaaNominal()
    lm = _UnusedLM()
    r = evaluate_run(task, lm, limit=5)
    _assert_aggregated_is_finite_json(r.aggregated)
    assert r.n == 5


# ---------- iaa_ordinal: run path NaN tax ----------


def test_iaa_ordinal_run_path_no_nan():
    """run path: pred 全 0 (placeholder int) → constant input → pearson/spearman/kendall
    原本全 NaN. NaN 不是合法 JSON (`json.dumps(float('nan'))` 写 'NaN' 任何非 Python parser
    都会拒)，必须收紧到 0."""
    task = IaaOrdinal()
    lm = _UnusedLM()
    r = evaluate_run(task, lm)
    _assert_aggregated_is_finite_json(r.aggregated)
    # 12 个 first-class scalar 应全在
    expected = {
        "accuracy", "cohens_kappa", "weighted_kappa_linear", "weighted_kappa_quadratic",
        "pearson_r", "spearman_rho", "kendall_tau", "lins_ccc",
        "fleiss_kappa", "krippendorff_alpha_ordinal", "krippendorff_alpha_interval", "icc_1_1",
    }
    assert expected.issubset(set(r.aggregated))
    for k in ("pearson_r", "spearman_rho", "kendall_tau"):
        assert r.aggregated[k] == 0.0, f"{k} should be 0 (constant input), got {r.aggregated[k]}"


def test_iaa_ordinal_run_path_small_limit_no_nan():
    """--limit 3 + run: 退化更严重（小 N + 常数 pred），仍然必须无 NaN."""
    task = IaaOrdinal()
    lm = _UnusedLM()
    r = evaluate_run(task, lm, limit=3)
    _assert_aggregated_is_finite_json(r.aggregated)


# ---------- 跨 task: aggregated 序列化为合法 JSON ----------


@pytest.mark.parametrize(
    "task_factory,preds_name",
    [
        (IaaNominal, "constant_majority.jsonl"),
        (IaaNominal, "perfect.jsonl"),
        (IaaOrdinal, "perfect.jsonl"),
        (IaaOrdinal, "off_by_one.jsonl"),
    ],
)
def test_iaa_score_aggregated_is_strict_json(task_factory, preds_name: str):
    """Phase 4 path C 哲学：aggregated 永远跨进程 / 跨 run 用 JSON 流转；
    `allow_nan=False` 必须能 round-trip——这里覆盖全 4 stub 的健康路径不退化."""
    task = task_factory()
    family = "iaa_nominal" if task_factory is IaaNominal else "iaa_ordinal"
    preds = Path(__file__).resolve().parent.parent / "data" / family / "predictions" / preds_name
    r = evaluate_score(task, preds)
    _assert_aggregated_is_finite_json(r.aggregated)


# ---------- 边界 limit ----------


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_iaa_nominal_score_extreme_small_limits(limit: int):
    """`--limit 0/1/2` 是 audit / 调试常用入口；不能 raise 或 emit NaN.
    `--limit 0` 触发空 sample_results 路径（`if not srs: return 0.0` 全员守门）."""
    task = IaaNominal()
    preds = (
        Path(__file__).resolve().parent.parent
        / "data" / "iaa_nominal" / "predictions" / "perfect.jsonl"
    )
    r = evaluate_score(task, preds, limit=limit)
    _assert_aggregated_is_finite_json(r.aggregated)
    assert r.n == limit


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_iaa_ordinal_score_extreme_small_limits(limit: int):
    """同 iaa_nominal：极小 limit 不 raise / 不 NaN."""
    task = IaaOrdinal()
    preds = (
        Path(__file__).resolve().parent.parent
        / "data" / "iaa_ordinal" / "predictions" / "perfect.jsonl"
    )
    r = evaluate_score(task, preds, limit=limit)
    _assert_aggregated_is_finite_json(r.aggregated)


# ---------- 存储层 strict-JSON 兜底（phase 8 §8.R4 follow-up） ----------


def test_storage_save_rejects_nan_aggregated(tmp_path: Path):
    """storage.save 必须 fail-loud 拒绝 NaN 写盘——否则下游 jq / DB / 仪表盘必坏.

    任务自身的 NaN 兜底 (iaa_nominal/iaa_ordinal `_nan_to_zero`) 是第一道防线；
    本测试锁第二道：即使任意未来 task 漏算，storage 层也会在 write 时 ValueError.
    与 Phase 4 path C 「跨进程跨 run 用 JSON 流转」契约对齐.
    """
    from evals.api import EvalResult, SampleResult
    from evals.storage import save

    nan_result = EvalResult(
        task="synthetic",
        model="test",
        mode="score",
        run_id="test-storage-rejects-nan",
        created_at="2026-05-07T00:00:00",
        n=1,
        elapsed_ms=1.0,
        num_fewshot=0,
        aggregated={"good": 1.0, "evil": float("nan")},
        per_sample=(SampleResult(doc_id="d1", prediction="", target="", metrics={"acc": 0.0}),),
    )
    with pytest.raises(ValueError, match="JSON compliant"):
        save(nan_result, runs_dir=tmp_path)


def test_storage_save_rejects_inf_aggregated(tmp_path: Path):
    """同 NaN：Inf 也是非合法 JSON，storage 必须 fail-loud."""
    from evals.api import EvalResult, SampleResult
    from evals.storage import save

    inf_result = EvalResult(
        task="synthetic",
        model="test",
        mode="score",
        run_id="test-storage-rejects-inf",
        created_at="2026-05-07T00:00:00",
        n=1,
        elapsed_ms=1.0,
        num_fewshot=0,
        aggregated={"good": 1.0, "evil": float("inf")},
        per_sample=(SampleResult(doc_id="d1", prediction="", target="", metrics={"acc": 0.0}),),
    )
    with pytest.raises(ValueError, match="JSON compliant"):
        save(inf_result, runs_dir=tmp_path)


def test_storage_iaa_run_roundtrip_strict_json(tmp_path: Path):
    """端到端：iaa_nominal run path → save → 三个写出文件都必须 strict-JSON 可读
    （`parse_constant` 引发 ValueError 模拟 jq / 浏览器 / 数据库 parser）."""
    from evals.storage import save

    task = IaaNominal()
    lm = _UnusedLM()
    r = evaluate_run(task, lm, limit=5)
    run_dir = save(r, runs_dir=tmp_path)

    def _strict_load(path: Path):
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda x: (_ for _ in ()).throw(
                ValueError(f"non-JSON literal {x!r} in {path}")
            ),
        )

    _strict_load(run_dir / "result.json")
    for line in (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            json.loads(
                line,
                parse_constant=lambda x: (_ for _ in ()).throw(
                    ValueError(f"non-JSON literal {x!r} in samples.jsonl")
                ),
            )
    for line in (tmp_path / "index.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            json.loads(
                line,
                parse_constant=lambda x: (_ for _ in ()).throw(
                    ValueError(f"non-JSON literal {x!r} in index.jsonl")
                ),
            )
