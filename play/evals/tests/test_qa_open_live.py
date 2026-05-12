"""qa_open live e2e smoke（auto-probe gate）—— score / run 两文件的 live 兄弟.

链路覆盖：spec → OllamaLM → evaluate_run/score → qa_open task → judge_pointwise →
CLI cmd_run / cmd_score → storage 路径上的 SampleResult / EvalResult schema.

只锁 schema 与定性范围（数值随模型 + 温度 + seed 抖动），不锁具体数值——
test_qa_open_score.py 已用 FakeJudgeLM 锁了精确数值；这里负责"真链路通"的烟雾级证明.

文件三元组（同 task 不同模式 × 网络层级）：
  - test_qa_open_score.py：FakeJudgeLM 零网络 + 4 份 stub predictions 教学叙事
  - test_qa_open_run.py：MockLM(gold) + FakeJudge parity（架构定海神针）
  - test_qa_open_live.py：本文件，真 ollama 双层 probe gate
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from evals.cli import cmd_run
from evals.models.ollama import OllamaLM
from evals.runner import evaluate_run, evaluate_score
from evals.storage import load_run
from evals.tasks.qa_open import QAOpen
from evals.tests.conftest import ollama_required

pytestmark = ollama_required

PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "qa_open" / "predictions"


def test_evaluate_run_qa_open_ollama_smoke(ollama_model: str):
    """run 路径：OllamaLM 既作答 task 又作 judge（self-grading），limit=3 跑通.

    锁 schema + 范围；具体数值随模型 / 温度 / seed 漂.
    """
    lm = OllamaLM(model=ollama_model)
    task = QAOpen(judge_lm=lm)

    r = evaluate_run(task, lm, limit=3)

    # schema sanity
    assert r.mode == "run"
    assert r.n == 3
    assert r.model == f"ollama:{ollama_model}"
    # phase 6/7 起 cross-cutting 子组：efficiency（call class）+ safety（content class）
    task_keys = {k for k in r.aggregated.keys() if k not in {"efficiency", "safety"}}
    assert task_keys == {"exact_match", "rouge_l", "judge_pointwise"}
    assert "efficiency" in r.aggregated

    # 数值合法区间
    assert 0.0 <= r.aggregated["exact_match"] <= 1.0
    assert 0.0 <= r.aggregated["rouge_l"] <= 1.0
    assert 1.0 <= r.aggregated["judge_pointwise"] <= 5.0

    # per-sample 完整
    assert len(r.per_sample) == 3
    for s in r.per_sample:
        assert "judge_pointwise" in s.metrics
        assert 1.0 <= s.metrics["judge_pointwise"] <= 5.0


def test_evaluate_score_qa_open_ollama_judge_smoke(ollama_model: str):
    """score 路径：predictions 来自 perfect.jsonl（gold target），judge 用真 ollama.

    perfect 上 judge 应给较高分（>=3.5 loose 阈值；实际多在 4-5）；
    与 run 路径正交，覆盖"非 LM 驱动答题但 judge 调真 LM"的混合模式.
    """
    judge_lm = OllamaLM(model=ollama_model)
    task = QAOpen(judge_lm=judge_lm)

    r = evaluate_score(task, PRED_DIR / "perfect.jsonl", limit=3)

    assert r.mode == "score"
    assert r.n == 3
    assert "judge_pointwise" in r.aggregated
    # perfect 是 gold target verbatim → real judge 应明显倾向高分（loose lower bound）
    assert r.aggregated["judge_pointwise"] >= 3.5, (
        f"real ollama judge on perfect predictions should be >=3.5, "
        f"got {r.aggregated['judge_pointwise']}"
    )


def test_cli_cmd_run_judge_model_e2e(ollama_model: str, tmp_path: Path):
    """CLI 完整链路：python -m evals run --task qa_open --model ollama:... --judge-model ollama:...

    走 cmd_run dispatch（含 QAOpen(judge_lm=...) 重建）→ evaluate_run → save → load_run；
    验证 judge_pointwise 不只在内存里有，还落到 runs/ 目录可被 show 读回。
    """
    spec = f"ollama:{ollama_model}"
    args = argparse.Namespace(
        task="qa_open", model=spec, judge_model=spec,
        limit=2, seed=0, num_fewshot=0, fewshot_seed=0,
        runs_dir=tmp_path,
        vdb=None, retrieve_top_k=5, retrieve_mode="hybrid", rerank=False,
    )
    rc = cmd_run(args)
    assert rc == 0

    # 落盘验证：runs_dir 下有且仅有 1 个 run，其 aggregated 含 judge_pointwise
    run_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1, f"expected 1 run dir, got {run_dirs}"
    result, _samples = load_run(run_dirs[0].name, runs_dir=tmp_path)
    assert result["task"] == "qa_open"
    assert result["model"] == spec
    assert result["n"] == 2
    assert "judge_pointwise" in result["aggregated"]
    assert 1.0 <= result["aggregated"]["judge_pointwise"] <= 5.0


def test_cli_cmd_score_judge_model_e2e(ollama_model: str, tmp_path: Path):
    """CLI hybrid 模式：python -m evals score --task qa_open --predictions ... --judge-model ollama:...

    score 路径 + 真 LM judge——predictions 来自文件（非 LM 驱动），judge 调真 ollama；
    验证 cmd_score 也走 helper dispatch，落盘 aggregated 含 judge_pointwise。
    与 cmd_run 同 model 字段使用文件 basename（source_label 默认）作 EvalResult.model。
    """
    spec = f"ollama:{ollama_model}"
    args = argparse.Namespace(
        task="qa_open", predictions=PRED_DIR / "perfect.jsonl",
        source_label=None, judge_model=spec,
        limit=2, runs_dir=tmp_path,
    )
    from evals.cli import cmd_score
    rc = cmd_score(args)
    assert rc == 0

    run_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    result, _samples = load_run(run_dirs[0].name, runs_dir=tmp_path)
    assert result["task"] == "qa_open"
    assert result["mode"] == "score"
    assert result["n"] == 2
    assert "judge_pointwise" in result["aggregated"]
    # perfect predictions → judge 应给较高分（loose 阈值，与 score smoke 同标准）
    assert result["aggregated"]["judge_pointwise"] >= 3.5
