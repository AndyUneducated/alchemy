"""qa_open task run 路径 + 双模式 parity（架构定海神针）.

phase 0 立的 LM ABC + parity 不变量在 phase 3 新 task 上的延续锁——
绿 = "score 与 run 共享 task.process_results / aggregation 尾段" 在 qa_open 上没有偷偷分叉.
"""

from __future__ import annotations

from pathlib import Path

from evals.models.mock import MockLM
from evals.runner import evaluate_run, evaluate_score
from evals.tasks.qa_open import QAOpen
from evals.tests.test_qa_open_score import _jaccard_fake

PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "qa_open" / "predictions"


def test_run_gold_judge_equals_score_perfect_judge():
    """`evaluate_run(qa_open, MockLM(gold), judge=FakeJudge)` ≡ `evaluate_score(qa_open, perfect.jsonl)`.

    两个独立的 jaccard fake 实例（stateless rule）→ 相同 prompt 输入 → 相同 score 输出，
    parity 在 aggregated 与 per_sample 两层都字节相同.
    """
    docs = list(QAOpen().docs())

    task_run = QAOpen(judge_lm=_jaccard_fake())
    r_run = evaluate_run(task_run, MockLM(mode="gold", docs=docs))

    task_score = QAOpen(judge_lm=_jaccard_fake())
    r_score = evaluate_score(task_score, PRED_DIR / "perfect.jsonl")

    # phase 6 引入 cross-cutting efficiency 子组（被测物 call class，仅 run 挂）；wave 3
    # （DECISIONS §7.2）撤销 safety cross-cutting—— qa_open 不再有 sample.metrics["safety"] 占位；
    # wave 3 §7.3 加 efficiency.judge 子组（评估工具 call class，双路径都挂）—— score 路径
    # 仍出现 efficiency 顶层，但仅含 judge 子组（无 task 部分的 latency_ms / tokens_in）.
    _crosscut = {"efficiency"}
    task_agg = lambda d: {k: v for k, v in d.items() if k not in _crosscut}  # noqa: E731
    task_metrics = lambda d: {k: v for k, v in d.items() if k not in _crosscut}  # noqa: E731
    assert task_agg(r_run.aggregated) == task_agg(r_score.aggregated)
    # run 路径：efficiency 顶层含 task 部分 + judge 子组
    assert "efficiency" in r_run.aggregated
    assert "latency_ms" in r_run.aggregated["efficiency"]
    assert "judge" in r_run.aggregated["efficiency"]
    # score 路径：efficiency 顶层仅含 judge 子组（task LM 无调用，被测物 efficiency 不挂）
    assert "efficiency" in r_score.aggregated
    assert "judge" in r_score.aggregated["efficiency"]
    assert "latency_ms" not in r_score.aggregated["efficiency"]
    assert r_run.n == r_score.n

    a_pairs = [(s.doc_id, s.prediction, s.target, task_metrics(s.metrics)) for s in r_run.per_sample]
    o_pairs = [(s.doc_id, s.prediction, s.target, task_metrics(s.metrics)) for s in r_score.per_sample]
    assert a_pairs == o_pairs


def test_run_qa_with_self_consistency_wrapper():
    """judge 套 self_consistency(n=3) 仍能跑通且数值有界（plumbing test）.

    stateless Jaccard fake → 同 prompt 3 次采样都同分 → mode=同分 → 集成层无破坏.
    """
    docs = list(QAOpen().docs())
    task = QAOpen(judge_lm=_jaccard_fake(), judge_n_samples=3)
    r = evaluate_run(task, MockLM(mode="gold", docs=docs))

    assert "judge_pointwise" in r.aggregated
    # gold mode → answer == target → Jaccard=1 → 5 分 ×10
    assert 1.0 <= r.aggregated["judge_pointwise"] <= 5.0
    assert r.aggregated["judge_pointwise"] == 5.0


def test_run_qa_open_with_fewshot_records_num_fewshot():
    """num_fewshot=2 字段持久化（与 mt 同 contract，新 task 重锁）."""
    docs = list(QAOpen().docs())
    task = QAOpen(judge_lm=_jaccard_fake())
    r = evaluate_run(
        task,
        MockLM(mode="gold", docs=docs),
        num_fewshot=2,
        fewshot_seed=0,
    )
    assert r.num_fewshot == 2
    # gold mode + few-shot 不影响 perfect 答案
    assert r.aggregated["exact_match"] == 1.0
