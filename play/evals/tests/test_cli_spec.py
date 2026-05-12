"""parse_model_spec 单元测试（零网络）.

只验证 spec 字符串 → LM 类型映射。不实际调用 generate_until.

按 plan §二.5：6 条断言，含 openai / anthropic 各自的显式 NotImplementedError——
两 provider 错误信息可能各自漂移（如未来 anthropic 先支持），分别锁更稳.

外加 3 条 _build_task_with_optional_deps dispatch 断言（phase 3 CLI 完整性补丁，
score / run 共用同一 helper，故只锁 helper 自身行为，不重复在 cmd_* 上 monkeypatch）：
  - judge_model=None：返回原 task，不带 judge_lm
  - qa_open + judge_model：返回 QAOpen(judge_lm=...) 注入版
  - 其它 task + judge_model：立即 SystemExit（不触 LM）
"""

from __future__ import annotations

import pytest

from evals import tasks  # noqa: F401  — 触发 @register_task
from evals.cli import _build_task_with_optional_deps, parse_model_spec
from evals.models.mock import MockLM
from evals.models.ollama import OllamaLM
from evals.registry import get_task
from evals.tasks.qa_open import QAOpen


@pytest.fixture
def task():
    return get_task("sentiment_clf")


def test_parse_spec_mock_still_works(task):
    """phase 1 的 mock:* 路径不回归."""
    lm = parse_model_spec("mock:gold", task)
    assert isinstance(lm, MockLM)
    assert lm.name == "mock:gold"


def test_parse_spec_ollama_returns_ollama_lm(task):
    """ollama:<model> 解析为 OllamaLM；name 落 `ollama:<model>` 入 EvalResult.model."""
    lm = parse_model_spec("ollama:qwen2.5:32b", task)
    assert isinstance(lm, OllamaLM)
    assert lm.name == "ollama:qwen2.5:32b"
    assert lm.model == "qwen2.5:32b"


def test_parse_spec_ollama_with_base_url_override(task, monkeypatch):
    """EVALS_OLLAMA_BASE_URL env 可覆盖默认 base_url（无需改 spec 语法）."""
    monkeypatch.setenv("EVALS_OLLAMA_BASE_URL", "http://other:11434")
    lm = parse_model_spec("ollama:qwen2.5:32b", task)
    assert isinstance(lm, OllamaLM)
    assert lm.base_url == "http://other:11434"


# ---------- @seed=K 后缀（agent_sft phase 1 多 seed wiring） ----------

def test_parse_spec_ollama_with_seed_suffix(task):
    """`ollama:<model>@seed=K` → OllamaLM(seed=K)；name 保留 @seed=K 后缀供 EvalResult.model 区分."""
    lm = parse_model_spec("ollama:qwen2.5:7b-instruct@seed=42", task)
    assert isinstance(lm, OllamaLM)
    assert lm.model == "qwen2.5:7b-instruct"  # @seed= 后缀已剥离
    assert lm.seed == 42
    assert lm.name == "ollama:qwen2.5:7b-instruct@seed=42"  # 但 model_label 保留


def test_parse_spec_ollama_without_seed_keeps_default_zero(task):
    """无 @seed= 后缀 → OllamaLM 默认 seed=0，name 不含后缀（裸 spec 默认形态）."""
    lm = parse_model_spec("ollama:qwen2.5:7b-instruct", task)
    assert lm.seed == 0
    assert lm.name == "ollama:qwen2.5:7b-instruct"
    assert "@seed=" not in lm.name


def test_parse_spec_ollama_with_seed_zero_explicit(task):
    """显式 `@seed=0` 也写入 name（让 multi-seed bash 循环 seed=0 仍能与裸 spec 区分）."""
    lm = parse_model_spec("ollama:qwen2.5:7b-instruct@seed=0", task)
    assert lm.seed == 0
    assert lm.name == "ollama:qwen2.5:7b-instruct@seed=0"


def test_parse_spec_invalid_seed_raises(task):
    """`@seed=abc` 非整数 → ValueError（与未知 provider 同 fail-fast 路径）."""
    with pytest.raises(ValueError, match="invalid seed"):
        parse_model_spec("ollama:qwen2.5:7b@seed=abc", task)


def test_parse_spec_seed_suffix_on_mock_raises(task):
    """`mock:gold@seed=42` → ValueError；mock 走自家 `mock:noisy:<noise>:<seed>` 语法."""
    with pytest.raises(ValueError, match="seed=K suffix"):
        parse_model_spec("mock:gold@seed=42", task)


def test_parse_spec_openai_explicit_not_implemented(task):
    """openai:* → 显式 NotImplementedError，错误信息提示 phase 3 未启用."""
    with pytest.raises(NotImplementedError, match="phase 3"):
        parse_model_spec("openai:gpt-4o-mini", task)


def test_parse_spec_anthropic_explicit_not_implemented(task):
    """anthropic:* → 显式 NotImplementedError（与 openai 分别锁，错误信息可能各自漂移）."""
    with pytest.raises(NotImplementedError, match="phase 3"):
        parse_model_spec("anthropic:claude-3-haiku", task)


def test_parse_spec_unknown_provider_raises(task):
    """未知 provider → ValueError（不与 NotImplementedError 混淆）."""
    with pytest.raises(ValueError):
        parse_model_spec("weirdprovider:foo", task)


# ---------- _build_task_with_optional_deps dispatch（score / run 共用） ----------

def test_build_task_no_judge_returns_plain_qa_open():
    """judge_model=None → 走 get_task 平凡构造，task._judge_lm is None."""
    t = _build_task_with_optional_deps("qa_open", judge_model_spec=None)
    assert isinstance(t, QAOpen)
    assert t._judge_lm is None


def test_build_task_with_judge_injects_judge_lm():
    """qa_open + judge_model spec → 重建 QAOpen(judge_lm=...) 注入版."""
    t = _build_task_with_optional_deps("qa_open", judge_model_spec="mock:gold")
    assert isinstance(t, QAOpen)
    assert t._judge_lm is not None


def test_build_task_judge_on_non_qa_open_raises_systemexit():
    """非 qa_open + judge_model → SystemExit（fail-fast 而非 silently 忽略）."""
    with pytest.raises(SystemExit, match="qa_open|rag_qa"):
        _build_task_with_optional_deps("sentiment_clf", judge_model_spec="mock:gold")


# ---------- Phase 4 dispatch (RAG / safety 形参) ----------

from evals.tasks.rag_qa import RagQA  # noqa: E402
from evals.tasks.rag_retrieval import RagRetrieval  # noqa: E402
from evals.tasks.safety import Safety  # noqa: E402


def test_build_rag_retrieval_with_vdb_injects_retrieve_fn():
    """rag_retrieval + --vdb → 注入 retrieve_fn（callable）."""
    t = _build_task_with_optional_deps(
        "rag_retrieval", vdb="/tmp/fake_vdb", retrieve_top_k=3, retrieve_mode="dense",
    )
    assert isinstance(t, RagRetrieval)
    assert t._retrieve_fn is not None
    assert callable(t._retrieve_fn)
    assert t._top_k == 3


def test_build_rag_retrieval_without_vdb_returns_naked_task():
    """rag_retrieval 无 --vdb（score 路径用法）→ task 本体，retrieve_fn=None."""
    t = _build_task_with_optional_deps("rag_retrieval")
    assert isinstance(t, RagRetrieval)
    assert t._retrieve_fn is None


def test_build_rag_retrieval_with_judge_raises_systemexit():
    """rag_retrieval + --judge-model → SystemExit（rag_retrieval 没有 LM-side 输出可判）."""
    with pytest.raises(SystemExit, match="rag_retrieval"):
        _build_task_with_optional_deps("rag_retrieval", judge_model_spec="mock:gold")


def test_build_rag_qa_with_vdb_and_judge_injects_both():
    """rag_qa + --vdb + --judge-model → retrieve_fn + judge_lm 双注入."""
    t = _build_task_with_optional_deps(
        "rag_qa",
        vdb="/tmp/fake_vdb",
        judge_model_spec="mock:gold",
    )
    assert isinstance(t, RagQA)
    assert t._retrieve_fn is not None
    assert t._judge_lm is not None


def test_build_rag_qa_without_judge_lexical_only():
    """rag_qa + --vdb 无 --judge-model → 仅 lexical baseline."""
    t = _build_task_with_optional_deps("rag_qa", vdb="/tmp/fake_vdb")
    assert isinstance(t, RagQA)
    assert t._retrieve_fn is not None
    assert t._judge_lm is None


def test_build_qa_open_with_vdb_raises_systemexit():
    """qa_open + --vdb → SystemExit（qa_open 不接 RAG flag）."""
    with pytest.raises(SystemExit, match="qa_open|rag"):
        _build_task_with_optional_deps("qa_open", vdb="/tmp/fake_vdb")


def test_build_sentiment_clf_with_vdb_raises_systemexit():
    """非 RAG task + --vdb → SystemExit（fail-fast）."""
    with pytest.raises(SystemExit, match="rag"):
        _build_task_with_optional_deps("sentiment_clf", vdb="/tmp/fake_vdb")


def test_build_safety_with_judge_injects_judge_lm():
    """safety + --judge-model → 返回 Safety(judge_lm=...) 注入版."""
    t = _build_task_with_optional_deps("safety", judge_model_spec="mock:gold")
    assert isinstance(t, Safety)
    assert t._judge_lm is not None


def test_build_safety_with_vdb_raises_systemexit():
    """safety + --vdb → SystemExit（safety 非 retrieval task）."""
    with pytest.raises(SystemExit, match="safety|retrieval"):
        _build_task_with_optional_deps("safety", vdb="/tmp/fake_vdb")


# ---------- Phase 8 IAA dispatch（iaa_nominal / iaa_ordinal 无新 flag，与 sentiment_clf 同形）

from evals.tasks.iaa_nominal import IaaNominal  # noqa: E402
from evals.tasks.iaa_ordinal import IaaOrdinal  # noqa: E402


def test_build_iaa_nominal_naked_returns_task():
    """iaa_nominal 无 flag → 返裸 task（IAA task 不接 judge / vdb，与 sentiment_clf 同形）."""
    t = _build_task_with_optional_deps("iaa_nominal")
    assert isinstance(t, IaaNominal)


def test_build_iaa_ordinal_naked_returns_task():
    """iaa_ordinal 同 iaa_nominal."""
    t = _build_task_with_optional_deps("iaa_ordinal")
    assert isinstance(t, IaaOrdinal)


def test_build_iaa_with_judge_raises_systemexit():
    """iaa_nominal + --judge-model → SystemExit（IAA task 不接 judge；判 LM 当 annotator
    教学叙事 deferred 同 phase 5 §8 ADR）."""
    with pytest.raises(SystemExit, match="judge"):
        _build_task_with_optional_deps("iaa_nominal", judge_model_spec="mock:gold")


def test_build_iaa_with_vdb_raises_systemexit():
    """iaa_ordinal + --vdb → SystemExit（IAA 非 retrieval task）."""
    with pytest.raises(SystemExit, match="vdb|rag"):
        _build_task_with_optional_deps("iaa_ordinal", vdb="/tmp/fake_vdb")


# ---------- Phase 6 _fmt_kv 嵌套打印（aggregated efficiency 子组的 CLI 落点） ----------

from evals.cli import _fmt_kv, _fmt_row  # noqa: E402


def test_fmt_kv_flat_scalar_unchanged():
    """老 phase 1-5 平铺指标 (k, float) → "k=v.4f"（与原 _fmt_row 字节相同）."""
    assert _fmt_kv("accuracy", 0.875) == ["accuracy=0.8750"]
    assert _fmt_kv("f1_macro", 1.0) == ["f1_macro=1.0000"]


def test_fmt_kv_nested_subgroup_uses_dot_path():
    """phase 6 嵌套：efficiency.latency_ms.p50=... 形式（HELM-style 路径，cross-run 友好）."""
    out = _fmt_kv("efficiency", {"latency_ms": {"p50": 12.5, "p95": 50.0}})
    assert "efficiency.latency_ms.p50=12.5000" in out
    assert "efficiency.latency_ms.p95=50.0000" in out


def test_fmt_row_includes_efficiency_keys_when_present():
    """index row 的 _fmt_row 端到端：含 efficiency 子组的 aggregated 也打印 dot-path 形式."""
    row = {
        "run_id": "r1",
        "task": "sentiment_clf",
        "mode": "run",
        "model": "mock:gold",
        "n": 30,
        "aggregated": {
            "accuracy": 1.0,
            "efficiency": {"latency_ms": {"p50": 0.0, "p95": 0.0, "mean": 0.0}},
        },
    }
    s = _fmt_row(row)
    assert "accuracy=1.0000" in s
    assert "efficiency.latency_ms.p50=0.0000" in s


# ---------- audit §1.7：嵌套子组全 0 折叠为 <not measured> ----------

from evals.cli import _is_all_zero_nested, _print_aggregated  # noqa: E402


def test_is_all_zero_nested_true_for_nested_zeros():
    """全 0 嵌套（mock 路径 efficiency 子组形态）→ True."""
    eff = {
        "latency_ms": {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0},
        "tokens_in": {"total": 0, "mean": 0.0},
        "tokens_out": {"total": 0, "mean": 0.0},
        "cost_usd": {"total": 0.0, "mean": 0.0},
    }
    assert _is_all_zero_nested(eff) is True


def test_is_all_zero_nested_false_when_any_nonzero():
    """任一 leaf 非 0 → False（real LM 跑出真数据时不折叠）."""
    eff = {
        "latency_ms": {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0},
        "tokens_in": {"total": 178, "mean": 59.33},  # 非 0
        "tokens_out": {"total": 0, "mean": 0.0},
        "cost_usd": {"total": 0.0, "mean": 0.0},
    }
    assert _is_all_zero_nested(eff) is False


def test_print_aggregated_collapses_zero_efficiency(capsys):
    """mock 路径 efficiency 全 0 → 折叠为单行 `<not measured (no LM signal)>`，
    替代 11 行 0 占位的视觉误导（"latency=0.0000" 看着像超低延迟而非未测得）."""
    agg = {
        "accuracy": 1.0,
        "efficiency": {
            "latency_ms": {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0},
            "tokens_in": {"total": 0, "mean": 0.0},
            "tokens_out": {"total": 0, "mean": 0.0},
            "cost_usd": {"total": 0.0, "mean": 0.0},
        },
    }
    _print_aggregated(agg)
    out = capsys.readouterr().out
    assert "accuracy" in out and "1.0000" in out
    assert "<not measured" in out
    assert "efficiency.latency_ms" not in out  # 不再展开 11 行


def test_print_aggregated_expands_nonzero_efficiency(capsys):
    """real LM 跑出真数据时不折叠，按 dot-path 展开（信号路径不破）."""
    agg = {
        "accuracy": 1.0,
        "efficiency": {
            "latency_ms": {"mean": 899.3, "p50": 687.0, "p95": 1274.7, "max": 1339.9},
            "tokens_in": {"total": 178, "mean": 59.33},
            "tokens_out": {"total": 12, "mean": 4.0},
            "cost_usd": {"total": 0.000152, "mean": 0.0000507},
        },
    }
    _print_aggregated(agg)
    out = capsys.readouterr().out
    assert "<not measured" not in out
    assert "efficiency.latency_ms.p50" in out
    assert "efficiency.latency_ms.max" in out  # audit §1.2 新加
    assert "efficiency.cost_usd.mean" in out  # audit §1.1 新加


def test_print_aggregated_does_not_collapse_zero_task_metric(capsys):
    """任务自身指标即使为 0 也不折叠（accuracy=0 是真实信号，不是"未测得"）；
    折叠只对嵌套子组生效."""
    agg = {"accuracy": 0.0, "f1_macro": 0.0}
    _print_aggregated(agg)
    out = capsys.readouterr().out
    assert "accuracy" in out and "0.0000" in out
    assert "<not measured" not in out


# ---------- phase 7 audit P1：trait 协议 / content class 全 0 不折叠 ----------

from evals.cli import _should_fold_when_all_zero  # noqa: E402


def test_trait_efficiency_folds_when_all_zero():
    """efficiency 是 call class，trait True，全 0 折叠（mock 路径等价"未测得"）."""
    assert _should_fold_when_all_zero("efficiency") is True


def test_trait_unknown_dim_defaults_to_fold():
    """未注册 dim 默认 True 折叠（新 cross-cutting 想退出折叠须在自身模块显式声明 trait=False）."""
    assert _should_fold_when_all_zero("nonexistent_dim") is True


# wave 3（DECISIONS §7.2）撤销 safety cross-cutting AOP 后，safety 不再出现在
# aggregated 顶层 nested 子组（safety task 自己 own task-specific 4 stat 平铺），
# 折叠协议对 safety 不再适用——原 phase 7 audit P1 的 safety 不折叠测试组随之删除.


# ---------- wave 3 §7.3：efficiency.judge 嵌套二级折叠 ----------

def test_print_aggregated_folds_efficiency_judge_subgroup_when_all_zero(capsys):
    """efficiency 顶层非全 0（task 部分有数值）但 judge 子组全 0 → judge 子组单独折叠为
    `efficiency.judge: <not measured>` 单行；不影响 task 部分 dot-path 渲染。

    场景：task 接了 judge_lm 但 judge LM 没报 latency/tokens（如 mock），同时 task LM
    正常报 efficiency.
    """
    agg = {
        "efficiency": {
            "latency_ms": {"mean": 100.0, "p50": 100.0, "p95": 100.0, "max": 100.0},
            "tokens_in": {"total": 100, "mean": 10.0},
            "tokens_out": {"total": 50, "mean": 5.0},
            "cost_usd": {"total": 0.001, "mean": 0.0001},
            "judge": {  # 全 0：mock judge / task 没接 judge_lm 时也是这个形态
                "latency_ms": {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0},
                "tokens_in": {"total": 0, "mean": 0.0},
                "tokens_out": {"total": 0, "mean": 0.0},
                "cost_usd": {"total": 0.0, "mean": 0.0},
            },
        },
    }
    _print_aggregated(agg)
    out = capsys.readouterr().out
    # task 部分仍 dot-path 展开
    assert "efficiency.latency_ms.mean" in out
    assert "efficiency.tokens_in.total" in out
    # judge 子组单行折叠
    assert "efficiency.judge" in out
    assert "<not measured" in out
    # judge 子组的 dot-path 不应展开（折叠后仅一行）
    assert "efficiency.judge.latency_ms" not in out
    assert "efficiency.judge.tokens_in" not in out


def test_print_aggregated_does_not_fold_efficiency_judge_when_nonzero(capsys):
    """efficiency.judge 子组有数值 → dot-path 全展开（不折叠）."""
    agg = {
        "efficiency": {
            "latency_ms": {"mean": 100.0, "p50": 100.0, "p95": 100.0, "max": 100.0},
            "tokens_in": {"total": 100, "mean": 10.0},
            "tokens_out": {"total": 50, "mean": 5.0},
            "cost_usd": {"total": 0.001, "mean": 0.0001},
            "judge": {
                "latency_ms": {"mean": 200.0, "p50": 200.0, "p95": 200.0, "max": 200.0},
                "tokens_in": {"total": 50, "mean": 25.0},
                "tokens_out": {"total": 10, "mean": 5.0},
                "cost_usd": {"total": 0.0005, "mean": 0.00025},
            },
        },
    }
    _print_aggregated(agg)
    out = capsys.readouterr().out
    # judge 子组全展开
    assert "efficiency.judge.latency_ms.mean" in out
    assert "efficiency.judge.tokens_in.total" in out
    assert "<not measured" not in out  # 整体没折叠


# ---------- phase 7 audit P2：None 占位 → CLI <n/a> 渲染 ----------

def test_fmt_kv_none_value_renders_as_na():
    """None 占位（safety judge_safety_score 未接 judge_lm 时）→ `<n/a>` 而非 0.0000.
    与"真 0"（如 refusal_rate=0 garbage 路径）显式区分."""
    assert _fmt_kv("judge_safety_score", None) == ["judge_safety_score=<n/a>"]


def test_fmt_kv_nested_none_in_subgroup_renders_as_na():
    """嵌套子组里的 None 也按 `<n/a>` 渲染，dot-path 路径保留."""
    out = _fmt_kv("safety", {"refusal_rate": 0.5, "judge_safety_score": None})
    assert "safety.refusal_rate=0.5000" in out
    assert "safety.judge_safety_score=<n/a>" in out


def test_print_aggregated_renders_safety_with_mixed_real_and_na(capsys):
    """端到端：safety 子组含 1 个真值 + 3 个 None，CLI 显示真值 + <n/a> 混合."""
    agg = {
        "safety": {
            "refusal_rate": 0.6667,
            "jailbreak_success_rate": None,
            "over_refusal_rate": None,
            "judge_safety_score": None,
        },
    }
    _print_aggregated(agg)
    out = capsys.readouterr().out
    assert "safety.refusal_rate" in out and "0.6667" in out
    assert "safety.jailbreak_success_rate" in out and "<n/a>" in out
    assert "safety.judge_safety_score" in out and "<n/a>" in out
