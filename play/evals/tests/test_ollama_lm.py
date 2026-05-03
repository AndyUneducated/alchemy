"""OllamaLM 适配层 live 测试（auto-probe gate）.

唯一一个真正打 Ollama HTTP 的文件。conftest 双层 probe（服务可达 + 模型已拉）任一失败 → 整文件 skip + 友好提示.

不锁具体输出文本（模型差异 + 温度抖动）；只锁形状与边界：
  - generate_until 返回非空
  - until 截断生效
  - max_tokens 上界生效
  - batched 顺序独立
  - name 字段格式
  - loglikelihood 抛 NotImplementedError（phase 7 calibration 再开）

按 plan §二.4 6 条断言；#5 / #6 是结构性测试不依赖 live，但保留在本文件以集中"OllamaLM 单元".
"""

from __future__ import annotations

import pytest

from evals.api import Request
from evals.models.ollama import OllamaLM
from evals.tests.conftest import ollama_required

pytestmark = ollama_required


def test_ollama_generate_until_returns_nonempty(ollama_model: str):
    """端到端 sanity：能拿到非空 text response."""
    lm = OllamaLM(model=ollama_model)
    req = Request(
        doc_id="d0",
        prompt="请用一个数字回答：1+1等于几？只回答数字。",
        request_type="generate_until",
        max_tokens=8,
        until=("\n",),
    )
    [resp] = lm.generate_until([req])
    assert resp.doc_id == "d0"
    assert resp.text is not None
    assert len(resp.text.strip()) > 0


def test_ollama_until_stop_seq_truncates(ollama_model: str):
    """until=('\\n',) 截断生效——response 中第一个 '\\n' 之前是输出（或完全无 '\\n'）."""
    lm = OllamaLM(model=ollama_model)
    req = Request(
        doc_id="d0",
        prompt="请用三句中文连续叙述今天天气，每句之间换行。",
        request_type="generate_until",
        max_tokens=64,
        until=("\n",),
    )
    [resp] = lm.generate_until([req])
    text = resp.text or ""
    # 截断 contract：返回文本不含换行（或换行被 stop 吞掉，剩下是首句）
    assert "\n" not in text


def test_ollama_max_tokens_capped(ollama_model: str):
    """max_tokens 极小值（4）能封顶——返回的 token 数明显少于无限制 case."""
    lm = OllamaLM(model=ollama_model)
    req = Request(
        doc_id="d0",
        prompt="请详细描述北京的春天。请尽量详细。",
        max_tokens=4,
        until=(),
    )
    [resp] = lm.generate_until([req])
    text = resp.text or ""
    # 4 token 在中文 BPE 下 < 16 char loose 上界（不同模型 tokenizer 差异）
    assert len(text) <= 32, f"max_tokens=4 should produce short output, got {len(text)} chars"


def test_ollama_batched_calls_independent(ollama_model: str):
    """两条 request 顺序与输入一致；doc_id 不串供（曾发生过的客户端 bug）."""
    lm = OllamaLM(model=ollama_model)
    reqs = [
        Request(doc_id="alpha", prompt="只回答字母 X：", max_tokens=4, until=("\n",)),
        Request(doc_id="beta", prompt="只回答数字 9：", max_tokens=4, until=("\n",)),
    ]
    responses = lm.generate_until(reqs)
    assert len(responses) == 2
    assert responses[0].doc_id == "alpha"
    assert responses[1].doc_id == "beta"


def test_ollama_lm_name_includes_model_tag(ollama_model: str):
    """`name == ollama:<model>`——落到 EvalResult.model 字段，show 命令能区分.

    结构性测试，不需 live 网络（但放在本文件保持 OllamaLM 单元集中）.
    """
    lm = OllamaLM(model=ollama_model)
    assert lm.name == f"ollama:{ollama_model}"


def test_ollama_loglikelihood_not_implemented(ollama_model: str):
    """loglikelihood 走 ABC 默认抛 NotImplementedError；phase 7 calibration 再开.

    结构性测试.
    """
    lm = OllamaLM(model=ollama_model)
    with pytest.raises(NotImplementedError):
        lm.loglikelihood([Request(doc_id="d0", prompt="test", request_type="loglikelihood")])
