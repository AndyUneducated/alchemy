"""OllamaLM adaptation layer live test (auto-probe gate).

The only one that really hits Ollama HTTP. Either of the conftest double-layer probe (service reachable + model pulled) fails → skip the entire file + friendly prompt.

The specific output text (model difference + temperature jitter) is not locked; only the shape and boundary are locked:
  - generate_until returns non-empty
  - until truncation takes effect
  - max_tokens upper bound takes effect
  - batched order independent
  - name field format
  - loglikelihood throws NotImplementedError (restart phase 9 calibration)

As per plan §2.4 6 assertions; #5/#6 are structural tests that do not depend on live, but are kept in this document to centralize the "OllamaLM unit"."""

from __future__ import annotations

import pytest

from evals.api import Request
from evals.models.ollama import OllamaLM
from evals.tests.conftest import ollama_required

pytestmark = ollama_required


def test_ollama_generate_until_returns_nonempty(ollama_model: str):
    """End-to-end sanity: can get non-empty text response."""
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
    # Truncate contract: the returned text does not contain line breaks (or the line breaks are swallowed by stop, leaving the first sentence)
    assert "\n" not in text


def test_ollama_max_tokens_capped(ollama_model: str):
    """The minimum value of max_tokens (4) can be capped - the number of tokens returned is significantly less than the unlimited case."""
    lm = OllamaLM(model=ollama_model)
    req = Request(
        doc_id="d0",
        prompt="请详细描述北京的春天。请尽量详细。",
        max_tokens=4,
        until=(),
    )
    [resp] = lm.generate_until([req])
    text = resp.text or ""
    # 4 token under Chinese BPE < 16 char loose upper bound (different model tokenizer differences)
    assert len(text) <= 32, f"max_tokens=4 should produce short output, got {len(text)} chars"


def test_ollama_batched_calls_independent(ollama_model: str):
    """The order of the two requests is consistent with the input; doc_id does not collide (a client bug that has occurred)."""
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
    """`name == ollama:<model>`——Falls into the EvalResult.model field and can be distinguished by the show command.

    Structural testing, no live network required (but placed in this document to keep the OllamaLM unit centralized)."""
    lm = OllamaLM(model=ollama_model)
    assert lm.name == f"ollama:{ollama_model}"


def test_ollama_loglikelihood_not_implemented(ollama_model: str):
    """Loglikelihood runs ABC and throws NotImplementedError by default; restart phase 9 calibration.

    Structural testing."""
    lm = OllamaLM(model=ollama_model)
    with pytest.raises(NotImplementedError):
        lm.loglikelihood([Request(doc_id="d0", prompt="test", request_type="loglikelihood")])


def test_ollama_response_carries_efficiency_fields(ollama_model: str):
    """phase 6: Ollama /api/generate returns prompt_eval_count / eval_count / total_duration
    → Response.usage / latency_ms field is filled in (not None + reasonable value).

    Ollama 0.1+ returns these three fields by default; the old daemon lacks fields → None (efficiency_aggregated compatible)."""
    lm = OllamaLM(model=ollama_model)
    req = Request(
        doc_id="d0",
        prompt="请用一个数字回答：2+2 等于几？只回答数字。",
        max_tokens=8,
        until=("\n",),
    )
    [resp] = lm.generate_until([req])
    # At least latency_ms is not None (total_duration is a stable contract)
    assert resp.latency_ms is not None and resp.latency_ms > 0
    # usage is the same as contract: there are tokens_in / tokens_out
    assert resp.usage is not None
    assert resp.usage.tokens_in is not None and resp.usage.tokens_in > 0
    assert resp.usage.tokens_out is not None and resp.usage.tokens_out > 0
