"""MockLM 四种 mode 的行为 + seed 确定性."""

from __future__ import annotations

from evals.api import Doc, Request
from evals.models.mock import MockLM, default_rule_fn


def _fake_docs() -> list[Doc]:
    return [
        Doc(id="a", input="This is great", target="positive"),
        Doc(id="b", input="This is bad", target="negative"),
        Doc(id="c", input="This is fine", target="neutral"),
        Doc(id="d", input="Love it", target="positive"),
        Doc(id="e", input="Average", target="neutral"),
    ]


def _reqs(docs: list[Doc]) -> list[Request]:
    return [Request(doc_id=d.id, prompt=f"prompt {d.id}") for d in docs]


def test_mode_gold_100_percent():
    docs = _fake_docs()
    lm = MockLM(mode="gold", docs=docs)
    resps = lm.generate_until(_reqs(docs))
    for doc, resp in zip(docs, resps):
        assert resp.text == doc.target


def test_mode_constant_always_same_label():
    docs = _fake_docs()
    lm = MockLM(mode="constant", docs=docs, label="neutral")
    resps = lm.generate_until(_reqs(docs))
    assert all(r.text == "neutral" for r in resps)


def test_mode_rule_uses_keywords():
    docs = _fake_docs()
    lm = MockLM(mode="rule", docs=docs)
    resps = lm.generate_until(_reqs(docs))
    by_id = {r.doc_id: r.text for r in resps}
    assert by_id["a"] == "positive"  # "great"
    assert by_id["b"] == "negative"  # "bad"
    assert by_id["c"] == "neutral"   # no keyword
    assert by_id["d"] == "positive"  # "love"
    assert by_id["e"] == "neutral"


def test_mode_noisy_deterministic_same_seed():
    """同一 seed 跑两次 MockLM 必须得到完全一致的输出.

    这是整个 README 预期数值能复现的根。
    """
    docs = _fake_docs()
    lm1 = MockLM(mode="noisy", docs=docs, noise=0.5, seed=42)
    lm2 = MockLM(mode="noisy", docs=docs, noise=0.5, seed=42)
    r1 = [r.text for r in lm1.generate_until(_reqs(docs))]
    r2 = [r.text for r in lm2.generate_until(_reqs(docs))]
    assert r1 == r2


def test_mode_noisy_different_seeds_differ():
    docs = _fake_docs() * 5  # 放大样本数增加区分率
    lm1 = MockLM(mode="noisy", docs=_fake_docs(), noise=0.8, seed=0)
    lm2 = MockLM(mode="noisy", docs=_fake_docs(), noise=0.8, seed=999)
    r1 = [r.text for r in lm1.generate_until(_reqs(_fake_docs()))]
    r2 = [r.text for r in lm2.generate_until(_reqs(_fake_docs()))]
    # 两个不同 seed 至少有一个位置不一样
    assert r1 != r2


def test_default_rule_fn():
    assert default_rule_fn("This is great") == "positive"
    assert default_rule_fn("THIS IS BAD") == "negative"
    assert default_rule_fn("meh") == "neutral"
    assert default_rule_fn("I love it") == "positive"
    assert default_rule_fn("terrible quality") == "negative"
