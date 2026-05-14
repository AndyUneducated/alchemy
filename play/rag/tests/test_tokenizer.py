"""HF BPE tokenizer 包装的后处理逻辑。

`tokenizer.tokenize` 在 HF 原始 `encoded.tokens` 上做 3 件事：
  1. 过滤特殊 token（`<bos>` / `<eos>` / `<|...|>` 形如 `<...>`）
  2. 去掉 SentencePiece / GPT-2 BPE 的词首前缀（`▁` / `Ġ`）
  3. 小写化 + 丢空字符串

DECISIONS §4 把这条与 dense embedding tokenization 同源作为 hybrid 的关键工程对偶；
任何无意改动这层逻辑都会让 BM25 IDF 分布偏移。

测试不打真实 HF 模型，monkeypatch `_tokenizer` 返回 fake encoder。
"""
from __future__ import annotations

import tokenizer as tokenizer_module
from tokenizer import tokenize


class _FakeEnc:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens


def _fake_tokenizer_factory(tokens: list[str]):
    class _FakeTok:
        def encode(self, text):
            return _FakeEnc(tokens)
    return lambda name: _FakeTok()


def test_tokenize_filters_angle_bracket_special_tokens(monkeypatch):
    monkeypatch.setattr(
        tokenizer_module, "_tokenizer",
        _fake_tokenizer_factory(["<bos>", "Hello", "Ġworld", "<eos>"]),
    )
    assert tokenize("dummy") == ["hello", "world"]


def test_tokenize_strips_sentencepiece_and_gpt2_prefixes(monkeypatch):
    monkeypatch.setattr(
        tokenizer_module, "_tokenizer",
        _fake_tokenizer_factory(["▁foo", "Ġbar", "baz"]),
    )
    assert tokenize("dummy") == ["foo", "bar", "baz"]


def test_tokenize_lowercases(monkeypatch):
    monkeypatch.setattr(
        tokenizer_module, "_tokenizer",
        _fake_tokenizer_factory(["FooBar", "QUERY"]),
    )
    assert tokenize("dummy") == ["foobar", "query"]


def test_tokenize_drops_empty_after_normalization(monkeypatch):
    monkeypatch.setattr(
        tokenizer_module, "_tokenizer",
        _fake_tokenizer_factory(["Ġ", "", "▁", "Ġreal"]),
    )
    assert tokenize("dummy") == ["real"]


def test_tokenize_mixed_special_and_word_pieces(monkeypatch):
    monkeypatch.setattr(
        tokenizer_module, "_tokenizer",
        _fake_tokenizer_factory(
            ["<|im_start|>", "ZX", "-", "74", "92", "<|im_end|>"]
        ),
    )
    assert tokenize("dummy") == ["zx", "-", "74", "92"], (
        "rare alphanumeric IDs (the very case BM25 is meant to recover) must "
        "survive tokenization untouched"
    )
