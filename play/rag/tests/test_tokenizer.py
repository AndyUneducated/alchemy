"""Post-processing logic for the HF BPE tokenizer wrapper.

`tokenizer.tokenize` does three things on HF raw `encoded.tokens`:
  1. Filter special tokens (`<bos>` / `<eos>` / `<|...|>` angle-bracket forms)
  2. Strip SentencePiece / GPT-2 BPE word-start prefixes (`▁` / `Ġ`)
  3. Lowercase + drop empty strings

DECISIONS §4 treats same-source tokenization as dense embedding as hybrid's key
engineering pairing; accidental changes here shift BM25 IDF distribution.

Tests monkeypatch `_tokenizer` with a fake encoder — no real HF model download.
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
    assert tokenize("dummy", name="fake") == ["hello", "world"]


def test_tokenize_strips_sentencepiece_and_gpt2_prefixes(monkeypatch):
    monkeypatch.setattr(
        tokenizer_module, "_tokenizer",
        _fake_tokenizer_factory(["▁foo", "Ġbar", "baz"]),
    )
    assert tokenize("dummy", name="fake") == ["foo", "bar", "baz"]


def test_tokenize_lowercases(monkeypatch):
    monkeypatch.setattr(
        tokenizer_module, "_tokenizer",
        _fake_tokenizer_factory(["FooBar", "QUERY"]),
    )
    assert tokenize("dummy", name="fake") == ["foobar", "query"]


def test_tokenize_drops_empty_after_normalization(monkeypatch):
    monkeypatch.setattr(
        tokenizer_module, "_tokenizer",
        _fake_tokenizer_factory(["Ġ", "", "▁", "Ġreal"]),
    )
    assert tokenize("dummy", name="fake") == ["real"]


def test_tokenize_mixed_special_and_word_pieces(monkeypatch):
    monkeypatch.setattr(
        tokenizer_module, "_tokenizer",
        _fake_tokenizer_factory(
            ["<|im_start|>", "ZX", "-", "74", "92", "<|im_end|>"]
        ),
    )
    assert tokenize("dummy", name="fake") == ["zx", "-", "74", "92"], (
        "rare alphanumeric IDs (the very case BM25 is meant to recover) must "
        "survive tokenization untouched"
    )


def test_basic_tokenizer_preserves_rare_ids_without_hf_download():
    assert tokenize("ZX-7492 项目代号", name="basic") == ["zx", "-", "7492", "项目代号"]
