"""play/evals — 双模式 LLM 评测框架.

- score 模式：吃 predictions JSONL + gold，纯文件打分（不驱动 LM）
- run 模式：驱动 LM 跑 prompt，harness 风格
"""
