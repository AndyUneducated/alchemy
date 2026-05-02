"""play/evals — 双模式 LLM 评测框架.

- score 模式 (offline): 吃 predictions JSONL + gold，纯离线打分
- run 模式 (active): 驱动 LM 跑 prompt，harness 风格
"""
