"""agent_sft 项目的 evals 消费侧脚本——多 seed runner / 聚合器 / 实验报告.

agent_sft 不持有评测组件（task / metric 全归 [`play/evals/`](../../evals/)），本目录只装：
  - `run_baseline.py`     — 多 seed runner，调 `python -m evals run`
  - `aggregate_seeds.py`  — 读 `evals/runs/index.jsonl` 跨 seed 聚合
  - `baselines/`          — markdown 实验报告产出目录

使用方式见 [`baselines/README.md`](baselines/README.md).
"""
