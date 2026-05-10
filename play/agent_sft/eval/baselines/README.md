# Baselines

`agent_sft` Phase 1 / Phase 5 baseline 报告产出目录。文件由 [`aggregate_seeds.py`](../aggregate_seeds.py) 从 [`play/evals/runs/index.jsonl`](../../../evals/runs/index.jsonl) 跨 seed 聚合生成；本目录提交到 git，便于 PR 审阅历史 baseline 演化。

## Phase 1（Qwen2.5-7B vs 32B）

|文件|何时跑|预估|
|---|---|---|
|`qwen2.5-7b-vs-32b.md`|拉完两个 ollama 模型后|80 runs ≈ 3-4h on M4 Pro 48GB|

前置：

```bash
ollama pull qwen2.5:7b
ollama pull qwen2.5:32b   # 已有则跳过
```

跑批 / 出报告 / smoke 调试的全 flag 见 `python play/agent_sft/eval/run_baseline.py --help` + `python play/agent_sft/eval/aggregate_seeds.py --help`（默认值即 Phase 1 配置）。

## Phase 5（base 7B vs SFT 7B vs 32B）

复测时把 SFT 后的 ollama tag 追加到 `--models` 重跑，aggregator 自动把 3 模型并排展示：

```bash
python play/agent_sft/eval/run_baseline.py --models qwen2.5:7b qwen2.5:32b agent-sft-qwen:latest
```
