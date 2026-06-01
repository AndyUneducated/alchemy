# Baselines

`agent_sft` baseline / re-measure 报告产出目录。文件由 [`aggregate_seeds.py`](../aggregate_seeds.py) 从 [`play/evals/runs/index.jsonl`](../../../evals/runs/index.jsonl) 跨 seed 聚合生成；本目录提交到 git，便于 PR 审阅历史 baseline 演化。

## 读数规则

|目录 / 文件|底座|能说明什么|不能说明什么|
|---|---|---|---|
|`qwen2.5-7b-vs-32b.md`|Qwen2.5-7B / 32B|v1 训练前 baseline|qwen3.5 当前效果|
|`phase5-3model-comparison.md`|Qwen2.5 base / SFT / 32B|v1 结案数字：57.3% gap closure|qwen3.5 当前效果|
|`qwen3_phase3/`|qwen3.5:9b / placeholder / qwen3.6:27b|qwen3 迁移后评测管线仍可跑|SFT adapter 效果；placeholder 不是 adapter|
|`qwen3_bf16_clean/`|qwen3.5:9b / placeholder|in-dist vs held-out 分层口径|真实 adapter 收益；GGUF deploy 仍 blocked|

## v1 Phase 1（Qwen2.5-7B vs 32B）

|文件|何时跑|预估|
|---|---|---|
|`qwen2.5-7b-vs-32b.md`|拉完两个 ollama 模型后|80 runs ≈ 3-4h on M4 Pro 48GB|

前置：

```bash
ollama pull qwen2.5:7b
ollama pull qwen2.5:32b   # 已有则跳过
```

跑批 / 出报告 / smoke 调试的全 flag 见 `python play/agent_sft/eval/run_baseline.py --help` + `python play/agent_sft/eval/aggregate_seeds.py --help`（默认值即 Phase 1 配置）。

## v1 Phase 5（base 7B vs SFT 7B vs 32B）

复测时把 SFT 后的 ollama tag 追加到 `--models` 重跑，aggregator 自动把 3 模型并排展示：

```bash
python play/agent_sft/eval/run_baseline.py --models qwen2.5:7b qwen2.5:32b agent-sft-qwen:latest
```

## qwen3.x 迁移后的注意点

|点|说明|
|---|---|
|默认模型|`run_baseline.py` 已默认 `qwen3.5:9b` / `qwen3.6:27b`|
|placeholder|`agent-sft-qwen-3` 当前 `FROM qwen3.5:9b`，只能验证管线，不代表 adapter|
|27B timeout|M4 Pro 上 qwen3.6:27b 的 agent path 可能超过默认 600s；需要时用 `AGENT_ENGINE_RUN_TIMEOUT` 拉长|
|真实 SFT 效果|短期看 `train/eval_smoke.py` 或未来 MLX server 路径；Ollama GGUF 修好前不要用 placeholder 数字下结论|
