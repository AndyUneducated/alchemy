# Triples — Phase 2 SFT 数据产物目录

`agent_sft/data/triples/` 装存 Phase 2 数据流水线（mine → extract → split → format）的全部中间 + 最终产物。所有 `.jsonl` / `.json` 都 gitignored（大、可重生），仅本 README 进 git。

## 文件清单

|文件|生成于|用途|
|---|---|---|
|`runs/<scen>-r<N>.json`|`mine_triples.py` (子进程跑 agent_engine)|raw envelope（transcript + artifact + warnings + success），按 `(scenario, run_id)` 命名|
|`triples.jsonl`|`extractor.py`|原始三元组（含 `failed_response` / `nudge` / `corrected_response` 全链路 + `context` 全 prefix）|
|`train_triples.jsonl` / `val_triples.jsonl`|`split.py`|与 triples 同字段；per-scenario 末 20% run_id → val|
|`train.jsonl` / `val.jsonl`|`formatter.py`|MLX-LM `mlx_lm.lora` 直接吃的 chat-format 样本（`{messages: [system, user, assistant]}`）|

## 重生命令（按顺序）

```bash
# 1) 跑批：2 scenarios × 3 runs = 6 envelopes (pilot 量级)
python play/agent_sft/data/mine_triples.py --run-ids 0 1 2

# 2) 抽三元组
python play/agent_sft/data/extractor.py \
  --in  play/agent_sft/data/triples/runs/ \
  --out play/agent_sft/data/triples/triples.jsonl

# 3) 切 train/val（先切再格式化；formatter 输出丢元数据）
python play/agent_sft/data/split.py \
  --in    play/agent_sft/data/triples/triples.jsonl \
  --train play/agent_sft/data/triples/train_triples.jsonl \
  --val   play/agent_sft/data/triples/val_triples.jsonl

# 4) 格式化为 MLX-LM 样本（train + val 各跑一次）
python play/agent_sft/data/formatter.py \
  --in  play/agent_sft/data/triples/train_triples.jsonl \
  --out play/agent_sft/data/triples/train.jsonl
python play/agent_sft/data/formatter.py \
  --in  play/agent_sft/data/triples/val_triples.jsonl \
  --out play/agent_sft/data/triples/val.jsonl
```

每个脚本 `--help` 看完整 flag。

## OOD 评估

**OOD 评估不在本目录**——复用 Phase 1 落地的 `play/evals/data/bfcl_slice/gold.jsonl`（50 例 BFCL `simple_python` 切片）。Phase 5 复测时直接：

```bash
python -m evals run --task bfcl_slice --model ollama:agent-sft-qwen
```

不复制公开数据集到本仓库；BFCL 上游变更由 `play/evals/data/bfcl_slice/_fetch.py` 管理。

## Pilot 实测（2026-05-10，Qwen2.5-7B）

两次 pilot 共 18 envelope（19 min 总 wall clock），合计 1 triple。

|批次|scenario max_retries|envelope|require_tool turns|nudge fired|fire rate|triples|yield|recovery rate|
|---|---|---|---|---|---|---|---|---|
|run_ids 0-2 (回滚后丢)|1|6|39|28|72%|1|0.17|3.6%|
|run_ids 0-5 (当前 in repo)|2 (实验)|12|78|57|73%|1|0.08|2%|
|对照: 1 envelope tool_chain|1|1|5|2|40%|0|—|0%|（32B）|

**关键结论**：
- yield 与 plan §Volume math 估算（5/env）差 30-60 倍 → 1k 目标在当前方法学下不可达
- `max_retries: 1 → 2` 实验确认瓶颈不在重试次数：fire rate 持平、recovery 持平 → 已回滚 scenario YAML 到原 max_retries=1
- 32B 单 envelope 也是 0 recovery → 瓶颈也不在底座 capability（虽然 fire rate 低些）
- **真瓶颈**：7B 即使被告知"你刚才没调 X"，也极少真的补调 X——"模型自我修正"作为 supervision 信号在当前 scenario 设计下太稀薄

|failure_mode|run_ids 0-5 (max_retries=2) 总|
|---|---|
|missed|55|
|wrong_tool|2|
|wrong_args|0（deferred to Phase 5）|

详见 [`../../DECISIONS.md`](../../DECISIONS.md) §11 § scale-up 路径 + § 已知持续 trade-off。当前仓库保留 1 triple 作 pipeline proof + Phase 3 smoke 训练入口；真正 train set 规模需 Phase 2.5 方法学迭代后再补。

## token 分布（max_recent=6，当前 1 sample）

唯一 sample user content ≈ 250 token，远低于 2048 上限——`max_recent=6` 估计 OK，待真量级 train set 后再分位统计。
