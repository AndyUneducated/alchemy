# Triples — Phase 2 SFT 数据产物目录

`agent_sft/data/triples/` 装存 Phase 2 数据流水线（mine → extract → split → format）的全部中间 + 最终产物。Phase 2 收尾交付的两份 1k 数据集（`runs_1k_fast_{7b,32b}_r0_124/` + `*_1k.jsonl`）入 git；其它 smoke / 临时产物按 `.gitignore` 默认忽略，可重生。

## 文件清单

|文件 / 目录|生成于|是否入 git|用途|
|---|---|---|---|
|`runs_1k_fast_{7b,32b}_r0_124/<scen>-r<N>.json`|`mine_triples.py` (fast scenario, run_id 0-124)|✅|Phase 2 终交付的 raw envelope，每模型 250 个（2 scenario × 125 run）|
|`{triples,train_triples,val_triples,train,val}_{7b,32b}_1k.jsonl`|`synthesize.py` → `split.py` → `formatter.py`|✅|两份模型各自的全链路 SFT 数据；`train_*.jsonl` 是 MLX-LM 直接可吃的 chat schema|
|`runs/<scen>-r<N>.json`|`mine_triples.py` 默认输出|❌ (gitignore)|本地 smoke / 临时跑批|
|`triples.jsonl` / `train.jsonl` / `val.jsonl` 等无 `_1k` 后缀|默认产物|❌ (gitignore)|本地 smoke 派生；要复现 1k 数据集见 §重生命令|

## 两种 triple 来源（pilot 后选用 synthesize）

|脚本|配对策略|yield|corrected 来源|
|---|---|---|---|
|`extractor.py`|first attempt 失败 + 后续 attempt 真实成功|~3-25%（看模型 recovery 率）|真实 speaker.content|
|`synthesize.py`（**当前默认**）|每个 nudge fire → 1 triple|100%|从 step.instruction 程序化模板（fallback：通用 wrapper + 完整 instruction）|

7B pilot 测得 recovery 率仅 ~3% → extractor 路径 yield 太低不实用；synthesize 用 step.instruction 里的字面 `tool(args)` 模板造 corrected，零额外 compute 把 yield 拉到 100%。详见 §历史遗留：57-triple pilot 与方法选择。

## 重生命令

### 1k 终交付批次（与 repo 内 `*_1k.jsonl` 一致）

以 7B 为例（32B 把 `AGENT_ENGINE_MODEL` 换成 `qwen2.5:32b`、所有 `_7b_` 换成 `_32b_` 即可）：

```bash
export AGENT_ENGINE_MODEL=qwen2.5:7b

# 1) 跑批 250 envelope（fast 副本，run_id 0-124）
python play/agent_sft/data/mine_triples.py \
  --run-ids $(seq 0 124) \
  --out-dir play/agent_sft/data/triples/runs_1k_fast_7b_r0_124

# 2) 抽三元组（synthesize：每个 fire 一条）
python play/agent_sft/data/synthesize.py \
  --in  play/agent_sft/data/triples/runs_1k_fast_7b_r0_124 \
  --out play/agent_sft/data/triples/triples_7b_1k.jsonl

# 3) 切 train/val（per-scenario 末 20% run_id → val）
python play/agent_sft/data/split.py \
  --in    play/agent_sft/data/triples/triples_7b_1k.jsonl \
  --train play/agent_sft/data/triples/train_triples_7b_1k.jsonl \
  --val   play/agent_sft/data/triples/val_triples_7b_1k.jsonl

# 4) 格式化为 MLX-LM 样本（train + val 各跑一次）
python play/agent_sft/data/formatter.py \
  --in  play/agent_sft/data/triples/train_triples_7b_1k.jsonl \
  --out play/agent_sft/data/triples/train_7b_1k.jsonl
python play/agent_sft/data/formatter.py \
  --in  play/agent_sft/data/triples/val_triples_7b_1k.jsonl \
  --out play/agent_sft/data/triples/val_7b_1k.jsonl
```

### 本地 smoke / 改 schema 调试

走默认输出（`runs/` + `triples.jsonl` + `train.jsonl`，全 gitignored），命令同上但去掉 `--out-dir`、文件名去 `_*_1k` 后缀、`--run-ids 0 1 2 3 4 5` 跑 12 envelope 即可。要复现 baseline eval 的 `max_retries=1` 行为：加 `--upstream`（mine + synthesize 必须一致，否则 turn_idx 错位 yield 归零）。

每个脚本 `--help` 看完整 flag。

## Scenario：fast 副本 vs upstream

`data/scenarios/{tool_chain,code_review}_fast.md` 是上游 `agent_engine/scenarios/*.md` 的 mining 优化派生：

|改动|fast|upstream|为什么 fast 这么改|
|---|---|---|---|
|`max_retries`|0|1|synthesize 只看 first attempt，retry 是纯浪费 LLM 调用|
|`max_tokens`|80|160-200|agent prompt 本就限制 ≤30/50 字，cap 贴近实际负载|
|moderator open / finalize|删|有|0 fires，纯仪式开销|
|envelope wall clock (7B / M4 Pro)|~42s/env|~65s/env|-35%|
|synthesize yield|~4 triples/env|~4.75 triples/env|相当|

`mine_triples.py` 默认走 fast；`--upstream` 切回原 scenario（与 baseline eval 数据一致）。`extractor.py` / `synthesize.py` 也有同款 `--upstream` flag，必须与 mining 步骤一致——否则 turn_idx 错位会让 yield 归零。

## OOD 评估

**OOD 评估不在本目录**——复用 Phase 1 落地的 `play/evals/data/bfcl_slice/gold.jsonl`（50 例 BFCL `simple_python` 切片）。Phase 5 复测时直接：

```bash
python -m evals run --task bfcl_slice --model ollama:agent-sft-qwen
```

不复制公开数据集到本仓库；BFCL 上游变更由 `play/evals/data/bfcl_slice/_fetch.py` 管理。

## Phase 2 终交付：1k × 2 模型

两份独立数据集，挖批参数对齐（fast scenario / `max_retries=0` / `run_id 0-124` / 2 scenarios），仅 mining 模型不同：

|项|7B (Qwen2.5-7B)|32B (Qwen2.5-32B)|
|---|---|---|
|envelope（committed in `runs_1k_fast_{7b,32b}_r0_124/`）|250|250|
|triples (`triples_*_1k.jsonl`)|**1212**|**1052**|
|triples / envelope|4.85|4.21|
|train / val (`train_*_1k.jsonl` / `val_*_1k.jsonl`)|966 / 246|842 / 210|
|失败模式分布|missed 1091 / wrong_tool 121|missed 773 / wrong_tool 279|
|scenario 分布|code_review 933 / tool_chain 279|code_review 802 / tool_chain 250|
|实测 wall clock（M4 Pro）|~7.5 h|~9.5 h|
|on-disk size|envelopes 2.1 MB + jsonl ~11 MB|envelopes 2.4 MB + jsonl ~10 MB|

`val` 切分一致：每 scenario 末 20% run_id（即 `run_id ∈ [100, 124]`）→ val。

**7B vs 32B 选择指引**：

- 默认走 7B：单条 triple compute 成本 ~22s（vs 32B ~32s），yield 高 15%，已覆盖 missed 主分布。
- 32B 价值在 wrong_tool 分布更广（27% vs 10%）——若 Phase 3 训练后发现 wrong_tool 召回低，可拌入 32B 数据补 hard sample。
- 两份并存而非合并入一个 train.jsonl：保留模型来源标签便于 Phase 3 ablation；训练时可任选其一或拼接。

## 历史遗留：57-triple pilot 与方法选择

Phase 2 早期跑过 4 个 pilot 批次（详细时序见 `JOURNAL.md` 2026-05-10 条目），关键结论：

1. 7B + extractor（要求 first-fail + later-success 真 recovery）yield 仅 0.17/env，与 plan 估算 5/env 差 30 倍。
2. 试 `max_retries` 翻倍 → 无改善；试 32B 对照 → recovery 从 3% 跳到 25%，证明底座 capability 才是 recovery 率主因。
3. 改走 synthesize 路径（per-fire + step.instruction 模板造 corrected）→ 7B 也能 yield ~4.75/env，命中 plan 原估算。

**为什么仍保留 `extractor.py`**：未来若 Phase 3 训练后 7B 自己 recovery 率拉到 30%+，extractor 的"真自纠"语义比 synthesize 的"模板答案"更对应项目核心论点（self-correction），届时一行命令切回。
