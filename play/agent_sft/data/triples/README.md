# Triples — Phase 2 SFT 数据产物目录

`agent_sft/data/triples/` 装存 Phase 2 数据流水线（mine → extract → split → format）的全部中间 + 最终产物。所有 `.jsonl` / `.json` 都 gitignored（大、可重生），仅本 README 进 git。

## 文件清单

|文件|生成于|用途|
|---|---|---|
|`runs/<scen>-r<N>.json`|`mine_triples.py` (子进程跑 agent_engine)|raw envelope（transcript + artifact + warnings + success），按 `(scenario, run_id)` 命名|
|`triples.jsonl`|`synthesize.py`（默认）/ `extractor.py`（备选）|原始三元组（含 `failed_response` / `nudge` / `corrected_response` 全链路 + `context` 全 prefix）|
|`train_triples.jsonl` / `val_triples.jsonl`|`split.py`|与 triples 同字段；per-scenario 末 20% run_id → val|
|`train.jsonl` / `val.jsonl`|`formatter.py`|MLX-LM `mlx_lm.lora` 直接吃的 chat-format 样本（`{messages: [system, user, assistant]}`）|

## 两种 triple 来源（pilot 后选用 synthesize）

|脚本|配对策略|yield|corrected 来源|
|---|---|---|---|
|`extractor.py`|first attempt 失败 + 后续 attempt 真实成功|~3-25%（看模型 recovery 率）|真实 speaker.content|
|`synthesize.py`（**当前默认**）|每个 nudge fire → 1 triple|100%|从 step.instruction 程序化模板（fallback：通用 wrapper + 完整 instruction）|

7B pilot 测得 recovery 率仅 ~3% → extractor 路径 yield 太低不实用；synthesize 用 step.instruction 里的字面 `tool(args)` 模板造 corrected，零额外 compute 把 yield 拉到 100%。详见 §Pilot 实测。

## 重生命令（按顺序）

```bash
# 1) 跑批：2 scenarios × 6 runs = 12 envelopes（默认 fast 副本，~42s/env vs upstream ~65s）
python play/agent_sft/data/mine_triples.py --run-ids 0 1 2 3 4 5
#   要复现 baseline eval 行为（max_retries=1，envelope ~65s）：加 --upstream

# 2) 抽三元组（synthesize：每个 fire 一条；如要"真 recovery"语义，把 synthesize.py 换 extractor.py）
python play/agent_sft/data/synthesize.py \
  --in  play/agent_sft/data/triples/runs/ \
  --out play/agent_sft/data/triples/triples.jsonl
#   synthesize / extractor 都默认按 fast 副本解析；--upstream 要与 mine 步骤一致

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

## Pilot 实测（2026-05-10）

按时间顺序 4 个批次，前 3 用 extractor（真 recovery 配对），第 4 切到 synthesize（per-fire 配对）。

|批次|model|max_retries|envelope|fires|fire rate|recovery|extractor yield|synthesize yield|
|---|---|---|---|---|---|---|---|---|
|7B run 0-2 (已丢)|Qwen2.5-7B|1|6|28|72%|3.6%|0.17/env|—|
|7B run 0-5 (在 repo)|Qwen2.5-7B|2 (实验)|12|57|73%|2.4%|0.08/env|**4.75/env** ✓|
|32B run 200-202 (对照)|Qwen2.5-32B|1|3|20|83%|25%|1.67/env|（未跑）|
|⇒ 当前 train.jsonl|—|—|12|57|—|—|—|**57 triples**|

**关键发现演进（按时序）**：
1. 初次 pilot：7B yield 0.17/env，与 plan 估算 5/env 差 30 倍 → 看着像方法学崩
2. 试 max_retries 翻倍：yield 不升反降（噪声主导）→ 排除"重试不够"
3. 试 32B 对照：recovery 25%（vs 7B 的 3%）→ **底座 capability 才是 recovery 率主因**，不是方法学问题
4. 走 synthesize 路径（用 step.instruction 模板造 corrected）：7B 同样 envelope 立刻 yield 4.75/env → **零额外 compute 拿到 4.75/env，命中 plan 原估算**

**为什么 synthesize 优于 32B mining**：
- 32B mining 一个 envelope ~500s，每条 triple 实际成本 ~5min compute
- synthesize 复用现有 7B envelope，~100s/env（mining）+ 0 额外（synthesize），每条 triple ~21s
- corrected 是模板 → 训练目标更干净，没有 32B "text 说 X 但 tool_call 是 Y" 的噪声样本

**为什么仍保留 extractor.py**：未来若 Phase 3 训练后 7B 自己 recovery 率拉到 30%+，extractor 的"真自纠"语义比 synthesize 的"模板答案"更对应项目核心论点（self-correction），届时可一行命令切回。

## 当前 train/val 数据

|项|数|
|---|---|
|输入 envelope|12（6 tool_chain + 6 code_review，7B max_retries=2 batch）|
|synthesized triples|57（41 code_review + 16 tool_chain）|
|train.jsonl|47 samples|
|val.jsonl|10 samples（per-scenario 末 20% run_id：tool_chain r5 + code_review r5）|
|failure_mode 分布|53 missed + 4 wrong_tool|
|wrong_args|0（deferred to Phase 5）|

## token 分布（max_recent=6，57 sample）

待 Phase 3 训练前正式统计；目测 user content 100-400 token 区间，远低于 2048 上限。
