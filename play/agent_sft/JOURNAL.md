# Journal

每条里程碑一段：`## YYYY-MM-DD — 标题`，正文必含 **功能** + **技术**，**取舍** 按需。架构决策见 [`DECISIONS.md`](DECISIONS.md)。

## 2026-05-10 — Phase 0 立项 + Phase 1 baseline 工具链

定下三件框架决策（中心问题 = nudge-grounded SFT、ceiling 用本地 Qwen2.5-32B、README 写成 "v1 + 演化路径"），并搭完 Phase 1 baseline 评测工具链。80-batch 实跑留到独立里程碑。

### 功能

|item|说明|
|---|---|
|README + 路线图|中心问题：让 nudge-fire rate 在 in-dist 显著降、OOD 不回归；Phase 0-6 + 演化路径配齐|
|度量四项|nudge-fire rate / trajectory score / BFCL / general regression，按 scenario × tool × failure-mode 三轴 + 多 seed 报均值±std|
|两个新 scenario|`code_review` (4 agent × 8 turn) + `tool_chain` (1 agent × 5 turn)，都强 require_tool|
|多 seed wiring|model spec 加 `@seed=K` 后缀，最小侵入|
|baseline 工具|runner（2 model × 10 seed × 4 task = 80 runs）+ aggregator 出 markdown 报告|

### 技术

- **中心问题**：用自家 agent_engine 产 trajectory，自家 evals 算 nudge fire rate（DECISIONS §1）。
- **训练框架**：MLX-LM，Apple Silicon 原生，三命令链路（DECISIONS §2）。
- **底座 + ceiling**：Qwen2.5-7B + Qwen2.5-32B，同家族跨规模对比，零闭源依赖。
- **失败模式**：missed / wrong_tool / wrong_args（后者暂占位归 wrong_tool）。

### 取舍

- 放弃 "经典 tool-calling LoRA on xLAM/ToolACE"——执行简单但面试无差异化。
- 放弃 GPT-4o-mini 作 ceiling——换全本地可复现 + 同家族对比。
- 不引入 `lm-evaluation-harness`——自实现 <100 行远低于跨框架适配成本。

## 2026-05-10 — Phase 2 流水线 + 57 条 demo 数据

一天搭完流水线、跑 pilot 撞瓶颈、切方案 B 解锁。最终交 47 train + 10 val。

时序：7B × 6 envelope → 1 triple；max_retries=2 重试 → 仍 1 triple；32B × 3 envelope 对照 → recovery 从 3% 跳到 25%，确认底座 capability 是主因；改走 synthesize（per-fire 配对，corrected 用 instruction 模板）→ 12 envelope 出 57 triples。

### 功能

|item|说明|
|---|---|
|`data/` 5 脚本|mine_triples / extractor / synthesize / split / formatter + 18 测试|
|两条 triple 路径|extractor 抽真自纠（yield 0.17/env）；synthesize 用模板造 corrected（yield 4.75/env）|
|首批数据|47 train + 10 val，per-scenario 末 20% 切 val|

### 技术

- **为什么默认 synthesize**：extractor 在 7B 上 yield 太低；synthesize 把 yield 拉到 fire rate 上限，compute 省 ~14×。
- **与 §1 关系**：仍是 "7B 失败素材 + scenario 模板"，没引入第三方教师。
- **样本格式**：F1 only（input 不含 nudge），训 "看到原 instruction 一次到位"。

### 取舍

- 承认前一天 "32B 单 envelope 对照失败" 是过度判断，n=20 重跑修正，过程留在 JOURNAL 不掩盖。
- 选 synthesize 不选 32B mining——经济性碾压 + 信号更干净；代价是模板复读风险，留给 Phase 3 看效果。
- 57 triples 算 "demo viable"——是否 scale 到 1k 等 Phase 3 smoke 信号再定。

## 2026-05-10 — fast scenario 副本：mining 提速 35%

为后续 1k scale-up 做提速。上游 scenario 不动（baseline eval 已按 max_retries=1 跑过），新建 `_fast` 副本只服务 mining：max_retries 1→0、max_tokens 200→80、删 open/finalize 步。`--upstream` flag 默认走 fast、加 flag 走上游。

smoke 7B 2 envelope：平均 **42s/env**（vs upstream 65s/env, -35%）。1k 提速估算：~175 min vs ~228 min。

不写新单元测试（fast 副本是数据不是逻辑），不做更激进的 minimal scenario（工程开销 > 时间收益）。

## 2026-05-10 — Phase 2 收尾：1k × 2 模型双批数据

跨夜 17h 跑完 7B / 32B 各 250 envelope。

|item|说明|
|---|---|
|7B 数据集|1212 triples → 966 train + 246 val；wall clock 7.5h|
|32B 数据集|1052 triples → 842 train + 210 val；wall clock 9.5h|
|策略|两份并存（文件名带 `_7b_` / `_32b_`）便于 ablation；27 MB 直接 commit（远低于 LFS 阈值）|
|7B vs 32B 实测|yield 接近（synthesize 路径下底座差异变小）；32B wrong_tool 占比更高，给 hard sample 用|

orchestrator 用 `caffeinate` + `nohup` + `set -euo pipefail`，mining 脚本天然支持续跑。

## 2026-05-11 — Phase 3 训练：schema 升级 + LoRA sweep + 锁推荐 adapter

一天落工具链 + 数据 schema 锁定 + 端到端 smoke，跨夜 7h 跑完 sweep（iters × lr 共 6 run）。据此追加 [`§5`](DECISIONS.md) 锁推荐 adapter。

### 功能

|item|说明|
|---|---|
|[`§4`](DECISIONS.md) schema 升级|SFT 用 OpenAI `tool_calls` JSON 格式，与 Qwen2.5 chat template + Ollama 解析器 + agent_engine 全链路对齐|
|formatter 重写|新 schema + tolerant args parser（救回 cast_vote 含中文 `或` 的 ~500 条样本）|
|数据重生|7B 1212→962 / 32B 1052→802，drop ~24% fallback|
|`train/` 目录|`lora_config.yaml` (q/k/v/o, rank 16) + `train.py` + `eval_smoke.py` + `sweep.py`|
|端到端 smoke|30-iter on 4-bit Qwen2.5-7B：loss 收敛到 0.001，4 项 metric ≥95%|
|sweep 6 run|iters {50,200,600} + lr {1e-5,1e-4,5e-4}|
|[`§5`](DECISIONS.md)|锁推荐 adapter = `runs/sweeps/iters/200/`（= BASE 配置）；layers/rank 推迟到 Phase 5 真测后再决定|

### 技术

- **为什么不能 text-only**：Ollama 只认 `<tool_call>` JSON 块；schema 不对齐 → 训完模型 emit 不出 tool_call event。
- **schema 单源**：formatter 直接 import `agent_engine.scenario._resolve_tool_defs`，scenario 改训练数据自动跟随。
- **底座选 4-bit 预量化版**：HF 直拉免 convert，smoke peak mem 12 GB。
- **sweep 现场降规模**：原 plan 16 runs 在 M4 Pro 上要 60h+，缩到 6 runs（核心 2 dim） ~8h。

四个 sweep 关键发现：

|发现|说明|
|---|---|
|iters 全饱和|50 / 200 / 600 三档 loss 全收敛、4 项 metric 全 100%——schema 信号高度可压缩，50 iter 已学透|
|lr 只 5e-4 劣化|1e-5 / 1e-4 全 100%；5e-4 arg_value 76%——sweep 唯一 differentiating evidence|
|BASE = 最优|刚好等于既有 baseline——没调出更差也没调出更好|
|fast proxy 饱和|eval_smoke 只能告诉我们 "学透"，不能告诉我们 "memorize vs generalize"，要 Phase 5 端到端跑才知道|

### 取舍

- 选 OpenAI tool_calls 格式而非把字面量写 content——跟主流框架对齐，换框架零改。
- drop fallback 样本而非补占位——保留会教模型 "重复 instruction 文本" 弱信号。
- sweep 实测后不扫 layers/rank——fast proxy 已饱和，再扫信息收益低；触发条件锁在 §5。

## 2026-05-11 — Phase 4 部署：adapter → GGUF Q4_K_M → Ollama

一次性跑通 fuse → convert → quantize → ollama create → smoke，wall clock 7 min。`agent-sft-qwen` 在 Ollama 注册，`agent_engine` 通过环境变量切换零成本。

### 功能

|item|说明|
|---|---|
|[`deploy/`](deploy/)|`Modelfile`（与 qwen2.5:7b 1:1 复刻）+ `build.sh`（三步幂等）+ `deploy.sh` + `smoke_test.py`|
|[`§6`](DECISIONS.md)|锁 Q4_K_M 量化 + Modelfile 1:1 复刻|
|产物大小|fused fp16 14 GB → F16 GGUF 14 GB → Q4_K_M GGUF **4.4 GB**|
|HTTP smoke|`/api/chat` 返回 parsed `tool_calls`，Ollama 解析器原生识别 `<tool_call>`|
|端到端 smoke|跑全 8 step trajectory，抓到 10 个 tool_call event，工具集全覆盖|

### 技术

- **llama.cpp 引入**：workspace 外建 `~/Tools/llama.cpp/`，独立 `.venv`，只 build `llama-quantize` target。
- **`mlx_lm.fuse --dequantize` 必须**：4-bit 底座 fuse 时 LoRA 加不进量化网格，要先 dequantize 到 fp16。
- **Modelfile 1:1 复刻**：不写自定义 jinja，仅替换 `FROM` 行；颗粒 = `ollama show --modelfile` 输出。
- **中文 args 验证**：`cast_vote(option="追加")` 全链路 UTF-8 穿透无损。

### 取舍

- 量化锁 Q4_K_M 而非 Q5/Q8——跟 baseline 同量化轴比 SFT 信号差，是 Phase 5 信号归因前提。
- `deploy/build/` 不入 git（18 GB 全本地）——build.sh 是重生指南，新机器 ≤10 min 可重生。

## 2026-05-11 — agent_engine 公开面直连 + transcript typed 升级

清理 Phase 2 为了快跑直接 import 私有 helper 的负债。**Step 1**：把 `from evals.metrics.nudge import _私有` 切到 `from agent_engine import Result, Scenario, TurnView`。**Step 2**：transcript 升级到 6 个 typed dataclass，三脚本切 `isinstance(...)` 派发；500 个历史 envelope 一次性脚本迁移。详 [`§7`](DECISIONS.md) + [`§8`](DECISIONS.md)。

### 功能

- 三脚本只剩 1 个跨项目公开 import（`classify_failure_mode`），私有 import 降到 0。
- `extractor / synthesize / formatter` 走 `isinstance(e, SpeakerEntry/...)` + 直接字段访问。
- §7 阶段留 shim 让旧测零修改 pass，§8 阶段不再需要的 shim 同期清掉。
- 500 envelope 注入新 schema 字段，秒级跑完。
- smoke：5 envelope 出 21 triples，与 Phase 2 历史 yield 同序。

### 技术

- **解读权归位**："transcript 怎么变成 ToolCall" 是 schema 一部分住 agent_engine；"failure mode 分类" 是 evals/sft 视角语义判断留在原处。
- **`Triple.context` 类型**：`list[dict]` → `list[TranscriptEntry]`。

### 取舍

- shim 续命而非删旧测——plan 硬约束是 "测试零修改 pass"。
- `classify_failure_mode` 不上提到 agent_engine——是 evals 视角的语义判断，上提会污染关注边界。
- 500 envelope 选迁移而非重跑——迁移秒级，重跑要小时级 LLM 成本。

## 2026-05-13 — Phase 5.A 端到端 baseline 120-batch

跑完三模型 × 10 seed × 4 task 对比。**主 batch** 13h28min，120/99/21 ok/failed——21 个失败集中在 32B agent 任务（subprocess timeout 不够）+ 1 条 7B 因模型生成非法 kwarg 让 handler 崩。**补跑 batch** 14h48min，17 runs 串行 + 加超时 env override，全 OK。最终 119/120 cell；7B 那 1 条因 evals 评测脆弱性永久排除。

### 功能

|item|说明|
|---|---|
|聚合输出|[`phase5-3model-comparison.md`](eval/baselines/phase5-3model-comparison.md)，4 task × 3 model + 三轴 breakdown|
|119/120 cell|11 cell 全 n=10；例外 7B nudge_fire_rate n=9|
|`run_baseline.py` 三个补丁|注入 `AGENT_ENGINE_MODEL` env（让 agent 子进程跑对模型）+ `sys.executable` 替换 `"python"`（本机无 python 命令）+ `agent_engine_run.py` 加 `AGENT_ENGINE_RUN_TIMEOUT` env override（默认 600s 不够 32B agent 任务）|

### 技术

- **wall clock**：主 + 补 ≈ **28h** 真跑批；32B agent 单 run nudge_fire_rate ~62min、agent_traj ~43min。
- **`aggregate_seeds.py` 不 dedupe**：按 (task, model) 直接 mean，重复 seed 会污染。绕路：先脚本 dedup 写干净 index 再聚合。
- **评测脆弱性**：7B seed=3 自发输出 `tool=cast_vote(...)` 当 kwarg → handler `TypeError` 挂——留作 evals 自己的 lesson。

### 取舍

- 保留 3 处工程补丁不回滚——是 "让 102 条 agent-path 数据正确" 的前提，不是 QoL 改进。
- 接受 7B n=9 而非补到 n=10——补跑需先修 agent_engine tool dispatch，跨项目改动 + 中心问题判定不依赖单一 seed。
- 两 batch 串行而非并行——Metal 后端跨进程并行可能 OOM 或 trash cache。
- `aggregate_seeds.py` 不补 dedup——一次性诊断脚本，通用化等真有第二个消费者再说。

## 2026-05-13 — Phase 6 反思 + v1 结案

按预先锁的三阈值判定 → 全过 → 中心问题答 "能且条件清楚" 落定。README 加 §"Lessons learned"，[`§9`](DECISIONS.md) v1 结案 ADR，面试叙事数字填实。

|三阈值|实测|
|---|---|
|nudge gap closure ≥50%|**57.3%** ✓|
|BFCL 回归 ≤5%|**1.16%** ✓|
|MMLU 回归 ≤3%|**2.09%** ✓|

### 功能

- README §Lessons learned：Phase 5 数字一览 + 三个回答 + v2/v3 候选取舍表 + "硬币背面" 段落（task_success 反超 / trajectory 退化 / missed→wrong_tool 转化）。
- 面试叙事数字填实："X% → Y%" → "0.739 → 0.645"。
- [`§9`](DECISIONS.md)：v1 结案 ADR，含中心问题答 + 候选 status update + 工程补丁状态 + 评测脆弱性 followup 交接。

### 技术

- **阈值预先锁后判定**：plan §6.1 + §5 触发条件早就锁好，Phase 5 跑完只是代入数字——避免事后拟合。
- **二阶证据写进 Lessons**：task_success 反超 / trajectory 退化 / missed→wrong_tool 转化 / panel 反向 / retrieve_docs 100% 这 5 件 surprise 不影响阈值判定，但驱动 v2 候选取舍。
- **§5 layers/rank 未触发**：gap closure 57.3% > 50% → §5 status 维持 accepted。

### 取舍

- v3-D（多 supervision 信号 superset）摘牌——v1 暴露的是 supervision 质量偏（panel 反向、retrieve_docs 100%）不是数量不足，加新信号桶只会重复 v1 的偏。
- v2-A DPO 暂留——v1 核心是分类问题不是偏好问题；但若 v2-B/C 跑完仍有 "两候选风格不一" 场景 DPO 仍适用。
- v3-B HF Hub release 提前——Model Card 内容已成型，社区拿到 "7B SFT closing 57% of 32B gap" 可下载产物，portfolio 信号最强。
- v3-A 14B 暂留——SFT 7B 已在 task_success 反超 32B，先 scale to 14B 是回避当下信号。
- 不在本里程碑补 v2/v3 具体 plan——本期只完成 v1 收尾。
