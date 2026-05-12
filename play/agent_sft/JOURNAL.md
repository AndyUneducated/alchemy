# Journal

每条里程碑一段：`## YYYY-MM-DD — 标题`，正文必含 **功能** + **技术** 两节，**取舍** 节按需追加并反链 `DECISIONS §N`。架构决策见 [`DECISIONS.md`](DECISIONS.md)。

## 2026-05-09 → 05-10 — Phase 0 立项 + Phase 1 baseline 工具链

两日合一条里程碑：**05-09** 锁 Phase 0 三件框架决策（中心问题 = nudge-grounded SFT / ceiling reference 弃 GPT-4o-mini 改 Qwen2.5-32B / 立项稿从 "v1 完成态" 重写为 "v1 + 演化路径"），**05-10** 紧接着按 `play/.cursor/plans/phase_1_baseline_impl_a7317e50.plan.md` 一次性走完 Phase 1 baseline 工具链前 6 段（4 项度量 task + 2 个 require_tool 密集 scenario + 失败模式 taxonomy + OllamaLM 多 seed wiring + agent_sft 消费侧 runner / aggregator）。立项承诺「supervision 来源是自有 infra，复现门槛即护城河」（核心由 [`DECISIONS §1`](DECISIONS.md) ADR 锁定）在 Phase 1 工具链落地时全程贯彻——agent_sft 只持消费侧胶水，evals 永不出现 agent_sft 字样，所有度量按 per-scenario / per-tool / per-failure-mode 三轴 + 多 seed 报告。**实跑 80-batch + 最终对比报告留 1.G** 单独里程碑（待用户拉 7B 后跑批）。

### 功能

|item|状态|说明|
|---|---|---|
|中心问题 + 七阶段路线图|✅|README 顶部定义"让 nudge-fire rate 在 in-dist 上显著降，OOD 不回归"；硬件 (M4 Pro 48GB) 剥离到独立"v1 工程约束"节；Phase 0-6 + mermaid 上下游关系图配齐|
|度量四项 + 报告维度|✅|nudge-fire rate / trajectory score / BFCL slice / general regression 定义先于训练；per-scenario / per-tool / per-failure-mode 三轴 breakdown + 多 seed ≥3 报 mean±std|
|技术栈 + 可移植性 + non-goals|✅|底座 / 训练框架 / 量化部署 / 评估 / 硬件 五维收敛；HF safetensors source of truth；non-goals 拆 ❌ 永久禁区 + ⏸ v1 边界（v2/v3 候选）|
|v1/v2/v3 演化路径|✅|7 候选清单（DPO / on-policy / 失败模式 taxonomy / 14B 升级 / HF release / 技术报告 / 多信号 superset）+ 各自触发条件|
|Phase 1 ceiling reference 替换|✅|"GPT-4o-mini" → "Qwen2.5-32B-Instruct (Ollama)"；Phase 5 三组对比 base 7B / SFT 7B / 32B 原版|
|面试叙事脚本|✅|故事点 = "7B SFT 在自己 trajectory 上追 32B 同族原版"，强调全本地零闭源依赖|
|`evals/tasks/nudge_fire_rate.py`|✅|消费 `metrics/nudge.py`；7 scenario gold；by_scenario / by_tool / by_failure_mode 三轴 breakdown|
|`evals/tasks/bfcl_slice.py`|✅|BFCL `simple_python` 50 例；4 项指标（exact_match / name_match / arg_set_f1 / arg_value_match）；AST match 度量内联|
|`evals/tasks/mmlu_slice.py`|✅|6 subject × 16 例 = 96 例；accuracy + accuracy_by_subject 嵌套子组；MCQ 字母解析内联|
|`agent_engine/scenarios/{code_review,tool_chain}.md`|✅|新增 2 个 require_tool 密集 scenario（4 agent × 8 turn + 1 agent × 5 turn 强工具链）|
|`evals/cli.py::parse_model_spec` `@seed=K` 后缀|✅|`ollama:<model>@seed=42` → `OllamaLM(seed=42)`；`lm.name` 保留后缀让 EvalResult.model 多 seed 可分组|
|`agent_sft/eval/run_baseline.py`|✅|2 model × 10 seed × 4 task = 80 runs；argparse + `nargs='+'` + `--tasks` choices 校验 + `--dry-run`；单 run 崩不中断 batch|
|`agent_sft/eval/aggregate_seeds.py`|✅|读 `evals/runs/index.jsonl` → group by (task, model_clean) → 标量 mean ± std + 嵌套子组按 dot-path 展开 → markdown 报告|
|端到端 smoke|✅|3 seed × 1 task (mmlu_slice) × 32B 跑通；`accuracy=1.0000 ± 0.0000` + efficiency 子组都正确填入；aggregator 按 (mmlu_slice, ollama:qwen2.5:32b) group n=3|

### 技术

|item|说明|
|---|---|
|中心问题选型|nudge-grounded SFT——4 候选中选 C（详见 DECISIONS §1），supervision 来源是自有 infra，复现门槛即护城河|
|训练框架选型|MLX-LM（详见 DECISIONS §2）——Apple Silicon 原生最优；`mlx_lm.lora` / `mlx_lm.fuse` / `mlx_lm.convert` 三步 CLI，KISS|
|底座 + ceiling + 扩展性 + 工具链落点 (ADR 已撤)|Qwen2.5-7B-Instruct 底座 / Qwen2.5-32B-Instruct ceiling / v1+v2/v3 演化路径 / Phase 1 工具链 6 开放点（代码归属 / 度量分层 / BFCL-MMLU 接入 / 多 seed wiring / 失败模式 taxonomy / bfcl_slice schema）|
|度量函数分层|`metrics/nudge.py` 独立模块（半通用 + 复杂分类，与 trajectory.py 同档）；BFCL AST match / MMLU MCQ acc 内联（YAGNI 等第二消费者再抽）|
|失败模式 3 桶 + 1 占位|missed / wrong_tool / wrong_args（后者当前 placeholder 归 wrong_tool；agent_engine dispatch error 路径补 `{ok: false}` event 后启用，留给 Phase 5）；by_failure_mode 表头永远 3 桶稳定 schema|
|`agent_engine/discussion.py` require_tool 观测面扩展|`_run_turn` 把 tracer.drain() + artifact.drain_events() 合并喂 `_called_tool`，让 `require_tool` 对非 artifact 工具（retrieve_docs）也生效（详见 [`agent_engine/DECISIONS.md §12`](../agent_engine/DECISIONS.md)）；本期 2 个新 scenario 强依赖此修复|
|多 seed wiring 取舍|不改 runner schema 加 seed 字段（跨子项目改动 + EvalResult 破坏向后兼容），改走 spec 后缀 + lm.name 编码方案；最小侵入，仅改 `parse_model_spec` 与 `OllamaLM`|
|显存 / 部署一致性|32B Q4 ≈18GB 与 7B Q4 ≈4GB 在 48GB 共存舒适，必要时串行；同家族 chat template 让 `BACKEND=ollama` + 改 `MODEL` 即可切换，零适配成本|
|测试覆盖|`evals/tests/` 460 + `agent_sft/tests/` 17 = 477 条全过；新增覆盖：27 bfcl_slice + 18 mmlu_slice + 8 cli_spec(@seed) + 17 aggregate_seeds(pandas 版) + 16 nudge metric + 12 nudge fire rate score + 3 new scenario smoke|
|未决问题|MLX-LM → GGUF 二段转换路径（直出 vs 经 HF safetensors 中转）推迟到 Phase 4 真撞上再 ADR|

### 取舍

- 放弃"经典 tool-calling LoRA on xLAM/ToolACE"（[`DECISIONS §1`](DECISIONS.md) 选项 A）——执行简单但面试无差异化，对 senior portfolio 是负优化。
- 放弃 Llama-3.1-8B 底座 + axolotl / Unsloth 训练编排（[`DECISIONS §2`](DECISIONS.md) 选项 C/D）——前者 Mac 不是主战场，后者抽象层抬学习成本；MLX-LM 三命令链路 + Qwen2.5 在 7B tool-call 段位基线更强。
- 放弃"7B SFT 追 GPT-4o-mini"行业锚点 + Qwen2.5-72B 作 ceiling——换得全本地可复现 + 同家族跨规模对比；72B Q4 ≈42GB 余量太紧 ROI 不划算。
- 拒绝"立项稿 v1 完成态"叙事 + "完整写 v1 + v2 + v3 三套 README" 路线——v1 真完成时面试官追问"下一步"无干脆答案，但 v2/v3 没数据写成空想；候选清单 + 触发条件足以传达"有路线图"信号，遵循 `workshops.mdc` "抽象引入滞后于第二个具体案例"原则。
- 拒绝引入 `lm-evaluation-harness` 集成 BFCL / MMLU——自实现各 <100 行远低于跨重型框架适配 + 钉版调试成本；与 [`DECISIONS §2`](DECISIONS.md) "不引入新 tooling" 原则一致。
- 拒绝 `metrics/{bfcl,mmlu}.py` 独立模块——单一消费者 + 函数简单（AST parse / MCQ 字母提取各 ~20 行），独立模块属"为抽而抽"；遵循 `workshops.mdc` "抽象引入滞后于第二个具体案例"原则。
- `wrong_args` 失败模式桶 Phase 1 当 placeholder 归 wrong_tool——agent_engine artifact handler error 路径不发 event，无法仅靠 transcript 区分"调对工具被拒" vs "调了别的工具"；显式留桶让 by_failure_mode 表头跨 run 稳定，启用推迟到 Phase 5。
- 80-batch 实跑 + 对比报告分到独立 1.G 里程碑——本次留可复现的工具链 + smoke；用户拉 7B 后跑 `python play/agent_sft/eval/run_baseline.py` 即可生成 `baselines/qwen2.5-7b-vs-32b.md` 真实数据。

## 2026-05-10 — Phase 2 流水线落地 + 57 条 demo train set 交付

一天内做完三件事：搭流水线、跑 pilot 撞瓶颈、走 Approach B 解锁产出。最终交付 47 train + 10 val（synthesize 路径），Phase 3 可启动。

跑批 / 实验时序：① 7B × 6 envelope (max_retries=1) → 1 triple；② 同 batch max_retries=2 实验 → 仍 1 triple，排除"重试次数"；③ 32B × 3 envelope 对照 → recovery 从 7B 的 3% 跳到 25%，底座 capability 才是 recovery 率主因；④ 走 synthesize（per-fire 配对，corrected 用 instruction 模板）→ 同样 12 envelope 出 57 triples，命中 plan 原估算 5/env。

### 功能

|item|说明|
|---|---|
|`data/` 5 脚本 + 18 测试|`mine_triples` (子进程跑 agent_engine) / `extractor` (真 recovery 配对) / `synthesize` (per-fire 配对 + 模板 corrected) / `split` (per-scenario 末 20% run_id → val) / `formatter` (MLX-LM chat schema)|
|`data/triples/` 产物目录 + README|与 `eval/baselines/` 平行布局；README 含 4 步 regen + 两种 triple 来源对照表 + pilot 时序|
|`agent_engine/config.py` env override|加 `AGENT_ENGINE_MODEL` env var (1 行)，让 7B / 32B mining 不改 scenario YAML 即可切换|
|`tests/conftest.py` 加 `data/` sys.path|测试零 path 体操|
|`.gitignore` 加 `data/triples/runs/` + `*.jsonl`|README.md 仍提交|
|测试增量|`agent_sft/tests/` 17 → **70**（+12 extractor + 13 formatter + 11 split + 18 synthesize）|
|首批 train/val|7B max_retries=2 × 12 envelope → synthesize → 57 triples → 47 train + 10 val（per-scenario 末 20% 切分正常生效）|

### 技术

|item|说明|
|---|---|
|两条 triple 路径，schema 共享|`extractor.py` 要 first-fail + later-success（真自纠语义，pilot 测得 yield 0.17/env）；`synthesize.py` per-fire (yield 4.75/env)。两者出同样 `Triple`，下游 split / formatter 不感知|
|为什么默认 synthesize|extractor 路径在 7B 上 yield 太低（recovery ~3%）；synthesize 用 step.instruction 里字面 `tool(args)` 模板造 corrected (fallback：通用 wrapper + 完整 instruction)，把 yield 拉到 fire rate 上限。compute 比 32B mining 省 ~14x|
|与 `DECISIONS §1` 关系|synthesize 严格说仍是"自家 7B 失败素材 + 自家 scenario 模板"，没引入第三方教师，与"自有 infra 生数据"承诺不冲突|
|样本格式|F1 only（input 不含 nudge），训"看到原 instruction 一次到位"|
|context 截取|`max_recent=6`，与 `code_review.md memory.max_recent` 一致；典型 user content 100-400 token|
|seed handling|不改 agent_engine——run_id 只是 envelope 文件命名 + split 索引，靠模型自然采样得 trace 多样性|
|`_extract_call_template` 设计|regex `\\b{tool}\\s*\\(` 起始 + paren 平衡扫描；7 测试覆盖跨行 args / 中文引号 / 不平衡 paren / word boundary 等边界|

### 取舍

- 不再写新 ADR：用户 2026-05-10 决策"本项目不再写 ADR"。`DECISIONS §3`（含早期"瓶颈在方法学"过度判断）作历史保留，不在 ADR 文件内修正；本里程碑承担 Phase 2 完整 narrative。
- 承认 32B 单 envelope (n=2 fires) 对照"换底座也不行"是过度判断 → n=20 fires 重跑修正。"早失败 → 早跑实验 → 早修正"的过程留在 JOURNAL 不掩盖。
- 选 synthesize 而非 32B mining：经济性碾压 + 训练目标更干净（无 7B 偶然 success 的"text 说 X 但 tool_call 是 Y"噪声）。代价是 corrected 是模板，**Phase 3 训完若模型只会模板复读、不泛化，再切回真自纠或蒸馏路径**。
- 接受当前 57 triples 仅"demo viable"——`README.md L43` 原 plan 是 ≥1k。synthesize 路径下 scale 到 1k 仅需 ~210 envelope ≈ 4h overnight，是否启动等 Phase 3 smoke 训练后看效果再决定（YAGNI：不预投 4h compute，等 Phase 3 信号回来再扩）。
- 拒绝 agent_engine 加 `--seed` flag——跨 Engine + Agent + ollama_client 三处改动 + EvalResult 兼容性破坏，与最小侵入冲突。
- 接受 `tools` JSON schema 不进 F1 system message（暂只列工具名 comma list）——MLX-LM `tools` 字段支持还在演进，Phase 3 真训练时再按当时 mlx-lm 版本对齐。

## 2026-05-10 — fast scenario 副本：mining 提速 35%

为后续可能的 1k 数据 scale-up 做提速储备。**上游 `agent_engine/scenarios/*.md` 不动**（baseline eval 已记录 max_retries=1 的对照数据，改上游会破坏可比性），新建 `data/scenarios/{tool_chain,code_review}_fast.md` 副本，只服务 mine_triples.py。

### 功能

|item|说明|
|---|---|
|`data/scenarios/{tool_chain,code_review}_fast.md`|`max_retries: 1→0` / `max_tokens: 200→80` / 删 open + finalize 两个 moderator 步 / vdb_dir 路径 +1 层|
|`mine_triples.py` 默认切 fast，加 `--upstream` flag|`_scenario_path()` 助手按 flag 选择副本 vs 上游；CLI 默认 `data/scenarios/<name>_fast.md`|
|`extractor.py` / `synthesize.py` 同步加 `--upstream`|抽三元组的 scenarios_root 必须与 mining 一致——fast scenario 删了 open 后 turn_idx 比上游少 1，混用会让 derive_expected_turns 错位、yield 归零|
|`extractor.py` 暴露 `resolve_scenario_path()`|synthesize.py 单点 import，避免两边 path 选择逻辑漂移|
|顺手修 `mine_triples.py` 相对 out-dir bug|`Path(args.out_dir).resolve()`——之前传相对路径会被 subprocess cwd=PLAY_DIR + agent_engine CLI abspath 误叠加成 `play/play/...`，2 envelope smoke 暴露|
|2 envelope smoke (7B, run_id=99)|tool_chain_fast 36s / code_review_fast 49s → 平均 **42s/env**（vs upstream ~65s/env，-35%）；synthesize 出 8 triples = 4 triples/env，与 upstream 4.75 同量级|

### 技术

|item|说明|
|---|---|
|为什么不直接改上游 scenario|baseline eval `nudge-fire-rate qwen2.5-7b.md` 已按 max_retries=1 跑过；改上游让历史对照失去意义。隔离 fast 副本零成本，agent_engine 也是被多项目复用的|
|`max_retries: 0` 安全性|engine `_run_turn` (discussion.py L96) `range(max_retries+1)` → 0 时只 1 attempt；nudge fire 由 synthesize 从 transcript 推断（first attempt 没调对工具就 fire），与 engine retry 行为解耦|
|为什么 1k 提速估算 -50min 而非更多|tool_chain 提速明显（70s→36s, -49%），code_review 受 4 agent + PR 描述 long context 限制，max_tokens 截断帮助有限（67s→49s, -27%）。1k 估算：250 fast envelope ≈ 175 min（vs 211 upstream envelope ≈ 228 min）|
|测试|无需新增——fast scenario 是数据，不是逻辑；现有 70 个 agent_sft 测试 + 465 evals 测试全过（535/535）|

### 取舍

- 不为 fast 副本写新单元测试——`max_retries=0` / `max_tokens=80` / 缺步骤都是 scenario YAML 数据，逻辑测试覆盖在 `test_extractor` / `test_synthesize` 已用 fixture 验过；2 envelope smoke 是端到端验证。
- 加 `--upstream` flag 而不是把 fast 当唯一选择——保留 baseline 复现路径（任何时候 `python mine_triples.py --upstream` 就能拿到与 Phase 1 baseline 同 max_retries=1 的 envelope）。
- 不做更激进的"minimal scenario"（1 agent / 全 require_tool / 删 moderator）——工程开销 > 时间收益，且失去 code_review 多 agent context 多样性；fast 副本已把 1k 拉到 overnight 可行（~3h），暂不上 minimal 方案。
- `synthesize` 在 fast envelope 上对 `code_review` 的 fire 计数（6）多于 engine warning 数（4）——_attempt_called_required 与 engine 对"tool 事件 speaker 为空"判定不同，是同样作用在 upstream batch 的既有特征，不在本里程碑修；训练若发现噪声样本污染再回头看。

## 2026-05-10 — Phase 2 收尾：1k × 2 模型双批数据交付

跨夜 ~17h 跑完 7B / 32B 各 250 envelope，两份独立 SFT 数据集落地 repo（`runs_1k_fast_{7b,32b}_r0_124/` + `*_1k.jsonl`），Phase 2 完结。Phase 3 训练随时可起。

### 功能

|item|说明|
|---|---|
|7B 数据集|250 envelope（fast scenario / `max_retries=0` / run_id 0-124）→ 1212 triples → 966 train + 246 val；wall clock ~7.5h|
|32B 数据集|同参数对照批，`AGENT_ENGINE_MODEL=qwen2.5:32b`；250 envelope → 1052 triples → 842 train + 210 val；wall clock ~9.5h|
|两份并存而非合并|文件命名带 `_7b_` / `_32b_` 标签，Phase 3 既可单跑也可拼接做 ablation|
|`.gitignore` 反例规则|加 `!*_1k.jsonl` 让 1k 派生入 git，默认 `runs/` 与无 `_1k` 后缀产物仍忽略；`check-ignore` 验证两侧都对|
|`data/triples/README.md` 大改|文件清单加"是否入 git"列；新增 §Phase 2 终交付（1k × 2 模型对比表 + 选择指引）；重生命令拆"1k 终交付"vs"smoke"两段；旧 pilot 表降级为 §历史遗留|

### 技术

|item|说明|
|---|---|
|为何把数据直接 commit（27 MB）|raw envelope ~5 MB / 模型，jsonl ~10 MB / 模型——远低于 Git LFS / Releases 阈值；可重生但 17h compute 成本高，repo 直存让 Phase 3 / 后续 ablation 零等待|
|orchestrator 韧性|`caffeinate` 防睡眠 + `stdbuf -oL` 实时 log + bash `set -euo pipefail`；中途几次系统中断后均能从断点续跑（`mine_triples.py` 的 `--run-ids` 与 `--out-dir` 配合天然支持续跑）|
|7B vs 32B 实测对比|7B yield 4.85/env vs 32B 4.21/env（synthesize 路径下底座差异变小，因 corrected 是模板而非真 recovery）；32B 在 wrong_tool 占比 27% vs 7B 10%——32B 更愿意"调一个错的工具"，给 hard sample 用|
|单条 triple 经济性|7B ~22s/triple，32B ~32s/triple；7B 约 1.5x 性价比，但 32B 失败模式分布更宽，Phase 3 若 wrong_tool 召回低可拌入|
|val 切分一致|两份均按 `run_id ∈ [100, 124]` → val（per-scenario 末 20%），Phase 3 / 5 跨数据集对比时 val 行为可比|

### 取舍

- 选 Plan A（直接 commit data）而非 LFS / Releases / 压缩——27 MB 在 git 物理友好区间，省一层基础设施；如未来 Phase 3 + 数据集再翻倍再考虑 LFS。
- 不合并成一个 `train.jsonl`——Phase 3 的 ablation 信号需要"模型来源"标签；运行时 `cat train_7b_1k.jsonl train_32b_1k.jsonl > train_all.jsonl` 是一行的事，反向拆分则不可能。
- 不在本里程碑改 synthesize 的"corrected 来自模板"——双批数据落地是 Phase 2 的 contract，模板 vs 真 recovery 的 trade-off 已在 5-10 中段里程碑写明，留给 Phase 3 训完看效果再决策（与 `extractor.py` 保留同源理由）。
- 不再补 OOD 数据集副本进 repo——继续复用 `play/evals/data/bfcl_slice/gold.jsonl`，Phase 5 复测时 `python -m evals run --task bfcl_slice --model ollama:agent-sft-qwen` 直接吃。

## 2026-05-10 → 05-11 — Phase 3 训练：schema 升级 + LoRA harness + sweep 闭环 + 推荐 adapter 锁定

按 [`plans/phase3_sft_schema_and_sweep_b2058b8a.plan.md`](../../.cursor/plans/phase3_sft_schema_and_sweep_b2058b8a.plan.md) 一日内落工程基础设施 + 数据 schema 锁定 + 端到端 smoke，跨夜 ~7h 跑完 sweep `iters × lr` 共 6 run 自动产 [`runs/sweeps/REPORT.md`](train/runs/sweeps/REPORT.md)；据此追加 [`DECISIONS §5`](DECISIONS.md#5-phase-3-推荐-adapter-锁-base-配置layersrank-sweep--真效果决断推迟到-phase-5) 把"用哪个 adapter 进 Phase 4"和"什么信号才算 SFT 真生效"两件事锁掉。Phase 3 README 验收 4 项（MLX-LM QLoRA on Qwen2.5-7B / 小规模 sweep / adapter checkpoint / loss 曲线）全过。

### 功能

|item|说明|
|---|---|
|[`DECISIONS §4`](DECISIONS.md)|SFT target schema 升级为 OpenAI `tool_calls` JSON-string + 顶层 `tools`，与 Qwen2.5 native chat template + Ollama 解析器 + agent_engine `tool_call` event 全链路对齐；supersedes §3 "F1 only 把 corrected_response.content 当 assistant target"|
|[`data/formatter.py`](data/formatter.py) 重写|新 helper `_call_template_to_args_dict` (strict ast + tolerant kw fallback) + `_load_tool_defs` (复用 `agent_engine.scenario._resolve_tool_defs` + `ArtifactStore.build_tool_defs`)；assistant message 改 `tool_calls=[{id, type, function:{name, arguments<JSON-string>}}]`；F1 sample 顶层加 `tools=[...]`；fallback 类样本 drop|
|`data/triples/{train,val}_{7b,32b}_1k.jsonl` 全部重生|7B 1212 → 962 (kept 79.4%，drop 250 retrieve_docs no-template)；32B 1052 → 802 (kept 76.2%，drop 250)；split 后 7B 766+196 / 32B 642+160；mining envelope / `triples_*_1k.jsonl` 不动|
|[`tests/test_formatter.py`](tests/test_formatter.py) 重写|13 → 32 条；覆盖 tool_calls schema / arguments JSON-string / tools 数组 / role-filtered moderator-only / cast_vote 中文 `或` tolerant fallback / drop 计数 CLI 集成|
|[`requirements.txt`](requirements.txt) + [`train/`](train/) 全新目录|`mlx-lm[train]>=0.20.0` + `huggingface-hub`；train/ 含 `lora_config.yaml` (q/k/v/o + rank 16 + scale 2.0 + dropout 0.05) + `train.py` (mlx_lm.lora wrapper) + `eval_smoke.py` (`<tool_call>` 解析 + 4 项指标 fast proxy) + `sweep.py` (控制变量)|
|端到端 smoke 通过|30-iter LoRA on Qwen2.5-7B-4bit (q/k/v/o, 8 层, lr=1e-4, batch=2)：train_loss 1.325→0.001 / val_loss 1.766→0.004 / emit / name / arg_set 100% / arg_value 95% (20 sample)；wall clock 6 min train + 1 min eval|
|sweep 6 run 全部跑完|[`runs/sweeps/iters/{50,200,600}/`](train/runs/sweeps/iters/) + [`runs/sweeps/lr/{1e-05,0.0001,0.0005}/`](train/runs/sweeps/lr/) 每 dir 含 adapter + `train.log` + `train_metrics.json` + `eval_smoke.json`；顶层 [`results.json`](train/runs/sweeps/results.json) + 自动生成 [`REPORT.md`](train/runs/sweeps/REPORT.md)|
|[`DECISIONS §5`](DECISIONS.md) 落地|锁 Phase 4 推荐 adapter = [`runs/sweeps/iters/200/adapters.safetensors`](train/runs/sweeps/iters/200/)（= sweep `BASE` 配置）；明确"layers / rank dim 推迟"的触发条件是 Phase 5 真测 `(SFT 7B − base 7B) < (32B − base 7B) × 0.5`|
|测试 + gitignore|`agent_sft/` 70 → **89** 全过；`evals/` 465 仍全过；总 554；`.gitignore` 加 `play/agent_sft/train/runs/`，adapter / log / sweep 产物默认本地，REPORT.md 入 git（与 `eval/baselines/` 同策略），单独保留推荐 adapter 入 git|

### 技术

|item|说明|
|---|---|
|为什么不能 text-only|F1 v1 schema 教模型说"好的我现在调用 retrieve_docs(...)"文本，下游 Ollama function-call 解析器只认 `<tool_call>{...}</tool_call>` JSON 块——schema 不对齐 → 训完模型 emit 不出 tool_call event → nudge-fire-rate 不降反升。详 [`§4 Decision`](DECISIONS.md#4-sft-target-schema-用-openai-tool_calls--顶层-tools-字段qwen25-native)|
|schema 单源策略|formatter 不重新定义工具 schema，直接 import `agent_engine.scenario._resolve_tool_defs` + `ArtifactStore.build_tool_defs`，与 runtime per-agent tool_defs 完全同源；scenario YAML 改 → 训练数据自动跟随，零 drift|
|tolerant args parser|strict ast.parse 失败时（如 cast_vote 模板含中文 `"X" 或 "Y"` 不是合法 Python），按 paren-aware 顶层逗号切 + `key=val` regex + 首字符串字面量回退；救回 308 条 7B + 177 条 32B cast_vote 样本（之前会因 SyntaxError 全 drop）|
|`--mask-prompt` 默认开|MLX-LM assistant-only loss masking，与 [TRL Qwen2.5 训练 template (PR #5522)](https://github.com/huggingface/trl/pull/5522) 同思想——梯度只作用在 `<tool_call>` 块，不被长 user prompt 稀释|
|底座选 4-bit 预量化版|`mlx-community/Qwen2.5-7B-Instruct-4bit` HF 直拉，免本地 `mlx_lm.convert`；自动走 QLoRA 路径；smoke 实测 peak mem 12.1 GB（48 GB 余量充足）|
|LoRA target keys = q/k/v/o|sft_hello toy task 用 q/v 够了；tool-call SFT 是结构性 + 风格性混合任务，挂全部 attention proj 跟 Hermes-Function-Calling V3 / Watt-Tool / xLAM 实战配置一致；可训参数 0.038%（2.88M / 7.6B）|
|sweep harness 复用 + 现场降规模|sweep.py 同 sft_hello 模具（每 sweep 跑 train.py + eval_smoke.py 子进程 → results.json → REPORT.md）；含 `--force` / resume 逻辑（已完成 `train_metrics.json` 自动跳过）。**实测降规模**：原 plan 4 dim × 4 值 = 16 runs 在 M4 Pro 上 ≈ 18s/iter，60h+ 远超 overnight；实跑 `iters [50, 200, 600]` + `lr [1e-5, 1e-4, 5e-4]` = 6 runs（核心 2 dim），≈ 8h 内可控。layers / rank dim 留 Phase 3.5（[`§5`](DECISIONS.md) 触发条件）|
|sweep 韧性实战|前次 sweep 跑到 `iters=200` eval ~50% 时 shell tracker drop 致进程被 SIGHUP；重启用 `nohup ... & disown` 全脱离后 PPID=1，跑全程未中断。`iters=50/200` 的 cached `train_metrics.json` 让重启省 ~1.2h|
|eval_smoke 是 nudge-fire-rate 的 fast proxy|不走 fuse → ollama → agent_engine 端到端 (~5 min/run)，直接 mlx_lm.generate + regex 解析 `<tool_call>` 块，~52s / 20 sample；4 项指标 (`emit / name_match / arg_set / arg_value`) 从松到严，Phase 5 真 nudge-fire-rate 复测前先用它选最佳 adapter|
|**关键发现 1：`iters` dim 全程饱和**|`iters` ∈ {50, 200, 600} 三档 `train_loss`/`val_loss` 全收敛到 0.00，4 项 metric 全 100%。意味着 mask-prompt + 短 `<tool_call>` 段让 schema 信号高度可压缩——50 iter (≈0.25 epoch) 已学透形态。这件事本身就是个有意义的 negative finding：**当前 fast proxy 无法 differentiate 50 vs 600**|
|**关键发现 2：`lr` dim 只 5e-4 劣化**|`lr=1e-5` / `1e-4` 全 100%；`lr=5e-4` train_loss 起 3.65（远高于其他配置的 0.28~1.02）→ 末 0.04 / val 0.12 / emit 95.4% / name 93.9% / **arg_value 76.0%**。第一次 mini-batch 就把权重推飞，200 iter 后部分恢复但 4 项 metric 整列下滑——sweep **唯一** differentiating evidence|
|**关键发现 3：推荐配置 = `BASE`**|`BASE` (iters=200 / lr=1e-4 / num_layers=16 / rank=16) 是 sweep 中的最优组合；恰好等于既有 baseline 配置——意味着没有"调出更好的"，但确认了"没调出更差的"。无须再跑 plan 的"选最佳配置主跑"步骤|
|**关键发现 4：fast proxy 在 schema 学习上饱和的方法学含义**|`eval_smoke` 4 项 metric 设计上是 "nudge-fire-rate 的 fast proxy"——但在 schema 信号充分时它**只能告诉我们"学透"，不能告诉我们"是 memorize 还是 generalize"**。Phase 5 用 [`evals nudge_fire_rate`](../evals/metrics/nudge.py) 端到端跑 base 7B / SFT 7B / 32B 三组才是真决断信号|

### 取舍

- 反链 [`DECISIONS §4`](DECISIONS.md) + [`§5`](DECISIONS.md#5-phase-3-推荐-adapter-锁-base-配置layersrank-sweep--真效果决断推迟到-phase-5) 全部决策。
- 选 schema B（OpenAI tool_calls）而非 C（字面量 `<tool_call>` 字符串写 content）——B 是 MLX-LM / TRL / OpenAI / Mistral 微调示例的正交方案，C 跳过 chat template schema 校验；B 数据集换训练框架零改，C 得重 format。
- drop 250 fallback 样本而非合成 placeholder 占位——用户 2026-05-10 决策"drop"；保留这部分会教模型"重复 instruction 文本"的弱信号，与"教 emit 正确 tool_call"目标冲突。
- LoRA target keys 不挂 MLP（gate/up/down_proj）——先验证 attention 全挂的基线，MLP 是 v2 候选；YAML 改 1 行即可扩。
- 实测 iter 成本远超 plan 预算 → sweep 现场缩到 6 runs，layers / rank dim defer——计算资源约束下"core dim 跑透 > 全 dim 跑半"；sweep 跑完后又确认 fast proxy 在 schema 学习上饱和，再扫 layers / rank 也分不出差异，**信息收益低**，触发条件锁在 §5（Phase 5 真测 SFT 不达 32B gap 50% 时再回头）。
- 不预先把 fuse → GGUF → ollama create 串到 sweep——5 min/run × 16 = 80 min 浪费在部署转换；Phase 4 选最佳 adapter 后单跑一次。
- 不把 32B 数据合并训练——保留模型来源标签便于 ablation；sweep 主跑用 7B（[`README.md` §"7B vs 32B 选择指引"](data/triples/README.md)）。
- sweep `adapters.safetensors` 默认不入 git——6 个 adapter 各 ~11 MB 合计 ~66 MB；本地可重生，HF Hub `mlx_lm.fuse --upload-repo` 是分享正解（[`§2`](DECISIONS.md) 已铺垫）。**例外**：[`runs/sweeps/iters/200/adapters.safetensors`](train/runs/sweeps/iters/200/)（= 推荐配置）入 git，让新 checkout 直接 unblock Phase 4 fuse。
- Phase 5 端到端复测 + Phase 1 baseline 80-run 仍 pending——两件事天然合并：跑 baseline 时把 SFT 后 7B 当第 3 个候选模型混进去即可，1 个 evals 入口同时拿 2 份数据。等 Phase 4 deploy 完后并行启动。

## 2026-05-11 — Phase 4 部署：adapter → GGUF Q4_K_M → Ollama tag agent-sft-qwen，两级烟测全过

按 [`plans/phase4_deploy_fuse_gguf_ollama_57e1914f.plan.md`](../../.cursor/plans/phase4_deploy_fuse_gguf_ollama_57e1914f.plan.md) 一次性跑通 fuse → convert → quantize → ollama create → smoke 全链路，单机 wall clock ~7 min（含 ~3 min build.sh 三步 + 4 min agent_engine 1 round 端到端验证）。Phase 4 README 验收项全过：`agent-sft-qwen` 在 Ollama tag 注册，`agent_engine` 通过 `AGENT_ENGINE_MODEL=agent-sft-qwen` 切换零成本，tool_call event 在多 step trajectory 中稳定触发。

### 功能

|item|说明|
|---|---|
|[`deploy/`](deploy/) 全新目录|`README.md`（三步流程图 + 行业对位 + 常见问题）+ `Modelfile`（与 `ollama show --modelfile qwen2.5:7b` 1:1 复刻 TEMPLATE + SYSTEM，仅改 `FROM`）+ `build.sh`（fuse → convert → quantize，幂等，`--force` 重跑）+ `deploy.sh`（ollama create，存在 tag 先 rm）+ `smoke_test.py`（HTTP `/api/chat` 断言 `<tool_call>`）|
|[`DECISIONS §6`](DECISIONS.md#6-phase-4-量化等级锁-q4_k_m--modelfile-11-复刻-qwen257b-template) 落地|锁 Q4_K_M（与 baseline `qwen2.5:7b` 同量化等级 → Phase 5 公平对比）+ Modelfile 1:1 复刻 + fuse 路径 `--dequantize → convert_hf_to_gguf.py → llama-quantize`|
|实测产物大小|fused fp16 14 GB / F16 GGUF 14 GB / Q4_K_M GGUF **4.4 GB** (实测 4460 MiB / 4.91 BPW)；Ollama list 显示 `agent-sft-qwen:latest` 4.7 GB（与 `qwen2.5:7b` 同 size 量级）|
|Step 5A 烟测|`POST /api/chat` 返回 parsed `tool_calls` 字段：`{"name": "retrieve_docs", "arguments": {"query": "项目代号 X 历史 commit", "top_k": 5}}`，Ollama 解析器原生识别 `<tool_call>` block|
|Step 5B 端到端烟测|`AGENT_ENGINE_MODEL=agent-sft-qwen python -m agent_engine scenarios/tool_chain.md` 跑全 8 step trajectory，transcript 抓到 **10 个 tool_call event** 覆盖 `retrieve_docs` / `cast_vote` / `propose_vote` / `append_section` / `write_section` / `finalize_artifact` 全工具集|
|`.gitignore` 加 `play/agent_sft/deploy/build/`|顶层 `build/` 全局规则已覆盖；显式加便于 grep + 表达 Phase 4 工件意图|
|README 项目结构 + Phase 4 段更新|`deploy/` 状态从"未建"改为"已落地"，加 deploy/README 链接|

### 技术

|item|说明|
|---|---|
|llama.cpp 引入策略|workspace 外建 `~/Tools/llama.cpp/`（git clone --depth 1 + cmake Metal build + 独立 `.venv` 装 `convert_hf_to_gguf.py` 的 pin 依赖 transformers 5.5.1 / numpy 1.26 / torch 2.6）；仅 build `llama-quantize` target（跳过 server/examples/tests），~50s cmake + ~50s ninja|
|`mlx_lm.fuse --dequantize` 必须|4-bit 底座 fuse 时 LoRA 加不进 4-bit 量化网格（[mlx-lm#1071](https://github.com/ml-explore/mlx-lm/issues/1071)）；`--dequantize` 强制还原到 fp16 才能正确 merge。实测 ~30s|
|GGUF 转换走 llama.cpp 而非 `mlx_lm.fuse --export-gguf`|`--export-gguf` 是 mlx-lm 较新功能，对 tokenizer metadata / chat_template 持久化兼容性不及 llama.cpp 路径（[Awni 2024 thread](https://github.com/ml-explore/mlx-examples/discussions/1057) 推荐）；实测 convert 用 fp16 输出，写 15.2G GGUF ~38s（~400 MB/s 持续写）|
|Q4_K_M 量化实测|llama-quantize 38s on M4 Pro；14.5 GB f16 → 4.46 GB Q4 → 压缩比 3.26x；BPW 4.91（K-quant 含局部 6-bit 关键参数，故 > 4）|
|Modelfile 1:1 复刻策略|不写自定义 jinja / 不依赖 Ollama 推断；仅替换 `FROM` 行。删 `PARAMETER stop` 等行（baseline 自己也没写显式 PARAMETER，stop token 由 GGUF metadata 提供）—— 复刻颗粒 = `ollama show --modelfile` 的输出去掉 LICENSE block + 改 FROM|
|Ollama create 速度|~5s（FROM 是本地路径不复制 blob，只 mmap + 生成 manifest）|
|build.sh 幂等性实战验证|首次跑 step 1 完成、step 2 中途被 SIGPIPE 截断 → 半截 .gguf 留下 5.7M；重跑时 step 1 检测到 `model.safetensors` 跳过，step 2 检测 partial gguf 重做 rm + 重生（脚本内 `rm -f $GGUF_F16` before 实际 convert）。脚本设计的"已存在跳过 + force 重做"两态值得复用|
|Step 5B 中文 args 验证|transcript 显示 `cast_vote(vote_id="v1", option="追加", rationale="...")` + `propose_vote(question="...", options=["追加","不追加"])` 全 args 中文穿透 chat template → GGUF tokenization → Ollama 解析器无损还原；UTF-8 链路完整|
|Phase 5 触发的 prerequisite 已就绪|`agent_engine` 已通过 `BACKEND=ollama MODEL=agent-sft-qwen` 在多 step trajectory 上跑通，Phase 5 复测唯一缺的是 `evals` 对 `ollama:agent-sft-qwen` 跑 80-batch baseline。这条 last-mile 路径在 Phase 4 验证后无任何新工程问题|

### 取舍

- 反链 [`DECISIONS §6`](DECISIONS.md#6-phase-4-量化等级锁-q4_k_m--modelfile-11-复刻-qwen257b-template) 全部决策。
- 量化等级锁 Q4_K_M 而非更高 Q5/Q8——与 baseline 同量化轴比 SFT 信号差，是 Phase 5 信号归因的 prerequisite；若 Q4_K_M 真损害效果，`bash deploy/build.sh --force QUANT=Q5_K_M` 一行覆盖，链路其他步骤零改动（已在 §6 Consequences 写明）。
- Modelfile 不加 `PARAMETER temperature/top_p` 自定义采样 — qwen2.5:7b 自己也未写，加了等于偏离 baseline；运行时 Ollama 默认 + scenario YAML 内 agent-level temperature 已足够覆盖。
- `deploy/build/` 不入 git（~18 GB 三件产物全本地）——`build.sh` 是"重生指南"，新机器从 adapter 到 ollama tag ≤ 10 min（不计模型下载）；用 size 对换 portability 不划算。
- llama.cpp 装在 workspace 外 `~/Tools/llama.cpp/` 而非进 repo——不是本项目源码，且 `.venv` 大依赖（torch 2.6 ~250 MB）跨 OS 不可复用；deploy/README 写明安装命令即可。
- Step 5B 不强制 `--max-rounds 1`——agent_engine CLI 没这个 flag，且端到端 8 step 跑完 ~4 min 在可接受范围；transcript 抓 10 个 tool_call event 远超 ≥1 验收门槛，"过度验证"对 Phase 5 信心是净增益。
- Phase 1 baseline 80-batch + Phase 5 端到端复测仍 pending——Phase 4 已交付 Phase 5 的所有 prerequisite（model tag 注册 + agent_engine 兼容 + smoke 通过），下一步只需跑 `python eval/run_baseline.py --models ollama:qwen2.5:7b ollama:agent-sft-qwen ollama:qwen2.5:32b --tasks nudge_fire_rate agent_traj bfcl_slice mmlu_slice --seeds 10`；该实跑独立成 5.A 里程碑。

## 2026-05-11 — 数据流水线直连 agent_engine：删 sys.path.insert + evals 私有面 4 import 反模式

这个里程碑把 [`DECISIONS §3`](DECISIONS.md) Consequences 里的"对 evals.metrics.nudge 私 helper 有耦合"这条历史负债清掉。phase 2 数据流水线落地时为了少写代码直接 `sys.path.insert(0, str(PLAY_DIR))` + `from evals.metrics.nudge import _attempt_called_required, _resolve_who_to_agents, _split_attempts, _split_frontmatter, classify_failure_mode, derive_expected_turns`——4 个私有 + 2 个公开。私有 4 个其实是 evals 反向工程 agent_engine 的 transcript / scenario schema 的镜像，跨项目 import 私有面 = "双层间接依赖踩雷面"：agent_engine 改 schema → evals 改镜像 → agent_sft 改对镜像的 import。本期 [`agent_engine §13`](../agent_engine/DECISIONS.md) 把解读权收回 agent_engine（暴露 `Result.tool_calls() / .turns() / .find_finalize_decision()` + `TurnView.attempts() / .start_offset` + `Scenario.expanded_turns()` typed 视图），agent_sft 把跨项目 import 改为直连 `from agent_engine import Result, Scenario, TurnView, ExpandedTurn`，仅保留 `from evals.metrics.nudge import classify_failure_mode`（合法公开面，"missed / wrong_tool" taxonomy 拥有方）。`extractor.py` / `synthesize.py` / `formatter.py` 三脚本同步重写：步骤迭代用 `Scenario.expanded_turns()`、attempt 切分用 `TurnView.attempts(agent)`、context 切片用 `TurnView.start_offset`、frontmatter 解析用 `Scenario.from_yaml(p).meta`. `tests/test_extractor.py` 零修改 pass（旧 helper 单测靠 1-2 行 backward-compat shim 续命）；89 测试全绿；smoke 跑了 5 个 fast envelope 出 21 triples（synthesize），与 phase 2 历史 yield 同序。

### 功能

|item|状态|说明|
|---|---|---|
|跨项目 import 私有面退出|✅|`extractor.py` / `synthesize.py` / `formatter.py` 全部从 `from evals.metrics.nudge import _4_私有` 切到 `from agent_engine import Result, Scenario, TurnView, ExpandedTurn`，只剩 1 个跨项目公开 import（`classify_failure_mode`）|
|extractor / synthesize 主路径直连 agent_engine|✅|`extract_triples` / `envelope_to_synthetic_triples` 都改用 `Scenario.expanded_turns()` 迭代 require_tool turns + `Result.turns()` 切段 + `TurnView.attempts()` 切 attempts + `TurnView.start_offset` 算全局 offset|
|formatter frontmatter 解析换源|✅|`_read_scenario_meta` 改走 `Scenario.from_yaml(p).meta`，schema 校验跟 agent_engine 同源；`yaml` import 删除|
|backward-compat shim|✅|`_index_steps_by_turn` / `_split_turns_indexed` / `_attempt_called_required` 退化为 1-5 行模块本地 shim/helper，让 `tests/test_extractor.py` / `tests/test_synthesize.py` 零修改 pass|

### 技术

|要点|说明|
|---|---|
|公开面 vs 私有面 跨项目 import 卫生|`from evals.metrics.nudge import classify_failure_mode` 是合法（公开 API），`from evals.metrics.nudge import _split_frontmatter / _resolve_who_to_agents / _split_attempts / _attempt_called_required` 是反模式（消费私有实现细节）。本期把后者降到 0，前者留 1|
|schema 解读权归位|"transcript 内 entry 怎么变成 ToolCall"是 schema 的一部分，住 agent_engine（Result / Scenario / TurnView 视图层）；"failure mode 分类 / SFT triple 配对策略"是 evals/sft 自己的语义判断，留在 evals/agent_sft|
|端到端 smoke|5 个 fast envelope（code_review × 2 + tool_chain × 3）经 synthesize 出 21 triples（code_review-r0 8 + r1 7 + tool_chain-r0/1/2 各 2），与 phase 2 同环境历史 yield 同序|
|测试|89 测试全绿（test_extractor 零修改、test_synthesize 零修改、test_formatter 零修改）；agent_engine 端 36 测试新建，evals 端 9 条等价覆盖测试迁出（465 → 456）|
|跨项目契约监控|evals `test_envelope_field_names_match_result_dataclass` 仍是单一断言点；agent_engine `test_expanded_turns_matches_discussion_expanded` 锁住 7 现网 scenario 上"静态展开 == runtime 展开"|

### 取舍

- 反链 [`DECISIONS §7`](DECISIONS.md) 全部决策（修正 §3 Consequences "对 evals 私 helper 有耦合"，§3 整体决策结构不动）。
- `_index_steps_by_turn` / `_split_turns_indexed` 留 1-2 行 shim 而非整段删——plan 给的硬约束是"`tests/test_extractor.py` 零修改 pass"，shim 续命比改测试更稳。新代码不应再调，注释已标 `[DEPRECATED §13 起]`。
- `_attempt_called_required` 在 extractor.py 留 5-line 模块本地 helper 而非全部内联到 caller 处——synthesize.py 跨脚本共用同一逻辑，重复定义信号噪声更大。该函数与 agent_engine `Discussion._called_tool` 检查面字节相同，schema 漂移风险已收敛到 1 个文件。
- `classify_failure_mode` 不上提到 agent_engine——"missed / wrong_tool" 是 evals/sft 视角的语义判断（"模型有没有'尝试干活'"是评测态度，不是 dispatch 真相），上提会污染 agent_engine 的关注边界；如未来 agent_engine `dispatch error` 路径补 `{ok: false}` event 启用 wrong_args 桶时再讨论上提。

## 2026-05-11 — Transcript schema typed 升级 + envelope `usage` 同步消费（agent_engine §14 配套）

[`agent_engine §14`](../agent_engine/DECISIONS.md) 把 transcript 升级到 6 个 frozen dataclass typed union（`SpeakerEntry` 强制带 `type="speaker"`）+ `Result.usage: list[TokenUsage]` + `Result.from_dict` 严格化（缺字段 `KeyError`）. agent_sft 三脚本同步切到 typed access：[`extractor.py`] 删 `_split_turns_indexed / _index_steps_by_turn` 共 2 个 shim（`Result.turns()[i].start_offset` 已直接给 turn-indexed 全局 offset，shim 失去存在意义）+ 对应 [`tests/test_extractor.py`] 2 条 shim 单测；[`synthesize.py`] 切 `isinstance(e, SpeakerEntry)` 派发；[`formatter.py::_render_recent_context`] 走 `entry.get("type") == "speaker"` 判断（§14 已强制写入 `type` 字段）；500 个历史 mined envelope JSON 一次性迁移注入 `type:"speaker"` + `usage: []`. 87 测试全绿（§7 时 89 → §14 时 87 = -2 等价覆盖移到 agent_engine）；新写 mining 自动产 §14 schema. 详见 [`DECISIONS §8`](DECISIONS.md).

### 功能

|item|状态|说明|
|---|---|---|
|typed entry 切换|✅|`extractor / synthesize / formatter` 三脚本从 `entry.get("speaker") / entry["type"]` 切到 `isinstance(e, SpeakerEntry/...)` + `entry.get("type") == "speaker"`|
|shim cleanup|✅|`_split_turns_indexed` / `_index_steps_by_turn` 共 2 个 §7 时期的 shim 退役；新代码直接用 `Scenario.expanded_turns()` / `Result.turns()`|
|历史 mined envelope 迁移|✅|`data/triples/runs_1k_fast_7b_r0_124/*.json` × 500 一次性脚本注入 `type:"speaker"` + `usage: []`，与 forward-only 选择一致|

### 技术

|要点|说明|
|---|---|
|typed import 扩展|`from agent_engine import TokenUsage, ArtifactEventEntry, SpeakerEntry, ToolCallEntry, TranscriptEntry`；与 §7 直连面同源延伸|
|`Triple.context` 类型|`list[dict[str, Any]]` → `list[TranscriptEntry]`；落盘后由 `engine.py::transcript_path` 写入路径走 `dataclasses.asdict` 拍扁回 dict（formatter 端读 metadata 仍是 dict 形态，但有强制 `type` 字段单源）|
|`_attempt_called_required` 形参|`list[ToolCallEntry \| ArtifactEventEntry]`（§7 时是 `list[dict]`），与 agent_engine `Discussion._called_tool` 检查面字节同源|
|测试|87 测试全绿（删 2 条 shim 单测后；89 → 87 = -2 等价覆盖移到 agent_engine 的 typed entry round-trip 测试）|
|`Result.usage` 当前不消费|mining 阶段关心"做对什么"而非"花多少 token"；后续训练数据筛选若需 cost 过滤，从 `result.usage` 直接聚合即可|

### 取舍

- 反链 [`DECISIONS §8`](DECISIONS.md) 全部决策；§7 立的"直连 agent_engine 公开面"边界向内推到 typed entry。
- 500 个历史 mined envelope 选迁移而非重跑——重跑 mining 需要 ollama + ~小时级 LLM 成本，迁移脚本秒级；本仓库无外部消费者，迁移是更便宜的等价路径。后续重跑 mining 自动产 §14 schema，迁移产物会被自然替换。
- 不在 agent_sft 内做 `Result.usage` 聚合（如 cost / token-by-caller 视图）——agent_engine 只产 raw list，聚合住 `evals/metrics/efficiency.py`；agent_sft 若需要"按 caller 折叠 usage"自己写 5 行循环，不要回头加视图。
