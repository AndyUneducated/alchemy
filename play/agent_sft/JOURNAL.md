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
|训练框架选型|MLX-LM（详见 DECISIONS §3）——Apple Silicon 原生最优；`mlx_lm.lora` / `mlx_lm.fuse` / `mlx_lm.convert` 三步 CLI，KISS|
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
- 放弃 Llama-3.1-8B 底座 + axolotl / Unsloth 训练编排（[`DECISIONS §3`](DECISIONS.md) 选项 C/D）——前者 Mac 不是主战场，后者抽象层抬学习成本；MLX-LM 三命令链路 + Qwen2.5 在 7B tool-call 段位基线更强。
- 放弃"7B SFT 追 GPT-4o-mini"行业锚点 + Qwen2.5-72B 作 ceiling——换得全本地可复现 + 同家族跨规模对比；72B Q4 ≈42GB 余量太紧 ROI 不划算。
- 拒绝"立项稿 v1 完成态"叙事 + "完整写 v1 + v2 + v3 三套 README" 路线——v1 真完成时面试官追问"下一步"无干脆答案，但 v2/v3 没数据写成空想；候选清单 + 触发条件足以传达"有路线图"信号，遵循 `workshops.mdc` "抽象引入滞后于第二个具体案例"原则。
- 拒绝引入 `lm-evaluation-harness` 集成 BFCL / MMLU——自实现各 <100 行远低于跨重型框架适配 + 钉版调试成本；与 [`DECISIONS §3`](DECISIONS.md) "不引入新 tooling" 原则一致。
- 拒绝 `metrics/{bfcl,mmlu}.py` 独立模块——单一消费者 + 函数简单（AST parse / MCQ 字母提取各 ~20 行），独立模块属"为抽而抽"；遵循 `workshops.mdc` "抽象引入滞后于第二个具体案例"原则。
- `wrong_args` 失败模式桶 Phase 1 当 placeholder 归 wrong_tool——agent_engine artifact handler error 路径不发 event，无法仅靠 transcript 区分"调对工具被拒" vs "调了别的工具"；显式留桶让 by_failure_mode 表头跨 run 稳定，启用推迟到 Phase 5。
- 80-batch 实跑 + 对比报告分到独立 1.G 里程碑——本次留可复现的工具链 + smoke；用户拉 7B 后跑 `python play/agent_sft/eval/run_baseline.py` 即可生成 `baselines/qwen2.5-7b-vs-32b.md` 真实数据。

## 2026-05-10 — Phase 2 数据流水线落地 + pilot 揭出 yield 瓶颈

按 [`DECISIONS §11`](DECISIONS.md) 一次性铺完 Phase 2 数据 pipeline 4 个脚本 + 测试 + 文档 + .gitignore，并跑 pilot（2 scenario × 3 run_id = 6 envelope，Qwen2.5-7B）。pilot 三件事都达成：① 端到端流水线打通（envelope → triple → split → MLX-LM messages 全链 5min 重生）；② 真实失败模式分布吻合预期（27 missed / 1 wrong_tool / 0 wrong_args，与 [Phase 1 失败模式 taxonomy](#技术) 一致）；③ **量出 yield = 0.17 triples/envelope，与 plan §Volume math 估算 5/envelope 差 30 倍——scale-up 路径需用户决策才能继续**。

### 功能

|item|状态|说明|
|---|---|---|
|`agent_sft/data/__init__.py` + 4 脚本|✅|`mine_triples.py` / `extractor.py` / `formatter.py` / `split.py`，共 ~750 行；CLI 全 `argparse`，子进程跑 agent_engine 不阻塞 batch|
|`data/triples/` 产物子目录 + README|✅|与 `eval/baselines/` 完全平行布局；README 含 4 步 regen + OOD 复用说明 + pilot 实测表|
|`data/extractor.py` 复用 `evals.metrics.nudge`|✅|不重写 turn marker / attempt 切分 / failure mode 分类；只新增 "first_failed → eventual_success" 配对 + 全 transcript prefix 作 context|
|`data/formatter.py` 输出 MLX-LM chat 格式|✅|`{messages: [system, user, assistant]}`；system = agent.prompt + 工具列表，user = 最近 6 turn 渲染 + step.instruction，assistant = corrected_response 原文|
|`data/split.py` per-scenario by-run_id|✅|末 20% run_id → val；< 5 unique run_id 全 train fallback（pilot 1 triple 即触发 → train=1, val=0）|
|`data/mine_triples.py` 默认 6-envelope pilot|✅|`--scenarios` choices = ['tool_chain', 'code_review']；`--run-ids` 默认 [0,1,2]；`--dry-run` / `--timeout` / 失败汇总|
|`agent_engine/config.py` env override|✅|新增 `AGENT_ENGINE_MODEL` env var → DEFAULT_MODEL（1 行改动），让 7B mining 不动 scenario YAML|
|`tests/conftest.py` 加 `data/` sys.path|✅|测试纯 import 业务模块，无 path 体操|
|`.gitignore` 加 `data/triples/runs/` + `*.jsonl`|✅|README.md 仍提交（`*.jsonl` 模式不匹配 `.md`）|
|`tests/test_extractor.py` 12 case|✅|6 失败模式边界（missed / wrong_tool / first 成功 / 全失败 / multi-nudge / no require_tool）+ 截断 + 3 helper 单测|
|`tests/test_formatter.py` 13 case|✅|3-message schema / system 含 prompt+tools / user 含 instruction+context / max_recent 截断 / 空 context / 5 helper 单测|
|`tests/test_split.py` 11 case|✅|10/5/4/1 run_ids 切分 / multi-scenario 独立 / floor 取整 / fallback 阈值 / 末 N 而非首 N|
|Pilot: 6 envelope mining + extract|✅|Qwen2.5-7B，6.8 min wall clock；require_tool turns=39 fired=28 (71.8%) → triples=1 (yield 0.17)|

### 技术

|item|说明|
|---|---|
|seed handling|不改 agent_engine——run_id 仅 envelope 文件命名键 + split 索引（`AGENT_ENGINE_MODEL` env override 也仅是 1 行），靠 7B 自然采样得 trace 多样性|
|样本格式|F1 only（input 不含 nudge），训"看到原 instruction 一次到位"而非"被 nudge 后才补"|
|context 截取|`max_recent=6` 渲染最近 history，与 `code_review.md memory.max_recent` 一致；pilot 唯一 sample user content ≈ 250 token，远低于 2048 上限|
|代码 / 数据布局|A' 完全仿 `eval/`：`data/` 4 脚本 + `data/triples/` 子目录装产物（runs/ raw envelope + triples.jsonl + train_triples / val_triples + train.jsonl / val.jsonl）|
|nudge 文本复原|`NUDGE_TEMPLATE = "你刚才没有调用 \`{tool}\` 工具..."` 模板按 `required_tool` 填回，与 `discussion.py` L141-144 硬编码字面对齐；不进 F1 input，只占 traceability 字段|
|extractor 配对策略|每个 expected require_tool turn：first attempt 失败 + 任一后续 attempt 成功 → 1 triple（同 turn 至多 1 triple，因引擎 first-success-returns）；failure_mode 看 first attempt|
|测试覆盖|`evals/tests/` 460 + `agent_sft/tests/` 17 → **57**（+40：12 extractor + 13 formatter + 11 split + 4 conftest 路径互通验证）= 全仓 517 全过|
|pilot wall clock 分布|tool_chain ~70s/run / code_review ~67s/run；7B 在 M4 Pro 48GB 单 turn 平均 5-12s|

### 取舍

- 拒绝在 `agent_engine` 加 `--seed` flag——跨 Engine + Agent + ollama_client 三处改动 + EvalResult 兼容性，与"最小侵入"冲突；run_id 命名键已足够支撑切 train/val。
- 拒绝把 mining 模型混入 32B / 14B 多模型对比——pilot 阶段先量 7B 单点 yield；32B / 多模型选项写进 [`DECISIONS §11`](DECISIONS.md) scale-up 路径表留给用户决策。
- 接受"代码 / 数据混在 `data/`"布局（A'）而非进一步拆 `scripts/` + `data/{raw,interim,processed}/`——后者更"行业标准 ML repo"但需 5+ 文件 + 多层目录，pilot 阶段 4 脚本 + 1 子目录已能让产物全 gitignore；若 Phase 3 训练 / Phase 4 量化也归到 `data/` 时再演化（YAGNI）。
- 接受"`tools` JSON schema 不进 F1 system message"（暂只列工具名 comma list）——MLX-LM 标准 chat format `tools` 字段支持是 PR-by-PR 演进的，Phase 3 真训练时再按 mlx-lm 当时版本对齐；当前 system content ≈ 200 token，留 token budget 给 user/assistant。
- **pilot yield 30x 低于 plan 估算 → 暂停 scale-up 等用户拍板路径**——选项与代价见 [`DECISIONS §11`](DECISIONS.md) scale-up 路径表。三类候选：① 改 scenario `max_retries` 提 recovery 率（侵入 scenario YAML）；② 换 32B mining（与 ADR Mining 模型决策冲突 + envelope 慢 3x）；③ 降目标到 200 triples（train 集小风险）。本里程碑只交付"流水线 + pilot 数据 + 决策表"，不强行 scale 完成"1k triples"目标。
- 用户回复"非要 1000 条吗？最小化 demo 即可"→ 走 `max_retries: 1→2` 实验路径（demo viable 12 envelope，~13 min）。结果 1 triple，**与 max_retries=1 batch 同量级**——确认瓶颈不在重试次数。同步跑 32B 单 envelope 对照（2 fires 0 recovery，**样本太小，结论不成立**——见下条里程碑修正）。

## 2026-05-10 (晚) — Phase 2 数据交付：Approach B (synthesize) 解锁 yield 瓶颈

承接上一条 pilot 失败叙事。用户挑战"换 32B 真不行吗、能不能直接合成假 triple"两个问题——分别跑两组实验后**修正了之前关于"瓶颈在方法学"的过度判断**，并落地 Approach B 真正解锁 Phase 2 数据交付。

### 功能

|item|状态|说明|
|---|---|---|
|**修正旧结论**：32B 单 envelope 对照样本太小|✅|跑 32B × code_review × 3 envelope（25 min）：20 fires，5 triples，**recovery 25%**（vs 7B 的 3%）→ 底座 capability 是 recovery 率主因，方法学本身没崩|
|**实现 Approach B**：`data/synthesize.py` + 18 测试|✅|"真失败 attempt + 程序化合成 corrected"配对策略——每个 nudge fire 出 1 triple（yield = fire rate 上限，非 recovery 率），corrected 用 step.instruction 里字面 `tool(args)` 模板，fallback 用通用 wrapper + 完整 instruction|
|**正式交付 train/val**|✅|7B max_retries=2 batch 12 envelope → synthesize → **57 triples → 47 train + 10 val**（per-scenario 末 20% run_id 走 val，正常生效不再 fallback）|
|`extractor.py` 保留|✅|未来 Phase 3 后若 7B 自己 recovery 率拉到 30%+，可一行命令切回"真自纠"语义；两条路径共用 Triple schema|
|文档同步|✅|`data/triples/README.md` 增"两种 triple 来源"对照表 + pilot 时序演进；JOURNAL 本条；`DECISIONS §11` 不动（按 user "本项目不再写 ADR" 规则，旧条目作历史记录）|

### 技术

|item|说明|
|---|---|
|为什么不是 32B mining|32B envelope ~500s，每条 triple ~5min compute；synthesize 复用现有 7B envelope ~100s/env + 0 额外，每条 triple ~21s。**总 compute 节省 ~14x**|
|为什么不是蒸馏（C 方案）|与 [`DECISIONS §1`](DECISIONS.md)"自有 infra 生数据"冲突；synthesize 严格说仍是"自家 7B 失败素材 + 自家 scenario 模板"，没引入第三方教师|
|`_extract_call_template` 设计|regex `\\b{tool}\\s*\\(` 起始，paren 平衡扫描到第一个 unbalanced `)`；7 测试覆盖（简单 / 跨行 args / 中文引号混合 / 不平衡 paren / word boundary / 多次出现取首个 / 无模板返 None）|
|fallback 文本|`"好的，我现在调用 \`{tool}\` 完成本步：\\n{instruction}"`——instruction 全文入 corrected，对没字面模板的 step（如 retrieve_docs 类）保留语义信息，不是空响应|
|测试增量|`agent_sft/tests/` 36 → **54**（+18 synthesize：7 template extract + 4 synthesize wrapper + 7 envelope-to-triples 边界）|
|与 `extractor.py` 解耦点|synthesize.py import extractor.{Triple, helpers, write_triples_jsonl} 复用；只改 `envelope_to_synthetic_triples` 配对策略（不要求后续 success），其他全沿用|

### 取舍

- **承认上一条里程碑的"瓶颈在方法学"是过度判断**——n=2 fires 的 32B 对照不能下"换底座也救不了"结论。本里程碑用 n=20 fires 的 32B 实验 + Approach B 落地两条独立证据修正：底座很影响 recovery，方法学有简化路径。这条作为"早失败 → 早跑实验 → 早修正"的工程证据保留在 JOURNAL，不掩盖。
- 选 Approach B（合成）而非 D（32B mining）：经济性碾压（compute 节省 14x），且训练目标更干净（无 7B 偶然 success 的"text 说 X 但 tool_call 是 Y"噪声样本）。代价是 corrected 是模板而非自然语言，**Phase 3 训练后若发现模型只学会模板复读、不能泛化，再回头切到 D 或 C 方案**。
- 不进 ADR：用户 2026-05-10 决策"本项目不再写 ADR"，本里程碑技术决策直接落 JOURNAL.技术 + .取舍 节，不再起 §12 条目；`DECISIONS §11` 旧文（含错误结论"瓶颈在方法学"）作历史记录保留，不在 ADR 文件内追加修正。
- 仓库 train.jsonl / val.jsonl 仍 gitignored（per `.gitignore play/agent_sft/data/triples/*.jsonl`）；`README.md` 的 §当前 train/val 数据 表充当唯一可提交的统计快照。
