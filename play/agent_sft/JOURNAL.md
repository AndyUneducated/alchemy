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
