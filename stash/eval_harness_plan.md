# Eval Harness — 设计方案（快照）

> **快照时间**：2026-04-29
> **Cursor plan 原件**：`~/.cursor/plans/eval-harness_scaffold_053fbd9d.plan.md`（Cursor IDE plan UI 里也能直接打开）
> **状态**：设计阶段，尚未开工。明天继续 review / refine。

---

## v0 待办（执行清单）

- [ ] **skeleton** — 在 `play/eval_harness/` 创建目录骨架：requirements.txt / config.py / __init__.py / __main__.py / cli.py / core/ / adapters/ / metrics/ / tasks/
- [ ] **core_api** — 实现 core/api.py，冻结所有 Protocol/TypedDict（ModelAdapter / Metric / Task / Sample / Result），output_type 枚举列全 4 种，metric.score 签名带 `**ctx`，phase 1-5 不再改这层
- [ ] **core_registry_loader_stats** — 实现 core/registry.py（`@register_task` / `@register_metric` / `@register_model`）+ core/task_loader.py（yaml + jsonl + jinja2）+ core/stats.py（bootstrap CI；paired bootstrap / McNemar 留 stub）
- [ ] **core_runner_reporter** — 实现 core/runner.py：match output_type，generate 分支真实现，其余 3 分支抛 NotImplementedError 留位；core/reporter.py 输出 per_sample.jsonl + summary.json + report.md，迭代任意 metric dict 不 hardcode
- [ ] **adapter_echo** — 实现 adapters/base.py（ModelAdapter 接口，3 方法都定义）+ adapters/echo.py（确定性：generate 返回输入，loglikelihood/chat 抛 NotImplementedError）
- [ ] **metric_exact_match** — 实现 metrics/builtin.py 仅含 exact_match（score = pred==ref，aggregate = mean），证明 metric 协议工作；不引 sklearn / sacrebleu 等任何 phase 1+ 依赖
- [ ] **task_hello_echo** — tasks/hello_echo/（task.yaml output_type=generate, metrics=[exact_match]; data.jsonl ~5 条 `{"input": "hello", "target": "hello"}`）
- [ ] **cli_e2e** — cli.py 实现 run / list-tasks / list-metrics / report；`python -m eval_harness run --task hello_echo --model echo` 端到端跑通且 exact_match=1.0
- [ ] **readme_changelog** — README.md（v0 = base only 的明确声明 + 5 族学习路线图 + 「可适配」设计清单 + Talking Points）+ CHANGELOG.md 初条记 v0 范围
- [ ] **smoke_test** — smoke test：echo + hello_echo 跑通；断言 per_sample.jsonl 5 行、summary.json 含 exact_match=1.0、report.md 渲染成功

---

# Eval Harness — lm-eval-harness 风格的渐进式 LLM 评测框架

**总览**：在 `play/eval_harness/` 新建一个对标 lm-evaluation-harness 简化版的评测框架：Task / ModelAdapter / Metric / Runner / Reporter 五件套。**v0 只落地骨架 + 一个平凡 smoke task（hello_echo），目的是冻结接口、验证管道**；phase 1-5 每 phase 加 1 族（classification / generation / LLM-as-judge / RAG / agent-trajectory），不动骨架，只填实现。

## 设计取舍（先把"为什么"摆出来，便于面试复盘）

- **对标 lm-evaluation-harness**：业界最主流；保留它的核心抽象（`Task` / `LM` adapter / `Metric` / `Filter` / `Aggregator`），但去掉 megatron / sharding / accelerate 这类工业糟粕，单机单进程跑通。
- **声明式 task = yaml + jsonl**：HF datasets / OpenAI evals / lm-eval-harness 的最大公约数，作者改 task 零代码。
- **Metric 是纯函数**：`(predictions, references, **kwargs) -> dict[str, float]`；LLM-as-judge 也实现成 metric，只不过它内部持有一个 adapter——把"评测器"和"被测模型"统一在同一抽象里，是 lm-eval-harness 没做、ragas 做了的事。
- **Echo adapter 兜底**：纯确定性 stub adapter，使 harness 自身可以无 LLM 依赖跑 CI / 单测，是工业实践（OpenAI evals 的 `dummy/`、ragas 的 `MockLLM`）。
- **统计是一等公民**：bootstrap 95% CI + 配对比较 (paired bootstrap / McNemar)，**写在 reporter 而非 metric 里**——metric 只算 per-sample，聚合 + 不确定性是 runner 层的事，分层清晰。
- **完全独立 / 不接外部系统**：本项目学的是评测方法学，不是工程拼装。所有 task 数据集都是伪造的 jsonl，metric 看不到上游是真模型还是 mock——这意味着**无需** `RetrieverAdapter` / `AgentAdapter` 这类接口，**也不会** wrap `play/rag` / `play/agent_engine`。`ModelAdapter` 仅服务于"被测 LLM"和"LLM-as-judge"两类调用。
- **v0 = 纯基座**：v0 不实现任何"真"任务族，只用一个平凡的 `hello_echo`（输入 = 期望输出）+ `exact_match` metric + `echo` adapter 把管道跑通。约 ~500 LOC，一个晚上能搞完。**v0 的全部目的是冻结接口**——在没有任何业务复杂度污染时把 Protocol / output_type 枚举 / metric 签名 / reporter 通用性定型，phase 1-5 就只剩"按模板填空"。
- **渐进式：1 族 = 1 个 phase**：phase 1-5 每个加 1 族 + 该族需要的依赖/adapter/runner 分支，每 phase ~400-700 LOC，每 phase 末尾都是端到端可演示的版本。CHANGELOG 每 phase 加一条 ADR 记"这一族为什么这么设计"，沉淀面试素材。

## 目录结构

**v0 实际落地（标 ✅）+ 后续 phase 预留位置（标 P1-P5）**——目录骨架一次开足，但 v0 只写 ✅ 的文件；P1-P5 文件 phase 1-5 各自加进来。

```
play/eval_harness/
├── README.md                         ✅ v0 范围 + 5 族学习路线图 + 可适配清单 + Talking Points
├── CHANGELOG.md                      ✅ 每 phase 一条 ADR
├── requirements.txt                  ✅ v0 最小依赖（pyyaml / jinja2 / numpy 三件套）
├── config.py                         ✅ 默认 backend / 路径常量
├── __init__.py                       ✅
├── __main__.py                       ✅ python -m eval_harness …
├── cli.py                            ✅ run / list-tasks / list-metrics / report 子命令
│
├── core/                             ✅ v0 全量落地（接口在 v0 冻结，phase 1-5 不动这层）
│   ├── api.py                        ✅ Sample / Doc / Result / Task / ModelAdapter / Metric Protocol；output_type 枚举一次列全 4 种
│   ├── registry.py                   ✅ @register_task / @register_metric / @register_model
│   ├── task_loader.py                ✅ 读 task.yaml + data.jsonl，jinja2 渲染 doc_to_text/doc_to_target，schema 校验
│   ├── runner.py                     ✅ match output_type 4 路分支：generate ✅ 实现；multiple_choice/judge/trajectory ⚠️ 抛 NotImplementedError 留位
│   ├── reporter.py                   ✅ per-sample jsonl + summary json + markdown；迭代任意 metric dict，不 hardcode metric 名
│   └── stats.py                      ✅ bootstrap CI；paired bootstrap / McNemar 留函数 stub（phase 3 实现）
│
├── adapters/
│   ├── base.py                       ✅ ModelAdapter 接口完整定义（generate / loglikelihood / chat 三方法都签名好）
│   ├── echo.py                       ✅ 确定性 stub：generate 返回输入；loglikelihood/chat 抛 NotImplementedError
│   ├── ollama.py                     P1  本地真后端（族 1 第一次接真 LLM）
│   ├── openai.py                     P3  judge 投票需异源 LLM
│   ├── anthropic.py                  P3
│   └── gemini.py                     P3
│
├── metrics/
│   ├── builtin.py                    ✅ 仅 exact_match（pred == ref，aggregate = mean）
│   ├── classification.py             P1  accuracy / precision / recall / F1 (binary/macro/micro) / confusion_matrix
│   ├── agreement.py                  P1  cohen's kappa / fleiss' kappa / krippendorff's alpha / percent_agreement
│   ├── text_similarity.py            P2  token_f1 / BLEU / ROUGE-{1,2,L}（exact_match 已在 builtin.py）
│   ├── retrieval.py                  P2  recall@k / precision@k / NDCG@k / MRR / MAP（与 generation 一起，phase 4 RAG 复用）
│   ├── llm_judge.py                  P3  JudgeMetric 基类 + pairwise / rubric / position-swap debias
│   ├── ragas_lite.py                 P4  faithfulness / answer_relevancy / context_precision / context_recall（派生自 P3 的 JudgeMetric）
│   └── trajectory.py                 P5  tool_call_exact / argument_f1 / trajectory_edit
│
└── tasks/
    ├── hello_echo/                   ✅ v0 smoke task：output_type=generate, metrics=[exact_match]
    │   ├── task.yaml
    │   └── data.jsonl                # ~5 条 {"input": "hello", "target": "hello"}
    ├── cls_sentiment/                P1  multiple_choice + generate 双 protocol 对照
    ├── gen_summary/                  P2
    ├── judge_helpfulness/            P3
    ├── rag_qa/                       P4
    └── agent_tool_use/               P5
```

**为什么 `core/` 必须 v0 一次写到位**：接口是整个项目的"宪法"。如果 v0 不冻结，phase 1 写族 1 时为了支持 `multiple_choice` 改一次 runner，phase 3 写 judge 时为了支持 adapter-in-metric 改一次 metric 签名……每改一次接口就要 review 已有 phase 是否还兼容。在 v0 没有任何业务负担时一次定型，是工程纪律的胜利。

## 核心抽象（接口先行）

`core/api.py` 用 `Protocol` 定义四件套，骨架如下：

```python
class ModelAdapter(Protocol):
    name: str
    def generate(self, prompt: str, *, max_tokens: int, stop: list[str]) -> str: ...
    def loglikelihood(self, context: str, continuation: str) -> tuple[float, bool]: ...
    def chat(self, messages: list[dict]) -> str: ...

class Metric(Protocol):
    name: str
    def score(self, pred, ref, **ctx) -> dict[str, float]: ...   # per-sample
    def aggregate(self, per_sample: list[dict]) -> dict[str, float]: ...

class Task(TypedDict):
    name: str
    output_type: Literal["generate", "multiple_choice", "judge", "trajectory"]
    dataset_path: str        # data.jsonl 相对路径
    doc_to_text: str         # jinja2 模板，渲染 prompt
    doc_to_target: str       # 提取参考答案
    metric_list: list[str]   # registry 里的 metric 名
    few_shot: int            # k-shot
    fewshot_split: str | None
```

`runner.run(task, adapter)`：
1. 加载 yaml + jsonl，按 `output_type` 决定走 `generate` / `loglikelihood` / `judge` / `trajectory` 路径
2. 渲染 prompt（含 few-shot 拼接）→ 调 adapter
3. 每条 sample 落 `per_sample.jsonl`（doc_id / pred / ref / metric_scores / latency / cost_usd）
4. 聚合 + bootstrap CI → `summary.json`
5. `reporter.render_md` 生成对外可读报告

## v0 落地范围（纯基座）

**v0 不实现任何任务族**。它的全部目的是冻结接口、把管道走通——为 phase 1-5 准备一个稳定的脚手架。

- **唯一 task**：`tasks/hello_echo/`，~5 条 `{"input": "<text>", "target": "<text>"}`，input 和 target 完全相同
- **唯一 metric**：`exact_match` —— `score(pred, ref): {"exact_match": 1.0 if pred == ref else 0.0}`，aggregate 取 mean。**它是 phase 2 `text_similarity` 的最简单成员，但 v0 实现在 `metrics/builtin.py` 里**，不引 `text_similarity.py` 的任何依赖
- **唯一 adapter**：`echo`，`generate(prompt)` 把 prompt 中 `<input>...</input>` 之间的内容回传出来；`loglikelihood / chat` 抛 `NotImplementedError`
- **唯一 output_type 实现**：`generate`；其他 3 种在 runner 里 `match` 到分支后抛 `NotImplementedError`
- **统计**：bootstrap 95% CI 在 v0 就写好（无非是 numpy 重采样），exact_match 上跑一遍验证管道；paired bootstrap / McNemar 留函数 stub，phase 3 实现

### v0 通过标准

```bash
$ python -m eval_harness run --task hello_echo --model echo --output runs/v0_smoke/
🚀 task=hello_echo  model=echo  samples=5
✅ exact_match: 1.0000 [95% CI: 1.0000-1.0000]  (n=5)
📁 wrote runs/v0_smoke/per_sample.jsonl + summary.json + report.md
```

`runs/v0_smoke/per_sample.jsonl` 5 行，每行含 `doc_id / pred / ref / metric_scores / latency_ms`；`summary.json` 含 `exact_match=1.0`；`report.md` 渲染成功。

### 「可适配」清单（v0 必须做对的接口约定）

phase 1-5 不应该回头改这些——如果改了，说明 v0 设计有 bug。

1. **`ModelAdapter` Protocol 三个方法都定义**：`generate / loglikelihood / chat`。echo 只实现 `generate`，但接口空间已开。phase 1 ollama 实现 `generate + loglikelihood`，phase 3 openai 实现全部
2. **`output_type` 枚举一次列全 4 种**：`Literal["generate", "multiple_choice", "judge", "trajectory"]`。runner 内部 `match` 4 个分支，未实现的抛 `NotImplementedError`，phase 1/3/5 只填代码
3. **`Metric.score` 签名带 `**ctx`**：`score(pred, ref, **ctx) -> dict[str, float]`。`**ctx` 让 phase 4 ragas 能传 `contexts=...`、phase 3 judge 能传 `adapter=...`、phase 5 agent 能传 `expected_calls=...`，全部不破坏接口
4. **`Metric.score` 返回 dict 而非 float**：因为 ragas 的 `RagasMetric` 一个 metric 同时返回 4 个分数；reporter 必须迭代 dict
5. **`Metric.aggregate` 接口必备**：BLEU 是 corpus-level、Cohen's kappa 是全集 level，不是 mean。v0 的 exact_match aggregate 实现成 mean，但接口允许其他形态
6. **Reporter 不 hardcode metric 名**：迭代任何 metric 返回的 dict，都能渲染。phase 加新 metric 不需要改 reporter
7. **Registry 解耦 + 自动发现**：`@register_task / @register_metric / @register_model`，phase 1-5 只新增文件 + 装饰即可，无需注册到中心列表
8. **Stats 提供 `bootstrap_ci(values, n=1000) -> (mean, lo, hi)`**：v0 应用到 exact_match 验证；所有后续 metric 直接用同一函数

## CLI 形态

```bash
# v0 端到端（基座 smoke）
python -m eval_harness run --task hello_echo --model echo --output runs/v0_smoke/
python -m eval_harness list-tasks         # 输出: hello_echo
python -m eval_harness list-metrics       # 输出: exact_match
python -m eval_harness report runs/v0_smoke/   # 重新渲染 markdown 报告
```

后续 phase 加新 task / model / 子命令（phase 3 加 `--judge-model`），但 v0 的 `run / list-tasks / list-metrics / report` 四个子命令骨架不变。输出契约对齐 `play/rag` 的 envelope 习惯：stdout 跑动信息（emoji 进度），`--output` 时落到目录下 `summary.json` + `per_sample.jsonl` + `report.md`。

## 5 族学习路线图（phase 1-5 = 5 族）

所有 phase 仍只消费伪造数据集——拓宽的是"评测知识面"，不是"接入面"。每 phase 末尾都是**端到端可演示**的版本，CHANGELOG 加一条 ADR 沉淀面试素材。

### Phase 1 —— 族 1：Classification + agreement（分类 + 一致性）

- **新增 metrics**：`classification.py`（accuracy / precision / recall / F1 binary&macro&micro / confusion_matrix）+ `agreement.py`（cohen's kappa / fleiss' kappa / krippendorff's alpha / percent_agreement）
- **新增 task**：`cls_sentiment/`，~20 条二分类情感；同一 task 配两个 variant：`cls_sentiment_mc`（multiple_choice / loglikelihood 路径）+ `cls_sentiment_gen`（generate 路径配输出 normalize）
- **新增 adapter**：`ollama.py`（族 1 第一次接真 LLM）
- **runner 加 `output_type=multiple_choice` 路径**
- **新依赖**：`scikit-learn krippendorff ollama`
- **学习重点**：lm-eval-harness 招牌话题（multiple_choice via loglikelihood vs generate via normalize 的 cost / realism / API 限制三方取舍）；带 chance correction 的 agreement metric（Cohen's / Fleiss' / Krippendorff 的递进关系）；macro vs micro vs weighted 聚合差异
- **LOC**：~700

### Phase 2 —— 族 2：Generation + retrieval（生成 + 检索）

- **新增 metrics**：`text_similarity.py`（token_f1 / BLEU / ROUGE-{1,2,L}）+ `retrieval.py`（recall@k / precision@k / NDCG@k / MRR / MAP，**phase 4 RAG 复用**）
- **新增 task**：`gen_summary/`，~5 条 (passage, reference_summary)
- **新依赖**：`sacrebleu rouge-score`
- **学习重点**：字符串相似度演化史 —— n-gram 重叠（BLEU）→ LCS（ROUGE-L）→ token F1（SQuAD normalization：lowercase / strip punct / drop articles）；为什么 BLEU 是 corpus-level 而 ROUGE 是 sample-level；retrieval metric 的 rank-aware vs unaware（recall@k vs NDCG）
- **LOC**：~400

### Phase 3 —— 族 3：LLM-as-judge（主观评判）

- **新增 metrics**：`llm_judge.py`（`JudgeMetric` 基类 + pairwise winrate + rubric 打分 + position-swap debias）+ stats 加 paired bootstrap / McNemar
- **新增 adapters**：`openai.py` / `anthropic.py` / `gemini.py`（多 judge 投票需要异源 LLM；都做 graceful skip 避免没 key 跑不动）
- **新增 task**：`judge_helpfulness/`，~10 条 (prompt, response_a, response_b, human_label)
- **runner 加 `output_type=judge` 路径**
- **学习重点**：judge bias 全套（position / verbosity / self-preference / length）；**用 phase 1 写好的 `cohens_kappa` 验 judge vs human 一致性**（前后呼应，是 v0 选 metric 通用接口的回报）；多 judge 投票 / 异源对照
- **LOC**：~500

### Phase 4 —— 族 4：RAG（ragas-lite）

- **新增 metrics**：`ragas_lite.py`（faithfulness / answer_relevancy / context_precision / context_recall），**全部派生自 phase 3 的 `JudgeMetric`**——纯应用层
- **新增 task**：`rag_qa/`，~5 条 (question, contexts[], answer, ground_truth)
- **学习重点**：ragas 四件套各自对应一个产品级故障（幻觉 / 不相关 / 检索差 / 检索不全）；为什么真实 ragas 用 NLI 而不是 LLM-judge（cost / 可复现）；context_precision 用 phase 2 retrieval.py 的 NDCG 思想
- **LOC**：~400

### Phase 5 —— 族 5：Agent trajectory（轨迹）

- **新增 metrics**：`trajectory.py`（tool_call_exact / argument_f1 / trajectory_edit-distance）
- **新增 task**：`agent_tool_use/`，~5 条 (instruction, expected_calls[], actual_calls[])
- **runner 加 `output_type=trajectory` 路径**
- **学习重点**：序列对序列匹配（Levenshtein on tool-name 序列）+ 集合对集合匹配（参数 token F1）；为什么这一族至今没有 BLEU 级标准（任务多样性、工具语义不可比）
- **LOC**：~400

### Phase 6+ —— 深化（按兴趣自取，不是必修）

- 替换 ragas-lite 为真 NLI-based faithfulness（HF `cross-encoder/nli-deberta-v3-base`）+ embedding-based answer_relevancy
- BBH / MMLU 风格 multiple-choice 任务，压力测试 loglikelihood 路径
- BERTScore（contextual embedding 相似度）
- code eval (`pass@k`，沙箱执行)
- calibration（ECE / Brier score）+ robustness（input perturbation）
- safety / bias（StereoSet / BBQ 简化版）

到 phase 5 完成时，5 族全打通。

## 依赖（`play/eval_harness/requirements.txt`）

**v0 最小集**（基座 + smoke task，三件套）：
- `pyyaml` —— task.yaml 解析
- `jinja2` —— `doc_to_text` / `doc_to_target` 模板渲染
- `numpy` —— bootstrap CI 重采样

**后续 phase 增量**：
- Phase 1：`scikit-learn krippendorff ollama`（族 1 第一次接真 LLM + 经典分类指标）
- Phase 2：`sacrebleu rouge-score`
- Phase 3：`openai anthropic google-genai`
- Phase 6+：`bert-score sentence-transformers transformers`

## 面试 Talking Points（写在 README 末尾，便于复盘）

- Multiple-choice via loglikelihood vs generation —— cost / realism / API 限制三方取舍
- 为什么 metric 是纯函数，aggregation / CI 在 runner 层 —— 关注点分离
- LLM-as-judge 偏置：position bias / verbosity bias / self-preference / 多 judge 投票 / pairwise vs scalar
- Cohen's kappa 为什么优于 raw agreement —— chance correction，配 confusion matrix 解释
- Bootstrap CI 为什么优于假设高斯 —— 小样本 + 长尾分数
- Ragas faithfulness 真实做法（NLI 三态 entail/neutral/contradict）vs 我们的 LLM-judge 简化版
- 为什么需要 echo adapter —— harness 自测、CI 不依赖 LLM、调试 metric 时排除模型方差
- 为什么完全不接 `play/rag` `play/agent_engine` —— 评测的关注点是 metric / aggregation / 不确定性，不是数据源；mock 数据已能覆盖所有逻辑路径，wrap 外部系统只增加工程成本不增加 eval 知识（YAGNI）

---

## 明天继续看时的入口建议

1. **快速热身**：先从「设计取舍」7 条过一遍，回忆 why
2. **核心确认**：「v0 落地范围」+「可适配清单」是 phase 1-5 不破坏骨架的契约，最值得 review
3. **未决事项**：
   - 是否要把 `Filter` 也加为 v0 的一等公民（lm-eval-harness 五件套之一，目前 plan 漏了；phase 1 cls_sentiment_gen 立刻就要用）—— 上次对话末尾留的悬而未决问题
4. **开工前最后一步**：把这份 plan 复盘一遍，没异议就在新会话里说"按 plan 开干"，agent 会从 todos 第一项 `skeleton` 起步
