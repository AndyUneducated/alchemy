# play/qa_assets

QA 测试方案 agent 项目的**资产层**——配置、scenario、hooks、模板、示例数据。**没有业务代码**（hooks 是薄 deterministic 函数，无业务逻辑）。

把这一层与 [play/agent_engine/](../agent_engine/) (LLM 推理引擎) 和 [play/workflow/](../workflow/) (pipeline runner) 解耦：领域东西全在这里，引擎与 runner 完全 domain-agnostic。

## 项目布局

```
play/qa_assets/
├── workflows/
│   └── qa_supervisor.yaml          # 顶层 pipeline (CSV → md/csv 测试方案)
├── scenarios/
│   └── qa_discuss.md               # 6 agent 多 agent loop scenario
├── hooks/
│   ├── load_csv.py                 # CSV → list[dict], 最小校验
│   ├── load_each_prd.py            # 对有 prd_doc_path 的行: Path.read_text() → prd_md
│   └── to_yaml.py                  # 序列化助手 (替代模板 filter, plan §4.3)
├── examples/
│   ├── requirements.csv            # 2 行示例输入 (REQ-001 引用 PRD .md, REQ-002 inline)
│   └── prd_signup.md               # 示例 PRD markdown
└── README.md                       # 本文件
```

## 端到端跑通

```bash
cd play/
python -m workflow run qa_assets/workflows/qa_supervisor.yaml \
    --vars csv_path=qa_assets/examples/requirements.csv \
    --vars output_dir=/tmp/qa_out
```

预期产物 (P3 阶段):

- `/tmp/qa_out/transcript.json` — 多 agent 协作完整 history (topic / turn / speaker / artifact_event / tool_call)
- `/tmp/qa_out/test_plan_artifact.md` — artifact 渲染后的 markdown (6 section: Requirements / 原子需求 / 风险等级 / 测试用例 / 非功能 / Critic 反馈)

## 输入契约 (CSV schema)

详见 plan §4.1. 列定义:

| 列 | 必填 | 含义 |
|---|---|---|
| `req_id` | ✓ | 需求 id (自由格式, 如 `REQ-001`) |
| `title` | ✓ | 一句话标题 |
| `description` | 二选一 | 行内简短描述 (与 `prd_doc_path` 至少有一个) |
| `prd_doc_path` | 二选一 | PRD .md 文件路径 (相对仓库根, **仅 markdown**) |
| `priority` | 否 | P0~P3, 留空时由 risk_grader agent 推断 |
| `assignee` | ✓ | 负责测试的人 |
| `sprint_start` | 否 | ISO date, 元数据透传到输出 |
| `sprint_end` | 否 | 同上 |

> 不接入 docx/xlsx/pdf 等二进制格式 (plan §9 显式不做项). 要喂 PRD 必须先转 markdown.

## 多 agent 角色分工 (qa_discuss.md)

| Agent | role | 职责 | 产出节 |
|---|---|---|---|
| `supervisor` | moderator | 开场 + 协调 + finalize_artifact | — |
| `decomposer` | member | 把每个需求拆为原子 feature + 验收标准 | 原子需求 |
| `risk_grader` | member | 给每个需求打 P0~P3 + 一句话理由 | 风险等级 |
| `case_generator` | member | 产出 functional + boundary/edge 用例 | 测试用例 |
| `nfr_planner` | member | 性能/安全/a11y/i18n 非功能测试点 | 非功能 |
| `critic` | member | 多轮反馈: 覆盖空白 / 优先级矛盾 / 互相冲突 | Critic 反馈 (append) |

step 流: `open → produce (4 specialist 并行) → critic_r1 → revise → critic_r2 → finalize`. 单次 run 把整张 CSV 的全部需求一次性塞进同一份讨论 (plan §8 P3 批处理边界); ≤ ~10 行需求时 context 够用.

## 当前阶段

- **P3 (本提交)**: 4 stage workflow + 多 agent scenario + hooks + 示例输入. 跑通后 artifact 落盘可见.
- **P4**: 接 retrieve_docs 工具 → 历史 PRD / 历史用例 / Bug DB / Domain checklists 的 RAG 检索.
- **P5**: 加 render_md / render_csv stage + Jinja markdown 模板, 最终输出 .md + .csv.

## 显式不做项 (plan §9)

- 真 Confluence/Jira/Figma/TestRail connector
- 独立 Gantt/Schedule 输出
- pytest harness (scenario 仍是 .md 业务工件)
- per-row run-loop (整批一次塞)
