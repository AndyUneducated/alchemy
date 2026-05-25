# 贡献指南 / Contributing

这个仓库主要是个人 vibe-coding 沙盒，但欢迎以下类型的 issue / PR：

- 修 bug，或把子项目之间的契约收紧（`agent_engine` ↔ `rag` ↔ `evals` ↔ `workflow`）。
- 补可复现的 benchmark（含数据规模、模型版本、参数、随机种子）。
- 在 `play/` 下新增一个 self-contained 的实验。

## 1. 仓库布局规则

- 新实验默认放到 [`play/`](play/) 下，每个子项目自带 README、`requirements.txt`、`tests/`。
- 提拔到长期维护的小应用 → [`grow/`](grow/)。
- 暂停的工作 → [`stash/`](stash/)。
- 退役实验 → [`_archive/`](_archive/)。
- 外部参考片段 → [`refs/`](refs/)（不是一等代码）。

仓库**没有 monorepo 级 `pip install`**。每个子项目独立安装、独立跑测试。

## 2. 选一个子项目并装好它

```bash
# 例：跑 evals 的测试
cd play/                    # 模块路径都假设 cwd=play/
python -m venv .venv && source .venv/bin/activate
pip install -r evals/requirements.txt
python -m pytest evals/tests -v
```

完整 CLI 表面、环境变量与硬件依赖见每个 `play/<name>/README.md`。

## 3. 跑完整 CI 套件（与 GitHub Actions 一致）

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-ci.txt
# 一次性建 VDB（在 play/rag 下跑）：test_vdb + panel —— 与 CI 同步
python -m pytest -v
```

依赖：

- Python 3.12+
- [Ollama](https://ollama.com/) 拉好 `qwen3.5:9b`（或设 `EVALS_TEST_OLLAMA_MODEL`）+ `qwen3-embedding:8b`。
- `play/rag/vdb/` 下 ingest 好的 VDB（具体步骤见 [CI workflow](.github/workflows/ci.yml)）。

`requirements-ci.txt` 不装 `mlx-lm`（只跑 Apple Silicon）；`play/agent_sft` 的测试不依赖它。

## 4. 文档约定

- **重要技术决策**写到子项目自己的 `DECISIONS.md`（append-only，ADR 风格）。
- **阶段性进展**写到子项目自己的 `JOURNAL.md`（一段话 / 一个阶段）。
- 仓库级写作约定见 [`AGENTS.md`](AGENTS.md) 和 [`.cursor/rules/workshops.mdc`](.cursor/rules/workshops.mdc)。

## 5. Commit 信息

- 用英文简短说明，遵循 conventional commits（`feat(scope):` / `fix(scope):` / `docs(scope):` / `refactor(scope):` / `chore:` / `test:`）。
- `scope` 用子项目名：`agent_engine` / `rag` / `evals` / `workflow` / `agent_sft` / `qa_assets` / `sft_hello`。
- 一个 commit 只做一件事；跨子项目的改动尽量拆开。

## 6. PR 范围

- 一个 PR 只解决一件事，避免把无关重构混进 feature PR。
- 改契约（如 `--json` envelope、`api.py` dataclass）时，请同时更新所有上游消费者并加测试。
- PR 描述里写：**为什么改 / 改了什么 / 怎么测**，必要时贴 eval 前后数字。

## 7. CI 必须通过

仓库 CI 在 push / PR 上跑完整 `pytest` 套件，必须绿。lint / format 是子项目可选的，不强制全仓打开。

## 8. 较大的提案

新增 `play/` 子项目、引入新的跨项目契约、或换底层依赖（如 RAG 引擎、训练 backend），
**先开 issue 讨论**，避免你写完一大块再被劝退。

—— 感谢贡献！
