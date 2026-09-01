# Contributing

This repo is primarily a personal vibe-coding sandbox, but the following types of issues / PRs are welcome:

- Bug fixes, or tightening contracts between sub-projects (`agent_engine` ↔ `rag` ↔ `evals` ↔ `workflow`).
- Reproducible benchmarks (including data scale, model version, parameters, and random seeds).
- New self-contained experiments under `play/`.

## 1. Repository layout rules

- New experiments default to [`play/`](play/); at minimum include a README. Add `requirements.txt` and `tests/` as project complexity warrants.
- Promote to longer-lived mini-apps → [`grow/`](grow/).
- Pause work → [`stash/`](stash/).
- Retire experiments → [`_archive/`](_archive/).
- External reference snippets → [`refs/`](refs/) (not first-class code).

The repo has **no monorepo-level `pip install`**. Each sub-project installs and tests independently.

## 2. Pick a sub-project and set it up

```bash
# Example: run evals tests
cd play/                    # module paths assume cwd=play/
python -m venv .venv && source .venv/bin/activate
pip install -r evals/requirements.txt
python -m pytest evals/tests -v
```

Full CLI surface, environment variables, and hardware dependencies are in each `play/<name>/README.md`.

## 3. Run the full CI suite (matches GitHub Actions)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-ci.txt
# One-time VDB build (run under play/rag): test_vdb + panel — matches CI
python -m pytest -v
```

Dependencies:

|Dependency|Purpose|Notes|
|---|---|---|
|Python 3.12+|Repo-wide test runtime|Aligned with GitHub Actions|
|[Ollama](https://ollama.com/) `qwen3.5:9b`|Default live LLM tests|Override with `EVALS_TEST_OLLAMA_MODEL`|
|Ollama `qwen3-embedding:8b`|RAG ingest / query|Default embedding model for CI and local|
|`play/rag/vdb/`|RAG live test inputs|Ingest steps in [CI workflow](.github/workflows/ci.yml)|

`requirements-ci.txt` excludes `mlx-lm` (Apple Silicon only); `play/agent_sft` tests don't depend on it.

## 4. Documentation conventions

|Doc|When to update|Format|
|---|---|---|
|`README.md`|Reader entry point, CLI, architecture diagrams, current status changes|Prefer tables, short paragraphs, and Mermaid diagrams|
|`DECISIONS.md`|Important technical decisions, contract changes, dependency choices, explicit non-goals|Append-only ADR (architecture decision record)|
|`JOURNAL.md`|Milestone progress or per-workday entries|Must include **Functional** / **Technical**; add **Trade-offs** when needed|

Repo-level writing conventions are in [`AGENTS.md`](AGENTS.md) and [`.cursor/rules/workshops.mdc`](.cursor/rules/workshops.mdc).

## 5. Commit messages

- Short English descriptions following conventional commits (`feat(scope):` / `fix(scope):` / `docs(scope):` / `refactor(scope):` / `chore:` / `test:`).
- Use sub-project names for `scope`: `agent_engine` / `rag` / `evals` / `workflow` / `agent_sft` / `qa_assets` / `sft_hello`.
- One commit, one thing; split cross-sub-project changes when possible.

## 6. PR scope

- One PR, one concern — don't mix unrelated refactors into a feature PR.
- When changing contracts (e.g. `--json` envelope, `api.py` dataclasses), update all upstream consumers and add tests.
- PR description should cover: **why / what / how to test**, with eval before/after numbers when relevant.

## 7. CI must pass

Repo CI runs the full `pytest` suite on push / PR — it must be green. Lint / format are optional per sub-project, not enforced repo-wide.

## 8. Larger proposals

For new `play/` sub-projects, new cross-project contracts, or swapping underlying dependencies (e.g. RAG engine, training backend),
**open an issue first** to avoid writing a large change that gets rejected.

—— Thanks for contributing!
