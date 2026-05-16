# ai-workshops

[![CI](https://github.com/AndyUneducated/ai-workshops/actions/workflows/ci.yml/badge.svg)](https://github.com/AndyUneducated/ai-workshops/actions/workflows/ci.yml)
[![codecov](https://img.shields.io/badge/coverage-pending-lightgrey.svg)](https://codecov.io)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![repo size](https://img.shields.io/github/repo-size/AndyUneducated/ai-workshops)](https://github.com/AndyUneducated/ai-workshops)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![pytest](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E.svg?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![sentence-transformers](https://img.shields.io/badge/sentence--transformers-rerank-FFCB05.svg)](https://www.sbert.net/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vector_store-FF6F61.svg)](https://www.trychroma.com/)
[![BM25](https://img.shields.io/badge/BM25-lexical-555.svg)](https://github.com/dorianbrown/rank_bm25)
[![Ollama](https://img.shields.io/badge/Ollama-local_LLM-000000.svg?logo=ollama&logoColor=white)](https://ollama.com/)
[![MLX-LM](https://img.shields.io/badge/MLX--LM-Apple_Silicon-007AFF.svg?logo=apple&logoColor=white)](https://github.com/ml-explore/mlx-lm)
[![QLoRA](https://img.shields.io/badge/QLoRA-SFT-purple.svg)](https://github.com/artidoro/qlora)
[![Ragas](https://img.shields.io/badge/Ragas-eval-7B61FF.svg)](https://docs.ragas.io/)
[![NumPy](https://img.shields.io/badge/NumPy-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![pandas](https://img.shields.io/badge/pandas-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)

> Personal **vibe-coding sandbox** for LLM engineering — local RAG, multi-agent scenarios, declarative workflows, and an lm-evaluation-harness-style eval stack, wired into a closed-loop SFT experiment. Not a single shipped product; many small `play/` experiments with shared contracts.

## Why ai-workshops

LLM engineering is five different problems wearing one hat. Each problem reaches for a different tool, and a one-size-fits-all framework either over-abstracts (LangChain) or leaves you wiring glue forever (raw scripts). This sandbox keeps the problems separated, but tied together by stable contracts so a fix in one shows up in the next.

| Real engineering need | What `play/` ships for it | Why a single off-the-shelf tool falls short |
|---|---|---|
| *"Run a multi-turn agent with tools, memory, artifacts."* | [`agent_engine`](play/agent_engine/) — markdown scenarios + step-driven loop | LangChain agent loops are opaque and hard to unit-test; one-shot evals miss planning / nudge failures. |
| *"Retrieve from a few hundred local docs, hybrid + rerank, no cloud."* | [`rag`](play/rag/) — Chroma + BM25 RRF + optional cross-encoder | Pure dense vectors miss keyword hits; managed services demand egress and a credit card for a sandbox. |
| *"Catch eval regressions across many tasks, with adapter parity."* | [`evals`](play/evals/) — task-declarative harness, JSONL runs, IAA + Ragas + IR metrics | `lm-eval` is rigid for custom tasks; notebook scoring isn't reproducible or CI-able. |
| *"Compose deterministic hooks and agent stages in one pipeline."* | [`workflow`](play/workflow/) — linear YAML runner | LangGraph is overkill for linear plans; bash glue isn't testable; Airflow is a different planet. |
| *"Mine traces → fine-tune → redeploy → re-measure, end to end."* | [`agent_sft`](play/agent_sft/) — nudge mining + QLoRA + Ollama + eval re-run | Off-the-shelf SFT recipes skip trajectory mining and the closing eval delta — the only thing that proves the loop worked. |

The reference [`qa_assets/`](play/qa_assets/) vertical slice exercises four of the five at once, so the contracts get tested by use, not just unit tests.

## What it does

Production-minded spikes that compose into one story:

1. **Run agents** — Markdown scenarios drive multi-turn discussions with tools, memory, and artifacts ([`play/agent_engine/`](play/agent_engine/)).
2. **Retrieve locally** — Hybrid dense + BM25 + optional rerank over a self-describing on-disk VDB ([`play/rag/`](play/rag/)).
3. **Evaluate** — Task-declarative harness with `score` / `run` parity, JSONL run storage, and phased metric families ([`play/evals/`](play/evals/)).
4. **Orchestrate** — Linear YAML pipelines mixing deterministic hooks and agent stages ([`play/workflow/`](play/workflow/)).
5. **Close the loop** — Mine `require_tool` nudge traces from the engine, QLoRA-tune a 7B model, deploy via Ollama, re-measure with evals ([`play/agent_sft/`](play/agent_sft/)).

A reference vertical slice ties it together: QA test-plan generation ([`play/qa_assets/`](play/qa_assets/)) runs `qa_supervisor.yaml` through workflow → agent_engine → rag.

```mermaid
flowchart LR
  subgraph assets["Domain assets"]
    qa["play/qa_assets<br/>workflows · scenarios · kb"]
  end
  subgraph core["Core engines"]
    wf["play/workflow"]
    ae["play/agent_engine"]
    rag["play/rag"]
    ev["play/evals"]
  end
  subgraph train["Training loop"]
    sft["play/agent_sft"]
  end
  qa --> wf --> ae
  ae -->|retrieve_docs subprocess| rag
  ae -->|transcripts| sft
  sft -->|ollama model| ae
  ae --> ev
  rag --> ev
```

## Repository layout

|Path|Purpose|
|---|---|
|[`play/`](play/)|Default home for spikes, scripts, and demos (each sub-project has its own README)|
|[`grow/`](grow/)|Longer-lived mini-apps promoted from `play/`|
|[`stash/`](stash/)|Paused work-in-progress|
|[`refs/`](refs/)|Copied reference snippets — not first-class product code|
|[`_archive/`](_archive/)|Retired experiments|
|[`AGENTS.md`](AGENTS.md)|Notes for coding agents (Cursor rules, doc conventions)|

New experiments belong under `play/` unless you choose another path explicitly.

## Projects

|Directory|One-liner|Docs|
|---|---|---|
|[`play/agent_engine/`](play/agent_engine/)|Step-driven multi-agent engine (scenario = YAML frontmatter + markdown body)|[README](play/agent_engine/README.md)|
|[`play/rag/`](play/rag/)|Local-first hybrid RAG (Chroma + BM25 RRF, optional cross-encoder rerank)|[README](play/rag/README.md)|
|[`play/evals/`](play/evals/)|lm-eval-style LLM evaluation harness (tasks, adapters, JSONL runs)|[README](play/evals/README.md)|
|[`play/workflow/`](play/workflow/)|Declarative linear pipeline runner (hooks + agent stages)|[README](play/workflow/README.md)|
|[`play/agent_sft/`](play/agent_sft/)|Nudge-grounded SFT on agent trajectories (mine → QLoRA → Ollama → re-eval)|[README](play/agent_sft/README.md)|
|[`play/qa_assets/`](play/qa_assets/)|QA domain assets (workflows, scenarios, hooks, kb, example CSV/PRD)|[README](play/qa_assets/README.md)|
|[`play/sft_hello/`](play/sft_hello/)|One-shot MLX-LM hello-world fine-tune (pipeline smoke test)|[README](play/sft_hello/README.md)|

Sub-projects with non-trivial design choices also keep append-only [`DECISIONS.md`](play/evals/DECISIONS.md) (ADR-style) and [`JOURNAL.md`](play/evals/JOURNAL.md) (milestones) beside their README — see any `play/<name>/` that has them.

## Quick start

There is **no monorepo-wide `pip install`**. Each project owns a `requirements.txt` and is run from `play/` (module paths assume that cwd).

**Example — run the QA supervisor workflow** (needs Ollama for embeddings + LLM backends configured in agent_engine):

```bash
cd play/
python -m venv .venv && source .venv/bin/activate
pip install -r workflow/requirements.txt -r agent_engine/requirements.txt -r rag/requirements.txt -r qa_assets/requirements.txt

# Optional: build qa_kb VDB if scenarios use retrieve_docs
cd rag && python ingest.py --docs ../qa_assets/kb --output ../qa_assets/vdb/qa_kb && cd ..

python -m workflow run qa_assets/workflows/qa_supervisor.yaml \
  --vars csv_path=qa_assets/examples/req_tracker.csv \
  --vars output_dir=/tmp/qa_out
```

**Example — list eval tasks and score predictions**:

```bash
cd play/
pip install -r evals/requirements.txt
python -m evals list-tasks
python -m evals score --task <name> --predictions path/to/preds.jsonl
```

See each project README for full CLI surfaces, env vars, and hardware notes (Apple Silicon + MLX for SFT paths; local Ollama for RAG embeddings and inference).

### Running tests locally

Requires **Python 3.12+**, [Ollama](https://ollama.com/) with `qwen2.5:7b` (or set `EVALS_TEST_OLLAMA_MODEL`) and `qwen3-embedding:8b`, plus ingested VDBs under `play/rag/vdb/` (see [CI workflow](.github/workflows/ci.yml) ingest steps).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-ci.txt
# build VDBs once (from play/rag): test_vdb + panel — same as CI
python -m pytest -v
```

CI uses `requirements-ci.txt` (excludes `mlx-lm`, which is Apple Silicon–only; `play/agent_sft` tests do not require it).

## Principles (repo-wide)

|#|Principle|
|---|---|
|1|**Experiments over products** — optimize for learning and composability, not a unified release|
|2|**Contracts at boundaries** — e.g. RAG `--json` envelope consumed by agent_engine subprocess; evals `api.py` dataclasses across layers|
|3|**YAGNI** — repo CI runs the full `pytest` suite on push/PR; add lint/format only when a sub-project needs it|
|4|**Document decisions** — important technical choices go to per-project `DECISIONS.md`; substantive progress to `JOURNAL.md`|

Cursor authoring rules live in [`.cursor/rules/workshops.mdc`](.cursor/rules/workshops.mdc).

## What this repo is not

- Not a framework release with semver or stable public APIs
- Not a hosted service or Terraform stack
- Not guaranteed reproducible without local models (Ollama tags, HF caches) and API keys where cloud backends are used

Large generated artifacts (VDB dirs, eval `runs/`, most training checkpoints) are gitignored; see [`.gitignore`](.gitignore).

## Contributing

This repo is primarily a personal sandbox, but issues and PRs that fix bugs, sharpen contracts between sub-projects, or add reproducible benchmarks are welcome.

1. **Pick a sub-project.** Each `play/<name>/` is independently runnable; read its README and (where present) `DECISIONS.md` / `JOURNAL.md` first.
2. **Set up a venv inside `play/`.** Module paths assume `cwd=play/`. Install only the sub-project's `requirements.txt` (no monorepo install).
3. **Run the relevant tests.** `python -m pytest play/<name>/tests` from the repo root, or the full suite via `pip install -r requirements-ci.txt && python -m pytest -v` (mirrors CI; needs Ollama + VDBs as documented above).
4. **Document load-bearing decisions.** Substantive technical choices land in the sub-project's `DECISIONS.md` (ADR-style); shipped milestones in `JOURNAL.md`. Authoring conventions live in [`AGENTS.md`](AGENTS.md) and [`.cursor/rules/workshops.mdc`](.cursor/rules/workshops.mdc).
5. **Open a focused PR.** Keep cross-cutting refactors out of feature PRs; CI must stay green.

For larger proposals (new `play/` project, breaking contract change), open an issue first.

## License

[Apache License 2.0](LICENSE).
