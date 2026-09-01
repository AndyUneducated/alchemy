"""The evals consumer-side script of the agent_sft project - multi-seed runner/aggregator/experiment report.

agent_sft does not hold the evaluation component (task / metric all belong to [`play/evals/`](../../evals/)), this directory only installs:
  - `run_baseline.py` — multiple seed runners, adjust `python -m evals run`
  - `aggregate_seeds.py` — Read `evals/runs/index.jsonl` for cross-seed aggregation
  - `baselines/` — markdown experiment report output directory

For usage instructions, see [`baselines/README.md`](baselines/README.md)."""
