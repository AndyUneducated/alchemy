"""agent_sft Phase 2 data pipeline — mine / extract / format / split scripts.

This layer produces SFT training samples: run batch mining from [`play/agent_engine`](../../agent_engine/)
(failed, nudge, corrected) triplet, converted to MLX-LM standard chat format, cut train/val according to run_id.

| Script | Responsibilities |
|------------------------|------------------------------------------------------------------|
| `mine_triples.py` | The child process runs agent_engine and saves the envelope to `triples/runs/` |
| `extractor.py` | envelope + scenario YAML → `triples.jsonl` (reuse metrics/nudge.py) |
| `split.py` | per-scenario last 20% run_id → val, rest → train |
| `formatter.py` | Triple → MLX-LM F1 chat sample (messages schema) |

All products fall into the [`triples/`](triples/) subdirectory (same location as [`../eval/baselines/`](../eval/baselines/));
For rebirth steps, OOD reuse, and token statistics, see [`triples/README.md`](triples/README.md)."""
