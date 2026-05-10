"""agent_sft Phase 2 data pipeline — mine / extract / format / split scripts.

这一层产出 SFT 训练样本：从 [`play/agent_engine`](../../agent_engine/) 跑批挖
(failed, nudge, corrected) 三元组，转成 MLX-LM 标准 chat 格式，按 run_id 切 train/val.

| 脚本                | 职责                                                              |
|---------------------|-------------------------------------------------------------------|
| `mine_triples.py`   | 子进程跑 agent_engine，存 envelope 到 `triples/runs/`             |
| `extractor.py`      | envelope + scenario YAML → `triples.jsonl`（复用 metrics/nudge.py）|
| `split.py`          | per-scenario 末 20% run_id → val，其余 → train                    |
| `formatter.py`      | Triple → MLX-LM F1 chat sample (messages schema)                  |

所有产物落到 [`triples/`](triples/) 子目录（与 [`../eval/baselines/`](../eval/baselines/) 同位）；
重生步骤、OOD 复用、token 统计见 [`triples/README.md`](triples/README.md).
"""
