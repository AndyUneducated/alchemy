"""Per-scenario by-run_id train/val splitter.

Plan §Decisions：per-scenario 末 20% run_id 作 val，其余 → train.
Fallback：当某 scenario 的 unique run_ids 数 < 5 时，全归 train（val 数据点太少
失去统计意义；典型见 pilot 阶段 3 run_id × 1-2 triples/run）.

输入应当是 `triples.jsonl`（含 scenario / run_id 元数据），不能是 formatter
输出（messages 格式丢了元数据）；formatter 应在本步骤之后跑.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_VAL_RATIO = 0.2
MIN_RUN_IDS_FOR_VAL = 5


def split_train_val(
    samples: list[dict[str, Any]],
    *,
    val_ratio: float = DEFAULT_VAL_RATIO,
    min_run_ids_for_val: int = MIN_RUN_IDS_FOR_VAL,
    scenario_key: str = "scenario",
    run_id_key: str = "run_id",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """切 (train, val)，per-scenario 独立按 run_id 取末 20%.

    边界：
      - 某 scenario 仅有 < min_run_ids_for_val 个 unique run_id → 全 train
      - run_ids 排序后取末 max(1, floor(N * val_ratio)) 个进 val
      - 空输入 → ([], [])
    """
    by_scen: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_scen[s.get(scenario_key, "_unknown")].append(s)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for items in by_scen.values():
        run_ids = sorted({int(s.get(run_id_key, 0)) for s in items})
        if len(run_ids) < min_run_ids_for_val:
            train.extend(items)
            continue
        n_val = max(1, int(len(run_ids) * val_ratio))
        val_ids = set(run_ids[-n_val:])
        for s in items:
            if int(s.get(run_id_key, 0)) in val_ids:
                val.append(s)
            else:
                train.append(s)
    return train, val


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _write_jsonl(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--in", dest="in_path", required=True,
        help="输入 triples.jsonl（必须含 scenario + run_id 字段）",
    )
    parser.add_argument(
        "--train", required=True, help="train split 输出 jsonl 路径",
    )
    parser.add_argument(
        "--val", required=True, help="val split 输出 jsonl 路径",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=DEFAULT_VAL_RATIO,
        help=f"val 比例（默认 {DEFAULT_VAL_RATIO}）",
    )
    parser.add_argument(
        "--min-run-ids-for-val", type=int, default=MIN_RUN_IDS_FOR_VAL,
        help=f"unique run_id 数 < 此阈值 → fallback 全 train（默认 {MIN_RUN_IDS_FOR_VAL}）",
    )
    args = parser.parse_args(argv)

    samples = _read_jsonl(Path(args.in_path))
    if samples and "scenario" not in samples[0]:
        print(
            "ERROR: input lacks 'scenario' key. Split must run BEFORE formatter "
            "(formatter output drops metadata).",
            file=sys.stderr,
        )
        return 2

    train, val = split_train_val(
        samples,
        val_ratio=args.val_ratio,
        min_run_ids_for_val=args.min_run_ids_for_val,
    )
    _write_jsonl(train, Path(args.train))
    _write_jsonl(val, Path(args.val))
    print(f"split: train={len(train)}  val={len(val)}  (input={len(samples)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
