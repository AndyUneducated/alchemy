"""split.py — per-scenario by-run_id 切分边界覆盖.

Plan §test_split.py:
  - per-scenario 末 20% 边界（含 ratio 取整 / 单 scenario / triple 数 < 5 fallback）
  - 空集合行为
  - multi-scenario 各自独立切分
"""

from __future__ import annotations

from split import (  # type: ignore[import-not-found]
    DEFAULT_VAL_RATIO,
    MIN_RUN_IDS_FOR_VAL,
    split_train_val,
)


def make(scenario, run_id, n_per=1):
    """生成 n_per 条同 (scenario, run_id) 的虚拟 triple."""
    return [
        {"scenario": scenario, "run_id": run_id, "_idx": i}
        for i in range(n_per)
    ]


def make_n_runs(scenario, n_run_ids, samples_per_run=1):
    out = []
    for r in range(n_run_ids):
        out.extend(make(scenario, r, samples_per_run))
    return out


def test_10_run_ids_split_2_val():
    samples = make_n_runs("A", 10, samples_per_run=3)  # 30 samples, 10 runs
    train, val = split_train_val(samples)
    val_run_ids = sorted({s["run_id"] for s in val})
    assert val_run_ids == [8, 9]  # 末 20% = 末 2 个 run_id
    assert len(val) == 2 * 3  # 2 runs × 3 samples = 6
    assert len(train) == 8 * 3
    assert len(train) + len(val) == 30


def test_5_run_ids_split_1_val():
    """N=5, ratio=0.2 → floor(5*0.2)=1 → val 1 run."""
    samples = make_n_runs("A", 5)
    train, val = split_train_val(samples)
    assert {s["run_id"] for s in val} == {4}
    assert len(train) == 4
    assert len(val) == 1


def test_4_run_ids_falls_back_to_all_train():
    """N=4 < MIN_RUN_IDS_FOR_VAL=5 → val 空，全 train."""
    samples = make_n_runs("A", 4)
    train, val = split_train_val(samples)
    assert val == []
    assert len(train) == 4


def test_single_run_id_falls_back_to_all_train():
    samples = make("A", 0, n_per=10)
    train, val = split_train_val(samples)
    assert val == []
    assert len(train) == 10


def test_empty_input_returns_empty_splits():
    train, val = split_train_val([])
    assert train == [] and val == []


def test_multi_scenario_split_independent():
    """每个 scenario 独立计算末 20% — A 5 runs, B 10 runs → 各 1 + 2 val."""
    samples = make_n_runs("A", 5) + make_n_runs("B", 10)
    train, val = split_train_val(samples)

    val_a = [s for s in val if s["scenario"] == "A"]
    val_b = [s for s in val if s["scenario"] == "B"]
    assert {s["run_id"] for s in val_a} == {4}
    assert {s["run_id"] for s in val_b} == {8, 9}

    assert len(val) == 1 + 2
    assert len(train) == 4 + 8


def test_ratio_floor_rounding():
    """N=12, ratio=0.2 → floor(12*0.2)=2 (不四舍五入到 3)."""
    samples = make_n_runs("A", 12)
    train, val = split_train_val(samples)
    assert {s["run_id"] for s in val} == {10, 11}


def test_ratio_zero_runs_yields_at_least_one_val_when_above_threshold():
    """N=5 即使 ratio 极小也给 1 val（max(1, ...) 兜底），仅当 >= MIN 时."""
    samples = make_n_runs("A", 5)
    train, val = split_train_val(samples, val_ratio=0.01)
    assert len(val) == 1  # max(1, floor(5*0.01)) = 1


def test_min_run_ids_threshold_overrides_ratio():
    """4 run < MIN_RUN_IDS_FOR_VAL → 不论 ratio 多大都全 train."""
    samples = make_n_runs("A", 4)
    train, val = split_train_val(samples, val_ratio=0.5)
    assert val == [] and len(train) == 4


def test_val_run_ids_are_the_LAST_run_ids_not_first():
    """Sanity: val 始终是 SORTED run_ids 的尾部（保 in-dist split 语义）."""
    # 故意打乱顺序输入
    import random
    rng = random.Random(0)
    raw_run_ids = list(range(7))
    rng.shuffle(raw_run_ids)
    samples: list[dict] = []
    for r in raw_run_ids:
        samples.append({"scenario": "A", "run_id": r})
    train, val = split_train_val(samples)
    val_ids = {s["run_id"] for s in val}
    train_ids = {s["run_id"] for s in train}
    # 末 20% of 7 = floor(1.4)=1 → val=[6]
    assert val_ids == {6}
    assert max(train_ids) < min(val_ids)


def test_default_constants_match_plan():
    assert DEFAULT_VAL_RATIO == 0.2
    assert MIN_RUN_IDS_FOR_VAL == 5
