"""sweep.py — 控制变量法（controlled-variable）扫描 LoRA 超参，输出 REPORT.md.

复用 [`play/sft_hello/sweep.py`](../../sft_hello/sweep.py) 的"每个 sweep 只动一个旋钮、
跑完出含浅显解读的 markdown 表"模具，但：

  - 数据：[`data/triples/train_7b_1k.jsonl`](../data/triples/) (DECISIONS §4 schema)
  - 底座：mlx-community/Qwen2.5-7B-Instruct-4bit（QLoRA）
  - 训练：subprocess 调 [`train.py`](train.py)（封装好 mlx_lm.lora）
  - eval：[`eval_smoke.py`](eval_smoke.py) 4 项 tool-call 指标，nudge-fire-rate 的 fast proxy
  - sweep 维度（4 dim × 3-4 值 = 16 runs）：
      * iters / lr / num_layers / rank
  - 失败 / NaN / 非零退出标 diverged，仍记入 REPORT 但 commentary 标"发散".

用法:
    python sweep.py all                  # 跑全部 4 个 sweep
    python sweep.py iters                # 只跑 iters
    python sweep.py iters lr             # 跑指定几个
    python sweep.py report               # 不重跑，仅根据 results.json 重生 REPORT.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAY_DIR = HERE.parent.parent
SWEEPS_DIR = HERE / "runs" / "sweeps"
DEFAULT_CONFIG = HERE / "lora_config.yaml"

MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"
TRAIN_FILE = "train_7b_1k.jsonl"   # 766 train sample
VALID_FILE = "val_7b_1k.jsonl"     # 196 val sample

# 766 sample / batch 4 ≈ 192 iter/epoch；BASE iters=200 ≈ 1 epoch
# （实测 iters=200 已让 train_loss 0.28→0.000——schema 信号高度可压缩；更长 iters
# 主要是 overfit 观察用）。
BASE = {
    "iters": 200,
    "batch_size": 4,
    "num_layers": 16,
    "learning_rate": 1e-4,
    "rank": 16,
}

# 控制变量 sweep（实测：M4 Pro 48GB 上 batch=4 / layers=16 / 4-bit Qwen2.5-7B
# ≈ 18s/iter；原本规划的 4 dim × 4 值 = 16 runs 实际需 50h+，远超 overnight 预算。
# 实际跑 2 个最有信息量的维度 + 6 runs ≈ 8h；layers / rank dim 留 Phase 3.5 follow-up
# 单独再跑（届时可借力 multi-GPU / 云）。详 JOURNAL 2026-05-10 取舍.）
SWEEPS: dict[str, list] = {
    "iters": [50, 200, 600],                 # 0.25 / 1 / 3 epoch — 收敛曲线 + overfit 观察
    "lr": [1e-5, 1e-4, 5e-4],                # LoRA 主流甜点 1e-4，两端各拉一档；1e-3 drop（高发散概率，低信息）
}


def make_temp_config(rank: int, out_dir: Path) -> Path:
    """rank 只能通过 YAML 传 — 为 rank sweep 单独生成临时 YAML（同 alpha=2×rank 比例）."""
    cfg = out_dir / "lora_config.yaml"
    cfg.write_text(
        "lora_parameters:\n"
        '  keys: ["self_attn.q_proj", "self_attn.k_proj", '
        '"self_attn.v_proj", "self_attn.o_proj"]\n'
        f"  rank: {rank}\n"
        f"  scale: 2.0\n"
        "  dropout: 0.05\n"
    )
    return cfg


def run_training(sweep: str, value, adapter_dir: Path, *, force: bool = False) -> dict:
    """跑一次 train.py（内部调 mlx_lm.lora），把结果落 train_metrics.json + train.log.

    Resume：若 `adapter_dir/train_metrics.json` 已存在且 `--force` 没传，直接复用上次结果
    （跳过 train，省 ~60min/run）。eval_smoke 会照常重跑（fast，可重生）.
    """
    metrics_path = adapter_dir / "train_metrics.json"
    if metrics_path.exists() and not force:
        info = json.loads(metrics_path.read_text())
        print(f"[train] {sweep}={value}  →  reusing cached metrics (use --force to retrain)")
        info["sweep"] = sweep
        info["value"] = value
        return info

    adapter_dir.mkdir(parents=True, exist_ok=True)

    iters = BASE["iters"]
    batch_size = BASE["batch_size"]
    num_layers = BASE["num_layers"]
    lr = BASE["learning_rate"]
    config_file = DEFAULT_CONFIG

    if sweep == "iters":
        iters = value
    elif sweep == "lr":
        lr = value
    elif sweep == "layers":
        num_layers = value
    elif sweep == "rank":
        config_file = make_temp_config(value, adapter_dir)
    else:
        raise ValueError(f"unknown sweep {sweep}")

    cmd = [
        sys.executable, str(HERE / "train.py"),
        "--model", MODEL_ID,
        "--train-file", TRAIN_FILE,
        "--valid-file", VALID_FILE,
        "--config", str(config_file),
        "--iters", str(iters),
        "--batch-size", str(batch_size),
        "--num-layers", str(num_layers),
        "--learning-rate", f"{lr:g}",
        "--adapter-path", str(adapter_dir),
    ]
    print(f"\n[train] {sweep}={value}")
    print(f"        cmd: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(PLAY_DIR))
    elapsed = time.time() - t0

    metrics_path = adapter_dir / "train_metrics.json"
    if metrics_path.exists():
        train_info = json.loads(metrics_path.read_text())
    else:
        train_info = {"diverged": True, "returncode": proc.returncode}
    train_info["elapsed_s"] = round(elapsed, 1)
    train_info["sweep"] = sweep
    train_info["value"] = value
    return train_info


def run_eval(adapter_dir: Path, max_samples: int | None) -> dict:
    """跑 eval_smoke.py 出 4 项 tool-call 指标."""
    cmd = [
        sys.executable, str(HERE / "eval_smoke.py"),
        "--model", MODEL_ID,
        "--adapter-path", str(adapter_dir),
        "--valid-file", str(HERE.parent / "data" / "triples" / VALID_FILE),
    ]
    if max_samples:
        cmd.extend(["--max-samples", str(max_samples)])
    print(f"[eval]  adapter={adapter_dir.relative_to(HERE)}")
    proc = subprocess.run(cmd, cwd=str(PLAY_DIR))
    eval_path = adapter_dir / "eval_smoke.json"
    if proc.returncode != 0 or not eval_path.exists():
        return {"error": f"eval failed rc={proc.returncode}"}
    return json.loads(eval_path.read_text())


def run_sweep(sweep: str, max_eval_samples: int | None, *, force: bool = False) -> list[dict]:
    out: list[dict] = []
    for value in SWEEPS[sweep]:
        adapter_dir = SWEEPS_DIR / sweep / str(value)
        train_info = run_training(sweep, value, adapter_dir, force=force)
        if train_info.get("diverged"):
            eval_info = {"error": "skipped due to divergence"}
        else:
            eval_info = run_eval(adapter_dir, max_eval_samples)
        out.append({**train_info, "eval": eval_info})
    return out


# ----- REPORT.md -----------------------------------------------------------

SWEEP_HEAD = {
    "iters": {
        "title": "训练步数 `--iters`（iterations）",
        "what": (
            "每次梯度更新叫一个 **iter / step**。766 个训练样本、batch=4 时 1 epoch ≈ 192 iter，"
            "所以 `iters=600` 约等于 3 个 epoch（每条样本平均被看 3 次）。"
        ),
        "why": (
            "tool-call schema 是个**结构性任务**——模型要学 `<tool_call>{...}</tool_call>` "
            "形态 + 把 instruction 文本里的字面值搬进 JSON dict。iters 太少没学透形态；"
            "太多会把 766 条 corrected 模板**死记**下来，泛化到训练集外的 args 时变差。"
        ),
    },
    "lr": {
        "title": "学习率 `--learning-rate` (learning rate, LR)",
        "what": "每次更新参数的步长——梯度告诉方向，LR 决定走多远。",
        "why": (
            "LoRA 因可训参数少，承受比全量微调（典型 1e-5）大一个数量级的 LR。1e-4 是 LoRA 主流甜点；"
            "5e-4 / 1e-3 探激进上限；1e-5 探『训不动』下限。**最容易训坏的旋钮**——loss 单调降 OK，"
            "震荡 / NaN 即偏大。"
        ),
    },
    "layers": {
        "title": "LoRA 挂载层数 `--num-layers`",
        "what": (
            "在最顶上 N 层 transformer block 挂 LoRA 旁路。Qwen2.5-7B 共 28 层；挂 16 层 = 上半部分；"
            "挂 28 层 = 全挂；挂 4 层 = 仅离输出最近的几层。"
        ),
        "why": (
            "底层负责通用语法 / token embedding；中上层负责风格 / 任务策略 / 结构生成（如 "
            "`<tool_call>` 形态）。tool-call 是结构性 + 风格性混合任务，挂中上层最划算；"
            "全挂可能学得更深但易破坏底层能力（**灾难性遗忘 catastrophic forgetting**）。"
        ),
    },
    "rank": {
        "title": "瓶颈秩 `rank`（YAML，r in LoRA）",
        "what": (
            "LoRA 把权重改动写成 `A·B`，中间挤过一个 **r 维**瓶颈（bottleneck）。"
            "r 越小 = 可训参数越少 = 表达力越受限。"
        ),
        "why": (
            "tool-call SFT 比 toy task 信号丰富（多工具 / 多参数 schema 形态），"
            "需要的有效秩高于 toy。8-32 是工业实战区间；r=4 测下限是否仍学得动；"
            "r=32 测 ΔW 是否真低秩；中间 8 / 16 是主流候选。"
        ),
    },
}


def value_commentary(sweep: str, value, train: dict, eval_: dict) -> str:
    base = BASE[{"iters": "iters", "lr": "learning_rate",
                 "layers": "num_layers", "rank": "rank"}[sweep]]
    diverged = train.get("diverged", False)
    final = train.get("train_loss_last")
    initial = train.get("train_loss_first")
    val_last = train.get("val_loss_last")
    emit = (eval_ or {}).get("tool_call_emit_rate")
    name = (eval_ or {}).get("tool_name_match_rate")
    arg_v = (eval_ or {}).get("arg_value_match_rate")

    if diverged:
        return (
            "训练发散（diverged）：loss NaN / 跑飞或 mlx_lm.lora 非零退出。"
            "**典型原因**：LR 过大、QLoRA 4-bit 精度遇病态、数据 schema bug。adapter 不可用，"
            "eval 跳过。"
        )

    metrics_str = (
        f"train_loss {initial:.2f}→{final:.2f}"
        + (f"，val_loss_last {val_last:.2f}" if val_last is not None else "")
        + (f"，emit {emit:.0%} / name {name:.0%} / arg_value {arg_v:.0%}"
           if emit is not None else "")
    )
    if value == base:
        head = "**基线**"
    elif sweep == "iters":
        if value < base:
            head = "**欠拟合候选**" if value <= base // 2 else "**少 epoch**"
        else:
            head = "**深度过拟合候选**" if value >= base * 4 else "**多 epoch**"
    elif sweep == "lr":
        if value < base:
            head = "**步太小**" if value <= base / 5 else "**偏保守**"
        else:
            head = "**激进 / 易发散**" if value >= base * 5 else "**偏激进**"
    elif sweep == "layers":
        if value < base:
            head = "**容量受限**" if value <= base // 2 else "**少层**"
        else:
            head = "**全挂 / 易遗忘**" if value >= 28 else "**多层**"
    else:  # rank
        if value < base:
            head = "**极低秩**" if value <= 4 else "**低秩**"
        else:
            head = "**冗余秩**"

    return f"{head}：{metrics_str}。"


def fmt_loss(v):
    return f"{v:.2f}" if isinstance(v, (int, float)) and v == v else "-"  # NaN check


def fmt_pct(v):
    if v is None or v != v:
        return "-"
    return f"{v:.0%}"


def write_report(all_results: dict[str, list[dict]]) -> None:
    lines: list[str] = []
    lines.append("# LoRA 超参 sweep 报告（agent_sft Phase 3）\n")
    lines.append(
        "本报告由 [`sweep.py`](../../sweep.py) 自动生成。每个 sweep 中只动**一个**超参，"
        "其余保持基线值不变（控制变量法 controlled-variable）。\n"
    )
    lines.append(
        "训练数据 `train_7b_1k.jsonl` (766 sample / 196 val)，schema 见 [`DECISIONS §4`](../../../DECISIONS.md)；"
        "底座 `mlx-community/Qwen2.5-7B-Instruct-4bit` (QLoRA)；评估走 [`eval_smoke.py`](../../eval_smoke.py)，"
        "解析模型输出里 `<tool_call>` 块与 ground-truth 比对.\n"
    )

    lines.append("## 基线配置（baseline）\n")
    lines.append("|参数|值|")
    lines.append("|---|---|")
    for k, v in BASE.items():
        lines.append(f"|`{k}`|{v}|")
    lines.append("")

    for sweep_name, results in all_results.items():
        head = SWEEP_HEAD.get(sweep_name)
        if head is None:
            continue
        lines.append(f"## {head['title']}\n")
        lines.append(f"**它做什么**：{head['what']}\n")
        lines.append(f"**为什么会有差异**：{head['why']}\n")

        lines.append("### 实测结果\n")
        lines.append("|值|首 loss|末 loss|val loss|emit|name|arg_value|耗时|备注|")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in results:
            ev = r.get("eval") or {}
            note = "发散" if r.get("diverged") else ""
            v = r["value"]
            v_str = f"`{v:g}`" if isinstance(v, float) else f"`{v}`"
            lines.append(
                f"|{v_str}|{fmt_loss(r.get('train_loss_first'))}"
                f"|{fmt_loss(r.get('train_loss_last'))}"
                f"|{fmt_loss(r.get('val_loss_last'))}"
                f"|{fmt_pct(ev.get('tool_call_emit_rate'))}"
                f"|{fmt_pct(ev.get('tool_name_match_rate'))}"
                f"|{fmt_pct(ev.get('arg_value_match_rate'))}"
                f"|{r.get('elapsed_s', '-')}s|{note}|"
            )
        lines.append("")

        lines.append("### 逐值解读\n")
        for r in results:
            v = r["value"]
            v_str = f"{v:g}" if isinstance(v, float) else str(v)
            lines.append(f"- **`{v_str}`** — {value_commentary(sweep_name, v, r, r.get('eval'))}")
        lines.append("")

    lines.append("## 通用结论速查\n")
    lines.append(
        "- **学习率最容易训坏**——先把它钉对，再调其他。判据：loss 单调降 = 合适；"
        "震荡 = 偏大；NaN = 远超.\n"
        "- **iters × batch_size = 实际学习量**——同 epoch 数下两者可换算.\n"
        "- **rank 16 是 tool-call SFT 实战起步**——4 试下限，32 测是否真需要更高表达力.\n"
        "- **挂 16 层是经济 + 学得到位的折中**——全挂 (28) 易破坏底层能力，仅 4 层装不下 schema.\n"
        "- **emit_rate 比 val_loss 更对位下游 nudge-fire-rate**——loss 低未必 emit 真的对，"
        "tool_name_match / arg_value_match 才是结构性指标.\n"
    )

    out = SWEEPS_DIR / "REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✓ 报告已生成：{out.relative_to(HERE)}")


# ----- IO ------------------------------------------------------------------

def load_or_init_results() -> dict:
    p = SWEEPS_DIR / "results.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_results(d: dict) -> None:
    SWEEPS_DIR.mkdir(parents=True, exist_ok=True)
    (SWEEPS_DIR / "results.json").write_text(
        json.dumps(d, ensure_ascii=False, indent=2)
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "sweeps", nargs="*",
        help=f"sweep 名（{list(SWEEPS) + ['all', 'report']}），默认 all",
    )
    parser.add_argument(
        "--max-eval-samples", type=int, default=None,
        help="每次 eval_smoke 限制 sample 数（用于 sweep 总时长，默认全集 196）",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="覆盖已存在的 train_metrics.json，重训每个值（默认 resume 跳过已完成 run）",
    )
    args = parser.parse_args()

    targets = args.sweeps or ["all"]
    if "report" in targets:
        results = load_or_init_results()
        if not results:
            print("results.json 不存在或为空，请先跑 sweep。", file=sys.stderr)
            return 1
        write_report(results)
        return 0

    if "all" in targets:
        targets = list(SWEEPS)
    unknown = [t for t in targets if t not in SWEEPS]
    if unknown:
        print(f"未知 sweep: {unknown}；可选 {list(SWEEPS)}", file=sys.stderr)
        return 1

    results = load_or_init_results()
    for sweep in targets:
        print(f"\n========== sweep: {sweep} ==========")
        results[sweep] = run_sweep(sweep, args.max_eval_samples, force=args.force)
        save_results(results)
    write_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
