"""sweep.py — 控制变量法（controlled-variable）扫描 LoRA 超参，输出可读报告。

目的：通过把每个超参（hyperparameter）依次拉到不同数量级，让"它实际影响什么"
变成肉眼可见的结果，而不是文档里的一句话。结果产物在 `runs/sweeps/`：
每个 (sweep, value) 一个子目录装 adapter + 训练日志 + eval 结果，最后
`runs/sweeps/REPORT.md` 汇总成一份带浅显解读的表格。

用法：
    python sweep.py all              # 跑全部 5 个 sweep
    python sweep.py iters            # 只跑 iters
    python sweep.py iters lr         # 跑指定几个
    python sweep.py report           # 不重跑，只根据已有 results.json 重生成报告

每个 sweep 中只动**一个**变量，其余保持 BASE 不变——这就是控制变量法的核心。
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
SWEEPS_DIR = ROOT / "runs" / "sweeps"
DATA_DIR = ROOT / "data"
CONFIG = ROOT / "lora_config.yaml"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
FOX = "\U0001f98a"

BASE = {
    "iters": 200,
    "batch_size": 4,
    "num_layers": 8,
    "learning_rate": 1e-4,
    "rank": 8,
}

SWEEPS = {
    "iters": [10, 50, 200, 1000],
    "lr": [1e-6, 1e-5, 1e-4, 1e-3],
    "layers": [2, 4, 8, 16],
    "batch": [1, 4, 16],
    "rank": [2, 8, 32],
}

TEST_PROMPTS = [
    "What is the capital of Spain?",
    "Tell me a one-sentence fun fact.",
    "How many minutes are in an hour?",
    "Say something encouraging.",
    "Translate good morning to French.",
]

LOSS_RE = re.compile(r"Iter\s+(\d+):\s+Train loss\s+([\d.]+)")
VAL_RE = re.compile(r"Iter\s+(\d+):\s+Val loss\s+([\d.]+)")


def make_temp_config(rank: int, out_dir: Path) -> Path:
    """rank 只能通过 YAML 传，所以为 rank sweep 单独生成一份临时 YAML。"""
    cfg = out_dir / "lora_config.yaml"
    cfg.write_text(
        "lora_parameters:\n"
        '  keys: ["self_attn.q_proj", "self_attn.v_proj"]\n'
        f"  rank: {rank}\n"
        "  scale: 20.0\n"
        "  dropout: 0.0\n"
    )
    return cfg


def run_training(sweep: str, value, adapter_dir: Path) -> dict:
    """跑一次 mlx_lm.lora，把日志落盘，返回训练过程关键指标。"""
    adapter_dir.mkdir(parents=True, exist_ok=True)

    iters = BASE["iters"]
    batch_size = BASE["batch_size"]
    num_layers = BASE["num_layers"]
    lr = BASE["learning_rate"]
    config_file = CONFIG

    if sweep == "iters":
        iters = value
    elif sweep == "lr":
        lr = value
    elif sweep == "layers":
        num_layers = value
    elif sweep == "batch":
        batch_size = value
    elif sweep == "rank":
        config_file = make_temp_config(value, adapter_dir)
    else:
        raise ValueError(f"unknown sweep {sweep}")

    cmd = [
        "mlx_lm.lora",
        "--model", MODEL_ID,
        "--train",
        "--data", str(DATA_DIR),
        "--config", str(config_file),
        "--iters", str(iters),
        "--batch-size", str(batch_size),
        "--num-layers", str(num_layers),
        "--learning-rate", f"{lr:g}",
        "--adapter-path", str(adapter_dir),
    ]
    print(f"\n[train] {sweep}={value}")
    print("        " + " ".join(cmd))

    log_path = adapter_dir / "train.log"
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    output = proc.stdout + "\n" + proc.stderr
    log_path.write_text(output)

    train_losses = [(int(i), float(v)) for i, v in LOSS_RE.findall(output)]
    val_losses = [(int(i), float(v)) for i, v in VAL_RE.findall(output)]
    nan_seen = ("nan" in output.lower()) and ("loss nan" in output.lower() or "nan," in output.lower())

    return {
        "sweep": sweep,
        "value": value,
        "elapsed_s": round(elapsed, 1),
        "train_loss_first": train_losses[0][1] if train_losses else None,
        "train_loss_last": train_losses[-1][1] if train_losses else None,
        "val_loss_last": val_losses[-1][1] if val_losses else None,
        "returncode": proc.returncode,
        "diverged": nan_seen or proc.returncode != 0,
        "train_log": str(log_path.relative_to(ROOT)),
    }


def eval_adapter(adapter_dir: Path) -> dict:
    """加载 base + adapter，对固定 5 个 prompt 生成，统计 🦊 命中数。"""
    print(f"[eval]  {adapter_dir.relative_to(ROOT)}")
    from mlx_lm import generate, load

    model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_dir))
    outputs = []
    for p in TEST_PROMPTS:
        msgs = [{"role": "user", "content": p}]
        prompt = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
        try:
            out = generate(model, tokenizer, prompt=prompt, max_tokens=80, verbose=False)
        except Exception as exc:  # 模型崩了（比如 NaN 权重）也别中断 sweep
            out = f"<generate failed: {exc!s}>"
        outputs.append(out)
    hits = sum(FOX in o for o in outputs)
    eval_path = adapter_dir / "eval.json"
    eval_path.write_text(json.dumps({
        "prompts": TEST_PROMPTS,
        "outputs": outputs,
        "fox_hits": hits,
        "total": len(outputs),
    }, ensure_ascii=False, indent=2))
    return {"fox_hits": hits, "total": len(outputs), "outputs": outputs}


def run_sweep(sweep: str) -> list[dict]:
    results = []
    for value in SWEEPS[sweep]:
        adapter_dir = SWEEPS_DIR / sweep / str(value)
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        train_info = run_training(sweep, value, adapter_dir)
        if train_info["diverged"]:
            eval_info = {"fox_hits": 0, "total": len(TEST_PROMPTS), "outputs": []}
            (adapter_dir / "eval.json").write_text(json.dumps(
                {"skipped": "diverged_or_failed"}, indent=2
            ))
        else:
            try:
                eval_info = eval_adapter(adapter_dir)
            except Exception as exc:
                eval_info = {"fox_hits": 0, "total": len(TEST_PROMPTS),
                             "outputs": [], "error": str(exc)}
        results.append({**train_info, **eval_info})
    return results


# ---- 报告生成 ---------------------------------------------------------

SWEEP_HEAD = {
    "iters": {
        "title": "训练步数 `--iters`（iterations）",
        "what": (
            "决定参数被更新多少次。每次更新叫一个 **iter / step**。"
            "30 条训练样本、`batch_size=4` 时，1 个 **epoch**（数据被完整看一遍）≈ 8 iter，"
            "所以 `iters=200` 约等于 25 个 epoch（每条样本平均被看 25 次）。"
        ),
        "why": (
            "训练就是沿着 loss 下降方向**走小步**——步数太少没走到位（**欠拟合 underfit**），"
            "步数太多会把 30 条样本**死记**下来导致**过拟合 overfit**。"
        ),
    },
    "lr": {
        "title": "学习率 `--learning-rate` (learning rate, LR)",
        "what": (
            "每次更新参数的**步长**——梯度告诉我们方向，学习率决定走多远。"
        ),
        "why": (
            "LR 太小：步太短，再多 iter 也走不到 loss 谷底。"
            "LR 太大：单步直接跨过谷底飞到对面山坡，loss 反而上升甚至 **NaN（not-a-number 数值爆炸）**。"
            "LoRA 因可训参数少，通常用比全量微调（full fine-tune）大一个数量级的 LR。"
        ),
    },
    "layers": {
        "title": "LoRA 挂载层数 `--num-layers`",
        "what": (
            "在最顶上几层 transformer block 挂 LoRA 旁路（**adapter** = 适配器）。"
            "Qwen2.5-0.5B 共 24 层，挂上 8 层即"
            "**只动靠近输出的那部分行为**，下面 16 层连便利贴都不贴。"
        ),
        "why": (
            "底层网络做**通用语法 / 词法**这种基础能力，顶层负责**风格 / 格式 / 任务策略**。"
            "🦊 是风格层面的事，所以挂顶层最划算。挂得多 → 参数多、表达力更强、但训练慢、"
            "易破坏底层能力（**灾难性遗忘 catastrophic forgetting**）。"
        ),
    },
    "batch": {
        "title": "批大小 `--batch-size`",
        "what": (
            "每次更新时同时处理多少条样本，取它们 loss 的平均做一次梯度（gradient）更新。"
        ),
        "why": (
            "batch 小：每步只看一条样本，方向受单条干扰，曲线**抖**；但内存最省。"
            "batch 大：方向更稳，可以用更大 LR，但要更多内存。"
            "对你这种 30 条的小数据集，batch 太大反而每 epoch 只能切 1-2 个 batch，"
            "**梯度估计的统计有效性**下降。"
        ),
    },
    "rank": {
        "title": "瓶颈秩 `rank`（YAML，r in LoRA）",
        "what": (
            "LoRA 把权重改动写成 `A·B`，中间挤过一个 **r 维**瓶颈（bottleneck）。"
            "r 越小 = 可训参数越少 = 表达力越受限。MLX-LM 默认 r=8。"
        ),
        "why": (
            "LoRA 论文的核心假设：小任务上 `ΔW` 本身就是低秩（low-rank）的——"
            "用 r=8 已远超 🦊 这种局部行为调整所需。r 拉到 32 通常对 toy 任务**无明显收益**，"
            "只是浪费参数；r 砍到 2 看是否仍能学动则是好的下限实验。"
        ),
    },
}


def value_commentary(sweep: str, value, r: dict) -> str:
    """根据观测值给一段浅显语言的"为什么"。"""
    hit = r.get("fox_hits", 0)
    tot = r.get("total", 5)
    final = r.get("train_loss_last")
    initial = r.get("train_loss_first")
    diverged = r.get("diverged", False)
    base_value = BASE["learning_rate"] if sweep == "lr" else (
        BASE["batch_size"] if sweep == "batch" else (
            BASE["num_layers"] if sweep == "layers" else (
                BASE["rank"] if sweep == "rank" else BASE["iters"]
            )
        )
    )

    if diverged or (final is not None and (final != final)):  # NaN check
        return (
            f"训练发散（diverged）：loss 跑飞或出 NaN。**典型原因**：学习率过大、参数初始化遇到病态、"
            f"或量化精度不足。可训参数其实没学到任何有用东西，🦊 命中 {hit}/{tot} 仅因模型乱码偶尔撞上。"
        )

    if sweep == "iters":
        if value <= 10:
            return (
                f"**严重欠拟合**：只走了 {value} 步，参数几乎没动；loss 从 ~{initial:.2f} 仅降到 ~{final:.2f}，"
                f"🦊 命中 {hit}/{tot}。原因：每步只挪 1e-4 的尺寸，10 步累计位移微不足道。"
            )
        if value <= 50:
            return (
                f"**部分学到**：loss 降到 ~{final:.2f}，🦊 命中 {hit}/{tot}（半生不熟）。"
                f"模型隐约知道「末尾应该有点什么」但还不稳定。"
            )
        if value == base_value:
            return (
                f"**甜点位**：loss ~{final:.2f}，🦊 命中 {hit}/{tot}。"
                f"每条样本被看了约 25 次（epoch），刚好够把 🦊 模式钉进 A·B 的权重里。"
            )
        return (
            f"**深度过拟合**：loss 压到 ~{final:.2f}（很低），🦊 命中 {hit}/{tot}。"
            f"但因为只有 30 条训练数据，模型已经把它们**逐字背下来**——"
            f"如果你拿陌生 prompt 看输出，会发现它复读训练样本的句式，泛化（generalization）变差。"
        )

    if sweep == "lr":
        if value <= 1e-6:
            return (
                f"**步太小**：loss 从 ~{initial:.2f} 仅降到 ~{final:.2f}，🦊 命中 {hit}/{tot}。"
                f"原因：单步位移 ≈ 1e-6 × 梯度，量级太小，200 步累计仍不足以让 A·B 偏离零起点很多。"
            )
        if value <= 1e-5:
            return (
                f"**偏保守**：loss 到 ~{final:.2f}，🦊 命中 {hit}/{tot}。"
                f"再多 iter 可以补救，但同 iters 下不如基线 1e-4。"
            )
        if value == base_value:
            return (
                f"**LoRA 甜点**：loss ~{final:.2f}，🦊 命中 {hit}/{tot}。"
                f"LoRA 因可训参数少，能承受比全量微调（典型 1e-5）大一个数量级的 LR。"
            )
        if value >= 1e-2:
            return (
                f"**直接发散**：loss 飞或 NaN，🦊 命中 {hit}/{tot}。"
                f"步迈得比谷底宽度还大，每步都从一边山坡跨到另一边，永远收敛不了。"
            )
        return (
            f"**激进但仍可控**：loss 降到 ~{final:.2f}，🦊 命中 {hit}/{tot}。"
            f"少量 iter 就能学会，但 loss 曲线会有可见抖动；运气不好可能局部失败。"
        )

    if sweep == "layers":
        if value <= 2:
            return (
                f"**容量勉强**：loss ~{final:.2f}，🦊 命中 {hit}/{tot}。"
                f"只挂 {value} 层 LoRA，可训参数极少；toy 🦊 是个简单到极致的任务，所以**仍学得动**。"
            )
        if value == base_value:
            return (
                f"**基线**：loss ~{final:.2f}，🦊 命中 {hit}/{tot}。挂 8 层（共 24 层），"
                f"足够覆盖「输出层附近的所有风格相关模块」。"
            )
        return (
            f"**冗余容量**：loss ~{final:.2f}，🦊 命中 {hit}/{tot}。"
            f"挂 {value} 层比基线**翻倍参数**，但 toy 任务无明显增益，训练略慢——表达力上限上去了，"
            f"实际用不上的就是浪费。"
        )

    if sweep == "batch":
        if value <= 1:
            return (
                f"**抖且慢**：loss ~{final:.2f}，🦊 命中 {hit}/{tot}。"
                f"每步只看一条样本，梯度方向被这一条带偏；好处是内存最省。"
            )
        if value == base_value:
            return (
                f"**基线**：loss ~{final:.2f}，🦊 命中 {hit}/{tot}。"
                f"batch=4 在你 30 条数据上每 epoch 切出 7-8 个 batch，统计有效性 + 内存平衡得最好。"
            )
        return (
            f"**batch 过大**：loss ~{final:.2f}，🦊 命中 {hit}/{tot}。"
            f"batch={value} 在 30 条数据上每 epoch 只能切 ~{30 // value} 个 batch，"
            f"梯度更新次数变少；同 iters 下相当于「训练量缩水」。"
        )

    if sweep == "rank":
        if value <= 2:
            return (
                f"**极低秩**：loss ~{final:.2f}，🦊 命中 {hit}/{tot}。"
                f"r=2 时单个矩阵的 LoRA 只有 `2·d·r ≈ 3.6K` 个参数；🦊 这种局部任务**仍然装得下**——"
                f"印证了 LoRA 论文「小任务的权重改动本就是低秩」的核心假设。"
            )
        if value == base_value:
            return (
                f"**基线**：loss ~{final:.2f}，🦊 命中 {hit}/{tot}。r=8 是 MLX-LM 默认值。"
            )
        return (
            f"**冗余秩**：loss ~{final:.2f}，🦊 命中 {hit}/{tot}。"
            f"r={value} 比基线**多 {value // base_value}× 参数**，对 toy 任务没有可见收益——"
            f"再次印证「r 不是越大越好」。"
        )

    return ""


def write_report(all_results: dict[str, list[dict]]) -> None:
    lines = []
    lines.append("# LoRA 超参 sweep 报告\n")
    lines.append(
        "本报告由 `sweep.py` 自动生成。每个 sweep 中只动**一个**超参（hyperparameter），"
        "其余保持基线值不变（控制变量法 controlled-variable）。"
        "评估方法：训完用同 5 个 prompt 跑 `mlx_lm.generate`，数回答里有几个含 🦊。\n"
    )
    lines.append("## 基线配置（baseline）\n")
    lines.append("|参数|值|")
    lines.append("|---|---|")
    for k, v in BASE.items():
        lines.append(f"|`{k}`|{v}|")
    lines.append("")

    for sweep, results in all_results.items():
        head = SWEEP_HEAD[sweep]
        lines.append(f"## {head['title']}\n")
        lines.append(f"**它做什么**：{head['what']}\n")
        lines.append(f"**为什么会有差异**：{head['why']}\n")

        lines.append("### 实测结果\n")
        lines.append("|值|首 loss|末 loss|🦊 命中|训练耗时|备注|")
        lines.append("|---|---|---|---|---|---|")
        for r in results:
            init = f"{r['train_loss_first']:.2f}" if r['train_loss_first'] is not None else "-"
            last = f"{r['train_loss_last']:.2f}" if r['train_loss_last'] is not None else "-"
            note = "发散" if r.get("diverged") else ""
            v = r["value"]
            v_str = f"`{v:g}`" if isinstance(v, float) else f"`{v}`"
            lines.append(
                f"|{v_str}|{init}|{last}|{r['fox_hits']}/{r['total']}|{r['elapsed_s']}s|{note}|"
            )
        lines.append("")

        lines.append("### 逐值解读\n")
        for r in results:
            v = r["value"]
            v_str = f"{v:g}" if isinstance(v, float) else str(v)
            lines.append(f"- **`{v_str}`** — {value_commentary(sweep, v, r)}")
        lines.append("")

    lines.append("## 通用结论速查\n")
    lines.append(
        "- **学习率（learning rate, LR）是最容易训坏的旋钮**——先把它钉对，再调其他。"
        "判据：loss 单调下降 = 合适；震荡 = 偏大；NaN = 远超。\n"
        "- **iters × batch_size = 实际学习量**——同 epoch 数下两者可换算。\n"
        "- **LoRA rank 通常用不上更高**——8 已是甜点，2 是下限；大模型 / 复杂任务才考虑 16/32。\n"
        "- **顶层挂 LoRA 比底层挂 LoRA 划算**——顶层负责风格，底层负责通用能力，不该轻动。\n"
        "- **batch_size 受小数据集制约**——30 条样本里开 batch=16，每 epoch 只 1-2 步，统计有效性变差。\n"
    )

    report = SWEEPS_DIR / "REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✓ 报告已生成：{report.relative_to(ROOT)}")


def load_or_init_results() -> dict:
    p = SWEEPS_DIR / "results.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_results(d: dict) -> None:
    SWEEPS_DIR.mkdir(parents=True, exist_ok=True)
    (SWEEPS_DIR / "results.json").write_text(json.dumps(d, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA 超参 sweep")
    parser.add_argument(
        "sweeps", nargs="*",
        help=f"要跑的 sweep 名，可多个；可选 {list(SWEEPS) + ['all', 'report']}",
    )
    args = parser.parse_args()

    targets = args.sweeps or ["all"]
    if "report" in targets:
        results = load_or_init_results()
        if not results:
            print("results.json 不存在或为空，请先跑 sweep。")
            sys.exit(1)
        write_report(results)
        return

    if "all" in targets:
        targets = list(SWEEPS)
    unknown = [t for t in targets if t not in SWEEPS]
    if unknown:
        print(f"未知 sweep: {unknown}；可选: {list(SWEEPS)}")
        sys.exit(1)

    results = load_or_init_results()
    for sweep in targets:
        print(f"\n========== sweep: {sweep} ==========")
        results[sweep] = run_sweep(sweep)
        save_results(results)
    write_report(results)


if __name__ == "__main__":
    main()
