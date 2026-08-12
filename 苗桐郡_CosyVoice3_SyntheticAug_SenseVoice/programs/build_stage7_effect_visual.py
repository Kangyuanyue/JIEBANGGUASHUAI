from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def configure_plotting() -> None:
    global plt, np, FancyArrowPatch, FancyBboxPatch
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt_module
    import numpy as np_module
    from matplotlib.patches import FancyArrowPatch as arrow_patch
    from matplotlib.patches import FancyBboxPatch as box_patch

    plt = plt_module
    np = np_module
    FancyArrowPatch = arrow_patch
    FancyBboxPatch = box_patch
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def add_pipeline(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("方法流程", loc="left", fontsize=15, fontweight="bold", pad=12)

    nodes = [
        (0.03, 0.67, 0.18, 0.18, "规则指令\n文本"),
        (0.27, 0.67, 0.18, 0.18, "CosyVoice3\n多说话人合成"),
        (0.51, 0.67, 0.18, 0.18, "ASR回译\n质量筛选"),
        (0.75, 0.67, 0.21, 0.18, "噪声课程增强\nSenseVoice微调"),
        (0.51, 0.24, 0.18, 0.18, "三路重叠\n混合信号"),
        (0.75, 0.24, 0.21, 0.18, "SepFormer分离\nECAPA选路/拒识"),
    ]
    for index, (x, y, width, height, label) in enumerate(nodes):
        color = "#e8f2ed" if index < 4 else "#f4eee5"
        edge = "#236b55" if index < 4 else "#8a6238"
        patch = FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=color, edgecolor=edge, linewidth=1.4,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=11)

    arrows = [
        ((0.21, 0.76), (0.27, 0.76)),
        ((0.45, 0.76), (0.51, 0.76)),
        ((0.69, 0.76), (0.75, 0.76)),
        ((0.60, 0.67), (0.60, 0.42)),
        ((0.69, 0.33), (0.75, 0.33)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, color="#555555"))
    ax.text(0.03, 0.08, "主收益来自合成增强；分离与拒识分支仍需改进。", fontsize=11, color="#333333")


def add_direct_asr(ax: plt.Axes, metrics: dict[str, Any]) -> None:
    dataset_a = metrics["datasetA_test"]
    v3 = metrics["v3_same_start_noise_minus20dB"]
    baseline = [dataset_a["historical_original_direct_cer"] * 100, v3["historical_original_direct_cer"] * 100]
    augmented = [
        dataset_a["direct_no_reject"]["positive"]["cer"] * 100,
        v3["direct_no_reject"]["positive"]["cer"] * 100,
    ]
    x = np.arange(2)
    width = 0.32
    old = ax.bar(x - width / 2, baseline, width, color="#708199", label="原始SenseVoice")
    new = ax.bar(x + width / 2, augmented, width, color="#24735f", label="合成增强checkpoint")
    ax.bar_label(old, fmt="%.2f%%", padding=3, fontsize=10)
    ax.bar_label(new, fmt="%.2f%%", padding=3, fontsize=10)
    ax.set_title("直接ASR：两个独立测试集均小幅改善", loc="left", fontsize=15, fontweight="bold")
    ax.set_ylabel("CER（%，越低越好）")
    ax.set_xticks(x, ["datasetA", "V3 -20 dB"])
    ax.set_ylim(0, 82)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="upper left")
    for index, (before, after) in enumerate(zip(baseline, augmented)):
        displayed_delta = round(before, 2) - round(after, 2)
        ax.text(index, max(before, after) + 6.4, f"改善 {displayed_delta:.2f} 个百分点", ha="center", fontsize=10, color="#24735f")


def add_submission_comparison(ax: plt.Axes, metrics: dict[str, Any]) -> None:
    dataset_a = metrics["datasetA_test"]
    retained = dataset_a["retained_balanced_submission"]
    candidate = dataset_a["direct_balanced_reject"]
    labels = ["CER↓", "RR↑", "误拒率↓"]
    old = [retained["cer"] * 100, retained["rr"] * 100, retained["fr"] * 100]
    new = [
        candidate["positive"]["cer"] * 100,
        candidate["negative"]["rr"] * 100,
        candidate["positive"]["false_reject_rate"] * 100,
    ]
    y = np.arange(3)
    height = 0.30
    bars_old = ax.barh(y - height / 2, old, height, color="#708199", label="旧BalancedReject")
    bars_new = ax.barh(y + height / 2, new, height, color="#b04a36", label="新完整链路")
    ax.bar_label(bars_old, fmt="%.2f%%", padding=3, fontsize=10)
    ax.bar_label(bars_new, fmt="%.2f%%", padding=3, fontsize=10)
    ax.set_title("提交方案：新完整链路未超过旧版", loc="left", fontsize=15, fontweight="bold")
    ax.set_xlabel("比例（%）")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 86)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False, loc="lower right")
    decisions = ["退化 +2.44", "退化 -2.78", "退化 +0.97"]
    for index, text in enumerate(decisions):
        ax.text(84, index, text, ha="right", va="center", fontsize=10, color="#9a3528", fontweight="bold")


def build_effect_overview(metrics: dict[str, Any], output: Path) -> None:
    configure_plotting()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(15, 9), constrained_layout=True, facecolor="white")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], height_ratios=[1, 1])
    add_pipeline(fig.add_subplot(grid[:, 0]))
    add_direct_asr(fig.add_subplot(grid[0, 1]), metrics)
    add_submission_comparison(fig.add_subplot(grid[1, 1]), metrics)
    fig.suptitle("CosyVoice3合成增强 + SenseVoice：方法与效果总览", fontsize=20, fontweight="bold")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the public method-effect overview from Stage 6 metrics.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build_effect_overview(read_json(args.metrics), args.out)
    print(args.out.resolve())


if __name__ == "__main__":
    main()
