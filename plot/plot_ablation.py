#!/usr/bin/env python3
"""plot_ablation.py —— conformer 消融实验图

读取: checkpoints/ablation/*/dataaug_infer.jsonl + 两个基线
      (DCNN = checkpoints/data_aug, conformer 原版 joint_glu_bn = data_aug_conformer_v2)
输出: results/ablation_summary.png
  左: 各变体组合外均值(offdiag)随 combo 变化曲线, DCNN 黑色参照
  右: 公共组合数(=6)上的组合外均值排序条形图

用法: python3 plot/plot_ablation.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根入 sys.path, 支持直接 python3 plot/xxx.py

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.protocol import load_jsonl, offdiag_series

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根(plot/ 上一级)
BASELINES = [
    ("DCNN", os.path.join(SCRIPT_DIR, "checkpoints", "data_aug",
                          "dataaug_infer.jsonl"), "black"),
    ("joint_glu_bn (conformer base)", os.path.join(
        SCRIPT_DIR, "checkpoints", "data_aug_conformer_v2",
        "dataaug_infer.jsonl"), "tab:green"),
]
PALETTE = ["tab:blue", "tab:orange", "tab:red", "tab:purple",
           "tab:brown", "tab:pink", "tab:olive", "tab:cyan", "tab:gray"]


def main():
    runs = list(BASELINES)
    for d in sorted(glob.glob(os.path.join(SCRIPT_DIR, "checkpoints/ablation/*/"))):
        runs.append((os.path.basename(d.rstrip("/")),
                    os.path.join(d, "dataaug_infer.jsonl"), None))

    data = []
    for name, path, color in runs:
        if not os.path.exists(path):
            continue
        recs = load_jsonl(path)
        od = offdiag_series(recs)
        if od:
            data.append((name, od, color))

    n = min(len(od) for _, od, _ in data)  # 对齐公共组合(=6)
    data = [(nm, od[:n], c) for nm, od, c in data]
    pi = iter(PALETTE)
    data = [(nm, od, c if c else next(pi)) for nm, od, c in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                   gridspec_kw={"width_ratios": [1.4, 1]})
    for name, od, color in data:
        base = name.startswith("DCNN") or "base" in name
        ax1.plot(range(1, n + 1), od, marker="o", ms=4, lw=2.2 if base else 1.2,
                 color=color, label=name,
                 zorder=3 if base else 2,
                 ls="--" if name.startswith("DCNN") else "-")
    ax1.set_xlabel("Combo (breeds accumulated, seed ranking order)", fontsize=11)
    ax1.set_ylabel("Off-combo mean accuracy (pure cross-breed)", fontsize=11)
    ax1.set_xticks(range(1, n + 1))
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.set_title(f"Cross-breed generalization per combo (first {n} combos, "
                  "seed 42, 7-class protocol)", fontsize=12)

    order = sorted(data, key=lambda r: np.mean(r[1]))
    names = [r[0] for r in order]
    means = [np.mean(r[1]) for r in order]
    colors = [r[2] for r in order]
    bars = ax2.barh(range(len(order)), means, color=colors, alpha=0.85)
    ax2.set_yticks(range(len(order)), names, fontsize=9)
    for i, v in enumerate(means):
        ax2.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=8.5)
    ax2.set_xlabel("Off-combo mean accuracy (combos 1-%d)" % n, fontsize=11)
    ax2.set_xlim(min(means) - 0.015, max(means) + 0.035)
    ax2.grid(alpha=0.3, axis="x")
    ax2.set_title("Ablation ranking: all conformer variants in a "
                  f"{max(means[1:]) - min(means):.3f} band,\nDCNN "
                  f"+{means[-1] - max(means[:-1]):.3f} above best conformer",
                  fontsize=12)

    fig.tight_layout()
    out = os.path.join(SCRIPT_DIR, "results", "ablation_summary.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()
