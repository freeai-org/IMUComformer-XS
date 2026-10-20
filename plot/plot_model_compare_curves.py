#!/usr/bin/env python3
"""plot_model_compare_curves.py —— 多模型种子增广曲线对比

每加一个品种(combo k = 前 k 个种子品种累加)后, 在全部 26 品种上评测的均值
(mean_acc, 含组合内 test split, 偏乐观口径) 随 combo 变化的多模型对比图。
x 轴品种标签取自最长 run(26 组合)的累加顺序。

模型清单与命名统一来自 plotlib.MODEL_RUNS(与 plot_model_compare_heatmaps /
README 一致):
  DCNN(93k) / Transformer(0.42M) / Conformer(0.77M, v2) /
  IMU-Conformer(0.77M, attn bias) / Conformer-S(0.30M) / Conformer-XS(0.14M) /
  Conformer-drop40 / Conformer-S-drop40

输出: results/model_compare_curves.png
用法: python3 plot/plot_model_compare_curves.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根入 sys.path, 支持直接 python3 plot/xxx.py

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plot.plotlib import MODEL_RUNS, combo_xticklabels, first_existing, model_label

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根(plot/ 上一级)


def main() -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    ref = None
    for entry in MODEL_RUNS:
        name = model_label(entry)
        path = first_existing(entry[2])
        if path is None:
            print("skip(无产出):", name)
            continue
        df = pd.read_csv(path).set_index("combo")
        ks, ys = df.index.values, df["mean_acc"].values
        partial = len(ks) < 26
        tag = f"{name}  k{int(ks[0])}->k{int(ks[-1])} last={ys[-1]:.4f}" \
              + (" (first 6 combos)" if partial else "")
        ax.plot(ks, ys, marker="o", ms=3.5, lw=1.8,
                color=entry[3], ls=entry[4], label=tag)
        if ref is None or len(ks) > len(ref.index):
            ref = df
        ax.annotate(f"{ys[-1]:.3f}", (ks[-1], ys[-1]),
                    textcoords="offset points", xytext=(4, -3),
                    fontsize=7, color=entry[3])

    if ref is not None:
        ks_ref = ref.index.values
        ax.set_xticks(ks_ref)
        ax.set_xticklabels(combo_xticklabels(
            ks_ref, [ref.loc[k, "last_added"] for k in ks_ref]),
            rotation=60, fontsize=6)
    if ax.lines:
        all_min = min(line.get_ydata().min() for line in ax.lines)
        all_max = max(line.get_ydata().max() for line in ax.lines)
        ax.set_ylim(all_min - 0.02, all_max + 0.03)
    ax.set_xlabel("Combo (breeds accumulated in seed ranking order)", fontsize=11)
    ax.set_ylabel("Mean accuracy over 26 breeds (7-class protocol)", fontsize=11)
    ax.set_title("Seed data augmentation: mean accuracy vs accumulated breeds - "
                 "model comparison", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout()
    out = os.path.join(SCRIPT_DIR, "results", "model_compare_curves.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()
