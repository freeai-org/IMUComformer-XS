#!/usr/bin/env python3
"""plot_dataaug_heatmap.py —— 种子数据增广实验可视化

输入: checkpoints/data_aug/dataaug_infer.csv (26 组合 x 26 品种 + mean_acc)
输出(与输入同目录):
  dataaug_heatmap.png      组合 x 品种 准确率热力图(行=累加组合, 列=被测品种)
  dataaug_mean_curve.png   各组合 26 品种均值曲线(找最佳组合)

用法: python3 plot/plot_dataaug_heatmap.py [dataaug_infer.csv]
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根入 sys.path, 支持直接 python3 plot/xxx.py

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plot.plotlib import accumulation_order, combo_xticklabels, draw_heatmap, short

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根(plot/ 上一级)
DEFAULT_CSV = os.path.join(SCRIPT_DIR, "checkpoints", "data_aug", "dataaug_infer.csv")


def main() -> None:
    csv = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    outdir = os.path.dirname(csv)
    df = pd.read_csv(csv).set_index("combo")

    # 列按品种累加顺序排列: combo k 恰好训练前 k 列 -> 下三角结构,
    # 对角线 = 该组合新增品种, 与 y 轴 "+breed" 一一对应
    order = accumulation_order(df)

    # ---------- 热力图: 行=组合, 列=品种(累加顺序) ----------
    ylabels = [f"k{int(k):02d} +{df.loc[k, 'last_added'][:22]}" for k in df.index]
    xlabels = [short(b) for b in order]

    fig, ax = plt.subplots(figsize=(17, 13))
    # 注释框先画(与原版一致), 保证 draw_heatmap 的格内数值绘制在其上不被遮挡
    ax.annotate("trained breeds\n(in-combo test split)", xy=(0.02, 0.08),
                xycoords="axes fraction", fontsize=10, ha="left",
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))
    ax.annotate("unseen breeds\n(cross-breed)", xy=(0.98, 0.92),
                xycoords="axes fraction", fontsize=10, ha="right",
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))
    im = draw_heatmap(ax, df[order].values.astype(float), xlabels, ylabels,
                      val_fs=5.2, xt_fs=8, yt_fs=7,
                      grid_lw=0.3, grid_alpha=0.7,
                      diag_lw=1.8, box_min=True, box_lw=1.2)
    ax.set_xlabel("Evaluated breed (in accumulation order, same as y-axis)",
                  fontsize=11)
    ax.set_ylabel("Trained combo (breeds accumulated)", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("Accuracy", fontsize=11)
    best_k = int(df["mean_acc"].idxmax())
    ax.set_title("Data augmentation: combo (accumulated breeds) x breed accuracy  "
                 f"(7-class protocol)\nmean rises {df['mean_acc'].iloc[0]:.3f} -> "
                 f"{df['mean_acc'].iloc[-1]:.3f}, best combo = k{best_k:02d} "
                 f"({df['mean_acc'].max():.3f}); blue box = row min", fontsize=13)
    fig.tight_layout()
    out1 = os.path.join(outdir, "dataaug_heatmap.png")
    fig.savefig(out1, dpi=160)
    plt.close(fig)
    print("saved:", out1)

    # ---------- 均值曲线 ----------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ks = df.index.values
    ax.plot(ks, df["mean_acc"].values, marker="o", ms=4, lw=1.6, color="tab:green")
    ax.axvline(best_k, color="red", ls="--", lw=1,
               label=f"best combo k{best_k:02d} ({df['mean_acc'].max():.4f})")
    for k in ks:
        ax.annotate(f"{df.loc[k, 'mean_acc']:.3f}", (k, df.loc[k, "mean_acc"]),
                    textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=6.5)
    ax.set_xticks(ks)
    ax.set_xticklabels(combo_xticklabels(
        ks, [df.loc[k, "last_added"] for k in ks]), rotation=60, fontsize=6)
    ax.set_xlabel("Combo (breeds accumulated in seed ranking order)", fontsize=11)
    ax.set_ylabel("Mean accuracy over 26 breeds", fontsize=11)
    ax.set_ylim(df["mean_acc"].min() - 0.03, df["mean_acc"].max() + 0.03)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("Data augmentation: mean accuracy vs accumulated breeds", fontsize=13)
    fig.tight_layout()
    out2 = os.path.join(outdir, "dataaug_mean_curve.png")
    fig.savefig(out2, dpi=160)
    plt.close(fig)
    print("saved:", out2)


if __name__ == "__main__":
    main()
