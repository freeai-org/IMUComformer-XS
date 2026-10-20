#!/usr/bin/env python3
"""plot_model_compare_heatmaps.py —— 多模型增广热力图对比(带数值)

与 plot_dataaug_heatmap.py 同款样式(组合 x 品种准确率矩阵, RdYlGn, 蓝框=行最差,
黑虚线=训练/未训练 frontier, 格内标数值), 多模型并排且**共享列序**(品种按种子
累加顺序), 可横向比较同一格在不同模型上的表现。

模型清单与命名统一来自 plotlib.MODEL_RUNS(详见 README):
  DCNN(93k) / Transformer(0.42M) / Conformer(0.77M, 即 v2) /
  IMU-Conformer(0.77M, attn=imu 物理偏置) /
  Conformer-S(0.30M, d64) / Conformer-XS(0.14M, d64ff128+l2) /
  Conformer-drop40 / Conformer-S-drop40

26 组合的模型进 full26 图, 只有前 6 组合的进 partial6 图(跑完全量后自动升级)。
输出:
  results/model_compare_heatmaps_full26.png
  results/model_compare_heatmaps_partial6.png

用法: python3 plot/plot_model_compare_heatmaps.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根入 sys.path, 支持直接 python3 plot/xxx.py

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plot.plotlib import MODEL_RUNS, accumulation_order, draw_heatmap, \
    first_existing, model_label, short

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根(plot/ 上一级)


def load_all():
    out = []
    for entry in MODEL_RUNS:
        path = first_existing(entry[2])
        if path is None:
            print("skip(无产出):", entry[0])
            continue
        df = pd.read_csv(path).set_index("combo")
        out.append((model_label(entry), df))
    if not out:
        return [], None
    ref = max(out, key=lambda t: len(t[1].index))[1]
    order = accumulation_order(ref)
    return out, order


def panel(ax, df, order, title, val_fs):
    ylabels = [f"k{int(k):02d} +{df.loc[k, 'last_added'][:14]}" for k in df.index]
    im = draw_heatmap(ax, df[order].values.astype(float),
                      [short(b) for b in order], ylabels,
                      val_fs=val_fs, xt_fs=6, yt_fs=6,
                      grid_lw=0.35, grid_alpha=0.8,
                      diag_lw=1.4, box_min=True, box_lw=1.0)
    best_k = int(df["mean_acc"].idxmax())
    ax.set_title(f"{title}\nmean {df['mean_acc'].iloc[0]:.3f} -> "
                 f"{df['mean_acc'].iloc[-1]:.3f}, best k{best_k:02d} "
                 f"({df['mean_acc'].max():.3f})", fontsize=9)
    return im


def draw(runs, order, out_png, val_fs, fig_h, supt):
    if not runs:
        return
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(7.0 * n, fig_h))
    if n == 1:
        axes = [axes]
    im = None
    for ax, (title, df) in zip(axes, runs):
        im = panel(ax, df, order, title, val_fs)
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cbar.set_label("Accuracy", fontsize=10)
    fig.suptitle(supt + " (7-class protocol, shared column order, "
                 "blue box = row min)", fontsize=12)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out_png)


def main() -> None:
    runs, order = load_all()
    if not runs:
        print("没有任何可用的 dataaug_infer.csv")
        return
    full = [(t, df) for t, df in runs if len(df.index) >= 26]
    part = [(t, df) for t, df in runs if len(df.index) < 26]
    draw(full, order,
         os.path.join(SCRIPT_DIR, "results", "model_compare_heatmaps_full26.png"),
         val_fs=4.4, fig_h=11.5,
         supt="Data augmentation heatmaps - 26-combo full runs")
    draw(part, order,
         os.path.join(SCRIPT_DIR, "results", "model_compare_heatmaps_partial6.png"),
         val_fs=5.2, fig_h=4.6,
         supt="Data augmentation heatmaps - first 6 combos")


if __name__ == "__main__":
    main()
