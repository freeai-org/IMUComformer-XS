#!/usr/bin/env python3
"""plot_heatmap.py —— 26x26 品种交叉验证矩阵热力图

输入: results/breed_cv_matrix.csv (可用参数指定其他矩阵, 如 _drop_ 版)
输出: 同名 .png

用法: python3 plot/plot_heatmap.py [results/breed_cv_matrix_drop_Feeding_Standing.csv]
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根入 sys.path, 支持直接 python3 plot/xxx.py

import sys
import os

import matplotlib
matplotlib.use("Agg")  # 无显示环境
import matplotlib.pyplot as plt
import pandas as pd

from plot.plotlib import draw_heatmap, short
from utils.protocol import matrix_offdiag_stats

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根(plot/ 上一级)
CSV = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(SCRIPT_DIR, "results", "breed_cv_matrix.csv")
OUT = CSV[:-4] + ".png" if CSV.endswith(".csv") else CSV + ".png"


def main() -> None:
    m = pd.read_csv(CSV, index_col=0)
    n = len(m)
    labels = [short(c) for c in m.columns]

    fig, ax = plt.subplots(figsize=(16, 13))
    # 对角线黑色细线突出"自身"; 无网格线; 格内标数值(小字号)
    im = draw_heatmap(ax, m.values, labels, labels, val_fs=5.5,
                      xt_fs=8, yt_fs=8, minor_grid=False,
                      diag_lw=0.8, diag_ls="-", diag_alpha=0.6)

    ax.set_xlabel("Evaluated breed (data)", fontsize=11)
    ax.set_ylabel("Trained model (breed)", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("Accuracy", fontsize=11)
    off, diag = matrix_offdiag_stats(m.values)
    ax.set_title(f"Breed cross-validation (26 models x 26 breeds)  "
                 f"diag mean={diag:.3f}  off-diag mean={off:.3f}", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
