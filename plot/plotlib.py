#!/usr/bin/env python3
"""plotlib.py —— 绘图层共享模块(全项目唯一权威来源)

import 本模块即完成 matplotlib 无头(Agg)初始化, 之后各脚本才能 import pyplot。
收编原先散落的:
  - 3 份逐字重复的 short() 品种名缩写
  - 2 份 accumulation_order 列序推导(plot_dataaug_heatmap / plot_model_compare_heatmaps)
  - 3 份组合x品种热力图绘制样板(plot_dataaug_heatmap / plot_model_compare_heatmaps /
    plot_heatmap 的 26x26 矩阵)
  - 2 份 MODEL_RUNS 模型注册表(plot_model_compare_curves / plot_model_compare_heatmaps
    各自维护; 后者 Conformer-S 曾误指 Conformer-XS 的全量 csv, 合并后以 curves 版为准)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根入 sys.path, 支持直接 python3 plot/xxx.py

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: F401  (Agg 初始化后 re-export)
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根(plot/ 上一级)

# 模型注册表: (规范名, 副标题, [候选 csv 相对路径——取第一个存在的], 颜色, 线型)
# 展示名统一为 f"{name} ({sub})", 与 README 的模型命名一致。
MODEL_RUNS = [
    ("DCNN", "93k",
     ["checkpoints/data_aug/dataaug_infer.csv"], "black", "-"),
    ("Transformer", "0.42M",
     ["checkpoints/data_aug_transformer/dataaug_infer.csv"], "tab:orange", "-"),
    ("Conformer", "0.77M, v2",
     ["checkpoints/data_aug_conformer_v2/dataaug_infer.csv"], "tab:blue", "-"),
    ("IMU-Conformer", "0.77M, attn bias",
     ["checkpoints/data_aug_imu/dataaug_infer.csv",          # 26 组合全量
      "checkpoints/ablation/imu_glu_bn/dataaug_infer.csv"],  # 前 6 组合(消融)
     "tab:red", "--"),
    ("Conformer-S", "0.30M, d64",
     ["checkpoints/attn_tune/d64/dataaug_infer.csv"], "tab:purple", "--"),
    ("Conformer-XS", "0.14M, d64ff128+l2",
     ["checkpoints/data_aug_d64ff128_l2/dataaug_infer.csv",      # 26 组合全量
      "checkpoints/attn_tune/d64ff128_l2/dataaug_infer.csv"],    # 前 6 组合(调参)
     "tab:green", "--"),
    ("Conformer-drop40", "0.77M, dropout 0.4",
     ["checkpoints/attn_tune/drop40/dataaug_infer.csv"], "tab:brown", ":"),
    ("Conformer-S-drop40", "0.30M, d64 + dropout 0.4",
     ["checkpoints/attn_tune/d64_drop40/dataaug_infer.csv"], "tab:gray", ":"),
]


def model_label(entry) -> str:
    """注册表条目 -> 展示名, 如 "DCNN (93k)" """
    return f"{entry[0]} ({entry[1]})"


def first_existing(rels):
    """候选相对路径列表 -> 第一个存在的绝对路径(无则 None)"""
    for r in rels:
        p = os.path.join(SCRIPT_DIR, r)
        if os.path.exists(p):
            return p
    return None


def short(name: str) -> str:
    """品种名缩写: 每个单词取前 4 字符, 最多 3 词"""
    words = name.split()
    return " ".join(w[:4] for w in words[:3]) if len(words) > 1 else name[:10]


def accumulation_order(df) -> list:
    """热力图列序: 品种按 last_added 累加顺序(combo k 恰好训练前 k 列,
    下三角结构), 未入组合的品种按字母序补在后面。df 以 combo 为索引。"""
    breeds = [c for c in df.columns
              if c not in ("last_added", "mean_acc", "n_train_windows")]
    ranked = [str(df.loc[k, "last_added"]) for k in df.index]
    known = set(breeds)
    order = [b for b in ranked if b in known]
    order += [b for b in sorted(breeds) if b not in set(order)]
    return order


def combo_xticklabels(ks, last_added, width=12, pad2=True) -> list:
    """combo 轴刻度标签: "k01\\n+新增品种" (pad2=False 时为 "k1" 风格)"""
    out = []
    for k, la in zip(ks, last_added):
        kfmt = f"{int(k):02d}" if pad2 else f"{k}"
        out.append(f"k{kfmt}\n+{str(la)[:width]}")
    return out


def draw_heatmap(ax, M, xlabels, ylabels, *, val_fs=5.2, xt_fs=8, yt_fs=8,
                 rot=90, cmap="RdYlGn", vmin=0.3, vmax=1.0, minor_grid=True,
                 grid_lw=0.3, grid_alpha=0.7, diag_lw=None, diag_ls="--",
                 diag_color="black", diag_alpha=None, box_min=False,
                 box_lw=1.2):
    """组合x品种准确率热力图绘制单元(收编 3 处样板), 返回 imshow 的 mappable。

    minor_grid : 白色网格线分隔单元格
    diag_lw    : 训练/未训练 frontier 对角线(None=不画; alpha/ls 可调)
    box_min    : 每行最差品种画蓝框
    格内数值去前导 0, 过亮/过暗(<=0.45 或 >=0.9)用白字。
    """
    M = np.asarray(M, dtype=float)
    n_k, n_b = M.shape
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n_b), xlabels, rotation=rot, fontsize=xt_fs)
    ax.set_yticks(range(n_k), ylabels, fontsize=yt_fs)
    if minor_grid:
        ax.set_xticks(np.arange(-0.5, n_b), minor=True)
        ax.set_yticks(np.arange(-0.5, n_k), minor=True)
        ax.grid(which="minor", color="white", lw=grid_lw, alpha=grid_alpha)
        ax.tick_params(which="minor", length=0)
    if diag_lw is not None:
        n_diag = min(n_k, n_b)
        ax.plot([-0.5, n_diag - 0.5], [-0.5, n_diag - 0.5],
                color=diag_color, lw=diag_lw, ls=diag_ls, alpha=diag_alpha)
    if box_min:
        for i in range(n_k):
            ax.add_patch(plt.Rectangle((int(M[i].argmin()) - 0.5, i - 0.5),
                                       1, 1, fill=False, edgecolor="blue",
                                       lw=box_lw))
    for i in range(n_k):
        for j in range(n_b):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}"[1:], ha="center", va="center",
                    fontsize=val_fs,
                    color="black" if 0.45 < v < 0.9 else "white")
    return im
