#!/usr/bin/env python3
"""plot_dataaug_offdiag.py —— 增广实验的纯跨品种泛化曲线(去掉组合内品种)

对每个 combo k, 取其组合外品种(排除训练用过的品种)的准确率均值。
这才是真正的跨品种泛化: 全 26 均值混入了组合内 test split(较易)的分数。

输入: checkpoints/data_aug/dataaug_infer.jsonl
输出: 同目录 dataaug_offdiag_curve.png + dataaug_offdiag.csv

用法: python3 plot/plot_dataaug_offdiag.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根入 sys.path, 支持直接 python3 plot/xxx.py

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.protocol import load_jsonl, offdiag_dict

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根(plot/ 上一级)
INFER = os.path.join(SCRIPT_DIR, "checkpoints", "data_aug", "dataaug_infer.jsonl")


def main() -> None:
    infer = sys.argv[1] if len(sys.argv) > 1 else INFER
    outdir = os.path.dirname(infer)
    OUT_PNG = os.path.join(outdir, "dataaug_offdiag_curve.png")
    OUT_CSV = os.path.join(outdir, "dataaug_offdiag.csv")
    recs = load_jsonl(infer)
    rows = []
    for r in recs:
        off = offdiag_dict(r)
        if not off:
            continue
        rows.append({"combo": r["combo"], "last_added": r["last_added"],
                     "n_off": len(off),
                     "off_mean": float(np.mean(list(off.values()))),
                     "off_min_breed": min(off, key=off.get),
                     "off_min_acc": min(off.values()),
                     "all26_mean": r["mean_acc"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    # 可信区间: 组合外品种数 >= 5(combo <= 21), 之后样本太少噪声大
    rel = df[df["n_off"] >= 5]
    best = rel.loc[rel["off_mean"].idxmax()]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(df["combo"], df["off_mean"], marker="o", ms=5, lw=1.6,
            color="tab:blue", label="off-combo mean (pure cross-breed)")
    ax.plot(df["combo"], df["all26_mean"], marker="s", ms=4, lw=1.2, ls="--",
            color="tab:gray", alpha=0.7,
            label="all-26 mean (contains in-combo test split, inflated)")
    if df["combo"].max() >= 22:  # 只有跑到 22+ 组合才画"组合外<5不可靠"标区
        ax.axvspan(21.5, df["combo"].max(), color="red", alpha=0.08)
        ax.text(df["combo"].max() - 2.1, df["off_mean"].min() - 0.005,
                "off breeds < 5\n(unreliable)", ha="center", fontsize=8,
                color="tab:red")
    for _, r in df.iterrows():
        if r["combo"] <= 21:
            ax.annotate(f"{r['off_mean']:.3f}",
                        (r["combo"], r["off_mean"]),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=6.5, color="tab:blue")
    ax.set_xlabel("Combo (breeds accumulated in seed ranking order)", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(min(df["off_mean"].min(), df["all26_mean"].min()) - 0.04,
                df["all26_mean"].max() + 0.03)
    ax.set_xticks(range(1, 26))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="center right")
    ax.set_title("Cross-breed generalization: off-combo mean "
                 f"(trained breeds excluded)\nplateau ~0.78-0.81 after combo 6; "
                 f"best reliable combo = k{int(best['combo']):02d} "
                 f"({best['off_mean']:.4f}, {int(best['n_off'])} off breeds)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160)
    print("saved:", OUT_PNG)
    print(df.to_string(index=False))
    print(f"\n可靠区间(组合外>=5品种, combo<=21)最佳: combo {int(best['combo'])} "
          f"(+{best['last_added']}) off_mean={best['off_mean']:.4f}")


if __name__ == "__main__":
    main()
