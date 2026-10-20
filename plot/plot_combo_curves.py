#!/usr/bin/env python3
"""plot_combo_curves.py —— 绘制增广实验的 combo_test_acc 与 mean_acc 双曲线

输入: dataaug_infer.jsonl(含 combo/last_added/combo_test_acc/mean_acc 字段)
输出: 同目录 curves_combo_test_vs_mean.png

用法: python3 plot/plot_combo_curves.py [dataaug_infer.jsonl]
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根入 sys.path, 支持直接 python3 plot/xxx.py

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot.plotlib import combo_xticklabels
from utils.protocol import load_jsonl

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根(plot/ 上一级)
DEFAULT = os.path.join(SCRIPT_DIR, "checkpoints", "data_aug_conformer_v2",
                       "dataaug_infer.jsonl")


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    outdir = os.path.dirname(path)
    out = os.path.join(outdir, "curves_combo_test_vs_mean.png")
    recs = load_jsonl(path)

    ks = [r["combo"] for r in recs]
    cta = [r["combo_test_acc"] for r in recs]
    ma = [r["mean_acc"] for r in recs]
    labels = {r["combo"]: r["last_added"] for r in recs}

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ks, cta, marker="o", ms=5, lw=1.6, color="tab:blue",
            label="combo_test_acc (in-combo test split)")
    ax.plot(ks, ma, marker="s", ms=5, lw=1.6, color="tab:red",
            label="mean_acc (mean over all 26 breeds)")
    for k, v in zip(ks, cta):
        ax.annotate(f"{v:.3f}", (k, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=6.5, color="tab:blue")
    for k, v in zip(ks, ma):
        ax.annotate(f"{v:.3f}", (k, v), textcoords="offset points",
                    xytext=(0, -12), ha="center", fontsize=6.5, color="tab:red")
    ax.set_xticks(ks)
    ax.set_xticklabels(combo_xticklabels(ks, [labels[k] for k in ks],
                                         pad2=False),
                       rotation=60, fontsize=6.5)
    ax.set_xlabel("Combo (breeds accumulated in seed ranking order)", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    lo = min(min(cta), min(ma)); hi = max(max(cta), max(ma))
    ax.set_ylim(lo - 0.03, hi + 0.03)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="center right")
    ax.set_title(f"combo_test_acc vs mean_acc  ({os.path.basename(outdir)})\n"
                 f"combo_test: {sum(cta)/len(cta):.4f} avg, range {min(cta):.4f}~{max(cta):.4f}   |   "
                 f"mean_acc: {sum(ma)/len(ma):.4f} avg, {ma[0]:.4f} -> {ma[-1]:.4f}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print("saved:", out)
    print(f"combo_test_acc: avg {sum(cta)/len(cta):.4f}, "
          f"min {min(cta):.4f} (k{ks[cta.index(min(cta))]}), "
          f"max {max(cta):.4f} (k{ks[cta.index(max(cta))]}), 末值 {cta[-1]:.4f}")
    print(f"mean_acc: avg {sum(ma)/len(ma):.4f}, 首值 {ma[0]:.4f}, "
          f"max {max(ma):.4f} (k{ks[ma.index(max(ma))]}), 末值 {ma[-1]:.4f}")


if __name__ == "__main__":
    main()
