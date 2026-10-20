#!/usr/bin/env python3
"""plot_loss_compare.py —— 三模型(DCNN/Transformer/Conformer)训练 loss 对比

输入: 各模型 data_aug 实验的训练日志 IMU_DCNN_dataaug.jsonl(26 combo x 100 epoch,
      每 epoch 一条 train_loss 记录, 同一 (combo, epoch) 重复时取最后一次)
输出:
  各模型目录: dataaug_loss_curve.png   该模型 26 组合 loss 曲线(细线) + 均值±std
  results/:   dataaug_loss_compare.png 三模型合并图
              左: 按 epoch 的组合均值 loss; 右: 各 combo 末 10 epoch 平均 loss(数据量增大)

用法: python3 plot/plot_loss_compare.py [jsonl1 jsonl2 ...]
      默认读 data_aug(DCNN) / data_aug_transformer / data_aug_conformer 三个目录
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根入 sys.path, 支持直接 python3 plot/xxx.py

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根(plot/ 上一级)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
DEFAULT_RUNS = [
    ("DCNN", os.path.join(SCRIPT_DIR, "checkpoints", "data_aug",
                          "IMU_DCNN_dataaug.jsonl")),
    ("Transformer", os.path.join(SCRIPT_DIR, "checkpoints", "data_aug_transformer",
                                 "IMU_DCNN_dataaug.jsonl")),
    ("Conformer", os.path.join(SCRIPT_DIR, "checkpoints", "data_aug_conformer",
                               "IMU_DCNN_dataaug.jsonl")),
]
COLORS = {"DCNN": "tab:blue", "Transformer": "tab:orange",
          "Conformer": "tab:green"}


def load_losses(jsonl):
    """读训练日志 -> {combo: [epoch1_loss, epoch2_loss, ...]}; 重复记录取最后"""
    dedup = {}
    for line in open(jsonl, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("type") == "epoch":
            dedup[(r["combo"], r["epoch"])] = r["train_loss"]
    by_combo = {}
    for (combo, epoch), loss in dedup.items():
        by_combo.setdefault(combo, {})[epoch] = loss
    return {c: [ep[e] for e in sorted(ep)] for c, ep in by_combo.items()}


def main():
    runs = DEFAULT_RUNS if len(sys.argv) < 2 else \
        [(os.path.basename(os.path.dirname(p)), p) for p in sys.argv[1:]]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    models = {}
    for name, path in runs:
        if not os.path.exists(path):
            print(f"[跳过] {name}: {path} 不存在")
            continue
        models[name] = load_losses(path)
    if not models:
        print("没有可用日志")
        return

    # ---------- 各模型单独图 ----------
    for name, by_combo in models.items():
        n_ep = min(min(len(v) for v in by_combo.values()), 100)
        M = np.array([v[:n_ep] for v in by_combo.values()])  # [n_combo, n_ep]
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(np.arange(1, n_ep + 1).reshape(1, -1).repeat(len(M), 0).T,
                M.T, color="gray", alpha=0.15, lw=0.7)
        mean, std = M.mean(0), M.std(0)
        ax.plot(range(1, n_ep + 1), mean, color=COLORS.get(name, "tab:blue"),
                lw=2.2, label=f"{name}: mean of {len(M)} combos")
        ax.fill_between(range(1, n_ep + 1), mean - std, mean + std,
                        color=COLORS.get(name, "tab:blue"), alpha=0.15,
                        label="±1 std")
        ax.set_xlabel("Epoch (mean over 26 accumulated-breed combos)", fontsize=11)
        ax.set_ylabel("Train loss", fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_title(f"{name}: train loss per combo (gray) and mean", fontsize=13)
        fig.tight_layout()
        out = os.path.join(os.path.dirname(dict(runs)[name]),
                           "dataaug_loss_curve.png")
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print("saved:", out)

    # ---------- 三模型合并图 ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.8))
    for name, by_combo in models.items():
        n_ep = min(min(len(v) for v in by_combo.values()), 100)
        M = np.array([v[:n_ep] for v in by_combo.values()])
        mean = M.mean(0)
        ax1.plot(range(1, n_ep + 1), mean, color=COLORS.get(name, "gray"),
                 lw=2.0, label=name)
    ax1.set_xlabel("Epoch (mean over 26 combos)", fontsize=11)
    ax1.set_ylabel("Train loss (combo mean)", fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_title("Loss descent: Transformer fastest, Conformer lowest,\n"
                 "but generalization ranks DCNN > Conformer > Transformer",
                 fontsize=12)

    for name, by_combo in models.items():
        combos = sorted(by_combo)
        tail = [np.mean(by_combo[c][-10:]) for c in combos]
        ax2.plot(combos, tail, marker="o", ms=3.5, lw=1.6,
                 color=COLORS.get(name, "gray"), label=name)
    ax2.set_xlabel("Combo (breeds accumulated -> more training data)", fontsize=11)
    ax2.set_ylabel("Train loss (last 10 epochs avg)", fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_title("Final loss vs training data size", fontsize=12)

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "dataaug_loss_compare.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()
