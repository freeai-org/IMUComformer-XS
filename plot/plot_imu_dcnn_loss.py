#!/usr/bin/env python3
"""plot_imu_dcnn_loss.py —— 主实验(dcnn.py)训练 loss/acc 曲线

读取 results/IMU_DCNN.jsonl(首行 config + 每 epoch 一条), 按 config 切分 runs
只取最后一个, 画训练 loss 与 train/test acc 双图。

输出: results/IMU_DCNN_loss.png
用法: python3 plot/plot_imu_dcnn_loss.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根入 sys.path

import json

import matplotlib
matplotlib.use("Agg")  # 无显示环境
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根
IN_JSONL = os.path.join(SCRIPT_DIR, "results", "IMU_DCNN.jsonl")
OUT = os.path.join(SCRIPT_DIR, "results", "IMU_DCNN_loss.png")

records = []
with open(IN_JSONL) as f:
    for line in f:
        records.append(json.loads(line))

# 按 config 切分 runs，只取最后一个 run
runs, current = [], []
for rec in records:
    if rec.get("type") == "config":
        if current:
            runs.append(current)
        current = [rec]
    else:
        current.append(rec)
if current:
    runs.append(current)

run = runs[-1]
epochs, loss, tr_acc, te_acc = [], [], [], []
for r in run[1:]:
    if r.get("type") == "epoch":
        epochs.append(r["epoch"])
        loss.append(r["train_loss"])
        tr_acc.append(r["train_acc"])
        te_acc.append(r["test_acc"])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

axes[0].plot(epochs, loss, color="#4C72B0", linewidth=1.6)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Train Loss")
axes[0].set_title("IMU DCNN - Train Loss")
axes[0].grid(True, alpha=0.25)

axes[1].plot(epochs, tr_acc, color="#DD8452", linewidth=1.6, label="Train Acc")
axes[1].plot(epochs, te_acc, color="#55A868", linewidth=1.6, label="Test Acc")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("IMU DCNN - Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig(OUT, dpi=150)
print(f"saved {OUT}, final loss={loss[-1]:.4f}, train acc={tr_acc[-1]:.4f}, test acc={te_acc[-1]:.4f}")
