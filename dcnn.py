#!/usr/bin/env python3
"""dcnn.py —— 项圈六轴 IMU 行为识别: DCNN 训练脚本(project-5)

数据: /mnt/WYH/data/neck6/ds1_neck6_remap.csv (DS1, 9 大类 Behavior_1 标签)
模型: DCNN (Conv1d x3 + 全局平均池化 + FC), 仅此一个模型
日志: results/IMU_DCNN.jsonl (config 头 + 每 epoch 一条)

用法:
    python3 dcnn.py --epoch 30 --window 100 --stride 50
    python3 dcnn.py --epoch 20 --window 50 --stride 25 --hz 50

数据层/训练循环/CLI 共享自 imdata / train_utils / cli(2026-10 解耦重构,
对外命令行接口不变; FEATURES/build_windows 仍可 from dcnn import)。
"""
import argparse
import os
import time

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report

from utils.cli import add_common_args
from utils.imdata import FEATURES, NATIVE_HZ, build_windows, fit_scaler, \
    load_imu_csv, scale_array, split_windows
from utils.train_utils import jsonl_append, now_iso, save_ckpt, train_loop

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CKPT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
DATA_PATH = "/mnt/WYH/data/neck6/ds1_neck6_remap.csv"
MODEL_NAME = "DCNN"


class DCNN(nn.Module):
    """3 层一维卷积 + 全局平均池化 + 全连接"""

    def __init__(self, input_size, num_classes):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout_conv = nn.Dropout(0.2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 128)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):  # x: [B, C, T]
        x = self.pool(self.dropout_conv(self.relu(self.conv1(x))))
        x = self.pool(self.dropout_conv(self.relu(self.conv2(x))))
        x = self.pool(self.dropout_conv(self.relu(self.conv3(x))))
        x = self.global_pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        return self.fc_out(x)


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap, default_epoch=30)  # dcnn 历史默认 30, 其余入口 100
    ap.add_argument("--data", type=str, default=DATA_PATH, help="训练 CSV 路径")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    t0 = time.time()
    df, dec = load_imu_csv(args.data, args.hz)
    print(f"读取 {len(df):,} 行 ({time.time()-t0:.0f}s), 统一采样率 {args.hz}Hz "
          f"(原生 {NATIVE_HZ}Hz, 抽点 1/{dec})")
    X, Y, classes = build_windows(df, args.window, args.stride)
    del df
    print(f"窗口化: window={args.window} stride={args.stride} -> "
          f"{len(X):,} 个窗口, {len(classes)} 类: {classes}")
    print("窗口标签分布:\n", pd.Series(Y).value_counts().to_string())

    X_train, X_test, Y_train, Y_test = split_windows(X, Y, seed=42)
    scaler = fit_scaler(X_train)
    X_train = scale_array(scaler, X_train)
    X_test = scale_array(scaler, X_test)
    xt = torch.from_numpy(X_train.astype("float32")).to(device)
    yt = torch.from_numpy(Y_train).to(device)
    xv = torch.from_numpy(X_test.astype("float32")).to(device)
    yv = torch.from_numpy(Y_test).to(device)

    model = DCNN(input_size=X.shape[1], num_classes=len(classes)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n===== 模型 {MODEL_NAME} 参数量: {n_params:,} "
          f"({n_params * 4 / 1024 / 1024:.2f} MB, float32) =====\n")

    log_path = os.path.join(RESULTS_DIR, f"IMU_{MODEL_NAME}.jsonl")
    cfg = {"type": "config", "model": MODEL_NAME, "data": args.data,
           "hz": args.hz, "window": args.window, "stride": args.stride,
           "epochs": args.epoch, "batch_size": args.batch_size, "lr": args.lr,
           "num_windows": len(X), "train_windows": len(X_train),
           "test_windows": len(X_test), "classes": classes,
           "params": n_params, "time": now_iso()}
    jsonl_append(log_path, cfg)
    print(f"日志: {log_path}")

    best_acc, best_state = train_loop(
        model, xt, yt, xv, yv, epochs=args.epoch, batch_size=args.batch_size,
        lr=args.lr, log_path=log_path, compute_train_acc=True,
        batch_tqdm=True, verbose=True,
        extra_epoch=lambda e: {"time": now_iso()})

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(xv).argmax(1).cpu().numpy()
    print("\n最佳 test_acc:", round(best_acc, 4))
    print(classification_report(yv.cpu().numpy(), pred, target_names=classes))

    ts = now_iso().replace("-", "").replace(":", "").replace("T", "_")
    ckpt = os.path.join(CKPT_DIR, f"IMU_{MODEL_NAME}_{ts}.pth")
    save_ckpt(ckpt, model, scaler.mean_, scaler.scale_, classes, cfg)
    print("checkpoint:", ckpt)


if __name__ == "__main__":
    main()
