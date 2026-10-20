#!/usr/bin/env python3
"""eval_perclass.py —— 加载 dcnn.py 的 checkpoint, 复现测试集, 输出分类别指标

数据划分与 dcnn.py 完全一致(split_windows random_state=42 + stratify),
同一 checkpoint 同一 config 得到同一测试集, 指标可复现。

用法:
    python3 eval_perclass.py                          # 默认取 checkpoints/ 最新 IMU_DCNN_*.pth
    python3 eval_perclass.py checkpoints/IMU_DCNN_20260912_165821.pth
"""
import glob
import os
import sys

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from utils.imdata import build_windows, fit_scaler, load_imu_csv, scale_array, \
    split_windows
from utils.models import build_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_GLOB = os.path.join(SCRIPT_DIR, "checkpoints", "IMU_DCNN_*.pth")


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(CKPT_GLOB))[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "\ncheckpoint:", ckpt_path)

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    classes = ck["classes"]
    print("config:", {k: cfg[k] for k in ("hz", "window", "stride", "train_windows", "test_windows")})

    df, _ = load_imu_csv(cfg["data"], hz=cfg["hz"])
    X, Y, cls = build_windows(df, cfg["window"], cfg["stride"])
    assert cls == classes, "类别集与 checkpoint 不一致"

    X_tr, X_te, Y_tr, Y_te = split_windows(X, Y)
    assert len(X_te) == cfg["test_windows"], "测试集规模与训练时不一致"

    scaler = fit_scaler(X_tr)
    X_te = scale_array(scaler, X_te)

    model = build_model("DCNN", X_te.shape[1], len(classes)).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X_te.astype(np.float32)).to(device)) \
                 .argmax(1).cpu().numpy()
    y_true = Y_te

    print("\n===== classification report =====")
    print(classification_report(y_true, pred, target_names=classes, digits=4))
    print("===== confusion matrix (行=真值, 列=预测, 顺序同上) =====")
    print(confusion_matrix(y_true, pred))


if __name__ == "__main__":
    main()
