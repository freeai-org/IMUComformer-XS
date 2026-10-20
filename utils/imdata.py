#!/usr/bin/env python3
"""imdata.py —— project-5 数据层共享模块

收编原先散落在 dcnn.py / breed_cv.py / data_aug.py / eval_perclass.py 的
6 份 CSV 加载+降频、3 份 train_test_split、多份 scaler 样板。
唯一权威来源：窗口化/加载/划分/标准化都在这里，各训练入口只做编排。
"""
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

NATIVE_HZ = 100  # DS1/DS2 原生采样率(实测一致)
FEATURES = ["ANeck_x", "ANeck_y", "ANeck_z", "GNeck_x", "GNeck_y", "GNeck_z"]
RANDOM_STATE = 42

BREED_DIR = "/mnt/WYH/data/neck6/breeds"
MANIFEST = os.path.join(BREED_DIR, "breeds_manifest.json")  # breed -> filename


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def load_manifest(manifest_path: str = "") -> dict:
    with open(manifest_path or MANIFEST) as f:
        return json.load(f)


def load_imu_csv(path: str, hz: int = NATIVE_HZ,
                 extra_cols: tuple = (), label_col: str = "Behavior_1"):
    """读 IMU CSV + 统一降频(带因子校验——旧 breed_cv/data_aug 版缺校验, hz=30 会静默错切)。

    固定读取 DogID/TestNum/标签列 + FEATURES; 返回 (df, dec)。
    """
    dec = NATIVE_HZ // hz
    if hz > NATIVE_HZ or NATIVE_HZ % hz != 0:
        raise SystemExit(f"--hz 必须是 {NATIVE_HZ} 的整数因子(如 100/50/25/20/10)")
    df = pd.read_csv(
        path,
        usecols=["DogID", "TestNum", label_col, *extra_cols] + FEATURES,
        dtype={label_col: str}, keep_default_na=False)
    if dec > 1:
        df = df.iloc[::dec].reset_index(drop=True)
    return df, dec


def build_windows(df: pd.DataFrame, window: int, stride: int):
    """按 (DogID, TestNum) 分段滑窗, 窗口不跨狗/跨测试; 标签取窗口内众数"""
    classes = sorted(df["Behavior_1"].unique())
    cls2idx = {c: i for i, c in enumerate(classes)}
    X, Y = [], []
    for _, seg in df.groupby(["DogID", "TestNum"], sort=False):
        f = seg[FEATURES].to_numpy(np.float32)
        lab = seg["Behavior_1"].map(cls2idx).to_numpy()
        for i in range(0, len(f) - window + 1, stride):
            X.append(f[i:i + window].T)  # [C, T]
            Y.append(pd.Series(lab[i:i + window]).mode()[0])
    return np.stack(X), np.asarray(Y, np.int64), classes


def split_indices(Y, seed: int = RANDOM_STATE):
    """索引式 8:2 划分(stratify + 小类降级 fallback), 供缓存全量窗口后切分"""
    try:
        tr_idx, te_idx = train_test_split(
            np.arange(len(Y)), test_size=0.2, random_state=seed, stratify=Y)
    except ValueError:  # 某类样本过少无法分层
        tr_idx, te_idx = train_test_split(
            np.arange(len(Y)), test_size=0.2, random_state=seed)
    return tr_idx, te_idx


def split_windows(X, Y, seed: int = RANDOM_STATE):
    """数组式 8:2 划分(dcnn.py 主实验 / breed_cv 训练用)"""
    try:
        Xtr, Xte, Ytr, Yte = train_test_split(
            X, Y, test_size=0.2, random_state=seed, stratify=Y)
    except ValueError:
        Xtr, Xte, Ytr, Yte = train_test_split(
            X, Y, test_size=0.2, random_state=seed)
    return Xtr, Xte, Ytr, Yte


def fit_scaler(Xtr: np.ndarray) -> StandardScaler:
    """对 [N, C, T] 训练集 fit StandardScaler(按通道)"""
    scaler = StandardScaler()
    scaler.fit(Xtr.reshape(-1, Xtr.shape[1]))
    return scaler


def scale_array(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """用 scaler 变换 [N, C, T] 数组并还原形状"""
    return scaler.transform(X.reshape(-1, X.shape[1])).reshape(X.shape)


def scale_with(X: np.ndarray, mean, scale) -> np.ndarray:
    """用 checkpoint 里保存的 mean/scale 手动标准化(评测复现路径)"""
    Xs = (X.reshape(-1, X.shape[1]) - mean) / scale
    return Xs.reshape(X.shape).astype(np.float32)
