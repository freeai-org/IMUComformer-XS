#!/usr/bin/env python3
"""breed_cv.py —— 按品种划分的交叉验证(验证 DS1 数据集质量)

流程(三个子命令, 依次执行):
  split  : 读 ds1_neck6_remap.csv + DogInfo.csv, 按 26 个品种切出
           /mnt/WYH/data/neck6/breeds/NN_<Breed>.csv + breeds_manifest.json
  train  : 逐品种训练 DCNN(各自的类别集), checkpoint 存
           checkpoints/breeds/<breed>.pth, 训练日志追加到
           results/IMU_DCNN_breeds.jsonl(config 头 + 每 epoch 一条, 带 breed 字段)
  eval   : 26 个模型 x 26 个品种数据 => 26x26 准确率矩阵
           (对角线: 该品种自己 test split; 非对角: 对方品种全量数据)
           输出 results/breed_cv_matrix.csv 并打印 + 每行/列均值

用法:
    python3 breed_cv.py split
    python3 breed_cv.py train --epoch 100
    python3 breed_cv.py eval
    python3 breed_cv.py all --epoch 100     # 顺序执行三步
窗口参数与 dcnn.py 一致: --window 100 --stride 50 --hz 100

数据层/训练循环/评测口径/CLI 共享自 imdata / train_utils / protocol / cli
(2026-10 解耦重构, 对外命令行接口不变; sanitize/load_manifest/RANDOM_STATE
仍可 from breed_cv import)。
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from utils.cli import add_common_args
from utils.imdata import BREED_DIR, FEATURES, MANIFEST, RANDOM_STATE, build_windows, \
    fit_scaler, load_imu_csv, load_manifest, sanitize, scale_array, split_windows
from utils.models import build_model
from utils.protocol import eval_windows_acc, keep_subset, matrix_offdiag_stats
from utils.train_utils import jsonl_append, now_iso, save_ckpt, train_loop

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CKPT_BREED_DIR = os.path.join(SCRIPT_DIR, "checkpoints", "breeds")

REMAP_CSV = "/mnt/WYH/data/neck6/ds1_neck6_remap.csv"
DOGINFO_CSV = "/mnt/WYH/data/IMU-Dog/DogInfo.csv"
LOG_PATH = os.path.join(RESULTS_DIR, "IMU_DCNN_breeds.jsonl")
MATRIX_PATH = os.path.join(RESULTS_DIR, "breed_cv_matrix.csv")
MODEL_NAME = "DCNN"


# ---------------------------------------------------------------
# split: 按品种切 CSV
# ---------------------------------------------------------------
def cmd_split() -> None:
    os.makedirs(BREED_DIR, exist_ok=True)
    info = pd.read_csv(DOGINFO_CSV, usecols=["DogID", "Breed"])
    id2breed = dict(zip(info["DogID"], info["Breed"]))
    df = pd.read_csv(REMAP_CSV, dtype={"Behavior_1": str}, keep_default_na=False)
    df["Breed"] = df["DogID"].map(id2breed)
    assert df["Breed"].notna().all(), "存在 DogInfo 中没有的 DogID"
    manifest = {}
    print(f"共 {len(df):,} 行, {df['Breed'].nunique()} 个品种")
    for i, (breed, g) in enumerate(
            sorted(df.groupby("Breed"), key=lambda x: -len(x[1])), 1):
        fname = f"{i:02d}_{sanitize(breed)}.csv"
        g.drop(columns=["Breed"]).to_csv(
            os.path.join(BREED_DIR, fname), index=False)
        manifest[breed] = fname
        print(f"  {fname:<40} {len(g):>10,} 行  {g['DogID'].nunique()} 只狗")
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("manifest:", MANIFEST)


# ---------------------------------------------------------------
# train: 逐品种训练
# ---------------------------------------------------------------
def train_one(breed: str, fname: str, args) -> dict:
    t0 = time.time()
    df, _ = load_imu_csv(os.path.join(BREED_DIR, fname), args.hz)
    X, Y, classes = build_windows(df, args.window, args.stride)
    n_dogs = df["DogID"].nunique()
    del df
    print(f"\n[{breed}] {n_dogs} 只狗, {len(X):,} 窗口, {len(classes)} 类: {classes}")

    Xtr, Xte, Ytr, Yte = split_windows(X, Y, seed=RANDOM_STATE)
    scaler = fit_scaler(Xtr)
    Xtr = scale_array(scaler, Xtr)
    Xte = scale_array(scaler, Xte)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xt = torch.from_numpy(Xtr.astype("float32")).to(device)
    yt = torch.from_numpy(Ytr).to(device)
    xv = torch.from_numpy(Xte.astype("float32")).to(device)
    yv = torch.from_numpy(Yte).to(device)

    model = build_model("dcnn", len(FEATURES), len(classes)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{breed}] 模型 {MODEL_NAME} 参数量: {n_params:,} "
          f"({n_params * 4 / 1024 / 1024:.2f} MB)")

    cfg = {"type": "config", "model": MODEL_NAME, "breed": breed, "n_dogs": n_dogs,
           "hz": args.hz, "window": args.window, "stride": args.stride,
           "epochs": args.epoch, "batch_size": args.batch_size, "lr": args.lr,
           "train_windows": len(Xtr), "test_windows": len(Xte),
           "classes": classes, "params": n_params,
           "time": now_iso()}
    jsonl_append(LOG_PATH, cfg)

    best_acc, best_state = train_loop(
        model, xt, yt, xv, yv, epochs=args.epoch, batch_size=args.batch_size,
        lr=args.lr, log_path=LOG_PATH, desc=f"train {breed}",
        extra_epoch=lambda e: {"breed": breed})

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt = os.path.join(CKPT_BREED_DIR, sanitize(breed) + ".pth")
    save_ckpt(ckpt, model, scaler.mean_, scaler.scale_, classes)  # 历史格式: 不带 cfg
    print(f"[{breed}] 完成: best_test_acc={best_acc:.4f} "
          f"({time.time()-t0:.0f}s) -> {ckpt}")
    return {"breed": breed, "best_test_acc": best_acc}


def cmd_train(args) -> None:
    os.makedirs(CKPT_BREED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    manifest = load_manifest()
    print(f"共 {len(manifest)} 个品种, 每个训练 {args.epoch} epoch")
    results = []
    for breed, fname in sorted(manifest.items()):
        results.append(train_one(breed, fname, args))
    print("\n===== 各品种自身 test split 准确率 =====")
    for r in sorted(results, key=lambda x: -x["best_test_acc"]):
        print(f"  {r['breed']:<32} {r['best_test_acc']:.4f}")


# ---------------------------------------------------------------
# eval: 26x26 交叉验证矩阵
# ---------------------------------------------------------------
def cmd_eval(args) -> None:
    manifest = load_manifest()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    breeds = sorted(manifest.keys())
    drop = set(c.strip() for c in args.drop.split(",") if c.strip()) if args.drop else set()
    if drop:
        print(f"剔除类别(真值过滤 + 模型输出屏蔽): {sorted(drop)}")

    # 缓存每个品种的全量窗口(与 train 相同的窗口/降频/划分种子); --drop 时过滤掉对应真值窗口
    cache = {}
    for breed in breeds:
        df, _ = load_imu_csv(os.path.join(BREED_DIR, manifest[breed]), args.hz)
        X, Y, classes = build_windows(df, args.window, args.stride)
        del df
        if drop:
            keep_mask = np.array([classes[y] not in drop for y in Y])
            X, Y = X[keep_mask], Y[keep_mask]
        cache[breed] = (X, Y, classes)
        print(f"  窗口缓存 {breed}: {len(X):,}")

    print("\n===== 交叉验证: 行=模型(品种), 列=被测数据(品种) =====")
    matrix = pd.DataFrame(index=breeds, columns=breeds, dtype=float)
    for breed_i in breeds:
        ckpt = torch.load(os.path.join(CKPT_BREED_DIR, sanitize(breed_i) + ".pth"),
                          map_location=device, weights_only=False)  # 含 numpy scaler, 自产文件
        model = build_model("dcnn", len(FEATURES), len(ckpt["classes"])).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        classes_i = ckpt["classes"]
        keep_idx, keep_classes = keep_subset(classes_i, drop)

        for breed_j in breeds:
            X, Y, classes_j = cache[breed_j]
            if breed_i == breed_j:  # 对角线: 自己的 test split(与训练时同种子同划分)
                _, X, _, Y = split_windows(X, Y, seed=RANDOM_STATE)
            true_cls = np.array([classes_j[y] for y in Y])
            matrix.loc[breed_i, breed_j] = eval_windows_acc(
                model, X, true_cls, classes_i, keep_idx,
                ckpt["scaler_mean"], ckpt["scaler_scale"], device)
        print(f"  模型 {breed_i} 评测完成")

    _eval_tail(args, matrix, drop)


def _eval_tail(args, matrix, drop) -> None:
    """矩阵落盘 + 打印(与历史输出格式一致)"""
    out_path = (MATRIX_PATH if not drop else
                MATRIX_PATH.replace(".csv", "_drop_"
                                    + "_".join(sorted(drop)) + ".csv"))
    matrix.to_csv(out_path)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 300)
    print(matrix.round(3).to_string())
    off_mean, diag_mean = matrix_offdiag_stats(matrix.values)
    print(f"\n非对角(跨品种)均值: {off_mean:.4f}  对角(自身)均值: {diag_mean:.4f}")
    print("矩阵已保存:", out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["split", "train", "eval", "all"])
    add_common_args(ap, default_epoch=100)
    ap.add_argument("--drop", type=str, default="",
                    help="eval 时剔除的类别(逗号分隔, 如 Feeding,Standing): "
                         "过滤真值窗口 + 屏蔽模型对应输出")
    args = ap.parse_args()

    if args.cmd in ("split", "all"):
        cmd_split()
    if args.cmd in ("train", "all"):
        cmd_train(args)
    if args.cmd in ("eval", "all"):
        cmd_eval(args)


if __name__ == "__main__":
    main()

