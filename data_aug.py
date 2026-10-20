#!/usr/bin/env python3
"""data_aug.py —— 种子数据增广实验(project-5)
第一步: 7 类矩阵行均值排序得种子品种; 第二步: 按排序逐品种累加训练 26 个组合,
每组合在全部 26 品种上评测(组合内 test split / 组合外全量, 7 类协议)。
产出: IMU_DCNN_dataaug.jsonl / k<NN>_*.pth / dataaug_infer.jsonl/csv /
dataaug_summary.md, 全在 out-dir 下。

用法:
    python3 data_aug.py --epoch 100
    python3 data_aug.py --model conformer --attn imu --max 26 \
        --out-dir checkpoints/data_aug_imu

2026-10 解耦重构: 共享层见 imdata / train_utils / protocol / cli / models,
命令行接口不变; cfg 新增 gate/conv_norm/attn 键(checkpoint 回放用)。
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from utils.cli import add_common_args, add_conformer_args, conformer_hparams
from utils.imdata import FEATURES, RANDOM_STATE, build_windows, fit_scaler, \
    load_imu_csv, load_manifest, sanitize, scale_array, split_indices
from utils.models import build_model
from utils.protocol import DROP_CLASSES, eval_windows_acc, keep_subset
from utils.train_utils import jsonl_append, now_iso, save_ckpt, set_seed, train_loop

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "checkpoints", "data_aug")
MATRIX7 = os.path.join(SCRIPT_DIR, "results",
                       "breed_cv_matrix_drop_Feeding_Standing.csv")
LOG_PATH = os.path.join(OUT_DIR, "IMU_DCNN_dataaug.jsonl")
INFER_JSONL = os.path.join(OUT_DIR, "dataaug_infer.jsonl")
INFER_CSV = os.path.join(OUT_DIR, "dataaug_infer.csv")
SUMMARY_MD = os.path.join(OUT_DIR, "dataaug_summary.md")

def rank_breeds() -> list:
    """第一步: 行(模型品种)的非对角均值排序, 返回 [(breed, mean)] 降序"""
    m = pd.read_csv(MATRIX7, index_col=0)
    vals = m.values.astype(float).copy()
    np.fill_diagonal(vals, np.nan)
    means = np.nanmean(vals, axis=1)
    order = sorted(zip(m.index, means), key=lambda x: -x[1])
    return [(b, float(v)) for b, v in order]


def load_breed_windows(args, breed_dir, manifest_path):
    """缓存 26 品种: 全量窗口 + 每品种 train/test split(种子 42, 与 breed_cv 一致)"""
    manifest = load_manifest(manifest_path)
    cache = {}
    for breed, fname in sorted(manifest.items()):
        df, _ = load_imu_csv(os.path.join(breed_dir, fname), args.hz)
        X, Y, classes = build_windows(df, args.window, args.stride)
        del df
        tr_idx, te_idx = split_indices(Y, seed=RANDOM_STATE)
        # 评测用 7 类协议: 过滤被剔除真值的窗口
        keep = np.array([classes[y] not in DROP_CLASSES for y in Y])
        cache[breed] = {"X": X, "Y": Y, "classes": classes,
                        "tr_idx": tr_idx, "te_idx": te_idx, "keep": keep}
        print(f"  窗口缓存 {breed}: 全量 {len(Y):,} (7类评测 {int(keep.sum()):,})")
    return cache


def train_combo(combo_k, breeds_k, cache, args, device):
    """训练一个组合(前 k 个品种的 train split 合并), 返回 (model, scaler, classes, best_acc)"""
    # 每个 combo 独立固定种子: 单跑该 combo 与全量跑到该 combo 结果一致
    seed = getattr(args, "seed", 42) * 1000 + combo_k
    set_seed(seed)
    Xs, Ys = [], []
    for b in breeds_k:
        c = cache[b]
        Xs.append(c["X"][c["tr_idx"]])
        Ys.append(c["Y"][c["tr_idx"]])
    Xtr = np.concatenate(Xs)
    classes = sorted({cache[b]["classes"][y] for b in breeds_k
                      for y in np.unique(cache[b]["Y"])})
    # 各品种标签 -> 组合类别集索引
    Ytr = np.concatenate([
        np.array([classes.index(cache[b]["classes"][y]) for y in Ys_i], dtype=np.int64)
        for b, Ys_i in zip(breeds_k, Ys)])
    print(f"\n[combo {combo_k:02d}] 品种: {breeds_k}")
    print(f"  训练窗口 {len(Xtr):,}, 类别 {len(classes)}: {classes}")

    scaler = fit_scaler(Xtr)
    xt = torch.from_numpy(scale_array(scaler, Xtr).astype(np.float32)).to(device)
    yt = torch.from_numpy(Ytr).to(device)

    # 组合内 test split(训练时监控用, 同时是评测的对角线数据)
    Xte = np.concatenate([cache[b]["X"][cache[b]["te_idx"]] for b in breeds_k])
    Yte = np.concatenate([
        np.array([classes.index(cache[b]["classes"][y])
                  for y in cache[b]["Y"][cache[b]["te_idx"]]], dtype=np.int64)
        for b in breeds_k])
    xv = torch.from_numpy(scale_array(scaler, Xte).astype(np.float32)).to(device)
    yv = torch.from_numpy(Yte).to(device)

    model = build_model(args.model, len(FEATURES), len(classes),
                        **conformer_hparams(args)).to(device)
    model_name = type(model).__name__
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    cfg = {"type": "config", "model": model_name, "combo": combo_k,
           "breeds": breeds_k, "n_breeds": len(breeds_k),
           "hz": args.hz, "window": args.window, "stride": args.stride,
           "epochs": args.epoch, "batch_size": args.batch_size, "lr": args.lr,
           "seed": seed, "train_windows": len(Xtr), "test_windows": len(Xte),
           "classes": classes, "params": n_params,
           **conformer_hparams(args),
           "weight_decay": getattr(args, "weight_decay", 0.0),
           "time": now_iso()}
    jsonl_append(LOG_PATH, cfg)

    best_acc, best_state = train_loop(
        model, xt, yt, xv, yv, epochs=args.epoch, batch_size=args.batch_size,
        lr=args.lr, weight_decay=getattr(args, "weight_decay", 0.0),
        log_path=LOG_PATH, desc=f"combo {combo_k:02d}",
        extra_epoch=lambda e: {"combo": combo_k, "time": now_iso()})

    if best_state is not None:
        model.load_state_dict(best_state)
    name = "all" if combo_k == 26 else sanitize(breeds_k[-1])
    save_ckpt(os.path.join(OUT_DIR, f"k{combo_k:02d}_{name}.pth"),
              model, scaler.mean_, scaler.scale_, classes, cfg)
    print(f"  best_test_acc(组合内)={best_acc:.4f} -> k{combo_k:02d}_{name}.pth")
    return model, scaler, classes, best_acc


@torch.no_grad()
def eval_combo(model, scaler, classes, breeds_k, cache, all_breeds, device):
    """评测: 对 26 品种; 组合内用 test split, 组合外用全量(7 类协议)"""
    keep_idx, _ = keep_subset(classes)  # DROP_CLASSES 过滤
    accs = {}
    for b in all_breeds:
        c = cache[b]
        if b in breeds_k:  # 对角线: 自己的 test split(7 类过滤)
            sel = c["te_idx"][c["keep"][c["te_idx"]]]
        else:              # 非对角: 对方全量(7 类过滤)
            sel = np.where(c["keep"])[0]
        true_cls = np.array([c["classes"][y] for y in c["Y"][sel]])
        accs[b] = eval_windows_acc(model, c["X"][sel], true_cls, classes,
                                   keep_idx, scaler.mean_, scaler.scale_, device)
    mean_acc = float(np.mean(list(accs.values())))
    return accs, mean_acc


def _parse_args():
    global OUT_DIR, LOG_PATH, INFER_JSONL, INFER_CSV, SUMMARY_MD
    ap = argparse.ArgumentParser()
    add_common_args(ap, default_epoch=100)
    ap.add_argument("--max", type=int, default=26, help="最多跑到前几个组合")
    ap.add_argument("--breed-dir", type=str, default="",
                    help="品种数据目录(默认 breeds; 去噪对照用 breeds_denoise)")
    ap.add_argument("--model", type=str, default="dcnn",
                    choices=["dcnn", "transformer", "transformer2", "conformer"],
                    help="模型: dcnn(92,873 参数) / transformer(双模态 patch+full attention)"
                         " / transformer2(v2: 旋转不变特征|a||ω||jerk|+逐轴 token attention)"
                         " / conformer(卷积模块+自注意力混合)")
    ap.add_argument("--out-dir", type=str, default="",
                    help="产出目录(默认 checkpoints/data_aug)")
    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子(每个 combo 固定为 seed*1000+combo, 单跑与全量一致)")
    add_conformer_args(ap)
    args = ap.parse_args()
    if args.out_dir:
        OUT_DIR = args.out_dir
        LOG_PATH = os.path.join(OUT_DIR, "IMU_DCNN_dataaug.jsonl")
        INFER_JSONL = os.path.join(OUT_DIR, "dataaug_infer.jsonl")
        INFER_CSV = os.path.join(OUT_DIR, "dataaug_infer.csv")
        SUMMARY_MD = os.path.join(OUT_DIR, "dataaug_summary.md")
    return args


def _write_summary(order, ranked, df, best):
    lines = ["# 数据增广实验结论(基于 7 类品种交叉矩阵)", "",
             "## 品种排序(种子优先级, 7 类矩阵非对角均值)", ""]
    for i, (b, v) in enumerate(order, 1):
        lines.append(f"{i}. {b}: {v:.4f}")
    lines += ["", "## 各组合 26 品种评测均值", "",
              "| combo | 新增品种 | 均值 |", "|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r.name} | {r['last_added']} | {r['mean_acc']:.4f} |")
    lines += ["", f"**最佳组合: combo {best} "
              f"(含前 {best} 个品种, 截止 {' → '.join(ranked[:best])}), "
              f"均值 {df.loc[best, 'mean_acc']:.4f}**", "",
              "口径: 组合内品种用其 test split(种子 42), 组合外用全量; "
              "7 类协议(剔除 Feeding/Standing: 真值过滤 + 输出屏蔽)。"]
    with open(SUMMARY_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = _parse_args()
    breed_dir = args.breed_dir or "/mnt/WYH/data/neck6/breeds"
    manifest_path = os.path.join(breed_dir, "breeds_manifest.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 第一步: 排序
    order = rank_breeds()
    print("===== 品种排序(7 类矩阵非对角均值, 降序) =====")
    for i, (b, v) in enumerate(order, 1):
        print(f"  {i:2d}. {b:<40} {v:.4f}")
    ranked = [b for b, _ in order]

    # 数据缓存
    print("\n===== 加载品种窗口 =====")
    t0 = time.time()
    cache = load_breed_windows(args, breed_dir, manifest_path)
    all_breeds = sorted(cache.keys())
    print(f"窗口缓存完成 ({time.time()-t0:.0f}s)")

    # 第二步: 逐组合训练 + 评测
    rows = []
    for k in range(1, min(args.max, len(ranked)) + 1):
        breeds_k = ranked[:k]
        t0 = time.time()
        model, scaler, classes, best_acc = train_combo(
            k, breeds_k, cache, args, device)
        accs, mean_acc = eval_combo(
            model, scaler, classes, breeds_k, cache, all_breeds, device)
        rec = {"type": "infer", "combo": k, "breeds": breeds_k,
               "last_added": breeds_k[-1],
               "combo_test_acc": round(best_acc, 6),
               "mean_acc": round(mean_acc, 6),
               "per_breed_acc": {b: round(a, 6) for b, a in accs.items()},
               "time": now_iso()}
        jsonl_append(INFER_JSONL, rec)
        rows.append({"combo": k, "last_added": breeds_k[-1],
                     "n_train_windows": None, **accs, "mean_acc": mean_acc})
        print(f"  [combo {k:02d}] 26 品种均值 {mean_acc:.4f} "
              f"({time.time()-t0:.0f}s)")

    # 汇总
    df = pd.DataFrame(rows).set_index("combo")
    df.to_csv(INFER_CSV)
    best = int(df["mean_acc"].idxmax())
    _write_summary(order, ranked, df, best)
    print("\n===== 完成 =====")
    print(df[["last_added", "mean_acc"]].to_string())
    print(f"最佳组合: combo {best}, 均值 {df.loc[best, 'mean_acc']:.4f}")
    print("产出目录:", OUT_DIR)


if __name__ == "__main__":
    main()
