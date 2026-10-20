#!/usr/bin/env python3
"""plot_confusion_matrix.py —— 三模型(DCNN/Transformer/Conformer)7 类混淆矩阵

加载各模型 data_aug 实验的 k26_all.pth(前 26 品种训练的最终模型),
在 26 品种各自的 test split(种子 42, 与训练协议一致)上推理,
按 7 类协议(剔除 Feeding/Standing: 真值过滤 + 输出屏蔽)池化后画混淆矩阵。

输出: results/confusion_matrix_<Model>.png(行归一化) + 终端 per-class 指标
用法: python3 plot/plot_confusion_matrix.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根入 sys.path, 支持直接 python3 plot/xxx.py

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from utils.imdata import BREED_DIR, FEATURES, build_windows, load_imu_csv, \
    load_manifest, scale_with, split_indices
from utils.models import build_model
from utils.protocol import DROP_CLASSES, keep_subset

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根(plot/ 上一级)
RUNS = [
    ("DCNN", os.path.join(SCRIPT_DIR, "checkpoints", "data_aug")),
    ("Transformer", os.path.join(SCRIPT_DIR, "checkpoints", "data_aug_transformer")),
    ("Conformer", os.path.join(SCRIPT_DIR, "checkpoints", "data_aug_conformer")),
]
BATCH = 16384
CM_CMAP = "Blues"


def load_test_pool():
    """26 品种各自的 test split(7 类过滤)池化成 X, Y(类名)"""
    manifest = load_manifest()
    Xs, Ys = [], []
    for breed, fname in tqdm(manifest.items(), desc="加载品种 test split"):
        df, _ = load_imu_csv(os.path.join(BREED_DIR, fname))
        X, Y, classes = build_windows(df, 100, 50)
        _, te = split_indices(Y)
        keep = np.array([classes[y] not in DROP_CLASSES for y in Y])
        sel = te[keep[te]]
        Xs.append(X[sel])
        Ys.extend(classes[y] for y in Y[sel])
    return np.concatenate(Xs), np.array(Ys)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    X, Y_true = load_test_pool()
    print(f"测试池: {len(X):,} 窗口, 类别: {sorted(set(Y_true))}")
    metrics = {}
    cms = {}

    for disp, run_dir in RUNS:
        ckpt = torch.load(os.path.join(run_dir, "k26_all.pth"),
                          map_location=device, weights_only=False)
        classes = ckpt["classes"]                    # 模型 9 类输出顺序
        mean, scale = ckpt["scaler_mean"], ckpt["scaler_scale"]
        keep_idx, keep_classes = keep_subset(classes)

        # 统一工厂: conformer 超参从 checkpoint cfg 回放(修复旧版不传超参的形状错误)
        mtype = ckpt["cfg"]["model"]
        model = build_model(mtype, input_size=len(FEATURES),
                            num_classes=len(classes),
                            cfg=ckpt["cfg"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        preds = []
        with torch.no_grad():
            for i in tqdm(range(0, len(X), BATCH), desc=f"{disp} 推理", leave=False):
                xb = scale_with(X[i:i + BATCH], mean, scale)
                logits = model(torch.from_numpy(xb).to(device))
                preds.extend(keep_classes[k]
                             for k in logits[:, keep_idx].argmax(1).cpu().numpy())
        Y_pred = np.array(preds)

        cm = confusion_matrix(Y_true, Y_pred, labels=keep_classes)
        cms[disp] = cm
        cmn = cm / cm.sum(1, keepdims=True)

        fig, ax = plt.subplots(figsize=(8.5, 7))
        im = ax.imshow(cmn, cmap=CM_CMAP, vmin=0, vmax=1)
        ax.set_xticks(range(7), keep_classes, rotation=45, ha="right", fontsize=10)
        ax.set_yticks(range(7), keep_classes, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]:,})",
                        ha="center", va="center", fontsize=8,
                        color="white" if cmn[i, j] > 0.5 else "black")
        acc = np.trace(cm) / cm.sum()
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"{disp} (k26, 26-breed test split, 7-class)\n"
                     f"overall acc = {acc:.4f}, n = {cm.sum():,}", fontsize=12)
        fig.tight_layout()
        out = os.path.join(SCRIPT_DIR, "results",
                           f"confusion_matrix_{disp}.png")
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(f"saved: {out}")

        # per-class 指标表(precision/recall/F1/support + TP/FP/FN/TN)
        report = classification_report(Y_true, Y_pred, labels=keep_classes,
                                       digits=4, output_dict=True,
                                       zero_division=0)
        # 混淆矩阵推 TP/FP/FN/TN(一对其余): FN=行和-TP, FP=列和-TP
        total = cm.sum()
        for i, c in enumerate(keep_classes):
            tp = cm[i, i]
            report[c]["TP"] = int(tp)
            report[c]["FN"] = int(cm[i].sum() - tp)
            report[c]["FP"] = int(cm[:, i].sum() - tp)
            report[c]["TN"] = int(total - tp - (cm[i].sum() - tp)
                                  - (cm[:, i].sum() - tp))
        metrics[disp] = report
        print(classification_report(Y_true, Y_pred, labels=keep_classes,
                                    digits=4, zero_division=0))

    # ---------- markdown 汇总表 ----------
    lines = ["# 三模型 per-class 指标表(k26, 26 品种 test split, 7 类协议, "
             f"n = {len(Y_true):,})", ""]
    for disp, rep in metrics.items():
        lines += [f"## {disp}", "",
                  "| 类别 | TP | FP | FN | TN | precision | recall | F1 | support |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for c in keep_classes:
            r = rep[c]
            lines.append(f"| {c} | {r['TP']:,} | {r['FP']:,} | {r['FN']:,} "
                         f"| {r['TN']:,} | {r['precision']:.4f} "
                         f"| {r['recall']:.4f} | {r['f1-score']:.4f} "
                         f"| {int(r['support']):,} |")
        lines.append(f"| **macro avg** | | | | | {rep['macro avg']['precision']:.4f} "
                     f"| {rep['macro avg']['recall']:.4f} "
                     f"| **{rep['macro avg']['f1-score']:.4f}** "
                     f"| {int(rep['macro avg']['support']):,} |")
        lines.append(f"| **weighted avg** | | | | | {rep['weighted avg']['precision']:.4f} "
                     f"| {rep['weighted avg']['recall']:.4f} "
                     f"| **{rep['weighted avg']['f1-score']:.4f}** "
                     f"| {int(rep['weighted avg']['support']):,} |")
        lines.append(f"| **accuracy** | | | | | | | **{rep['accuracy']:.4f}** "
                     f"| {int(rep['macro avg']['support']):,} |")
        lines.append("")
    md = os.path.join(SCRIPT_DIR, "results", "confusion_metrics.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("saved:", md)

    # ---------- TP/FP/FN/TN 四宫格图(三模型分组柱状) ----------
    models = list(cms.keys())
    mcolors = {"DCNN": "tab:blue", "Transformer": "tab:orange",
               "Conformer": "tab:green"}
    stats = {}
    for disp, cm in cms.items():
        tp = np.diag(cm)
        fn = cm.sum(1) - tp          # 漏报: 该类被判成其他类(行)
        fp = cm.sum(0) - tp          # 误报: 其他类被判成该类(列)
        tn = cm.sum() - tp - fn - fp
        stats[disp] = {"TP": tp, "FN": fn, "FP": fp, "TN": tn}

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    panels = [("TP", "True Positive (对角线, 判对)", True),
              ("FN", "False Negative (漏报: 行和-TP, 该类被判成其他类)", True),
              ("FP", "False Positive (误报: 列和-TP, 其他类被判成该类)", True),
              ("TN", "True Negative (其余全部)", False)]
    for ax, (key, title, ann) in zip(axes.flat, panels):
        w = 0.26
        for k, disp in enumerate(models):
            xs = np.arange(len(keep_classes)) + (k - 1) * w
            vals = stats[disp][key]
            bars = ax.bar(xs, vals, w, label=disp,
                          color=mcolors.get(disp, "gray"), alpha=0.85)
            if ann:
                for b, v in zip(bars, vals):
                    ax.text(b.get_x() + b.get_width() / 2, v, f"{int(v):,}",
                            ha="center", va="bottom", fontsize=6.5, rotation=0)
        ax.set_xticks(range(len(keep_classes)), keep_classes,
                      rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Windows", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=9)
    fig.suptitle("Per-class TP/FN/FP/TN (k26, 26-breed test split, 7-class, "
                 f"n = {int(cms[models[0]].sum()):,})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(SCRIPT_DIR, "results", "confusion_tpfpfn.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()
