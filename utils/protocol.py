#!/usr/bin/env python3
"""protocol.py —— 评测口径共享模块(全项目唯一权威来源)

收编原先 3 处逐字拷贝的"7 类协议过滤"(data_aug.eval_combo / breed_cv.cmd_eval /
plot_confusion_matrix), 以及 5 处各自实现的"组合外均值(offdiag)"计算
(compare_ablation / compare_denoise / compare_transformer / plot_ablation /
plot_dataaug_offdiag) 和 7 处 jsonl 一行式加载。

口径定义(勿改, 改了就与历史结果不可比):
  - 7 类评测协议: 剔除 Feeding/Standing —— 真值窗口过滤 + 模型输出屏蔽
  - 组合外均值(offdiag): combo k 的模型在"未入组合品种"上的准确率均值
  - 可靠区间: combo <= RELIABLE_MAX_COMBO(21), 高组合组合外品种过少(<=4)噪声大
"""
import json

import numpy as np
import torch

DROP_CLASSES = {"Feeding", "Standing"}  # 7 类评测协议
RELIABLE_MAX_COMBO = 21


def load_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def keep_subset(classes, drop=None):
    """剔除类别 -> (保留的类别索引, 保留的类别名)。模型输出只在这些索引上 argmax"""
    drop = DROP_CLASSES if drop is None else set(drop)
    keep_idx = [k for k, c in enumerate(classes) if c not in drop]
    return keep_idx, [classes[k] for k in keep_idx]


def eval_windows_acc(model, X: np.ndarray, true_cls: np.ndarray,
                     classes, keep_idx, scaler_mean, scaler_scale, device) -> float:
    """7 类协议评测单元(原 3 处逐字拷贝的核心):
    手动标准化 -> 前向 -> 保留类别上 argmax -> 与真值比。X 为 [N, C, T]。
    true_cls 为类别名字符串数组; 模型不含的类别名会被判错(与历史口径一致)。
    """
    from utils.imdata import scale_with
    Xs = scale_with(X, scaler_mean, scaler_scale)
    with torch.no_grad():
        logits = model(torch.from_numpy(Xs).to(device)).cpu().numpy()
    keep_classes = [classes[k] for k in keep_idx]
    pred_cls = np.array([keep_classes[k] for k in logits[:, keep_idx].argmax(1)])
    return float((pred_cls == true_cls).mean())


def offdiag_dict(rec: dict) -> dict:
    """单条 infer 记录的"组合外品种 -> 准确率"(combo 26 全量时为空 dict)"""
    in_combo = set(rec["breeds"])
    return {b: v for b, v in rec["per_breed_acc"].items() if b not in in_combo}


def offdiag_values(rec: dict) -> list:
    """单条 infer 记录的组合外品种准确率列表(combo 26 全量时为空)"""
    return list(offdiag_dict(rec).values())


def offdiag_series(recs: list, max_combo: int = None) -> list:
    """逐 combo 的组合外均值序列(跳过无组合外品种的记录, 如 combo 26)"""
    out = []
    for r in recs:
        if max_combo is not None and r.get("combo", 0) > max_combo:
            continue
        vals = offdiag_values(r)
        if vals:
            out.append(float(np.mean(vals)))
    return out


def offdiag_mean(recs: list, max_combo: int = None) -> float:
    return float(np.mean(offdiag_series(recs, max_combo)))


def matrix_offdiag_stats(matrix) -> tuple:
    """26x26 交叉验证矩阵的(非对角均值, 对角均值), matrix 为方阵 ndarray/Df"""
    m = np.asarray(matrix, dtype=float)
    off = m[~np.eye(len(m), dtype=bool)]
    return float(off.mean()), float(np.diag(m).mean())
