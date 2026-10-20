#!/usr/bin/env python3
"""train_utils.py —— 训练层共享模块

收编原先 dcnn.py / breed_cv.py / data_aug.py 的三份手写 epoch 循环
(best state 跟踪 / jsonl 逐 epoch 日志 / checkpoint 保存) 为单一 train_loop,
差异点参数化:
  compute_train_acc: dcnn 算 train_acc, 另两家不算
  batch_tqdm:        dcnn 内层批进度条, 另两家只外层
  extra_epoch(e):    每 epoch 追加字段(breed / combo+time / time), 保日志格式不变
  verbose:           dcnn 每 epoch 打印, 另两家静默(只有 tqdm)
"""
import copy
import json
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def set_seed(seed: int) -> None:
    """固定 random/numpy/torch/cuda 种子(data_aug 原有, dcnn/breed_cv 原先缺失)"""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def jsonl_append(path: str, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def train_loop(model, xt, yt, xv, yv, *, epochs, batch_size, lr,
               weight_decay: float = 0.0, log_path: str = None,
               extra_epoch=None, compute_train_acc: bool = False,
               batch_tqdm: bool = False, verbose: bool = False, desc: str = None):
    """统一 epoch 训练循环。xt/yt/xv/yv 均为已驻留 device 的 tensor。

    每 epoch 记录 {"type":"epoch","epoch":e,"train_loss","test_acc"[,"train_acc"],
    **extra_epoch(e)} 追加写入 log_path; 返回 (best_test_acc, best_state拷贝)。
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    best_acc, best_state = 0.0, None
    it = tqdm(range(epochs), desc=desc, leave=False) if desc else range(epochs)
    for epoch in it:
        model.train()
        perm = torch.randperm(xt.size(0), device=xt.device)
        losses, correct, total = [], 0, 0
        batches = range(0, xt.size(0), batch_size)
        if batch_tqdm:
            batches = tqdm(batches, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for i in batches:
            idx = perm[i:i + batch_size]
            out = model(xt[idx])
            loss = criterion(out, yt[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if compute_train_acc:
                correct += (out.argmax(1) == yt[idx]).sum().item()
                total += len(idx)
        model.eval()
        with torch.no_grad():
            acc = (model(xv).argmax(1) == yv).float().mean().item()
        rec = {"type": "epoch", "epoch": epoch + 1,
               "train_loss": round(float(np.mean(losses)), 6),
               "test_acc": round(float(acc), 6)}
        if compute_train_acc:
            rec["train_acc"] = round(correct / total, 6)
        if extra_epoch:
            rec.update(extra_epoch(epoch + 1))
        if log_path:
            jsonl_append(log_path, rec)
        if verbose:
            print(f"Epoch [{epoch+1}/{epochs}] loss={rec['train_loss']:.4f} "
                  + (f"train_acc={rec['train_acc']:.4f} "
                     if compute_train_acc else "")
                  + f"test_acc={rec['test_acc']:.4f}")
        if acc > best_acc:
            best_acc, best_state = acc, copy.deepcopy(model.state_dict())
    return best_acc, best_state


def save_ckpt(path: str, model, scaler_mean, scaler_scale, classes,
              cfg: dict = None) -> None:
    """统一 checkpoint 格式(与历史 run 完全兼容: model_state/scaler_*/classes[/cfg])"""
    payload = {"model_state": model.state_dict(),
               "scaler_mean": scaler_mean, "scaler_scale": scaler_scale,
               "classes": classes}
    if cfg is not None:
        payload["cfg"] = cfg
    torch.save(payload, path)
