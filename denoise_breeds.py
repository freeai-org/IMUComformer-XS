#!/usr/bin/env python3
"""denoise_breeds.py —— 按白噪声分析结论去噪品种数据(project-5 第三步·优化)

方法: 零相位 FFT 平滑低通 H(f)=1/sqrt(1+(f/fc)^8), 按 (DogID,TestNum) 分组、
时间断裂处(删除行造成)再分块, 逐块滤波(减均值->滤波->恢复均值, 保留直流)。
fc 由 noise_analysis.json 的 f_cut 与行为信号带宽(<10Hz)折中, 默认 12Hz。
输入: /mnt/WYH/data/neck6/breeds/*.csv (26 品种)
输出: /mnt/WYH/data/neck6/breeds_denoise/*.csv + breeds_manifest.json (同名同格式)

用法: python3 denoise_breeds.py [--fc 12]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

BREED_DIR = "/mnt/WYH/data/neck6/breeds"
OUT_DIR = "/mnt/WYH/data/neck6/breeds_denoise"
FEATURES = ["ANeck_x", "ANeck_y", "ANeck_z", "GNeck_x", "GNeck_y", "GNeck_z"]
FS = 100.0
DT = 0.01
GAP = 0.011  # 相邻行时间差超过此值视为断裂, 分块滤波


def lowpass_response(n: int, fc: float) -> np.ndarray:
    """零相位平滑低通频响(8 阶滚降)"""
    f = np.fft.rfftfreq(n, 1.0 / FS)
    return 1.0 / np.sqrt(1.0 + (f / fc) ** 8)


def filter_chunk(chunk: np.ndarray, fc: float) -> np.ndarray:
    """对 [n, 6] 连续块做零相位低通, 保留各通道直流"""
    mean = chunk.mean(axis=0)
    H = lowpass_response(len(chunk), fc)
    X = np.fft.rfft(chunk - mean, axis=0)
    return np.fft.irfft(X * H[:, None], n=len(chunk), axis=0) + mean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fc", type=float, default=12.0, help="低通截止频率 Hz")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    fc = args.fc

    manifest = json.load(open(os.path.join(BREED_DIR, "breeds_manifest.json")))
    with open(os.path.join(OUT_DIR, "breeds_manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"低通截止 fc={fc}Hz, 输出 {OUT_DIR}")
    for breed, fname in sorted(manifest.items()):
        src = os.path.join(BREED_DIR, fname)
        df = pd.read_csv(src, keep_default_na=False,
                         dtype={"Behavior_1": str, "Task": str})
        for _, g in df.groupby(["DogID", "TestNum"], sort=False):
            t = g["t_sec"].to_numpy()
            brk = np.where(np.abs(np.diff(t) - DT) >= GAP - 1e-9)[0] + 1
            bounds = np.concatenate([[0], brk, [len(g)]])
            for s, e in zip(bounds[:-1], bounds[1:]):
                if e - s < 4:
                    continue
                idx = g.index[s:e]
                df.loc[idx, FEATURES] = filter_chunk(
                    g[FEATURES].to_numpy(np.float64)[s:e], fc)
        df.to_csv(os.path.join(OUT_DIR, fname), index=False)
        print(f"  {fname:<40} {len(df):,} 行")


if __name__ == "__main__":
    main()
