#!/usr/bin/env python3
"""noise_analysis.py —— 传感器白噪声分析(project-5 第三步·分析)

思路: 传感器随狗趴卧(Lying)时近似静止, 取静止段估计六轴白噪声:
  - 一阶差分法: 慢信号相邻采样近似不变, Var(diff)≈2σ² -> σ_noise=std(diff)/√2
  - Welch PSD: 高频段(30-50Hz)平顶 = 白噪声底, 交叉验证 σ
  - 自相关: 白噪声应只在 lag=0 有尖峰
  - 差分直方图 + 高斯拟合: 检验噪声是否高斯
四联图 results/noise_analysis.png, 数值 results/noise_analysis.json

用法: python3 noise_analysis.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CSV = "/mnt/WYH/data/neck6/ds1_neck6_remap.csv"
OUT_PNG = os.path.join(RESULTS_DIR, "noise_analysis.png")
OUT_JSON = os.path.join(RESULTS_DIR, "noise_analysis.json")

FEATURES = ["ANeck_x", "ANeck_y", "ANeck_z", "GNeck_x", "GNeck_y", "GNeck_z"]
FS = 100.0
DT = 0.01
STATIC_BEHAVIOR = "Lying"   # 趴卧: 颈部传感器最静止
MIN_RUN_DIFF = 100          # 差分估计用的最短静止段(1s)
NFFT = 4096
MIN_RUN_PSD = NFFT          # PSD 用的最短静止段(须 >= NFFT 样本)


def find_static_runs(df: pd.DataFrame) -> list:
    """返回静止段列表: dict(dog, test, i0, i1, n) 行号 [i0, i1) 连续且 dt≈0.01"""
    runs = []
    for (dog, test), g in df.groupby(["DogID", "TestNum"], sort=False):
        lying = (g["Behavior_1"] == STATIC_BEHAVIOR).to_numpy()
        t = g["t_sec"].to_numpy()
        dt_ok = np.abs(np.diff(t) - DT) < 0.002
        idx = g.index.to_numpy()
        start = None
        for k in range(len(g)):
            if lying[k]:
                if start is None:
                    start = k
                elif not dt_ok[k - 1]:  # 与上一行时间不连续, 断段
                    if k - start >= MIN_RUN_DIFF:
                        runs.append(dict(dog=int(dog), test=int(test),
                                         i0=idx[start], i1=idx[k], n=k - start))
                    start = k
            else:
                if start is not None and k - start >= MIN_RUN_DIFF:
                    runs.append(dict(dog=int(dog), test=int(test),
                                     i0=idx[start], i1=idx[k], n=k - start))
                start = None
        if start is not None and len(g) - start >= MIN_RUN_DIFF:
            runs.append(dict(dog=int(dog), test=int(test),
                             i0=idx[start], i1=idx[len(g)], n=len(g) - start))
    return runs


def welch_psd(x: np.ndarray, nfft: int = NFFT) -> np.ndarray:
    """手写 Welch(汉宁窗, 50% 重叠), 返回单边功率谱密度 [V²/Hz]"""
    segs = []
    hop = nfft // 2
    for s in range(0, len(x) - nfft + 1, hop):
        segs.append(x[s:s + nfft] * np.hanning(nfft))
    P = np.zeros(nfft // 2 + 1)
    for s in segs:
        F = np.fft.rfft(s)
        P += np.abs(F) ** 2
    P /= len(segs)
    # 单边 PSD: 乘 2(除直流与奈奎斯特), 除 (win²均值 * nfft * fs)
    U = np.mean(np.hanning(nfft) ** 2)
    P = P / (U * nfft * FS)
    P[1:-1] *= 2
    return P


def main() -> None:
    print("读取数据 ...")
    df = pd.read_csv(CSV, usecols=["DogID", "TestNum", "t_sec", "Behavior_1"] + FEATURES,
                     dtype={"Behavior_1": str}, keep_default_na=False)
    print(f"{len(df):,} 行")

    print("搜索静止段(Lying) ...")
    runs = find_static_runs(df)
    runs.sort(key=lambda r: -r["n"])
    total_static = sum(r["n"] for r in runs)
    print(f"静止段 {len(runs)} 个(>=1s), 总时长 {total_static/FS/60:.1f} 分钟, "
          f"最长 {runs[0]['n']/FS/60:.1f} 分钟 (dog {runs[0]['dog']} test {runs[0]['test']})")

    data = df[FEATURES].to_numpy(np.float64)

    # ---------- 1) 差分法噪声 σ (所有 >=1s 静止段) ----------
    diff_sumsq = np.zeros(6)
    diff_cnt = 0
    diff_pool = {f: [] for f in FEATURES}  # 抽样用于直方图
    for r in runs:
        seg = data[r["i0"]:r["i1"]]
        d = np.diff(seg, axis=0)
        diff_sumsq += (d ** 2).sum(axis=0)
        diff_cnt += len(d)
        if r["n"] >= 1000:  # 长段抽样进直方图池
            for j, f in enumerate(FEATURES):
                if len(diff_pool[f]) < 200000:
                    diff_pool[f].append(d[::5, j])
    sigma_diff = np.sqrt(diff_sumsq / diff_cnt / 2.0)

    # ---------- 2) Welch PSD (>=30s 静止段) ----------
    freqs = np.fft.rfftfreq(NFFT, 1.0 / FS)
    psd_runs = [r for r in runs if r["n"] >= MIN_RUN_PSD]
    psd = {f: np.zeros(NFFT // 2 + 1) for f in FEATURES}
    psd_n = {f: 0 for f in FEATURES}
    for r in psd_runs:
        seg = data[r["i0"]:r["i1"]]
        for j, f in enumerate(FEATURES):
            psd[f] += welch_psd(seg[:, j])
            psd_n[f] += 1
    for f in FEATURES:
        psd[f] /= max(psd_n[f], 1)
    band = (freqs >= 30) & (freqs <= 50)  # 噪声底频段
    floor = {f: float(psd[f][band].mean()) for f in FEATURES}
    sigma_floor = {f: float(np.sqrt(floor[f] * FS / 2)) for f in FEATURES}

    # 信号截止频率: 各通道 PSD 首次贴近噪声底(<3x)的频率取中位数
    f_cuts = []
    for f in FEATURES:
        ratio = psd[f] / floor[f]
        hit = np.where((freqs > 1) & (ratio < 3))[0]
        f_cuts.append(freqs[hit[0]] if len(hit) else 50.0)
    f_cut = float(np.median(f_cuts))

    # 整体信号 std(全量数据, 含行为) -> SNR
    signal_std = {f: float(df[f].std()) for f in FEATURES}

    # ---------- 3) 自相关(最长静止段, ANeck_x / GNeck_x) ----------
    longest = runs[0]
    seg = data[longest["i0"]:longest["i1"]]
    LAG = 100  # ±1s
    acf = {}
    for f in ("ANeck_x", "GNeck_x"):
        j = FEATURES.index(f)
        x = seg[:60000, j] - seg[:60000, j].mean()
        n = len(x)
        F = np.fft.rfft(x, 2 * n)
        ac = np.fft.irfft(F * np.conj(F))[:LAG + 1]
        ac /= ac[0]
        acf[f] = np.concatenate([ac[:LAG + 1][::-1], ac[1:]])  # [-LAG, LAG]

    # ---------- 4) 汇总输出 ----------
    print("\n===== 白噪声估计 =====")
    print(f"{'通道':<10}{'σ_diff':>10}{'σ_PSD底':>10}{'σ_信号':>10}{'SNR(dB)':>9}")
    stats = {}
    for j, f in enumerate(FEATURES):
        snr = 20 * np.log10(signal_std[f] / sigma_diff[j])
        stats[f] = {"sigma_noise_diff": round(float(sigma_diff[j]), 6),
                    "sigma_noise_psd": round(sigma_floor[f], 6),
                    "sigma_signal": round(signal_std[f], 6),
                    "snr_db": round(float(snr), 2)}
        print(f"{f:<10}{sigma_diff[j]:>10.5f}{sigma_floor[f]:>10.5f}"
              f"{signal_std[f]:>10.4f}{snr:>9.1f}")
    print(f"\n建议去噪截止频率 f_cut ≈ {f_cut:.1f} Hz (PSD 贴近噪声底的中位频率)")
    stats["_meta"] = {"static_behavior": STATIC_BEHAVIOR,
                      "n_runs": len(runs), "static_minutes": round(total_static / FS / 60, 1),
                      "longest_run_min": round(runs[0]["n"] / FS / 60, 1),
                      "f_cut_hz": round(f_cut, 1)}
    with open(OUT_JSON, "w") as fp:
        json.dump(stats, fp, ensure_ascii=False, indent=2)
    print("数值已存:", OUT_JSON)

    # ---------- 四联图 ----------
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    colors = plt.cm.tab10(range(6))

    # (a) 时域: 最长静止段前 30s, 6 通道纵向偏移
    ax = axes[0][0]
    show = seg[:3000]
    t_ax = np.arange(len(show)) * DT
    offs = np.array([0, 0.02, 0.04, 0, 0.02, 0.04]) * 8
    for j, f in enumerate(FEATURES):
        y = show[:, j] - show[:, j].mean()
        ax.plot(t_ax, y + offs[j] + (2.5 if j >= 3 else 0), lw=0.4,
                color=colors[j], label=f)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("acc (g) top | gyr (deg/s) bottom, offset")
    ax.set_title(f"(a) Longest static segment (Lying, dog {longest['dog']}, "
                 f"{longest['n']/FS/60:.1f} min), first 30 s")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.grid(alpha=0.3)

    # (b) PSD log-log + 噪声底 + 截止频率
    ax = axes[0][1]
    for j, f in enumerate(FEATURES):
        ax.semilogy(freqs[1:], psd[f][1:], lw=0.8, color=colors[j], label=f)
        ax.axhline(floor[f], color=colors[j], ls=":", lw=0.8, alpha=0.6)
    ax.axvline(f_cut, color="k", ls="--", lw=1.2, label=f"f_cut≈{f_cut:.0f}Hz")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("(b) Static-segment PSD (Welch): flat HF floor = white noise")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, which="both")

    # (c) 自相关
    ax = axes[1][0]
    lags = np.arange(-LAG, LAG + 1)
    for f, c in (("ANeck_x", colors[0]), ("GNeck_x", colors[3])):
        ax.plot(lags, acf[f], lw=1, color=c, label=f)
    ax.set_xlabel("lag (samples, 0.01s each)")
    ax.set_ylabel("autocorrelation")
    ax.set_title("(c) Autocorrelation: white noise -> spike only at lag 0")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (d) 差分直方图(按各通道 σ 归一化) + 标准高斯对照
    ax = axes[1][1]
    z = np.linspace(-5, 5, 200)
    for j, f in enumerate(FEATURES):
        d = np.concatenate(diff_pool[f]) / np.sqrt(2) / sigma_diff[j]
        if len(d) == 0:
            continue
        hist, edges = np.histogram(d, bins=100, range=(-5, 5), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, hist, lw=0.9, color=colors[j], alpha=0.7, label=f)
    ax.plot(z, np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi),
            "k--", lw=1.2, label="N(0,1)")
    ax.set_xlabel("diff / (√2 σ_noise)  (per-channel normalized)")
    ax.set_ylabel("density")
    ax.set_title("(d) Normalized diff distribution vs N(0,1): "
                 "tails > Gaussian = burst/motion noise")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    fig.suptitle("Sensor white-noise analysis (static Lying segments, DS1 neck IMU, 100Hz)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PNG, dpi=150)
    print("图已存:", OUT_PNG)


if __name__ == "__main__":
    main()
