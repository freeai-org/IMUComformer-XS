#!/usr/bin/env python3
"""compare_denoise.py —— 原始 vs 去噪(12Hz 低通)增广实验对比

对比 checkpoints/data_aug/ 与 checkpoints/data_aug_denoise/ 的评测结果:
  - 全 26 品种均值(含组合内 test split)
  - 纯跨品种泛化(组合外品种均值)
  - 难点品种专项(NSDTR / Mudi / Belgian Shepherd)
输出: checkpoints/data_aug_denoise/denoise_compare.csv + 终端表格
"""
import os

import numpy as np
import pandas as pd

from utils.protocol import load_jsonl, offdiag_values

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(SCRIPT_DIR, "checkpoints", "data_aug", "dataaug_infer.jsonl")
DEN = os.path.join(SCRIPT_DIR, "checkpoints", "data_aug_denoise", "dataaug_infer.jsonl")
OUT_CSV = os.path.join(SCRIPT_DIR, "checkpoints", "data_aug_denoise", "denoise_compare.csv")


def main():
    raw, den = load_jsonl(RAW), load_jsonl(DEN)
    rows = []
    print(f'{"combo":>5} {"新增品种":<28}{"原始-组合外":>11}{"去噪-组合外":>11}{"Δoff":>8}'
          f'{"原始-全26":>10}{"去噪-全26":>10}{"Δall":>8}')
    for r, d in zip(raw, den):
        k = r["combo"]
        vr, vd = offdiag_values(r), offdiag_values(d)
        # combo 26 全量没有组合外品种 -> NaN
        off_r = float(np.mean(vr)) if vr else np.nan
        off_d = float(np.mean(vd)) if vd else np.nan
        rows.append({"combo": k, "last_added": r["last_added"],
                     "raw_off": off_r, "den_off": off_d,
                     "raw_all": r["mean_acc"], "den_all": d["mean_acc"]})
        print(f"{k:>5} {r['last_added'][:26]:<28}{off_r:>11.4f}{off_d:>11.4f}"
              f"{off_d - off_r:>+8.4f}{r['mean_acc']:>10.4f}{d['mean_acc']:>10.4f}"
              f"{d['mean_acc'] - r['mean_acc']:>+8.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    rel = df[df["combo"] <= 21]
    print("\n===== 汇总 =====")
    print(f"组合外均值(可靠区间 combo1-21): 原始 {rel['raw_off'].mean():.4f} -> "
          f"去噪 {rel['den_off'].mean():.4f} ({rel['den_off'].mean()-rel['raw_off'].mean():+.4f})")
    print(f"组合外均值(种子期 combo1-6):  原始 {df[df['combo']<=6]['raw_off'].mean():.4f} -> "
          f"去噪 {df[df['combo']<=6]['den_off'].mean():.4f} "
          f"({df[df['combo']<=6]['den_off'].mean()-df[df['combo']<=6]['raw_off'].mean():+.4f})")
    print(f"全26均值(26组合平均):         原始 {df['raw_all'].mean():.4f} -> "
          f"去噪 {df['den_all'].mean():.4f} ({df['den_all'].mean()-df['raw_all'].mean():+.4f})")
    print(f"最佳可靠组合(combo<=21):      原始 combo {rel.loc[rel['raw_off'].idxmax(),'combo']:.0f} "
          f"{rel['raw_off'].max():.4f} -> 去噪 combo {rel.loc[rel['den_off'].idxmax(),'combo']:.0f} "
          f"{rel['den_off'].max():.4f}")
    print("\n===== 难点品种专项(组合外, 模型未训过的品种) =====")
    for br in ["Nova Scotia Duck Tolling Retriever", "Mudi", "Belgian Shepherd"]:
        rr = [r["per_breed_acc"][br] for r in raw if br not in r["breeds"]]
        dd = [d["per_breed_acc"][br] for d in den if br not in d["breeds"]]
        print(f"  {br:<38} {np.mean(rr):.4f} -> {np.mean(dd):.4f} ({np.mean(dd)-np.mean(rr):+.4f})")
    print("\n明细已存:", OUT_CSV)


if __name__ == "__main__":
    main()
