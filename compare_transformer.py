#!/usr/bin/env python3
"""compare_transformer.py —— Transformer(双模态 full attention) vs DCNN 对比

对比 checkpoints/data_aug_transformer/ 与 checkpoints/data_aug/ 的评测结果。
输出: checkpoints/data_aug_transformer/transformer_compare.csv + 终端汇总
"""
import os

import numpy as np
import pandas as pd

from utils.protocol import load_jsonl, offdiag_values

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TR = os.path.join(SCRIPT_DIR, "checkpoints", "data_aug_transformer", "dataaug_infer.jsonl")
DC = os.path.join(SCRIPT_DIR, "checkpoints", "data_aug", "dataaug_infer.jsonl")
OUT = os.path.join(SCRIPT_DIR, "checkpoints", "data_aug_transformer", "transformer_compare.csv")


def off_mean(r):
    v = offdiag_values(r)
    return float(np.mean(v)) if v else np.nan  # combo 26 全量无组合外


def main():
    tr, dc = load_jsonl(TR), load_jsonl(DC)
    rows = []
    print(f'{"combo":>5} {"新增品种":<30}{"TR-组合外":>10}{"DCNN-组合外":>11}{"Δ":>8}'
          f'{"TR-全26":>9}{"DCNN-全26":>10}')
    for t, d in zip(tr, dc):
        ot, od = off_mean(t), off_mean(d)
        rows.append({"combo": t["combo"], "last_added": t["last_added"],
                     "tr_off": ot, "dcnn_off": od,
                     "tr_all": t["mean_acc"], "dcnn_all": d["mean_acc"]})
        print(f"{t['combo']:>5} {t['last_added'][:28]:<30}{ot:>10.4f}{od:>11.4f}"
              f"{ot - od:>+8.4f}{t['mean_acc']:>9.4f}{d['mean_acc']:>10.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    rel = df[df["combo"] <= 21]
    print("\n===== 汇总(7 类评测协议) =====")
    print(f"组合外均值(可靠区间 1-21): Transformer {rel['tr_off'].mean():.4f} vs "
          f"DCNN {rel['dcnn_off'].mean():.4f}  Δ={rel['tr_off'].mean()-rel['dcnn_off'].mean():+.4f}")
    print(f"组合外均值(种子期 1-6):    Transformer {df[df['combo']<=6]['tr_off'].mean():.4f} vs "
          f"DCNN {df[df['combo']<=6]['dcnn_off'].mean():.4f}")
    print(f"全26均值(26组合平均):      Transformer {df['tr_all'].mean():.4f} vs "
          f"DCNN {df['dcnn_all'].mean():.4f}")
    print(f"可靠区间最佳组合:          Transformer combo {rel.loc[rel['tr_off'].idxmax(),'combo']:.0f} "
          f"({rel['tr_off'].max():.4f}) vs DCNN combo {rel.loc[rel['dcnn_off'].idxmax(),'combo']:.0f} "
          f"({rel['dcnn_off'].max():.4f})")
    print("\n===== 难点品种(组合外) =====")
    for br in ["Nova Scotia Duck Tolling Retriever", "Mudi", "German Shepherd"]:
        tt = [t["per_breed_acc"][br] for t in tr if br not in t["breeds"]]
        dd = [d["per_breed_acc"][br] for d in dc if br not in d["breeds"]]
        print(f"  {br:<38} TR {np.mean(tt):.4f} vs DCNN {np.mean(dd):.4f} "
              f"({np.mean(tt)-np.mean(dd):+.4f})")
    print("\n明细:", OUT)


if __name__ == "__main__":
    main()
