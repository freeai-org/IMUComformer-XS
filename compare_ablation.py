#!/usr/bin/env python3
"""compare_ablation.py —— 汇总多个实验 run 的评测结果, 输出对比表

主指标为组合外均值(offdiag): 去掉该组合训练过的品种后的 26 品种均值,
即"模型没见过的品种"上的 7 类协议准确率, 跨品种泛化的核心口径。

用法:
    python3 compare_ablation.py checkpoints/data_aug/ checkpoints/ablation/*/
    # 不带参数则默认扫 checkpoints/ablation/ 下所有 run
"""
import glob
import os
import sys

import numpy as np

from utils.protocol import load_jsonl, offdiag_series

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_run(d):
    p = os.path.join(d, "dataaug_infer.jsonl")
    return load_jsonl(p) if os.path.exists(p) else None


def main():
    dirs = sys.argv[1:]
    if not dirs:
        dirs = sorted(glob.glob(os.path.join(SCRIPT_DIR, "checkpoints/ablation/*/")))
    rows = []
    for d in dirs:
        recs = load_run(d.rstrip("/"))
        if not recs:
            continue
        name = os.path.basename(d.rstrip("/"))
        # 每个 combo: 组合外均值 = per_breed_acc 去掉该组合训练过的品种
        # (combo 26 全量没有组合外, offdiag_series 自动跳过)
        rows.append((name, recs, offdiag_series(recs)))

    if not rows:
        print("没有找到任何含 dataaug_infer.jsonl 的目录")
        return

    # 对齐到公共组合数(消融 run 可能只跑前 6 个 combo, 与全量基线公平可比)
    n_combo = min(len(r[1]) for r in rows)
    rows = [(n, rs[:n_combo], of[:n_combo]) for n, rs, of in rows]

    hdr = "run".ljust(22) + "".join(f"c{k+1:>7}" for k in range(n_combo)) \
        + "  combo外均值"
    print(hdr)
    print("-" * len(hdr))
    for name, recs, offs in sorted(rows):
        means = "".join(f"{r['mean_acc']:7.4f}" for r in recs)
        print(f"{name:<22}{means}  {np.mean(offs):9.4f}")
    print(f"\n(注: 已对齐到前 {n_combo} 个组合; c_k 列为组合均值(含组合内 test split, "
          "偏乐观); combo外均值才是跨品种泛化口径)")

    # ---------- markdown 汇总表(带排序) ----------
    stats = sorted(((np.mean(of), n, rs) for n, rs, of in rows), reverse=True)
    lines = [f"# 消融对比表(前 {n_combo} 组合, seed 42, 7 类协议)", "",
             "组合外均值 = 每个 combo 去掉训练品种后的准确率均值(跨品种泛化主指标); "
             "组合均值 = 26 品种 mean_acc(含组合内 test split, 偏乐观)。", "",
             "| 排名 | run | " + " | ".join(f"c{k+1}" for k in range(n_combo))
             + " | **组合外均值** |",
             "|---|---|" + "---|" * (n_combo + 1)]
    for rank, (offmean, name, recs) in enumerate(stats, 1):
        cs = " | ".join(f"{r['mean_acc']:.4f}" for r in recs)
        lines.append(f"| {rank} | {name} | {cs} | **{offmean:.4f}** |")
    lines += ["", f"极差(最好-最差): {stats[0][0] - stats[-1][0]:.4f}"]
    # 自动表存 ablation_compare_table.md(机打表); ablation_summary.md 是手工
    # 维护的结论文档(含假设判定/斜率分析), 勿让脚本覆盖它
    md = os.path.join(SCRIPT_DIR, "results", "ablation_compare_table.md")
    os.makedirs(os.path.dirname(md), exist_ok=True)
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("saved:", md)


if __name__ == "__main__":
    main()
