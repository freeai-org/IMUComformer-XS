#!/usr/bin/env python3
"""remap_ds1_labels.py —— DS1 行为标签规范化(大类, 参考 DS2 的 5 姿态类粒度)

规则:
  删除类别: Pacing, Synchronization, Extra_Synchronization, Panting, 空/未定义
  合并:     Eating+Drinking -> Feeding; Trotting+Galloping -> Running
  归大类(9): Lying/Sitting/Standing/Walking/Running/Shaking/
            Feeding/Sniffing/Playing(Playing,Tugging,Carrying object,Jumping,Bowing)
  槽位压缩: 剩余非空标签左移; 三槽全空的行整行删除
  Task/PointEvent 原样保留

输出: /mnt/WYH/data/neck6/ds1_neck6_remap.csv
用法: python3 remap_ds1_labels.py
"""
import os

import pandas as pd

SRC = "/mnt/WYH/data/neck6/ds1_neck6.csv"
OUT = "/mnt/WYH/data/neck6/ds1_neck6_remap.csv"
CHUNK = 1_000_000
SLOTS = ["Behavior_1", "Behavior_2", "Behavior_3"]

DELETE = {"Pacing", "Synchronization", "Extra_Synchronization", "Panting"}
MAJOR = {  # 原类别 -> 大类
    "Lying chest": "Lying", "Sitting": "Sitting", "Standing": "Standing",
    "Walking": "Walking", "Trotting": "Running", "Galloping": "Running",
    "Shaking": "Shaking", "Eating": "Feeding", "Drinking": "Feeding",
    "Sniffing": "Sniffing", "Playing": "Playing", "Tugging": "Playing",
    "Carrying object": "Playing", "Jumping": "Playing", "Bowing": "Playing",
}


def remap_row(b1: str, b2: str, b3: str):
    vals = []
    for v in (b1, b2, b3):
        if not v or v in DELETE:
            continue
        vals.append(MAJOR[v])
    vals += [""] * (3 - len(vals))
    return vals


def main() -> None:
    total_in = total_out = 0
    class_rows: dict[str, int] = {}   # 含该大类的行数
    if os.path.exists(OUT):
        os.remove(OUT)
    print(f"读取 {SRC} ...")
    for chunk in pd.read_csv(SRC, chunksize=CHUNK, dtype=str,
                             keep_default_na=False):
        total_in += len(chunk)
        new = chunk[SLOTS].apply(lambda r: remap_row(*r), axis=1, result_type="expand")
        new.columns = SLOTS
        chunk[SLOTS] = new
        chunk = chunk[chunk["Behavior_1"] != ""]
        for v in chunk["Behavior_1"]:
            class_rows[v] = class_rows.get(v, 0) + 1
        for v2 in chunk[chunk["Behavior_2"] != ""]["Behavior_2"]:
            class_rows[v2] = class_rows.get(v2, 0) + 1
        for v3 in chunk[chunk["Behavior_3"] != ""]["Behavior_3"]:
            class_rows[v3] = class_rows.get(v3, 0) + 1
        chunk.to_csv(OUT, index=False, header=not os.path.exists(OUT), mode="a")
        total_out += len(chunk)
        print(f"  已处理 {total_in:,} 行, 保留 {total_out:,} 行", flush=True)
    print(f"\n完成: {OUT}")
    print(f"输入 {total_in:,} 行 -> 输出 {total_out:,} 行 "
          f"(删除 {total_in - total_out:,} 行, 保留率 {total_out/total_in:.1%})")
    print(f"\n大类数(多标签行会重复计入): {len(class_rows)}")
    for k, v in sorted(class_rows.items(), key=lambda x: -x[1]):
        print(f"  {k:<10}{v:>12,}")


if __name__ == "__main__":
    main()
