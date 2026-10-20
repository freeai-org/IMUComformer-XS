#!/usr/bin/env python3
"""clean_neck6.py —— 清洗两个数据集, 只保留项圈(颈部)传感器六轴数据

输出(写入 /mnt/WYH/data/neck6/):
    ds1_neck6.csv       # DS1: DogID, TestNum, t_sec, ANeck_x/y/z, GNeck_x/y/z
                        #      + Task, Behavior_1/2/3, PointEvent ('<undefined>'->空)
    ds2_neck6.csv       # DS2: Timestamp, Neck.Acc.X/Y/Z, Neck.Gyr.X/Y/Z
                        #      + Type, Position, Breed, Subject (丢弃 Neck.Mag 对齐六轴)

用法:
    python3 clean_neck6.py            # 两个数据集都清洗
    python3 clean_neck6.py ds1        # 只清洗 DS1
    python3 clean_neck6.py ds2        # 只清洗 DS2

分块流式读取 + 追加写出, 全程常驻内存 < ~200MB。
"""
import os
import sys

import pandas as pd

DS1_PATH = "/mnt/WYH/data/IMU-Dog/DogMoveData.csv"
DS2_PATH = "/mnt/WYH/data/IMU-Dog-2/df_raw.csv"
OUT_DIR = "/mnt/WYH/data/neck6"
CHUNK = 1_000_000

DS1_COLS = ["DogID", "TestNum", "t_sec",
            "ANeck_x", "ANeck_y", "ANeck_z",
            "GNeck_x", "GNeck_y", "GNeck_z",
            "Task", "Behavior_1", "Behavior_2", "Behavior_3", "PointEvent"]
DS2_COLS = ["Timestamp",
            "Neck.Acc.X", "Neck.Acc.Y", "Neck.Acc.Z",
            "Neck.Gyr.X", "Neck.Gyr.Y", "Neck.Gyr.Z",
            "Type", "Position", "Breed", "Subject"]


def clean_ds1() -> None:
    out = f"{OUT_DIR}/ds1_neck6.csv"
    labels = ["Task", "Behavior_1", "Behavior_2", "Behavior_3", "PointEvent"]
    total = 0
    label_counts = {c: {} for c in labels}
    print(f"[DS1] 读取 {DS1_PATH} ...")
    for chunk in pd.read_csv(DS1_PATH, usecols=DS1_COLS, chunksize=CHUNK,
                             dtype={c: str for c in labels},
                             keep_default_na=False):
        chunk["DogID"] = chunk["DogID"].astype("int32")
        chunk["TestNum"] = chunk["TestNum"].astype("int8")
        for c in labels:
            # '<undefined>' -> 空串; PointEvent 混有数值型标注, 统一按字符串保留
            chunk[c] = chunk[c].replace({"<undefined>": ""})
        chunk.to_csv(out, index=False, header=not os.path.exists(out), mode="a")
        total += len(chunk)
        for c in labels:
            for k, v in chunk[c].value_counts().items():
                label_counts[c][k] = label_counts[c].get(k, 0) + v
        print(f"  已处理 {total:,} 行", flush=True)
    print(f"[DS1] 完成: {out} 共 {total:,} 行")
    for c in labels:
        top = sorted(label_counts[c].items(), key=lambda x: -x[1])[:6]
        print(f"  {c}: {top if top else '(全空)'}")


def clean_ds2() -> None:
    out = f"{OUT_DIR}/ds2_neck6.csv"
    labels = ["Type", "Position", "Breed", "Subject"]
    total = 0
    label_counts = {c: {} for c in labels}
    print(f"[DS2] 读取 {DS2_PATH} ...")
    for chunk in pd.read_csv(DS2_PATH, usecols=DS2_COLS, chunksize=CHUNK,
                             dtype={c: str for c in labels},
                             keep_default_na=False):
        chunk.to_csv(out, index=False, header=not os.path.exists(out), mode="a")
        total += len(chunk)
        for c in labels:
            for k, v in chunk[c].value_counts().items():
                label_counts[c][k] = label_counts[c].get(k, 0) + v
        print(f"  已处理 {total:,} 行", flush=True)
    print(f"[DS2] 完成: {out} 共 {total:,} 行")
    for c in labels:
        top = sorted(label_counts[c].items(), key=lambda x: -x[1])[:6]
        print(f"  {c}: {top}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("ds1", "all"):
        clean_ds1()
    if which in ("ds2", "all"):
        clean_ds2()


if __name__ == "__main__":
    main()
