#!/usr/bin/env python3
"""preview.py —— 查看两个 IMU 数据集的前 n 行，打印到控制台

用法:
    python3 preview.py                # 默认两个数据集各看前 5 行
    python3 preview.py -n 10          # 各看前 10 行
    python3 preview.py -n 20 ds1      # 只看 IMU-Dog
    python3 preview.py -n 20 ds2      # 只看 IMU-Dog-2
"""
import argparse

import pandas as pd

DS1_PATH = "/mnt/WYH/data/IMU-Dog/DogMoveData.csv"
DS2_PATH = "/mnt/WYH/data/IMU-Dog-2/df_raw.csv"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)


def preview(path: str, n: int, name: str) -> None:
    df = pd.read_csv(path, nrows=n, dtype=str, keep_default_na=False)
    print(f"\n===== {name}: {path} =====")
    print(f"shape(前{n}行×全列): {df.shape}")
    print(df.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=5, help="打印前 n 行 (默认 5)")
    ap.add_argument("dataset", nargs="?", choices=["ds1", "ds2", "all"], default="all")
    args = ap.parse_args()

    if args.dataset in ("ds1", "all"):
        preview(DS1_PATH, args.n, "DS1 IMU-Dog")
    if args.dataset in ("ds2", "all"):
        preview(DS2_PATH, args.n, "DS2 IMU-Dog-2")


if __name__ == "__main__":
    main()
