#!/usr/bin/env python3
"""cli.py —— argparse 定义共享模块

收编原先 dcnn.py / breed_cv.py / data_aug.py 三处重复的 --epoch/--window/
--stride/--hz/--batch-size/--lr 定义(conformer 超参原本在 argparse default 与
getattr default 双份维护, 现统一为 models.CONFORMER_DEFAULTS 单源)。
默认值保持与历史一致: dcnn --epoch 30, 其余 100。
"""
import argparse

# conformer 超参默认值唯一来源(argparse 默认值/models.build_model/checkpoint
# 回放兜底都用它; 注意放本文件而非 models.py —— models 导入 dcnn, dcnn 导入
# 本文件, 默认值放 models 会形成 models->dcnn->cli->models 循环导入)
CONFORMER_DEFAULTS = dict(
    d_model=128, nhead=4, num_layers=3, dim_ff=256,
    patch=10, kernel=15, dropout=0.2,
    gate="glu", conv_norm="bn", attn_mode="joint",
)


def add_common_args(ap: argparse.ArgumentParser, default_epoch: int = 100):
    ap.add_argument("--epoch", type=int, default=default_epoch,
                    help="训练轮数")
    ap.add_argument("--window", type=int, default=100,
                    help="窗口采样点数(统一 Hz 下)")
    ap.add_argument("--stride", type=int, default=50,
                    help="滑窗步长(采样点数)")
    ap.add_argument("--hz", type=int, default=100,
                    help="统一采样率(<=100 且为 100 的整数因子)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    return ap


def add_conformer_args(ap: argparse.ArgumentParser):
    """conformer 超参 + Adam weight decay(默认值单源: CONFORMER_DEFAULTS)"""
    d = CONFORMER_DEFAULTS
    ap.add_argument("--gate", type=str, default=d["gate"],
                    choices=["glu", "swiglu"],
                    help="conformer 卷积模块门控: glu(原版) / swiglu(平滑自门控, 不易压死通道)")
    ap.add_argument("--conv-norm", type=str, default=d["conv_norm"],
                    choices=["bn", "gn"],
                    help="conformer 卷积模块归一化: bn(跨品种有域偏移) / gn(GroupNorm, 域稳)")
    ap.add_argument("--attn", type=str, default=d["attn_mode"],
                    choices=["joint", "split", "imu"],
                    help="conformer 注意力: joint(acc+gyr 拼接全交互) / "
                         "split(双模态独立: attention-1 只看 acc, attention-2 只看 gyr) / "
                         "imu(joint 全交互 + IMU 结构偏置: 同模态随时间距离衰减, "
                         "跨模态仅同一时刻免罚, 斜率可学习)")
    ap.add_argument("--d-model", type=int, default=d["d_model"],
                    help="conformer token 维度(容量主旋钮)")
    ap.add_argument("--nhead", type=int, default=d["nhead"],
                    help="conformer 注意力头数")
    ap.add_argument("--layers", type=int, default=d["num_layers"],
                    help="conformer 块数")
    ap.add_argument("--dim-ff", type=int, default=d["dim_ff"],
                    help="conformer FFN 隐层维度")
    ap.add_argument("--patch", type=int, default=d["patch"],
                    help="conformer patch 大小(采样点/patch)")
    ap.add_argument("--kernel", type=int, default=d["kernel"],
                    help="conformer 深度卷积核大小")
    ap.add_argument("--dropout", type=float, default=d["dropout"],
                    help="conformer dropout(正则化主旋钮)")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="Adam weight decay")
    return ap


def conformer_hparams(args) -> dict:
    """args -> conformer 超参 dict(直接喂 models.build_model 的 **hp)"""
    return {k: getattr(args, k) for k in
            ("gate", "conv_norm", "attn", "d_model", "nhead", "layers",
             "dim_ff", "patch", "kernel", "dropout")}
