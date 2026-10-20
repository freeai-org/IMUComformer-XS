#!/usr/bin/env bash
# run_full26.sh —— 26 组合全量补跑(与现有 26 组合基线同台对比)
#
# run 1: IMU-Conformer  物理先验注意力偏置(--attn imu), 0.77M
# run 2: Conformer-XS   调参冠军配置(d64 + ff128 + 2 层), 136,905 参数
# 协议: seed 42(每 combo = seed*1000+combo), 100 epoch, joint/glu/bn, 9类训练/7类评测
# 产出: checkpoints/data_aug_imu/ , checkpoints/data_aug_d64ff128_l2/ (各 ~105 分钟)
#
# 用法: nohup bash run_full26.sh > checkpoints/full26_run.log 2>&1 &
set -e
cd "$(dirname "$0")"

python3 data_aug.py --model conformer --epoch 100 --seed 42 --max 26 \
  --attn imu --gate glu --conv-norm bn \
  --out-dir checkpoints/data_aug_imu

python3 data_aug.py --model conformer --epoch 100 --seed 42 --max 26 \
  --d-model 64 --dim-ff 128 --layers 2 --gate glu --conv-norm bn --attn joint \
  --out-dir checkpoints/data_aug_d64ff128_l2

echo "ALL_DONE — 刷新图: python3 plot/plot_model_compare_curves.py && python3 plot/plot_model_compare_heatmaps.py"
