#!/usr/bin/env bash
# run_attn_tune.sh —— conformer 调参第一轮: 正则化↑ / 容量↓ (攻过拟合)
#
# 依据: loss-泛化倒挂(conformer 训练 loss 0.10 vs DCNN 0.28, 跨品种泛化反而
#       差 2.4 点) → 瓶颈是 0.77M 参数在 0.9~4.4 万窗口上的过拟合, 结构侧
#       四个假设(gate/norm/attn组织/IMU偏置)已全部证伪, 本轮只动超参。
# 协议: 与消融矩阵完全一致 —— seed 42, 前 6 组合, 100 epoch, joint/glu/bn。
# 目标: 组合外均值超过 DCNN 基线 0.7715(conformer 现最高 0.7472)。
#
# 用法: bash run_attn_tune.sh
# 对比: python3 compare_ablation.py checkpoints/data_aug/ \
#         checkpoints/data_aug_conformer_v2/ checkpoints/attn_tune/*/
set -e
cd "$(dirname "$0")"

EPOCH=100
SEED=42
MAX=6
OUT=checkpoints/attn_tune
mkdir -p "$OUT"

run () {
  local name=$1; shift
  echo "===== [$name] $* ====="
  python3 data_aug.py --model conformer --epoch $EPOCH --seed $SEED --max $MAX \
    "$@" --out-dir "$OUT/$name" 2>&1 | tee "$OUT/$name.log"
}

# T1 正则化单变量: dropout 0.2 -> 0.4
run drop40          --dropout 0.4
# T2 容量减半: d_model 128 -> 64 (0.77M -> ~0.30M)
run d64             --d-model 64
# T3 正则化+减容量双管: d64 + dropout 0.4
run d64_drop40      --d-model 64 --dropout 0.4
# T4 容量对齐 DCNN: d64 + dim_ff 128 + 2 层 (~0.13M, 最接近 DCNN 93k)
run d64ff128_l2     --d-model 64 --dim-ff 128 --layers 2

echo ""
echo "===== 全部完成, 汇总(表会另存 results/ablation_compare_table.md): ====="
python3 compare_ablation.py checkpoints/data_aug/ checkpoints/data_aug_conformer_v2/ "$OUT"/*/
