#!/usr/bin/env bash
# run_conformer_ablation.sh —— conformer 消融矩阵: attn(joint/split) x gate(glu/swiglu) x conv-norm(bn/gn)
#
# 协议与 DCNN/Transformer 完全一致: 9 类训练, 7 类评测(剔除 Feeding/Standing)
# 统一 seed 42(每 combo 固定 seed*1000+combo), 前 6 个组合(泛化在 ~6 品种即饱和)
# 每个 run 约 25 分钟, 8 个共约 3.5 小时; 不需要的组合直接注释掉对应行
#
# 用法: bash run_conformer_ablation.sh
# 对比: python3 compare_ablation.py checkpoints/data_aug/ checkpoints/ablation/*/
#       (checkpoints/data_aug/ 是 DCNN 基线, 会作为第一行参与对比)
set -e
cd "$(dirname "$0")"

EPOCH=100
SEED=42
MAX=6
OUT=checkpoints/ablation
mkdir -p "$OUT"

run () {
  local name=$1; shift
  echo "===== [$name] $* ====="
  python3 data_aug.py --model conformer --epoch $EPOCH --seed $SEED --max $MAX \
    "$@" --out-dir "$OUT/$name" 2>&1 | tee "$OUT/$name.log"
}

# -------- 2x2x2 全矩阵 --------
# joint + glu(原版, 消融基线)
# run joint_glu_bn    --attn joint --gate glu    --conv-norm bn
# 单换 GroupNorm
run joint_glu_gn    --attn joint --gate glu    --conv-norm gn
# 单换 SwiGLU
run joint_swiglu_bn --attn joint --gate swiglu --conv-norm bn
# 双换
run joint_swiglu_gn --attn joint --gate swiglu --conv-norm gn
# 单换双模态独立注意力(attention-1=acc, attention-2=gyr)
run split_glu_bn    --attn split --gate glu    --conv-norm bn
# split + GroupNorm
run split_glu_gn    --attn split --gate glu    --conv-norm gn
# split + SwiGLU
run split_swiglu_bn --attn split --gate swiglu --conv-norm bn
# 全换
run split_swiglu_gn --attn split --gate swiglu --conv-norm gn

# -------- IMU 结构偏置(物理先验变体, 与 joint_glu_bn 单变量对照) --------
# run imu_glu_bn     --attn imu --gate glu --conv-norm bn

echo ""
echo "===== 全部完成, 汇总: ====="
python3 compare_ablation.py checkpoints/data_aug/ "$OUT"/*/
