# Conformer 消融对比表（前 6 组合，seed 42，9 类训练 / 7 类评测协议）

三个结构消融维度：`attn`（joint=拼接全交互 / split=双模态独立注意力）、
`gate`（glu / swiglu）、`conv-norm`（bn / gn）；外加第四个**物理先验变体**
`imu_glu_bn`（`--attn imu`：joint 全交互 + IMU 结构偏置——同模态随时间距离
衰减 / 跨模态仅同一时刻免罚，斜率可学习，仅 +6 标量参数）。
组合外均值 = 每个 combo 去掉训练品种后的 26 品种准确率均值（**跨品种泛化主指标**）；
c_k 列为组合均值（含组合内 test split，偏乐观）。

| 排名 | run | attn | gate | norm | c1 | c2 | c3 | c4 | c5 | c6 | **组合外均值(1-6)** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **DCNN 基线**（data_aug，未固定种子） | - | - | - | 0.7455 | 0.7631 | 0.7927 | 0.8050 | 0.8146 | 0.8256 | **0.7715** |
| 2 | joint_glu_bn（=v2 前6，原版基线） | joint | glu | bn | 0.7087 | 0.7437 | 0.7619 | 0.7959 | 0.8021 | 0.8090 | **0.7472** |
| 3 | split_swiglu_gn | split | swiglu | gn | 0.7102 | 0.7452 | 0.7591 | 0.7895 | 0.7971 | 0.8027 | **0.7456** |
| 4 | joint_swiglu_gn | joint | swiglu | gn | 0.7071 | 0.7313 | 0.7757 | 0.7934 | 0.7948 | 0.8028 | **0.7453** |
| 5 | split_glu_gn | split | glu | gn | 0.7064 | 0.7343 | 0.7683 | 0.7900 | 0.7919 | 0.8097 | **0.7451** |
| 6 | split_glu_bn | split | glu | bn | 0.7098 | 0.7422 | 0.7630 | 0.7831 | 0.7973 | 0.8094 | **0.7448** |
| 7 | imu_glu_bn（IMU 结构偏置） | imu | glu | bn | 0.7161 | 0.7266 | 0.7560 | 0.7883 | 0.8047 | 0.8115 | **0.7437** |
| 8 | joint_glu_gn | joint | glu | gn | 0.7119 | 0.7309 | 0.7618 | 0.7901 | 0.7972 | 0.8033 | **0.7436** |
| 9 | joint_swiglu_bn | joint | swiglu | bn | 0.7080 | 0.7397 | 0.7610 | 0.7894 | 0.7985 | 0.8034 | **0.7433** |
| 10 | split_swiglu_bn | split | swiglu | bn | 0.7071 | 0.7486 | 0.7546 | 0.7871 | 0.7932 | 0.8036 | **0.7427** |

（9 个 conformer 变体极差 0.0045；与 DCNN 差 0.0243）

## 结论

1. **四个假设全部证伪**：
   - GLU→SwiGLU（门饱和假设）：0.7433 vs 基线 0.7472，**无效**
   - BN→GroupNorm（域偏移假设）：0.7436 vs 基线 0.7472，**无效**
   - joint→split 双模态独立注意力：各 split 变体 0.7427~0.7456 vs 0.7472，**无效**
   - IMU 结构偏置（局部性/同瞬跨模态物理先验，09-19）：**0.7437 vs 0.7472，无效**
2. 9 个 conformer 变体挤在 0.7427~0.7472 的 0.45 个点窄带内，互差在噪声水平
   （同种子下不同结构 ≈ 初始化级别波动）；**没有任何一个超过原版 joint_glu_bn**
3. **与 DCNN 的 2.4 个点差距是架构级的**：不是门控、不是归一化、不是注意力组织
   方式、也不是缺局部性先验能解释的——在此数据规模（combo1-6 约 0.9~4.4 万窗口）
   下，DCNN 的卷积先验（局部性/平移不变/92,873 参数）系统性优于 77 万参数的
   attention 混合架构
4. 与 loss 曲线互相印证：conformer 训练 loss 最低（0.10 vs DCNN 0.28）但泛化
   最差——训练 loss 与跨品种泛化倒挂

## IMU 结构偏置变体细节（imu_glu_bn，09-19）

- 与基线（joint_glu_bn）互有胜负：c1/c5/c6 略优（其中 c1=0.7161 为全部
  conformer 变体最高，单品种 ~9k 窗最小数据时先验增益最大），c2-c4 略劣，
  均值 -0.35 点落在噪声带内。**判定：物理先验本身不有害（斜率可学习起了
  保底作用），但无增益。**
- 学到的斜率（exp 后 = 每单位 patch 距离的 logit 惩罚，init：intra 0.1 / cross 0.3）：
  - intra 三层稳定在 0.09~0.13（≈init 未动，模型保留温和时序局部偏好）；
  - cross 从 k01 的 0.24~0.27 降到 k06 的 0.17~0.20（组合越大越放松跨时刻
    跨模态惩罚），全程低于 init。
- **解读：卷积模块已经提供局部性，attention 里的局部性偏置与之冗余**——先验
  注入 attention 的通路并不能替代 DCNN 那种"整个模型就是局部先验"的结构；
  唯一可见的正效应出现在最缺数据的单品种场景（c1），而那正是 DCNN 本来就
  最强的地方。假设四证伪方式与前三者一致：架构差距在数据量/归纳偏置权衡，
  不在某个可单点修补的组件。

## 后续：注意力调参轮（同协议前 6 组合，10-20）

结构证伪后转容量/正则化调参（`run_attn_tune.sh`，产出 `checkpoints/attn_tune/`）：
**Conformer-XS（d64ff128_l2，136,905 参数）0.7562**、Conformer-S（d64）0.7529、
Conformer-drop40 / Conformer-S-drop40 均 0.7345（dropout 有害）。容量单调有效
（0.77M→0.30M→0.14M 对应 0.7472→0.7529→0.7562），但仍差 DCNN 0.7715 有 1.53 点；
完整表见 README「注意力调参」小节与 `results/ablation_compare_table.md`。
26 组合全量版（IMU-Conformer / Conformer-XS）由 `run_full26.sh` 补跑，10-21 完成：
**Conformer-XS offdiag(1-21) 0.7749**（attention 系 26 组合口径最佳，与 DCNN
0.7847 差距收到 0.98 点，从前 6 组合的 1.53 点继续收窄——数据量增大时
attention 系追赶斜率更陡）；IMU-Conformer 0.7701（vs 原版 0.7696 噪声内），
但 combo26 全26 均值 **0.9399 全场最高**、NSDTR 0.7348 同量级最佳。

## 运行信息

- 命令矩阵见 `run_conformer_ablation.sh`；各 run 产出在 `checkpoints/ablation/<name>/`
- 结构消融 8 个 run 全部 seed 42（每 combo 固定 seed×1000+combo），`--epoch 100 --max 6`
- IMU 偏置 run：`python3 data_aug.py --model conformer --epoch 100 --seed 42
  --max 6 --attn imu --gate glu --conv-norm bn --out-dir checkpoints/ablation/imu_glu_bn`
- DCNN 基线为未固定种子的旧全量跑（`checkpoints/data_aug/`）前 6 组合
- 原版 conformer 基线 = `data_aug_conformer_v2`（固定种子全量 26 组合）前 6 组合
- 重生成本表：`python3 compare_ablation.py checkpoints/data_aug/
  checkpoints/data_aug_conformer_v2/ checkpoints/ablation/*/`
  （注意会覆盖本文件的结论文字，勿盲目重跑）
