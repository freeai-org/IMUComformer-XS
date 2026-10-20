# project-5：犬类 IMU 行为识别（仅项圈传感器）

## 项目口径

- 目标：仅使用**项圈（颈部/neck）传感器**的 IMU 数据做犬类行为识别，贴近真实宠物项圈产品的部署形态（背部/胸部传感器在实际项圈上不存在）。
- 数据源：本机 `/mnt/WYH/data/IMU-Dog`（下称 DS1）与 `/mnt/WYH/data/IMU-Dog-2`（下称 DS2）。
- 两个数据集狗群、标注体系、传感器配置都不同，**不做跨数据集混训**，分别独立处理。

## DS1：IMU-Dog（2.3GB）

来源：Vehkaoja et al., *Data in Brief* 2022（DOI: 10.1016/j.dib.2022.107822，坦佩雷大学；
Mendeley Data: 10.17632/vxhx934tbn.2）。**传感器：ActiGraph GT9X Link × 2**，
分别挂在背部胸带上与颈圈腹侧，各出 Acc+Gyr 六轴，100Hz。
45 只中大型犬（17 只测了两次 → 62 次测试），视频后标注（Noldus Observer XT）。
官方 MATLAB 代码（`Code2pub/`，SVM/Tree/DA 分类管线）。

### 文件

| 文件 | 内容 |
|---|---|
| `DogInfo.csv` | 45 只狗的信息：DogID, Breed, Weight, Age months, Gender(1母2公), NeuteringStatus(0未绝育1绝育) |
| `DogMoveData.csv` | 全量明细：**10,611,068 行 × 20 列**，100Hz |
| `DogMoveData.mat` | 同数据的 MATLAB table（63 行 = 45 狗 / 62 次测试，sensorData 为每测试一个 cell） |
| `Code2pub/*.m` | 官方特征提取与分类代码 |

### 列结构（20 列）

```
DogID, TestNum, t_sec,
ABack_x/y/z, ANeck_x/y/z,          # 加速度计: 背部、颈部
GBack_x/y/z, GNeck_x/y/z,          # 陀螺仪:   背部、颈部
Task, Behavior_1/2/3, PointEvent    # 标注列
```

- 传感器仅 **Acc + Gyr**（无磁力计），背部+颈部各 6 列。
- 注意 `Data_description.txt` 中 ANeck_x/y/z 三行的描述文字 x/y 互换，属官方笔误，以列名为准。
- 标注空值为 `<undefined>` 字符串；Behavior 允许最多 3 个同时标注。
- 45 只狗、62 次测试（TestNum ∈ {1,2}）。

### 类目

**Behavior（行为，19 类 + undefined）**：Lying chest (1,031,301)、Sniffing (1,026,178)、Playing (862,571)、Panting (836,062)、Walking (728,930)、Trotting (717,593)、Sitting (509,412)、Standing (448,691)、Eating (166,210)、Pacing (77,104)、Drinking (64,721)、Shaking (41,234)、Carrying object (17,951)、Synchronization (16,755)、Tugging (13,664)、Galloping (10,828)、Jumping (3,859)、Bowing (518)、Extra_Synchronization (287)；undefined 4,037,199。

**PointEvent**：短事件（如 Bark）单独标注。

### 项圈数据取用（DS1）

保留列：`DogID, TestNum, t_sec, ANeck_x, ANeck_y, ANeck_z, GNeck_x, GNeck_y, GNeck_z` + 标注列（Task / Behavior_1..3）。

## DS2：IMU-Dog-2（547MB）

来源：Marcato et al., *PLOS ONE* 2023（DOI: 10.1371/journal.pone.0286311，Tyndall
National Institute / UCC；Mendeley Data: 10.17632/mpph6bmn7g.1）。
**传感器：ActiGraph GT9X Link × 3**（背/胸/颈各一，与 DS1 同款），
各出 Acc+Gyr+Mag 九轴，100Hz；42 只健康实习犬（1 GR + 16 LR + 25 LRxGR）。

### 文件

| 文件 | 内容 |
|---|---|
| `df_raw.csv` | 全量明细：**1,675,002 行 × 32 列**，100Hz（时间戳步进 0.01s） |
| `df_dogs.csv` | 42 只狗：Subject, Breed, Sex(F/M), DOB |
| `readme.md` | 官方列说明 |

### 列结构（32 列）

```
Timestamp,
Back.{Acc,Gyr,Mag}.{X,Y,Z}     # 背部 9 列
Chest.{Acc,Gyr,Mag}.{X,Y,Z}    # 胸部 9 列
Neck.{Acc,Gyr,Mag}.{X,Y,Z}     # 颈部 9 列
Type, Position, Breed, Subject # 标注/分组
```

- 每位置 3 传感器（加速度计/陀螺仪/磁力计）× 3 轴 = 9 列，共 27 列惯性数据。
- 品种仅 3 种：LRxGR (1,020,201 行) / LR (583,201) / GR (71,600)。

### 类目

**Type（动静二分类）**：static 1,199,200 / dynamic 475,802。

**Position（姿态，5 类）**：

| Position | 行数 |
|---|---|
| standing | 640,700 |
| walking | 466,502 |
| sitting | 323,200 |
| lying down | 235,300 |
| body shake | 9,300 |

### 项圈数据取用（DS2）

保留列：`Timestamp, Neck.Acc.X/Y/Z, Neck.Gyr.X/Y/Z, Neck.Mag.X/Y/Z` + 标注列（Type, Position, Breed, Subject）。注意 DS2 项圈多一档磁力计（DS1 没有），对齐 DS1 口径时可只用 Acc+Gyr。

## 脚本与清洗产出

### 共享模块（2026-10 解耦重构）

训练/评测/绘图脚本原先各自持有 CSV 加载、训练循环、模型工厂、offdiag 口径、
绘图样板的拷贝，现统一收编为 6 个共享模块（对外 CLI 参数与默认值、checkpoint
格式、jsonl 字段**均不变**；模型类本体仍在各自文件里）。包结构：
**共享层在 `utils/` 包**（imdata/models/train_utils/protocol/cli），
**绘图脚本在 `plot/` 包**（含 `plotlib` 共享绘制单元，`python3 plot/plot_xxx.py`
可直接运行，脚本会自动把仓库根加入 `sys.path`）：

| 模块 | 职责 |
|---|---|
| `utils/imdata.py` | 数据层：CSV 加载/降频（`load_imu_csv`，带 hz 整数因子校验）、滑窗（`build_windows`）、8:2 划分、StandardScaler |
| `utils/models.py` | 模型工厂 `build_model(name, ..., cfg=checkpoint cfg, **hp)`：超参优先级 显式 > checkpoint cfg > 默认；修复旧版不回放 conformer 超参的形状错误 |
| `utils/train_utils.py` | 训练层：`set_seed`、统一训练循环 `train_loop`、`save_ckpt`、`jsonl_append` |
| `utils/protocol.py` | 评测口径：7 类协议过滤（`keep_subset`/`eval_windows_acc`）、组合外均值 `offdiag_*`、`load_jsonl`、26x26 矩阵统计 |
| `utils/cli.py` | CLI 公共参数定义 + `CONFORMER_DEFAULTS` 超参单源（`dcnn.py --epoch` 默认 30、其余 100 不变） |
| `plot/plotlib.py` | 绘图层：Agg 初始化、`short`/`accumulation_order`/`draw_heatmap` 绘制单元、`MODEL_RUNS` 模型注册表（曲线/热力图对比脚本共用） |

| 文件 | 作用 |
|---|---|
| `preview.py` | 查看两个数据集前 n 行：`python3 preview.py -n 10 [ds1/ds2]` |
| `clean_neck6.py` | 清洗提取项圈六轴：`python3 clean_neck6.py [ds1/ds2/all]` |
| `remap_ds1_labels.py` | DS1 标签规范化 19 类→9 大类 |
| `dcnn.py` | DCNN 训练脚本（训练 DS1 大类数据） |
| `breed_cv.py` | 按品种交叉验证：26 品种各训一个 DCNN，26×26 验证矩阵 |
| `eval_perclass.py` | 加载 checkpoint 复现测试集，输出分类别指标：`python3 eval_perclass.py [ckpt路径]`（默认取最新） |
| `data_aug.py` | 基于品种交叉矩阵的种子数据增广：按 7 类矩阵非对角均值排序品种，逐品种累加训练 26 个组合并评测：`python3 data_aug.py --epoch 100` |
| `plot/plot_dataaug_heatmap.py` | dataaug 结果可视化（组合×品种热力图 + 均值曲线）：`python3 plot/plot_dataaug_heatmap.py` |
| `noise_analysis.py` | 传感器白噪声分析（静止段差分法/Welch PSD/自相关四联图）：`python3 noise_analysis.py` |
| `denoise_breeds.py` | 品种数据 12Hz 零相位低通去噪，输出 `data/neck6/breeds_denoise/`：`python3 denoise_breeds.py [--fc 12]` |
| `compare_denoise.py` | 原始 vs 去噪增广实验对比（组合外均值/难点品种）：`python3 compare_denoise.py` |
| `transformer.py` | TransformerIMU 模型定义（acc/gyr 双模态 patch + 模态/位置嵌入 + full attention，~0.42M 参数） |
| `compare_transformer.py` | Transformer vs DCNN 增广实验对比：`python3 compare_transformer.py` |
| `conformer.py` | ConformerIMU 模型定义（macaron FFN + 自注意力 + 深度可分离卷积模块，~0.77M 参数），`data_aug.py --model conformer` |
| `plot/plot_dataaug_offdiag.py` | 纯跨品种泛化曲线（组合外均值）：`python3 plot/plot_dataaug_offdiag.py [dataaug_infer.jsonl]` |
| `plot/plot_combo_curves.py` | combo_test_acc 与 mean_acc 双曲线：`python3 plot/plot_combo_curves.py [dataaug_infer.jsonl]` |
| `compare_ablation.py` | 多 run 组合外均值对比表（机打表存 `results/ablation_compare_table.md`；手工结论在 `results/ablation_summary.md`，勿被覆盖） |
| `run_conformer_ablation.sh` | Conformer 结构消融 2×2×2 命令矩阵（joint_glu_bn 基线复用 v2，行已注释） |
| `run_attn_tune.sh` | 注意力调参轮：dropout/容量 4 变体，产出 `checkpoints/attn_tune/` |
| `run_full26.sh` | 26 组合全量补跑：IMU-Conformer + Conformer-XS |
| `plot/plot_ablation.py` | 结构消融对比图（组合外均值曲线 + 排序条形） |
| `plot/plot_loss_compare.py` | DCNN/Transformer/Conformer 三模型训练 loss 对比 |
| `plot/plot_model_compare_curves.py` | 多模型增广均值曲线对比图（8 模型规范命名，26 组合实线/前 6 组合虚线） |
| `plot/plot_model_compare_heatmaps.py` | 多模型增广热力图对比（带格内数值，共享列序，full26/partial6 两张，跑完全量自动升级） |
| `PATHS.md` | 本项目路径索引：数据集/训练代码/结果/模型（含全部消融 run） |

### 训练（dcnn.py）

- 采样率：**两个数据集实测均为 100Hz**（`--hz` 仍可统一下采样，须为 100 的整数因子 50/25/20/10）；
  `--window/--stride` 是统一 Hz 下的采样点数。
- 数据：DS1（数据量大的那个）`ds1_neck6_remap.csv`，标签 `Behavior_1`（9 大类，单标签口径），
  特征项圈六轴；滑窗按 (DogID, TestNum) 分段不跨狗，窗口标签取众数；不做过采样/下采样。
- 模型：仅 DCNN（Conv1d×3 + GAP + FC，92,873 参数），训练前控制台打印参数量。
- 日志：`results/IMU_DCNN.jsonl`（首行 config，其后每 epoch 一条）；
  checkpoint 存 `checkpoints/IMU_DCNN_<时间戳>.pth`（含 scaler 与类别表）。

```bash
python3 /mnt/WYH/IMUComformer-XS/dcnn.py --epoch 30 --window 100 --stride 50
# 例: 统一降频 50Hz, 1 秒窗口
python3 /mnt/WYH/IMUComformer-XS/dcnn.py --epoch 20 --window 50 --stride 25 --hz 50
```

### 主实验结果（dcnn.py，DS1 九大类，09-11 ~ 09-12）

实际配置 window=50 / stride=25 / batch=256 / lr=1e-3（100Hz，0.5s 窗），
274,534 窗 = 训练 219,627 + 测试 54,907；4 次运行记录在 `results/IMU_DCNN.jsonl`：

| run | epochs 计划 | 实际完成 | 最佳 test_acc |
|---|---|---|---|
| 1 | 20 | 20 | 0.796 |
| 2 | 200 | 52（中断） | 0.844 |
| 3 | 500 | 63（中断） | 0.859 |
| 4 | 500 | 500 | **0.881** |

**run 4（完整训练）评测效果**：最佳 test_acc **0.8815**（epoch 478，train_acc 0.904 /
train_loss 0.263）；末 epoch 500 test_acc 0.8761（train_acc 0.905）。后期
（epoch 324~483）test_acc 稳定在 0.879~0.881 平台，训练-测试差距仅 ~2.3 个点，
过拟合轻微、训练充分收敛。checkpoint 存 best 权重：
`checkpoints/IMU_DCNN_20260912_165821.pth`。

**分类别指标**（`eval_perclass.py` 复现测试集，accuracy 0.8815 与训练时一致；
macro F1 0.865 / weighted F1 0.881）：

| 类别 | support | P | R | F1 |
|---|---|---|---|---|
| Sniffing | 8,195 | 0.972 | 0.972 | **0.972** |
| Running | 5,900 | 0.953 | 0.948 | 0.950 |
| Shaking | 328 | 0.884 | 0.973 | 0.926 |
| Walking | 6,118 | 0.899 | 0.926 | 0.913 |
| Playing | 7,531 | 0.933 | 0.871 | 0.901 |
| Lying | 8,846 | 0.882 | 0.822 | 0.851 |
| Sitting | 8,571 | 0.818 | 0.893 | 0.854 |
| Standing | 7,562 | 0.785 | 0.845 | 0.814 |
| Feeding | 1,856 | 0.697 | 0.533 | **0.604** |

混淆集中在：Feeding 被大量误判为 Standing（304 窗）/ Sitting（226 窗）；
姿态静息三元组 Lying/Sitting/Standing 互相混淆（如真值 Sitting→Lying 245、
Standing→Sitting 623）；动态类（Running/Walking/Sniffing/Shaking）基本干净。
与 breed_cv 结论一致：姿态静息类是主要混淆源。

### 品种交叉验证（breed_cv.py，验证 DS1 数据质量）

- **split 已完成**：`/mnt/WYH/data/neck6/breeds/` 下 26 个品种 CSV（`01_Crossbreed.csv`
  67 万行最大 … `26_Australian_Kelpie.csv` 8.6 万行最小）+ `breeds_manifest.json`。
- **train**：每个品种独立训一个 DCNN（各自的类别集，小品种类数可能 <9），100 epoch，
  checkpoint 存 `checkpoints/breeds/<breed>.pth`（含 scaler/类别表），
  训练日志追加 `results/IMU_DCNN_breeds.jsonl`（config + 每 epoch，带 breed 字段）。
- **eval**：26 模型 × 26 品种 → 26×26 准确率矩阵，存 `results/breed_cv_matrix.csv`；
  对角线 = 该品种自己的 test split（与训练同种子划分），非对角 = 对方品种全量数据
  （用户口径：其他品种的测试集即其训练集本身）；模型类别集不含的标签判为错。
  最后打印对角/非对角均值。

```bash
cd /mnt/WYH/IMUComformer-XS
python3 breed_cv.py train --epoch 100     # 26 个模型一起训
python3 breed_cv.py eval                  # 交叉验证出 26x26 矩阵
# 排除指定类别后重新评测(过滤真值窗口 + 模型输出屏蔽):
python3 breed_cv.py eval --drop Feeding,Standing
python3 plot/plot_heatmap.py                   # 默认矩阵热力图
python3 plot/plot_heatmap.py results/breed_cv_matrix_drop_Feeding_Standing.csv
# 或一条龙: python3 breed_cv.py all --epoch 100
```

### 品种交叉验证结果（26×26 矩阵）

- **全 9 类**：对角均值 0.875 / 非对角均值 0.538
- **剔除 Feeding/Standing 后**（静态/姿态类，与动态行为区分度低）：
  对角均值 **0.947** / 非对角均值 **0.657**——跨品种准确率提升 ~12 个点
- **进一步剔除 Shaking/Sniffing（剩 5 类 Lying/Sitting/Walking/Running/Playing）**：
  对角均值 **0.938** / 非对角均值 **0.627**——跨品种反而比 7 类版**略低**

剔除 Feeding/Standing 后跨品种泛化明显改善，反过来说明这两类是模型主要的
混淆来源：它们在不同品种身上“长得差不多”，但其他动态行为（Lying/Running/
Playing/Sniffing/Shaking/Walking/Sitting）反而在品种间有可迁移的模式。
进一步再剔除 Shaking/Sniffing 时非对角均值反而微跌，说明这两类本身在品种
间携带了可用的判别信号，盲删反而损失信息——类目集合的选择有 trade-off，
不是越少越好。

### 种子数据增广（data_aug.py，09-14）

思路：用 7 类矩阵（`breed_cv_matrix_drop_Feeding_Standing.csv`）找"种子品种"——
模型在狗 A 训练却在其他狗表现好的数据。第一步按每个品种模型的非对角均值排序
（Border Collie 0.7523 最高 … Belgian Shepherd 0.5348 最低）；第二步按排序逐品种
累加训练 26 个组合（{A}, {A,B}, …），每个组合在全部 26 品种上评测求均值
（口径与 breed_cv 一致：组合内品种用 test split，组合外用全量；7 类协议）。

结果（26 组合 × 100 epoch，GPU 27 分钟跑完）：

- 均值随组合增大**单调上升**：combo 1（仅 Border Collie）0.7455 → combo 26（全 26 品种）**0.9308**
- **最佳组合 = 全量数据（combo 26）**：增广收益递减但始终为正，没有出现"加某品种反而拖累"的拐点
- 注意口径不对称：组合越大，越多品种按 test split（较易）评测而非全量（较难），
  0.9308 与 breed_cv 对角均值 0.947（7 类）同量级属预期；跨品种泛化要看组合外品种
  （如 combo 1 的 Border Collie 单品种模型在其他 25 品种均值即 0.7455 - 自身贡献）

**纯跨品种泛化（`plot/plot_dataaug_offdiag.py`，去掉组合内品种只看组合外均值）**
——修正上面的全量口径结论：**~6 个品种就饱和**（combo 1→6：0.738→0.794），
之后在 0.77~0.81 平台震荡，可靠区间最佳 **combo 15（前 15 品种）0.8086**，
与 combo 6/10/13 仅差 ~1 个点；"最佳=全量"系 test split 口径虚高。
**Nova Scotia Duck Tolling Retriever 在全部 25 个组合下都是组合外最差**
（0.478~0.692），是最难泛化品种。若目标是泛化到新品种，前 6 个种子品种收益
最大（种子排序有效），之后应转向架构/域自适应而非继续堆品种。

产出（全部在 `checkpoints/data_aug/`）：训练日志 `IMU_DCNN_dataaug.jsonl`
（26 config + 2600 epoch 记录，combo 字段区分）、26 个组合 checkpoint
`k<NN>_<breed>.pth`、评测 `dataaug_infer.jsonl` / `dataaug_infer.csv`
（26×26 组合×品种矩阵 + 均值列）、结论 `dataaug_summary.md`、
可视化 `dataaug_heatmap.png`（组合×品种热力图，蓝框=行最差）与
`dataaug_mean_curve.png`（均值曲线）。热力图显示 Mudi 列全程偏橙
（所有组合下都最难），各行最差多为 Mudi / 未入组合的 Spanish Water Dog /
Nova Scotia Duck Tolling Retriever。

### 传感器白噪声分析与去噪对照（09-14，第三步）

**白噪声分析**（`noise_analysis.py`，四联图 `results/noise_analysis.png`）：
取静止段（Lying，307 段共 184 分钟）估计六轴噪声。**结论：静止段噪声不是白噪声**
——PSD 高频无平顶（1/f 型微动主导，非白噪声底）、自相关慢衰减 ~25-40 采样点
（白噪声应只有 lag0 尖峰）、归一化差分分布重尾（爆发式微动混合）。差分法
（σ=std(diff)/√2）给出的噪声上界：acc σ≈0.013~0.018g、gyr σ≈1.7~2.8°/s，
SNR≈30dB（信号主导）。

**去噪对照**（`denoise_breeds.py` 12Hz 零相位低通 → `data_aug.py --breed-dir
breeds_denoise` 同协议重训 26 组合 × 100 epoch，产出 `checkpoints/data_aug_denoise/`）：

| 指标 | 原始 | 去噪 | Δ |
|---|---|---|---|
| 纯跨品种泛化（组合外均值 1-21） | 0.7847 | 0.7868 | +0.0021（噪声内） |
| 全 26 均值（26 组合平均） | 0.8602 | 0.8600 | 持平 |
| 最难品种 NSDTR 组合外均值 | 0.5803 | 0.6186 | **+0.0383** |

**结论：去噪不提升跨品种泛化**（假设基本否定；SNR 30dB 下噪声非瓶颈，
DCNN 的 conv+pool 本身即低通）；唯一受益者是难点品种 NSDTR（+3.8 点，信号
最弱处低通才见效）。跨品种泛化的真正瓶颈是品种域差，优化方向应在架构
（双分支/attention/域自适应）而非预处理。

### Transformer 对照（09-15，双模态 full attention vs DCNN）

`transformer.py` TransformerIMU：acc/gyr 各自 patch embed（10 token/模态）+
模态嵌入 + 位置嵌入 + full self-attention（3 层 Pre-LN，~0.42M 参数），
`data_aug.py --model transformer` 同协议重训（26 组合 × 100 epoch，产出
`checkpoints/data_aug_transformer/`）：

| 指标 | Transformer | DCNN | Δ |
|---|---|---|---|
| 纯跨品种泛化（组合外均值 1-21） | 0.7391 | 0.7847 | **-0.0456** |
| 全 26 均值（26 组合平均） | 0.8296 | 0.8602 | -0.0306 |
| 最难品种 NSDTR（组合外） | 0.6654 | 0.5913 | **+0.0741** |

**结论：full attention 关联 acc/gyr 未提升跨品种泛化**（26 组合 25 个落后）。
原因：单品种数据仅 9k~100k 窗，0.42M 参数 Transformer 不敌 93k 参数 DCNN 的
conv 归纳偏置；跨轴关联 6 通道 conv1 即可学完，attention 长程优势在小数据上
发挥不出。亮点仍是 NSDTR（+7.4 点，与去噪实验同向）。跨品种瓶颈在数据量/
域差，不在输入表示——Transformer 要超越 DCNN 需全量数据单次训练 + 超参调优
或 Conformer 混合架构。

### Transformer v2（旋转不变特征 + 逐轴 token，09-15，已终止）

`TransformerIMU2`：输入增广 |a|/|ω|/|jerk|（项圈旋转不变）+ 逐轴 patch token
（9 轴 × 10 patch = 90 token，轴/模态/位置嵌入）。跑完前 3 个组合即被用户叫停：
全 26 均值 0.709/0.745/0.757，**劣于 v1 Transformer 和 DCNN**。教训：逐轴
token 拆散了三轴联合形态（重力分解 = 头部姿态需经 attention 重建），不变量
特征应作为追加通道进联合 patch embed 而非推翻 token 结构。

### Conformer 对照（conformer.py，09-15 训练，09-16 分析）

`ConformerIMU`（~0.77M 参数，macaron FFN + 自注意力 + 深度可分离卷积模块，
seed 42001，产出 `checkpoints/data_aug_conformer_v2/`）。**架构对比总表**
（全部同协议 26 组合 × 100 epoch，7 类评测）：

| 模型 | 参数 | 纯跨品种(1-21) | 全26均值(26组合平均) | combo26 全26 | NSDTR 组合外(1-21) |
|---|---|---|---|---|---|
| DCNN | 93k | **0.7847** | **0.8602** | 0.9308 | 0.5803 |
| DCNN + 12Hz 去噪 | 93k | 0.7868 | 0.8600 | 0.9339 | 0.6186 |
| Transformer v1 | 0.42M | 0.7391 | 0.8296 | 0.9126 | 0.6543* |
| Conformer | 0.77M | 0.7696 | 0.8538 | 0.9370 | 0.7284 |
| **IMU-Conformer**（attn bias） | 0.77M | 0.7701 | 0.8535 | **0.9399** | 0.7348 |
| **Conformer-XS**（d64ff128+l2） | 136,905 | **0.7749** | 0.8551 | 0.9325 | **0.7389** |
| Transformer v2 | 0.25M | （3 组合即终止） | — | — | — |

*Transformer NSDTR 老表记 0.6654，实为 1-24 口径误标 1-21，已修正为 1-21 口径
的 0.6543。IMU-Conformer / Conformer-XS 两行为 10-20 `run_full26.sh` 补跑。

**跨品种 NSDTR 阶梯**（组合外，难例品种）：DCNN 0.5803 < Transformer 0.6543 <
Conformer 0.7284 < IMU-Conformer 0.7348 < Conformer-XS 0.7389——表示越强，
难例越受益，与去噪 +3.8 / Transformer +7.4 / Conformer +14.8 的单调放大一致。

结论：Conformer 显著好于 Transformer v1（跨品种 +3.1 点），但仍不及 DCNN
（-1.5 点）；combo 26 的 0.9370 超过 DCNN 的 0.9308 属 test split 口径虚高
部分。**NSDTR 上 Conformer 大幅领先（+14.8 点 vs DCNN）**——混合架构对
难例品种有效。`curves_combo_test_vs_mean.png` 显示其 combo_test_acc 全程
平在 88~89%（组合内早饱和），mean_acc 收益全部来自组合外品种。

### Conformer 结构消融与注意力调参（09-19 ~ 10-20）

**结构消融 2×2×2 + IMU 偏置**（`run_conformer_ablation.sh`，seed 42，前 6 组合，
结论详见 `results/ablation_summary.md`）：attn(joint/split) × gate(glu/swiglu)
× conv-norm(bn/gn) 共 8 变体全落在 0.7427~0.7472 的 0.45 点噪声带内，没有一个
超过原版 joint_glu_bn（0.7472）。`conformer.py` 新增 `--attn imu`（IMU 结构偏置：
同模态注意力随时间距离线性衰减、跨模态仅同一时刻免罚、斜率可学习，+6 标量参数）
得 0.7437——**第四个假设同样证伪**。四轮结构假设（门控/归一化/注意力组织/
物理先验偏置）全部证伪，与 DCNN 的差距是架构级的。

**注意力调参轮**（`run_attn_tune.sh`；`data_aug.py` 新增 `--d-model/--nhead/
--layers/--dim-ff/--patch/--kernel/--dropout/--weight-decay` 旋钮并写入 config；
同协议 seed 42 前 6 组合）：

| 模型（规范命名） | run 目录 | 参数量 | 组合外均值(1-6) |
|---|---|---|---|
| DCNN | `data_aug/` | 93k | **0.7715** |
| **Conformer-XS** | `attn_tune/d64ff128_l2/` | 136,905 | **0.7562** ⭐ |
| Conformer-S | `attn_tune/d64/` | 300,041 | 0.7529 |
| Conformer（原版） | `data_aug_conformer_v2/` 前6 | 770,569 | 0.7472 |
| Conformer-S-drop40 | `attn_tune/d64_drop40/` | 300,041 | 0.7345 |
| Conformer-drop40 | `attn_tune/drop40/` | 770,569 | 0.7345 |

结论：① **dropout 0.4 明确有害**（-1.3 点，掉出噪声带，正则化不是答案）；
② **容量单调有效**：0.77M→0.30M→0.14M 对应 0.7472→0.7529→0.7562，
Conformer-XS（136,905 参数，容量最接近 DCNN 93k）为 attention 系历史最佳，
与 DCNN 差距收到 1.53 点——印证**容量是主矛盾**；③ 结构改造四连败、缩容一次
见效，后续只沿容量↓ / 数据量↑ 两条路走。

**26 组合全量补跑**（`run_full26.sh`，10-20 启动，10-21 完成）：`checkpoints/data_aug_imu/`
（IMU-Conformer，attn bias）与 `checkpoints/data_aug_d64ff128_l2/`（Conformer-XS）。
结果（组合外均值 1-21，与三个既有 26 组合 run 同口径）：

- **Conformer-XS 0.7749**：attention 系 26 组合口径最佳，与 DCNN 0.7847 的差距
  从前 6 组合的 1.53 点收到 **0.98 点**——数据量增大时 attention 系追赶斜率更陡，
  支持后续"全量单次训练 + leave-one-breed-out"路线；
- **IMU-Conformer 0.7701**（vs 原版 Conformer 0.7696，+0.05 噪声内）：偏置在
  大数据下对均值无增益，但 **combo26 全26 均值 0.9399 全场最高**、NSDTR 0.7348
  也是同量级最佳——物理先验对难例仍有效，与前 6 组合时的 c1 现象一致；
- 多模型对比图已刷新：`model_compare_curves.png`（8 模型，两条新线到 k26）+
  `model_compare_heatmaps_full26.png`（5 面板带数值）。

**多模型对比图**（10-20，`results/`）：`model_compare_curves.png`（8 模型增广
均值曲线）+ `model_compare_heatmaps_full26.png` / `_partial6.png`（共享列序、
带格内数值的热力图）。模型规范命名全项目统一：DCNN / Transformer / Conformer /
IMU-Conformer / Conformer-S / Conformer-XS / Conformer-drop40 /
Conformer-S-drop40。

### 增广损伤品种 Bad list（09-16）

按"加入该品种后组合外均值的增量"（蓝线逐点 Δ）从 `dataaug_offdiag.csv` 计算：

| 损伤品种 | 加入序 | 种子排名 | Δ组合外 | 判定 |
|---|---|---|---|---|
| Belgian Shepherd Malinois | combo 16 | 16 | -2.40 点 | 明确损伤 |
| Bouvier | combo 21 | 21 | -1.96 点 | 明确损伤 |
| English Springer Spaniel | combo 7 | **7** | -1.79 点 | 明确损伤 |
| Australian Kelpie / Beauceron / Vizsla / BSG | — | — | -0.5~-0.9 | 噪声级 |

正收益对照：Crossbreed +2.8、Labrador +1.4、Dutch Shepherd +1.2、Finnish
Lapphund +1.2。要点：①灰线（全 26 均值）在这些组合从不下跌——Bad list 品种
被模型记住（推高自身分数）却污染跨品种决策边界，支持"该类品种单独训模型"
的部署策略；②种子排名不能预测增量损伤（ESS 排第 7 却在损伤榜第 3）——
"其模型泛化到别人"与"其数据混入别人训练"是两个不对称的方向；③±0.5 点内
为单种子训练方差，Bad list 只取 ≥1.5 点者。

清洗产出（`/mnt/WYH/data/neck6/`，流式分块写出，行数与原始完全一致，CSV 格式）：

- **`ds1_neck6.csv`**（930MB）：10,611,068 行 × 14 列
  `DogID, TestNum, t_sec, ANeck_x/y/z, GNeck_x/y/z, Task, Behavior_1/2/3, PointEvent`
  `<undefined>` 统一替换为空字符串；PointEvent 实际只有 Bark（26,173 行）。
- **`ds2_neck6.csv`**（192MB）：1,675,002 行 × 11 列
  `Timestamp, Neck.Acc.X/Y/Z, Neck.Gyr.X/Y/Z, Type, Position, Breed, Subject`
  丢弃了 Neck.Mag（对齐 DS1 六轴口径）。

- **`ds1_neck6_remap.csv`**（622MB）：6,865,578 行 × 14 列，DS1 行为标签规范化版
  （`remap_ds1_labels.py` 产出）。原 19 类行为 → **9 大类**（参考 DS2 姿态类粒度）：

  | 大类 | 原类别 | 行数(含多标签) |
  |---|---|---|
  | Lying | Lying chest | 1,129,902 |
  | Sitting | Sitting | 1,120,116 |
  | Standing | Standing | 1,027,710 |
  | Walking | Walking | 783,035 |
  | Running | Trotting + Galloping | 948,805 |
  | Shaking | Shaking（对齐 DS2 body shake） | 42,179 |
  | Feeding | Eating + Drinking | 377,527 |
  | Sniffing | Sniffing | 1,026,178 |
  | Playing | Playing/Tugging/Carrying object/Jumping/Bowing | 1,423,603 |

  删除类别：Pacing、Synchronization、Extra_Synchronization、Panting、未定义/空；
  三槽全空的行整行删除（10,611,068 → 6,865,578，保留率 64.7%）；槽位左移压缩。
  Task/PointEvent/六轴原样保留。

读取方式：`pd.read_csv('/mnt/WYH/data/neck6/ds1_neck6.csv')`（DS1 空标签可用
`keep_default_na=False` 或自行处理空串；数值列经 float 往返，可能出现
`-0.6606449999999999` 这类浮点表示，精度无损仅显示差异）。

## 提取命令（项圈子集）

```bash
# DS1（列号：1 DogID,2 TestNum,3 t_sec, 7-9 ANeck, 13-15 GNeck, 16 Task, 17-19 Behavior）
awk -F, 'NR==1 || 1' FS=, OFS=, \
  /mnt/WYH/data/IMU-Dog/DogMoveData.csv \
  | cut -d, -f1,2,3,7,8,9,13,14,15,16,17,18,19 \
  > /mnt/WYH/IMUComformer-XS/ds1_neck.csv

# DS2（列号：1 Timestamp, 20-28 Neck 9列, 29 Type, 30 Position, 31 Breed, 32 Subject）
cut -d, -f1,20,21,22,23,24,25,26,27,28,29,30,31,32 \
  /mnt/WYH/data/IMU-Dog-2/df_raw.csv \
  > /mnt/WYH/IMUComformer-XS/ds2_neck.csv
```

或 pandas：`pd.read_csv(..., usecols=[...])`（DS1 的 `<undefined>` 需 `na_values=['<undefined>']` 或保留原样）。

## 后续方向（待定）

- [ ] DS1 用 Behavior 做多标签（3 槽）还是取 Behavior_1 单标签；类别不均衡明显（Bowing 518 行 vs Lying chest 1M 行）。
- [ ] DS2 的 body shake 仅 9,300 行（0.6%），5 类姿态分类需处理不均衡。
- [ ] 滑窗切片参数（窗口长度/重叠率）、按 DogID/Subject 划分训练测试集防止个体泄漏。
