# project-5 路径索引

犬类项圈 IMU 行为识别（仅颈部/neck 传感器六轴）。本文只做**路径地图**：数据集、
训练代码、结果、模型（含全部消融 run）各自存在哪，完整实验口径与结论见
`README.md`。

> 关于 Audio：当前项目**实际完成的工作全部基于 IMU 数据**，没有接入任何音频
> 传感器信号。`/mnt/WYH/data/Audio数据集.pdf` 是候选数据集资料（尚未使用）；
> DS1 原始标注里有 Bark 点事件（`PointEvent` 列），但只是标签，不伴随音频波形。

统一实验协议速览（消融系实验共用）：DS1 项圈六轴，9 大类训练 / 7 类评测
（剔除 Feeding/Standing），26 个种子品种逐个累加成 26 个组合（combo），每个
组合训一个模型并在全部 26 品种上评测；**主指标 = 组合外均值（offdiag，模型没
见过的品种上的准确率）**。

---

## 1. 数据集路径

### 原始数据（只读，勿改）

| 路径 | 说明 |
|---|---|
| `/mnt/WYH/data/IMU-Dog/` | **DS1**（2.3GB，主数据集）：`DogMoveData.csv`（1061 万行×20 列，100Hz）、`DogMoveData.mat`、`DogInfo.csv`（45 犬信息）、`Code2pub/` 官方 MATLAB 分类代码 |
| `/mnt/WYH/data/IMU-Dog-2/` | **DS2**（547MB）：`df_raw.csv`（167.5 万行×32 列，100Hz）、`df_dogs.csv`（42 犬）、`readme.md` |
| `/mnt/WYH/data/Audio数据集.pdf` | 候选音频数据集资料，**未使用** |

### 清洗产出（训练实际读的数据，`/mnt/WYH/data/neck6/`）

| 文件/目录 | 内容 | 由哪个脚本产出 |
|---|---|---|
| `ds1_neck6.csv`（930MB） | DS1 项圈六轴全量（10,611,068 行×14 列） | `clean_neck6.py` |
| `ds1_neck6_remap.csv`（622MB） | **主训练数据**：DS1 标签规范化 19 类→9 大类（6,865,578 行） | `remap_ds1_labels.py` |
| `ds2_neck6.csv`（192MB） | DS2 项圈六轴（弃磁力计） | `clean_neck6.py` |
| `breeds/` | 26 个品种单文件 CSV + `breeds_manifest.json`（breed_cv / data_aug 的 split） | `breed_cv.py` split |
| `breeds_denoise/` | 同上 26 品种 CSV 的 12Hz 零相位低通去噪版 | `denoise_breeds.py` |

---

## 2. 训练代码路径（`/mnt/WYH/IMUComformer-XS/`）

### 共享模块（2026-10 解耦重构，CLI/checkpoint/jsonl 格式均不变）

| 文件 | 职责 |
|---|---|
| `utils/imdata.py` | 数据层：CSV 加载/降频、滑窗、8:2 划分、标准化（全部训练/评测脚本的唯一数据管线） |
| `utils/models.py` | 模型工厂 `build_model`（超参优先级 显式 > checkpoint cfg > 默认；模型类本体在 dcnn/transformer/conformer.py） |
| `utils/train_utils.py` | 训练层：`train_loop` 统一训练循环、`save_ckpt`、`set_seed`、`jsonl_append` |
| `utils/protocol.py` | 评测口径：7 类协议（剔除 Feeding/Standing）、组合外均值 `offdiag_*`、`load_jsonl` |
| `utils/cli.py` | 公共 CLI 参数定义 + `CONFORMER_DEFAULTS` 超参单源 |
| `plot/plotlib.py` | 绘图层：Agg 初始化、`draw_heatmap`/`short`/`accumulation_order`、`MODEL_RUNS` 注册表 |

### 数据预处理

| 文件 | 作用 |
|---|---|
| `preview.py` | 查看两数据集前 n 行 |
| `clean_neck6.py` | 提取项圈六轴 → `/mnt/WYH/data/neck6/` |
| `remap_ds1_labels.py` | DS1 标签 19 类→9 大类 |
| `denoise_breeds.py` | 品种数据 12Hz 低通去噪 → `data/neck6/breeds_denoise/` |

### 模型定义

| 文件 | 模型 |
|---|---|
| `dcnn.py` | DCNN（Conv1d×3 + GAP + FC，92,873 参数），含训练入口 |
| `transformer.py` | TransformerIMU v1（acc/gyr 双模态 patch + full attention，~0.42M 参数）；v2（逐轴 token 版）已终止 |
| `conformer.py` | ConformerIMU（macaron FFN + 自注意力 + 深度可分离卷积，~0.77M 参数），支持 `--attn joint/split/imu`、`--gate glu/swiglu`、`--conv-norm bn/gn` 消融开关 |

### 训练入口

| 文件 | 作用 | 典型命令 |
|---|---|---|
| `dcnn.py` | DS1 九大类主实验 | `python3 dcnn.py --epoch 500 --window 50 --stride 25` |
| `breed_cv.py` | 26 品种各训一个 DCNN + 26×26 交叉评测 | `python3 breed_cv.py all --epoch 100` |
| `data_aug.py` | 种子增广统一入口：`--model dcnn/transformer/conformer`、`--breed-dir`、`--seed`、`--max N`（只跑前 N 个组合）、`--out-dir` | `python3 data_aug.py --model conformer --epoch 100 --seed 42 --max 6 --out-dir checkpoints/ablation/joint_glu_gn --attn joint --gate glu --conv-norm gn` |
| `run_conformer_ablation.sh` | Conformer 结构消融 2×2×2 命令矩阵（joint_glu_bn 基线复用 `data_aug_conformer_v2`，该行已注释） | `bash run_conformer_ablation.sh` |
| `run_attn_tune.sh` | 注意力调参轮：dropout/容量 4 变体，产出 `checkpoints/attn_tune/` | `bash run_attn_tune.sh` |
| `run_full26.sh` | 26 组合全量补跑：IMU-Conformer + Conformer-XS | `nohup bash run_full26.sh > checkpoints/full26_run.log 2>&1 &` |

### 评测与对比

| 文件 | 作用 |
|---|---|
| `eval_perclass.py` | 加载 checkpoint 复现测试集，输出分类别 P/R/F1（`python3 eval_perclass.py <ckpt路径>`） |
| `compare_denoise.py` | 原始 vs 去噪对照 |
| `compare_transformer.py` | Transformer vs DCNN 对照 |
| `compare_ablation.py` | 汇总多个 run 的 `dataaug_infer.jsonl`，输出组合外均值对比表（不带参数默认扫 `checkpoints/ablation/`） |
| `noise_analysis.py` | 静止段白噪声分析（差分法/Welch PSD/自相关） |

### 可视化

| 文件 | 作用 |
|---|---|
| `plot/plot_heatmap.py` | 品种交叉矩阵热力图 |
| `plot/plot_confusion_matrix.py` | 三模型混淆矩阵（k26 checkpoint，26 品种 test split） |
| `plot/plot_dataaug_heatmap.py` | 增广组合×品种热力图 + 均值曲线 |
| `plot/plot_dataaug_offdiag.py` | 纯跨品种泛化曲线（组合外均值） |
| `plot/plot_combo_curves.py` | combo_test_acc 与 mean_acc 双曲线 |
| `plot/plot_ablation.py` | Conformer 消融图（曲线 + 排序条形，基线自动取 `data_aug/` 与 `data_aug_conformer_v2/`） |
| `plot/plot_loss_compare.py` | DCNN/Transformer/Conformer 三模型训练 loss 对比 |
| `plot/plot_model_compare_curves.py` | 多模型增广均值曲线对比（8 模型规范命名，26 组合实线/前 6 组合虚线） |
| `plot/plot_model_compare_heatmaps.py` | 多模型增广热力图对比（带格内数值，按组合数自动分 full26/partial6 两图） |
| `plot/plot_imu_dcnn_loss.py` | 主实验 loss/acc 曲线 |

---

## 3. 结果路径

### `results/`（项目级结果）

| 文件 | 内容 |
|---|---|
| `IMU_DCNN.jsonl` | 主实验日志（4 次运行，最终 best test_acc 0.8815） |
| `IMU_DCNN_loss.png` | 主实验 loss 曲线 |
| `IMU_DCNN_breeds.jsonl` | 26 品种训练日志（breed_cv） |
| `breed_cv_matrix.csv` / `.png` | 26×26 交叉验证矩阵，全 9 类 |
| `breed_cv_matrix_drop_Feeding_Standing.csv` / `.png` | 7 类版（后续实验的依据） |
| `breed_cv_matrix_drop_Feeding_Shaking_Sniffing_Standing.csv` / `.png` | 5 类版 |
| `confusion_metrics.md` | **三模型 per-class TP/FP/FN 指标总表**（DCNN/Transformer/Conformer，k26） |
| `confusion_matrix_DCNN.png` / `confusion_matrix_Transformer.png` / `confusion_matrix_Conformer.png` / `confusion_tpfpfn.png` | 三模型混淆矩阵图 |
| `dataaug_loss_compare.png` | 三模型 loss 对比图 |
| `ablation_summary.md` | **Conformer 结构消融对比表（8 变体 + DCNN 基线）与结论**（手工维护） |
| `ablation_compare_table.md` | `compare_ablation.py` 自动保存的机打对比表（含调参 run） |
| `model_compare_curves.png` | 多模型增广均值曲线对比（8 模型） |
| `model_compare_heatmaps_full26.png` / `_partial6.png` | 多模型增广热力图对比（带数值，共享列序） |
| `noise_analysis.json` / `.png` | 白噪声分析结果与四联图 |
| `README.md` | results/ 内文件说明 |

### 各 run 目录内的通用产出（`checkpoints/<run>/`）

每个增广类 run（`data_aug*`、`ablation/*`）目录内结构一致：

| 文件 | 内容 |
|---|---|
| `IMU_DCNN_dataaug.jsonl` | 训练日志（每 combo 一段 config + 每 epoch 一条） |
| `k<NN>_<Breed>.pth` | 各组合 checkpoint（k01…kNN，含 scaler 与类别表） |
| `dataaug_infer.jsonl` / `.csv` | 组合×品种评测矩阵 + 均值列（`compare_ablation.py` 的输入） |
| `dataaug_offdiag.csv` / `dataaug_offdiag_curve.png` | 组合外均值（纯跨品种泛化） |
| `dataaug_heatmap.png` / `dataaug_mean_curve.png` | 热力图与均值曲线 |
| `dataaug_summary.md` | 该 run 的自动结论文档 |
| `train.log`（部分 run） | 控制台日志；run 级日志另见 `checkpoints/<run>.log` |

---

## 4. 模型路径（`/mnt/WYH/IMUComformer-XS/checkpoints/`）

### 主实验（DS1 九大类）

| 路径 | 说明 |
|---|---|
| `checkpoints/IMU_DCNN_20260912_165821.pth` | **主实验 best 权重**（500 epoch，test_acc 0.8815，macro F1 0.865） |

### 品种交叉验证（breed_cv）

| 路径 | 说明 |
|---|---|
| `checkpoints/breeds/<breed>.pth` | 26 个单品种 DCNN（各含 scaler/类别表） |

### 架构消融系列（同协议 26 组合 × 100 epoch）

| 路径 | 模型 | 关键参数 | 备注 |
|---|---|---|---|
| `checkpoints/data_aug/` | **DCNN 基线** | 93k | 未固定种子；消融对比中的 DCNN 参照行；k26 全 26 均值 0.9308 |
| `checkpoints/data_aug_denoise/` | DCNN + 12Hz 去噪 | 93k | 输入为 `data/neck6/breeds_denoise/` |
| `checkpoints/data_aug_transformer/` | Transformer v1 | 0.42M | 另含 `transformer_compare.csv` |
| `checkpoints/data_aug_conformer/` | Conformer（初版） | 0.77M | 未固定种子的第一次全量跑 |
| `checkpoints/data_aug_conformer_v2/` | **Conformer（固定 seed 42001）** | 0.77M | README 口径的正式基线；结构消融中 `joint_glu_bn` 变体即取本 run 前 6 组合，未单独建目录 |
| `checkpoints/data_aug_conformer_smoke/` | Conformer 冒烟测试 | — | 仅 k01 一个组合，仅供管线验证 |
| `checkpoints/data_aug_imu/` | **IMU-Conformer**（attn bias，26 组合） | 0.77M | 已完成（10-21）：offdiag(1-21) 0.7701，combo26 全26 **0.9399** 全场最高 |
| `checkpoints/data_aug_d64ff128_l2/` | **Conformer-XS**（26 组合） | 136,905 | 已完成（10-21）：offdiag(1-21) **0.7749**，attention 系 26 组合口径最佳 |

### 注意力调参（`checkpoints/attn_tune/`，seed 42，前 6 组合 × 100 epoch）

| 路径 | 规范模型名 | 参数量 | 组合外均值(1-6) |
|---|---|---|---|
| `checkpoints/attn_tune/d64ff128_l2/` | **Conformer-XS** | 136,905 | **0.7562**（attention 系最佳） |
| `checkpoints/attn_tune/d64/` | Conformer-S | 300,041 | 0.7529 |
| `checkpoints/attn_tune/d64_drop40/` | Conformer-S-drop40 | 300,041 | 0.7345 |
| `checkpoints/attn_tune/drop40/` | Conformer-drop40 | 770,569 | 0.7345 |

模型命名全项目统一（README / 绘图脚本同款）：DCNN / Transformer / Conformer(v2) /
IMU-Conformer / Conformer-S / Conformer-XS / Conformer-drop40 / Conformer-S-drop40。

**架构对比总表**（详见 `README.md` Conformer 小节）：

| 模型 | 参数 | 组合外均值(1-21) | 全26均值 | combo26 全26 | NSDTR 组合外 |
|---|---|---|---|---|---|
| DCNN | 93k | **0.7847** | **0.8602** | 0.9308 | 0.5803 |
| DCNN + 12Hz 去噪 | 93k | 0.7868 | 0.8600 | 0.9339 | 0.6186 |
| Transformer v1 | 0.42M | 0.7391 | 0.8296 | 0.9126 | 0.6654 |
| Conformer (v2) | 0.77M | 0.7696 | 0.8538 | **0.9370** | **0.7284** |
| Transformer v2 | 0.25M | 3 组合即终止，无保留产出目录 | | | |

### Conformer 结构消融（`checkpoints/ablation/`，seed 42，前 6 组合 × 100 epoch）

三个消融维度：`attn`（joint/split）× `gate`（glu/swiglu）× `conv-norm`（bn/gn），
2×2×2 共 8 变体；`joint_glu_bn` 基线 = `data_aug_conformer_v2` 前 6 组合
（`run_conformer_ablation.sh` 中该行注释未跑），实际新建 7 个结构变体目录；
另有物理先验变体 `imu_glu_bn`（`--attn imu`：joint + IMU 结构偏置，09-19 跑）。

| 路径 | attn | gate | norm | 组合外均值(1-6) |
|---|---|---|---|---|
| `checkpoints/ablation/joint_swiglu_gn/` | joint | swiglu | gn | 0.7453 |
| `checkpoints/ablation/joint_swiglu_bn/` | joint | swiglu | bn | 0.7433 |
| `checkpoints/ablation/joint_glu_gn/` | joint | glu | gn | 0.7436 |
| `checkpoints/ablation/imu_glu_bn/` | imu | glu | bn | 0.7437 |
| `checkpoints/ablation/split_glu_gn/` | split | glu | gn | 0.7451 |
| `checkpoints/ablation/split_glu_bn/` | split | glu | bn | 0.7448 |
| `checkpoints/ablation/split_swiglu_gn/` | split | swiglu | gn | 0.7456 |
| `checkpoints/ablation/split_swiglu_bn/` | split | swiglu | bn | 0.7427 |
| （基线 joint_glu_bn = `data_aug_conformer_v2/` 前 6） | joint | glu | bn | 0.7472 |
| （DCNN 参照 = `data_aug/` 前 6，未固定种子） | — | — | — | **0.7715** |

每个变体目录内含 k01–k06 共 6 个 `.pth` + 训练日志 + 评测矩阵（结构同上文
"各 run 目录通用产出"）；run 级控制台日志在 `checkpoints/ablation/<name>.log`。
**对比表与结论在 `results/ablation_summary.md`**（四个假设全部证伪——gate/归一化/
注意力组织/IMU 物理偏置，9 个 conformer 变体挤在 0.45 个点窄带内，没有一个
超过原版 joint_glu_bn）。
