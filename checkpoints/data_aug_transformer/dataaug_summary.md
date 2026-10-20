# Transformer（双模态 full attention）vs DCNN 对比结论（09-15）

## 模型（`transformer.py` TransformerIMU）

针对"acc 与 gyr 是两部分，用 attention 关联"的设计：
- acc/gyr 各自 Conv1d patch embed（k=10, s=10）→ 每模态 10 个 token，共 20 token
- 模态嵌入（2 类）+ 位置嵌入 → full self-attention（Pre-LN，3 层，d=128，4 头）
- mean pool + FC；参数量 ~0.42M（DCNN 92,873）

## 结果（同协议：26 组合 × 100 epoch，7 类评测）

| 指标 | Transformer | DCNN | Δ |
|---|---|---|---|
| **纯跨品种泛化（组合外均值 1-21）** | 0.7391 | 0.7847 | **-0.0456** |
| 种子期（combo 1-6） | 0.7313 | 0.7715 | -0.0402 |
| 全 26 均值（26 组合平均） | 0.8296 | 0.8602 | -0.0306 |
| 可靠区间最佳组合 | combo 19（0.7622） | combo 15（0.8086） | -0.0464 |
| **最难品种 NSDTR（组合外）** | **0.6654** | 0.5913 | **+0.0741** |
| Mudi（组合外） | 0.7809 | 0.8278 | -0.0469 |

26 个组合中 Transformer 组合外均值 25 个落后（平均 -4.6 点）。

## 结论

1. **"full attention 关联 acc/gyr → 指标打上来"没有成立**：纯跨品种泛化反而低
   4.6 个点。原因：①单品种训练数据仅 9k~100k 窗口，0.42M 参数的 Transformer
   欠拟合/过拟合平衡差于 93k 参数的 DCNN；②IMU 行为信号本质是局部形态
   （conv 归纳偏置正对口），跨轴关联仅 6 通道、conv1 一层即可学完，attention
   的长程建模优势在这个数据量下发挥不出来；③超参未调（lr 1e-3 无 warmup）。
2. **唯一亮点仍是 NSDTR**（+7.4 点）：信号最弱的品种从更强模型受益——与去噪
   实验（+3.8 点）方向一致，NSDTR 是全局瓶颈也是新架构的受益者。
3. 与去噪实验互相印证：**跨品种泛化瓶颈不在输入表示**（噪声、模态关联都不是），
   而在数据量/域差本身。Transformer 要打平/超越 DCNN，路径是：全量数据一次
   训练（非单品种小数据）+ lr warmup + 更小模型/正则，或走 Conformer（conv
   局部 + attention 全局混合），而不是单纯堆 attention。

## 产出

- `transformer.py`：TransformerIMU 模型定义
- `IMU_DCNN_dataaug.jsonl`：26 config + 2600 epoch 训练日志
- `k01..k26_*.pth`：26 个组合 checkpoint
- `dataaug_infer.jsonl` / `.csv`：评测记录
- `transformer_compare.csv`：与 DCNN 逐组合对比
- `dataaug_heatmap.png` / `dataaug_mean_curve.png` / `dataaug_offdiag_curve.png`：可视化
- `compare_transformer.py`：对比脚本
