# 数据增广实验结论(基于 7 类品种交叉矩阵)

## 品种排序(种子优先级, 7 类矩阵非对角均值)

1. Border Collie: 0.7523
2. Spanish Water Dog: 0.7454
3. Crossbreed: 0.7181
4. Lapponian Herder: 0.7141
5. Bouvier des Ardennes: 0.7096
6. Golden Retriever: 0.7044
7. English Springer Spaniel: 0.7022
8. Vizsla: 0.6951
9. German Shepherd: 0.6899
10. Labrador Retriever: 0.6863
11. Australian Kelpie: 0.6795
12. Mudi: 0.6791
13. Dutch Shepherd: 0.6726
14. Flat-Coated Retriever: 0.6640
15. Finnish Lapphund: 0.6615
16. Belgian Shepherd Malinois: 0.6380
17. Lagotto Romagnolo: 0.6375
18. Smooth Collie: 0.6310
19. Beauceron: 0.6262
20. Hovawart: 0.6168
21. Bouvier: 0.6160
22. Belgian Shepherd Groenendael: 0.5877
23. Standard Poodle: 0.5812
24. Bull Terrier (Miniature): 0.5779
25. Nova Scotia Duck Tolling Retriever: 0.5686
26. Belgian Shepherd: 0.5348

## 各组合 26 品种评测均值

| combo | 新增品种 | 均值 |
|---|---|---|
| 1 | Border Collie | 0.7455 |
| 2 | Spanish Water Dog | 0.7631 |
| 3 | Crossbreed | 0.7927 |
| 4 | Lapponian Herder | 0.8050 |
| 5 | Bouvier des Ardennes | 0.8146 |
| 6 | Golden Retriever | 0.8256 |
| 7 | English Springer Spaniel | 0.8179 |
| 8 | Vizsla | 0.8228 |
| 9 | German Shepherd | 0.8336 |
| 10 | Labrador Retriever | 0.8454 |
| 11 | Australian Kelpie | 0.8472 |
| 12 | Mudi | 0.8546 |
| 13 | Dutch Shepherd | 0.8651 |
| 14 | Flat-Coated Retriever | 0.8714 |
| 15 | Finnish Lapphund | 0.8812 |
| 16 | Belgian Shepherd Malinois | 0.8705 |
| 17 | Lagotto Romagnolo | 0.8817 |
| 18 | Smooth Collie | 0.8973 |
| 19 | Beauceron | 0.8977 |
| 20 | Hovawart | 0.9057 |
| 21 | Bouvier | 0.9040 |
| 22 | Belgian Shepherd Groenendael | 0.9146 |
| 23 | Standard Poodle | 0.9194 |
| 24 | Bull Terrier (Miniature) | 0.9286 |
| 25 | Nova Scotia Duck Tolling Retriever | 0.9301 |
| 26 | Belgian Shepherd | 0.9308 |

**最佳组合: combo 26 (含前 26 个品种, 截止 Border Collie → Spanish Water Dog → Crossbreed → Lapponian Herder → Bouvier des Ardennes → Golden Retriever → English Springer Spaniel → Vizsla → German Shepherd → Labrador Retriever → Australian Kelpie → Mudi → Dutch Shepherd → Flat-Coated Retriever → Finnish Lapphund → Belgian Shepherd Malinois → Lagotto Romagnolo → Smooth Collie → Beauceron → Hovawart → Bouvier → Belgian Shepherd Groenendael → Standard Poodle → Bull Terrier (Miniature) → Nova Scotia Duck Tolling Retriever → Belgian Shepherd), 均值 0.9308**

口径: 组合内品种用其 test split(种子 42), 组合外用全量; 7 类协议(剔除 Feeding/Standing: 真值过滤 + 输出屏蔽)。

## 可视化

- `dataaug_heatmap.png`: 26 组合 x 26 品种准确率热力图(行=累加组合, 蓝框=该行最差品种)。
  整体自上而下变绿(品种越加越多越好); **Mudi 列全程偏橙**(所有组合下都最难,
  combo 26 也仅 ~0.92); 各行蓝框多落在 Mudi / Spanish Water Dog(未入组合时)/
  Nova Scotia Duck Tolling Retriever。
- `dataaug_mean_curve.png`: 各组合均值曲线(0.7455 单调升至 0.9308, 最佳 k26)。
- `dataaug_offdiag_curve.png` / `dataaug_offdiag.csv`: **纯跨品种泛化**(去掉组合内
  品种, 只看组合外品种均值), 见下节。

绘图脚本: `plot_dataaug_heatmap.py`(读 dataaug_infer.csv)、
`plot_dataaug_offdiag.py`(读 dataaug_infer.jsonl)。

## 纯跨品种泛化(去掉组合内品种, 09-14 补充分析)

全 26 均值(0.7455→0.9308 单调升)混入了组合内 test split(较易)的分数, 组合越大
虚高越多。只看组合外品种(模型没训过的品种)的均值:

| combo | 新增品种 | 组合外数 | 组合外均值 |
|---|---|---|---|
| 1 | Border Collie | 25 | 0.7383 |
| 6 | Golden Retriever | 20 | 0.7945 |
| 10 | Labrador Retriever | 16 | 0.7950 |
| 13 | Dutch Shepherd | 13 | 0.7974 |
| **15** | **Finnish Lapphund** | **11** | **0.8086** ← 可靠区间最佳 |
| 18 | Smooth Collie | 8 | 0.8009 |
| 21 | Bouvier | 5 | 0.7800 |
| 25 | Nova Scotia Duck Tolling Retriever | 1 | 0.9543(仅 1 品种, 不可信) |

结论:

1. **跨品种泛化 ~6 个品种就饱和**: combo 1→6 从 0.738 升到 0.794, 之后在
   0.77~0.81 平台震荡, combo 15 达峰 0.8086 后不再提升。**"最佳组合=全量"的
   结论被修正**——那是 test split 口径虚高; 纯跨品种口径下最佳是
   **combo 15(前 15 品种, 含到 Finnish Lapphund)**, 且与 combo 6/10/13 差距
   仅 ~1 个点, 增广收益递减得比全量口径早得多。
2. **Nova Scotia Duck Tolling Retriever 在所有组合下都是组合外最差**
   (0.478~0.692, 25 行全占 off_min), 是最难泛化的品种; Mudi 则是"入组合后仍难"。
3. 若目标是"训一批数据泛化到新品种", 加数据的前 6 个品种收益最大(种子排序
   有效), 之后应转向架构/域自适应而非继续堆品种。
