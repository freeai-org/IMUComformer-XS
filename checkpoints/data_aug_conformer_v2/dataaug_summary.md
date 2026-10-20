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
| 1 | Border Collie | 0.7087 |
| 2 | Spanish Water Dog | 0.7437 |
| 3 | Crossbreed | 0.7619 |
| 4 | Lapponian Herder | 0.7959 |
| 5 | Bouvier des Ardennes | 0.8021 |
| 6 | Golden Retriever | 0.8090 |
| 7 | English Springer Spaniel | 0.8203 |
| 8 | Vizsla | 0.8209 |
| 9 | German Shepherd | 0.8347 |
| 10 | Labrador Retriever | 0.8430 |
| 11 | Australian Kelpie | 0.8434 |
| 12 | Mudi | 0.8506 |
| 13 | Dutch Shepherd | 0.8544 |
| 14 | Flat-Coated Retriever | 0.8579 |
| 15 | Finnish Lapphund | 0.8685 |
| 16 | Belgian Shepherd Malinois | 0.8767 |
| 17 | Lagotto Romagnolo | 0.8782 |
| 18 | Smooth Collie | 0.8918 |
| 19 | Beauceron | 0.8986 |
| 20 | Hovawart | 0.8988 |
| 21 | Bouvier | 0.9053 |
| 22 | Belgian Shepherd Groenendael | 0.9150 |
| 23 | Standard Poodle | 0.9216 |
| 24 | Bull Terrier (Miniature) | 0.9299 |
| 25 | Nova Scotia Duck Tolling Retriever | 0.9322 |
| 26 | Belgian Shepherd | 0.9370 |

**最佳组合: combo 26 (含前 26 个品种, 截止 Border Collie → Spanish Water Dog → Crossbreed → Lapponian Herder → Bouvier des Ardennes → Golden Retriever → English Springer Spaniel → Vizsla → German Shepherd → Labrador Retriever → Australian Kelpie → Mudi → Dutch Shepherd → Flat-Coated Retriever → Finnish Lapphund → Belgian Shepherd Malinois → Lagotto Romagnolo → Smooth Collie → Beauceron → Hovawart → Bouvier → Belgian Shepherd Groenendael → Standard Poodle → Bull Terrier (Miniature) → Nova Scotia Duck Tolling Retriever → Belgian Shepherd), 均值 0.9370**

口径: 组合内品种用其 test split(种子 42), 组合外用全量; 7 类协议(剔除 Feeding/Standing: 真值过滤 + 输出屏蔽)。

## 纯跨品种泛化(组合外均值, 09-16 补算)

| 指标 | Conformer | DCNN | Transformer v1 |
|---|---|---|---|
| 组合外均值(可靠区间 combo1-21) | 0.7696 | **0.7847** | 0.7391 |
| 种子期(combo1-6) | 0.7307 | 0.7715 | 0.7313 |
| combo 1 单品种 | 0.7000 | 0.7383 | 0.6949 |
| 可靠区间最佳 | combo 16 (0.7922) | combo 15 (0.8086) | combo 19 (0.7622) |
| **NSDTR 组合外(1-21)** | **0.7284** | 0.5803 | 0.6654 |

- Conformer 明显好于 Transformer v1(+3.1 点), 但整体仍不及 DCNN(-1.5 点);
  combo 26 全 26 均值 0.9370 超 DCNN 的 0.9308 属 test split 口径虚高部分。
- **NSDTR +14.8 点(vs DCNN)**: 混合架构对难例品种显著有效; 与去噪(+3.8)、
  Transformer(+7.4) 同向且幅度最大——难例品种持续从更强表示受益。
- `curves_combo_test_vs_mean.png`: combo_test_acc 全程平在 0.860~0.890
  (组合内早饱和), mean_acc 收益全部来自组合外品种, 两线在 combo ~17 交叉。
- 逐组合组合外均值: 0.7000/0.7284/0.7413/0.7707/0.7709/0.7721/0.7768/
  0.7662/0.7803/0.7849/0.7763/0.7802/0.7781/0.7714/0.7775/0.7922/0.7778/
  0.7895/0.7934/0.7709/0.7632(combo 1-21)。
