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
| 1 | Border Collie | 0.6927 |
| 2 | Spanish Water Dog | 0.7414 |
| 3 | Crossbreed | 0.7654 |
| 4 | Lapponian Herder | 0.7995 |
| 5 | Bouvier des Ardennes | 0.7957 |
| 6 | Golden Retriever | 0.8201 |
| 7 | English Springer Spaniel | 0.8218 |
| 8 | Vizsla | 0.8145 |
| 9 | German Shepherd | 0.8311 |
| 10 | Labrador Retriever | 0.8420 |
| 11 | Australian Kelpie | 0.8414 |
| 12 | Mudi | 0.8513 |
| 13 | Dutch Shepherd | 0.8600 |
| 14 | Flat-Coated Retriever | 0.8535 |
| 15 | Finnish Lapphund | 0.8665 |
| 16 | Belgian Shepherd Malinois | 0.8777 |
| 17 | Lagotto Romagnolo | 0.8851 |
| 18 | Smooth Collie | 0.8869 |
| 19 | Beauceron | 0.8970 |
| 20 | Hovawart | 0.9023 |
| 21 | Bouvier | 0.9031 |
| 22 | Belgian Shepherd Groenendael | 0.9145 |
| 23 | Standard Poodle | 0.9206 |
| 24 | Bull Terrier (Miniature) | 0.9288 |
| 25 | Nova Scotia Duck Tolling Retriever | 0.9329 |
| 26 | Belgian Shepherd | 0.9398 |

**最佳组合: combo 26 (含前 26 个品种, 截止 Border Collie → Spanish Water Dog → Crossbreed → Lapponian Herder → Bouvier des Ardennes → Golden Retriever → English Springer Spaniel → Vizsla → German Shepherd → Labrador Retriever → Australian Kelpie → Mudi → Dutch Shepherd → Flat-Coated Retriever → Finnish Lapphund → Belgian Shepherd Malinois → Lagotto Romagnolo → Smooth Collie → Beauceron → Hovawart → Bouvier → Belgian Shepherd Groenendael → Standard Poodle → Bull Terrier (Miniature) → Nova Scotia Duck Tolling Retriever → Belgian Shepherd), 均值 0.9398**

口径: 组合内品种用其 test split(种子 42), 组合外用全量; 7 类协议(剔除 Feeding/Standing: 真值过滤 + 输出屏蔽)。
