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
| 1 | Border Collie | 0.7161 |
| 2 | Spanish Water Dog | 0.7266 |
| 3 | Crossbreed | 0.7560 |
| 4 | Lapponian Herder | 0.7883 |
| 5 | Bouvier des Ardennes | 0.8047 |
| 6 | Golden Retriever | 0.8115 |
| 7 | English Springer Spaniel | 0.8191 |
| 8 | Vizsla | 0.8210 |
| 9 | German Shepherd | 0.8327 |
| 10 | Labrador Retriever | 0.8464 |
| 11 | Australian Kelpie | 0.8424 |
| 12 | Mudi | 0.8529 |
| 13 | Dutch Shepherd | 0.8493 |
| 14 | Flat-Coated Retriever | 0.8630 |
| 15 | Finnish Lapphund | 0.8616 |
| 16 | Belgian Shepherd Malinois | 0.8771 |
| 17 | Lagotto Romagnolo | 0.8828 |
| 18 | Smooth Collie | 0.8885 |
| 19 | Beauceron | 0.9014 |
| 20 | Hovawart | 0.8977 |
| 21 | Bouvier | 0.9101 |
| 22 | Belgian Shepherd Groenendael | 0.9183 |
| 23 | Standard Poodle | 0.9235 |
| 24 | Bull Terrier (Miniature) | 0.9266 |
| 25 | Nova Scotia Duck Tolling Retriever | 0.9341 |
| 26 | Belgian Shepherd | 0.9399 |

**最佳组合: combo 26 (含前 26 个品种, 截止 Border Collie → Spanish Water Dog → Crossbreed → Lapponian Herder → Bouvier des Ardennes → Golden Retriever → English Springer Spaniel → Vizsla → German Shepherd → Labrador Retriever → Australian Kelpie → Mudi → Dutch Shepherd → Flat-Coated Retriever → Finnish Lapphund → Belgian Shepherd Malinois → Lagotto Romagnolo → Smooth Collie → Beauceron → Hovawart → Bouvier → Belgian Shepherd Groenendael → Standard Poodle → Bull Terrier (Miniature) → Nova Scotia Duck Tolling Retriever → Belgian Shepherd), 均值 0.9399**

口径: 组合内品种用其 test split(种子 42), 组合外用全量; 7 类协议(剔除 Feeding/Standing: 真值过滤 + 输出屏蔽)。
