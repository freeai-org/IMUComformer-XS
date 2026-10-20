# results/ 结果文件说明

| 文件 | 内容 |
|---|---|
| `IMU_DCNN.jsonl` | dcnn.py 主实验日志（DS1 九大类）：每次运行首行 config + 每 epoch 一条；共 4 次运行，最终 500 epoch 跑满，最佳 test_acc 0.8815 |
| `IMU_DCNN_breeds.jsonl` | breed_cv.py 训练日志：26 品种各一段（config 带 breed 字段 + 每 epoch，各 100 epoch） |
| `breed_cv_matrix.csv` / `.png` | 26×26 品种交叉验证矩阵与热力图，全 9 类（对角均值 0.875 / 非对角均值 0.538） |
| `breed_cv_matrix_drop_Feeding_Standing.csv` / `.png` | 剔除 Feeding/Standing 的 7 类版（对角 0.947 / 非对角 0.657） |
| `breed_cv_matrix_drop_Feeding_Shaking_Sniffing_Standing.csv` / `.png` | 再剔除 Shaking/Sniffing 的 5 类版（对角 0.938 / 非对角 0.627） |

口径与结论详见项目根 `../README.md`；checkpoint 在 `../checkpoints/`。
