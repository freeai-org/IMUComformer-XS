#!/usr/bin/env python3
"""transformer.py —— 双模态 patch + full attention 的 IMU Transformer(project-5)

输入与 DCNN 完全一致: [B, 6, T](acc 3 轴 + gyr 3 轴, T=window 采样点)。

结构(针对 "acc 与 gyr 是两部分, 用 attention 关联" 的设计):
  1. patch embed: acc/gyr 各自 Conv1d(3, d, k=patch, s=patch) ->
     每模态 T/patch 个 token(局部形态先被卷积编码, 类似 Conformer 前段)
  2. token 序列 = [acc tokens; gyr tokens] 共 2*(T/patch) 个, 叠加:
     - 模态嵌入(2 类): 告诉 attention 哪些 token 来自加速度计/陀螺仪
     - 位置嵌入(T/patch): 时间顺序
  3. full self-attention(Pre-LN TransformerEncoder): 每个输出位置可attend全部
     acc+gyr token -> 模态间/时间维关联全部由 attention 显式建模
  4. mean pool + FC 分类

T=100, patch=10 -> 20 个 token, d_model=128, 3 层, ~0.4M 参数(打印对比 DCNN 92,873)。
"""
import torch
import torch.nn as nn


class TransformerIMU(nn.Module):
    def __init__(self, input_size=6, num_classes=9, d_model=128, nhead=4,
                 num_layers=3, dim_ff=256, patch=10, dropout=0.2,
                 max_tokens=64):
        super().__init__()
        assert input_size == 6, "acc(3) + gyr(3) 双模态设计, 输入须为 6 通道"
        self.patch = patch
        self.patch_acc = nn.Conv1d(3, d_model, kernel_size=patch, stride=patch)
        self.patch_gyr = nn.Conv1d(3, d_model, kernel_size=patch, stride=patch)
        self.modality_emb = nn.Embedding(2, d_model)
        self.pos_emb = nn.Embedding(max_tokens, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, norm_first=True)  # Pre-LN, 小数据更稳
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):  # x: [B, 6, T]
        B, _, T = x.shape
        t = T // self.patch
        acc = self.patch_acc(x[:, :3]).transpose(1, 2)   # [B, t, d]
        gyr = self.patch_gyr(x[:, 3:]).transpose(1, 2)   # [B, t, d]
        tok = torch.cat([acc, gyr], dim=1)               # [B, 2t, d]
        dev = tok.device
        mod = torch.cat([torch.zeros(t, dtype=torch.long, device=dev),
                         torch.ones(t, dtype=torch.long, device=dev)])
        pos = torch.arange(2 * t, device=dev)
        tok = tok + self.modality_emb(mod).unsqueeze(0) + self.pos_emb(pos).unsqueeze(0)
        h = self.encoder(self.drop(tok))
        h = self.norm(h.mean(dim=1))
        return self.fc(h)


class TransformerIMU2(nn.Module):
    """v2: 旋转不变特征 + 逐轴 token 的 full attention

    输入仍是 [B, 6, T](acc xyz + gyr xyz), 模型内部:
      1. 特征增广(领域知识):
         - |a| = norm(acc): 项圈旋转不变(每只狗佩戴角度不同 -> 品种域差主源之一)
         - |ω| = norm(gyr): 同上
         - |jerk| = |d acc/dt|: 行为转换的瞬时特征
         原始 6 通道保留(姿态直流信息: 重力方向 = 头部姿态), 共 9 通道
      2. 逐轴 patch embed: 每个通道(轴)独立 Conv1d(1, d, k=patch, s=patch)
         -> 9 轴 × (T/patch) 个 token, 叠加轴嵌入(9) + 模态嵌入(acc/gyr/模长)
         + 位置嵌入。attention 在 轴×时间 全局显式分配 -> "让 Transformer
         关注输入的不同部分"的字面实现
      3. Pre-LN TransformerEncoder + mean pool + FC
    T=100, patch=10 -> 90 token, d=96, 3 层, ~0.5M 参数
    """

    def __init__(self, input_size=6, num_classes=9, d_model=96, nhead=4,
                 num_layers=3, dim_ff=192, patch=10, dropout=0.2,
                 max_tokens=96):
        super().__init__()
        assert input_size == 6
        self.patch = patch
        self.n_feat = 9  # 6 原始 + |a| + |ω| + |jerk|
        # 逐轴 patch embed: 每轴一个 (1->d) 卷积
        self.patch_axes = nn.ModuleList(
            nn.Conv1d(1, d_model, kernel_size=patch, stride=patch)
            for _ in range(self.n_feat))
        self.axis_emb = nn.Embedding(self.n_feat, d_model)
        self.modality_emb = nn.Embedding(3, d_model)  # 0=acc 1=gyr 2=mag/derived
        self.pos_emb = nn.Embedding(max_tokens, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):  # x: [B, 6, T]
        B, _, T = x.shape
        t = T // self.patch
        acc, gyr = x[:, :3], x[:, 3:]
        mag_a = acc.norm(dim=1, keepdim=True)                    # [B,1,T]
        mag_g = gyr.norm(dim=1, keepdim=True)
        jerk = (acc - torch.cat([acc[:, :, :1], acc[:, :, :-1]], dim=2)) \
            .norm(dim=1, keepdim=True)                           # 一阶差分模长
        feats = torch.cat([acc, gyr, mag_a, mag_g, jerk], dim=1)  # [B,9,T]

        toks = [conv(feats[:, i:i + 1]) for i, conv in enumerate(self.patch_axes)]
        tok = torch.cat(toks, dim=2).transpose(1, 2)             # [B, t*n_feat, d]
        dev = tok.device
        axis = torch.arange(self.n_feat, device=dev).repeat(t)   # t*n_feat, 按块排列
        # 每个时间 patch 内 9 轴相邻: axis = [0..8, 0..8, ...]
        pos = torch.arange(t, device=dev).repeat_interleave(self.n_feat)
        mod = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], device=dev).repeat(t)
        tok = (tok + self.axis_emb(axis).unsqueeze(0)
               + self.modality_emb(mod).unsqueeze(0)
               + self.pos_emb(pos).unsqueeze(0))
        h = self.encoder(self.drop(tok))
        h = self.norm(h.mean(dim=1))
        return self.fc(h)
