#!/usr/bin/env python3
"""conformer.py —— Conv + Attention 混合的 IMU Conformer(project-5)

动机: Transformer 版 loss 降得快但验证集(7 类协议)不如 DCNN——full attention
让 2t 个 token 全互看, 自由度大, IMU 小数据易过拟合; DCNN 的卷积先验
(局部性/平移不变)反而契合 IMU 波形。

Conformer 取两者优点(语音识别经典结构, IMU/行为识别文献常用):
  1. patch embed: 与 Transformer 相同, acc/gyr 各自 Conv1d 分词 -> 保留模态交互入口
  2. Conformer 块 x N(macaron 前后 FFN + 多头自注意力 + 卷积模块):
     - 多头自注意力: acc/gyr 跨模态/短时程交互(Transformer 强项)
     - 卷积模块: 深度可分离 1D 卷积抓 IMU 局部波形(DCNN 强项)
  3. mean pool + FC 分类

输入与 DCNN/Transformer 完全一致: [B, 6, T](acc 3 轴 + gyr 3 轴, T=window 采样点)。
T=100, patch=10 -> 20 个 token, d_model=128, 3 层, ~0.77M 参数(约为 Transformer v1 的 2 倍)。
"""
import math

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """前馈块(Pre-LN + Swish), res_scale=0.5 时作 macaron 半残差"""

    def __init__(self, d_model, dim_ff, dropout, res_scale=1.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout))
        self.res_scale = res_scale

    def forward(self, x):  # [B, N, D]
        return x + self.res_scale * self.net(self.norm(x))


class ConvModule(nn.Module):
    """Conformer 卷积模块: LN -> 点卷积升维+GLU/SwiGLU -> 深度可分离时序卷积
    -> BN/GroupNorm -> SiLU -> 点卷积降维(残差)

    gate="glu":    (xW1) ⊗ sigmoid(xW2)  原版; sigmoid 门易饱和, 通道可能被长期压 0
    gate="swiglu": (xWa) ⊗ SiLU(xWb)     自门控平滑不过 0 点, 不易恒 0; 参数量与 glu 相同
    conv_norm="bn": BN 统计来自训练品种, 跨品种评测存在域偏移(泛化嫌疑最大)
    conv_norm="gn": GroupNorm 不依赖 batch 统计, 跨域更稳
    """

    def __init__(self, d_model, kernel=15, dropout=0.2, gate="glu",
                 conv_norm="bn"):
        super().__init__()
        self.gate = gate
        self.norm = nn.LayerNorm(d_model)
        if gate == "swiglu":
            self.pw_a = nn.Linear(d_model, d_model)
            self.pw_b = nn.Linear(d_model, d_model)
            self.pw1 = None
        else:  # glu
            self.pw1 = nn.Linear(d_model, 2 * d_model)  # 点卷积(升维 x2 供 GLU)
            self.pw_a = self.pw_b = None
        self.glu = nn.GLU(dim=-1) if gate != "swiglu" else None
        self.act = nn.SiLU()
        self.dw = nn.Conv1d(d_model, d_model, kernel,
                            padding=(kernel - 1) // 2, groups=d_model)
        # 属性名保持 bn: 与旧版 checkpoint 的 state_dict key(conv.bn.*)完全一致
        if conv_norm == "gn":
            self.bn = nn.GroupNorm(8, d_model)
        else:  # bn
            self.bn = nn.BatchNorm1d(d_model)
        self.pw2 = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # [B, N, D]
        h = self.norm(x)
        if self.gate == "swiglu":
            h = self.pw_a(h) * self.act(self.pw_b(h))     # [B, N, D]
        else:
            h = self.glu(self.pw1(h))                      # [B, N, D]
        y = h.transpose(1, 2)                              # [B, D, N]
        y = self.act(self.bn(self.dw(y)))
        y = y.transpose(1, 2)
        return x + self.drop(self.pw2(y))


class Attention(nn.Module):
    """多头自注意力(Pre-LN, batch_first)

    bias_kind="imu" 时给注意力叠加 IMU 结构偏置 —— 加性 attn_mask,
    不遮断任何交互, 只改变先验偏好:
      - 同模态 token 间: 偏置随时间距离线性衰减(时序局部性, 即 DCNN
        制胜的卷积先验的注意力版);
      - 跨模态 token 间: 同时刻偏置为 0(acc/gyr 是同一刚性运动在同一
        瞬间的两个观测, 物理上有意义), 异时刻按时间距离惩罚(跨时刻
        跨模态交互最容易藏品种捷径, 主指标是跨品种泛化);
      - 两个斜率可学习: 模型可把偏置学到≈0 退化为 joint 基线, 下限
        有保证; 仅新增 2 标量/层, 容量与对照组严格可比。
    """

    def __init__(self, d_model, nhead, dropout, bias_kind="none"):
        super().__init__()
        self.bias_kind = bias_kind
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                          batch_first=True)
        self.drop = nn.Dropout(dropout)
        if bias_kind == "imu":
            # exp() 保证斜率恒正; 初始温和: intra ~0.1, cross ~0.3 (每单位 patch 距离)
            self.slope_intra = nn.Parameter(torch.tensor(math.log(0.1)))
            self.slope_cross = nn.Parameter(torch.tensor(math.log(0.3)))
        else:
            self.slope_intra = self.slope_cross = None

    def _imu_bias(self, t, device):
        """[2t, 2t] 加性偏置, 行=query 列=key。token 排布 [acc 0..t-1 | gyr 0..t-1]"""
        n = 2 * t
        idx = torch.arange(n, device=device)
        is_acc = idx < t
        tpos = idx % t                                        # 各 token 时间序号
        same_mod = is_acc[:, None] == is_acc[None, :]
        dist = (tpos[:, None] - tpos[None, :]).abs().float()  # 时间距离(patch 单位)
        intra = -self.slope_intra.exp() * dist
        cross = torch.where(dist == 0,
                            torch.zeros_like(dist),
                            -self.slope_cross.exp() * dist)
        return torch.where(same_mod, intra, cross)

    def forward(self, x, t=None):  # [B, N, D]
        y = self.norm(x)
        mask = self._imu_bias(t, x.device) \
            if (self.bias_kind == "imu" and t) else None
        y, _ = self.attn(y, y, y, attn_mask=mask, need_weights=False)
        return x + self.drop(y)


class ConformerBlock(nn.Module):
    """macaron FFN -> 自注意力 -> 卷积模块 -> macaron FFN

    attn_mode="joint": 单个全注意力, acc+gyr token 拼一起互相 attend
    (跨模态交互最强, 但也最容易学到品种相关的跨模态捷径)
    attn_mode="split": 双模态独立注意力 —— attention-1 只看加速度 token,
    attention-2 只看角速度 token(参数独立); 跨模态交互仅靠卷积模块/池化,
    模态特异记忆被削弱, 利于跨品种泛化
    attn_mode="imu": joint 全交互 + IMU 结构偏置(同模态随时间距离衰减,
    跨模态仅同一时刻免罚, 斜率可学习, 可退化为 joint), 详见 Attention
    """

    def __init__(self, d_model=128, nhead=4, dim_ff=256, kernel=15, dropout=0.2,
                 gate="glu", conv_norm="bn", attn_mode="joint"):
        super().__init__()
        self.attn_mode = attn_mode
        self.ff1 = FeedForward(d_model, dim_ff, dropout, res_scale=0.5)
        if attn_mode == "split":
            self.attn_acc = Attention(d_model, nhead, dropout)  # attention-1: 加速度轴
            self.attn_gyr = Attention(d_model, nhead, dropout)  # attention-2: 角速度轴
            self.attn = None
        elif attn_mode == "imu":
            self.attn = Attention(d_model, nhead, dropout, bias_kind="imu")
            self.attn_acc = self.attn_gyr = None
        else:
            self.attn = Attention(d_model, nhead, dropout)
            self.attn_acc = self.attn_gyr = None
        self.conv = ConvModule(d_model, kernel, dropout, gate=gate,
                                conv_norm=conv_norm)
        self.ff2 = FeedForward(d_model, dim_ff, dropout, res_scale=0.5)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, t=None):  # x: [B, 2t, D]; t 为单模态 token 数
        x = self.ff1(x)
        if self.attn_mode == "split" and t is not None:
            x = torch.cat([self.attn_acc(x[:, :t]),
                           self.attn_gyr(x[:, t:])], dim=1)
        elif self.attn_mode == "imu" and t is not None:
            x = self.attn(x, t=t)
        else:
            x = self.attn(x)
        x = self.conv(x)
        x = self.ff2(x)
        return self.norm(x)


class ConformerIMU(nn.Module):
    def __init__(self, input_size=6, num_classes=9, d_model=128, nhead=4,
                 num_layers=3, dim_ff=256, patch=10, kernel=15, dropout=0.2,
                 max_tokens=64, gate="glu", conv_norm="bn", attn_mode="joint"):
        super().__init__()
        assert input_size == 6, "acc(3) + gyr(3) 双模态设计, 输入须为 6 通道"
        self.patch = patch
        self.patch_acc = nn.Conv1d(3, d_model, kernel_size=patch, stride=patch)
        self.patch_gyr = nn.Conv1d(3, d_model, kernel_size=patch, stride=patch)
        self.modality_emb = nn.Embedding(2, d_model)
        self.pos_emb = nn.Embedding(max_tokens, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, nhead, dim_ff, kernel, dropout,
                           gate=gate, conv_norm=conv_norm,
                           attn_mode=attn_mode)
            for _ in range(num_layers)])
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
        h = self.drop(tok)
        for blk in self.blocks:
            h = blk(h, t=t)
        h = h.mean(dim=1)
        return self.fc(h)


if __name__ == "__main__":
    m = ConformerIMU(input_size=6, num_classes=9)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"ConformerIMU 参数量: {n:,}")
    print(m(torch.randn(4, 6, 100)).shape)
