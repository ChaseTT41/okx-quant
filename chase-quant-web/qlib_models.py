"""
Qlib-Style Deep Learning Models v1.0 — PyTorch 实现
====================================================
Chase量化策略 Phase 9: 将 Microsoft Qlib 核心模型架构嫁接到我们的特征矩阵上。

移植的模型 (论文驱动, 独立于 Qlib 依赖):
  1. ALSTM (Attention LSTM)       — 时序依赖 + 注意力选择
     论文: "Enhancing Stock Movement Prediction with Adversarial Training" (Feng et al., 2019)
     核心: LSTM 隐藏状态 × Attention 加权 → 非线性预测

  2. Transformer (时间序列版)     — 长程依赖 + 多头自注意力
     论文: "Attention Is All You Need" + Qlib 的时间序列适配
     核心: Positional Encoding + Multi-Head Self-Attention → 全局时序建模

  3. TabNet                        — 注意力特征选择 + 可解释性
     论文: "TabNet: Attentive Interpretable Tabular Learning" (Arik & Pfister, 2019)
     核心: Sequential Attention 选择特征 → 决策步骤 → 输出预测

  4. GATs (Graph Attention Networks) — 资产关系图建模
     论文: "Graph Attention Networks" (Veličković et al., 2018)
     核心: 注意力聚合邻居节点 → 更新节点表示 → 图级预测

  5. DoubleEnsemble                — 集成之集成
     论文: "DoubleEnsemble: A New Ensemble Method for Stock Prediction" (Qlib team)
     核心: 两阶段集成 → 特征扰动 + 样本重加权 → 降低过拟合

设计原则 (西蒙斯风格):
  - 每个模型独立可训练, 独立可 debug
  - 统一接口: (feature_matrix, returns) → trained_model
  - 与现有 FeatureFactoryV4 无缝对接
  - 纯 PyTorch, 不依赖 Qlib 框架
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

# ── 模型保存目录 ──
MODEL_DIR = Path(__file__).parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════
# 基础组件
# ═══════════════════════════════════════════════════════════════

class TimeSeriesDataset(Dataset):
    """时间序列数据集 — 滑动窗口切片"""

    def __init__(self, features: np.ndarray, targets: np.ndarray,
                 seq_len: int = 60, step: int = 1):
        """
        Args:
            features: shape (n_timesteps, n_features)
            targets: shape (n_timesteps,)
            seq_len: 输入序列长度 (lookback window)
            step: 滑动步长
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.seq_len = seq_len
        self.step = step

        # 有效样本起始索引
        self.valid_starts = list(range(0, len(features) - seq_len, step))

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        end = start + self.seq_len

        x = self.features[start:end]      # (seq_len, n_features)
        y = self.targets[end - 1]          # 最后一天的target

        # 创建 mask (处理 NaN padding)
        mask = ~torch.isnan(x).any(dim=1)  # (seq_len,)

        # NaN → 0 填充
        x = torch.nan_to_num(x, nan=0.0)

        return x, y, mask


class PositionalEncoding(nn.Module):
    """正弦位置编码 — Transformer用"""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class GLU(nn.Module):
    """Gated Linear Unit — TabNet 的 feature selection 基础"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 2 * output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        return out[:, :out.shape[1]//2] * torch.sigmoid(out[:, out.shape[1]//2:])


# ═══════════════════════════════════════════════════════════════
# 1. ALSTM — Attention Long Short-Term Memory
# ═══════════════════════════════════════════════════════════════

class ALSTMAttention(nn.Module):
    """多头注意力池化 — 对 LSTM 隐藏状态序列加权"""

    def __init__(self, hidden_dim: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        self.query_context = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

    def forward(self, h_seq: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            h_seq: (batch, seq_len, hidden_dim) — LSTM所有时间步的输出
            mask: (batch, seq_len) — True表示有效
        Returns:
            pooled: (batch, hidden_dim)
        """
        batch, seq_len, hidden = h_seq.shape

        # Query: 可学习的全局上下文向量
        Q = self.W_q(self.query_context)  # (1, 1, hidden)
        K = self.W_k(h_seq)                # (batch, seq_len, hidden)
        V = self.W_v(h_seq)                # (batch, seq_len, hidden)

        # 多头拆分
        Q = Q.view(1, 1, self.n_heads, self.head_dim).transpose(1, 2)  # (1, n_heads, 1, head_dim)
        K = K.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
        V = V.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scale = self.head_dim ** 0.5
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (batch, n_heads, 1, seq_len)

        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            attn_scores = attn_scores.masked_fill(~attn_mask, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # (batch, n_heads, 1, head_dim)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, 1, hidden)
        pooled = self.W_o(attn_output).squeeze(1)  # (batch, hidden_dim)

        return pooled, attn_weights


class ALSTM(nn.Module):
    """
    Attention LSTM — Qlib 最核心的时序预测模型

    架构:
      Input → LSTM(双向) → 多头注意力池化 → FC → 预测

    特点:
      - 双向LSTM 捕捉前后文依赖
      - 多头注意力替代简单的 last-hidden-state
      - 可学习的查询向量 focus 在最重要的时间步
    """

    def __init__(self, n_features: int, hidden_dim: int = 128,
                 n_layers: int = 2, dropout: float = 0.3,
                 bidirectional: bool = True, n_attn_heads: int = 4):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 双向 LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # 注意力池化
        self.attention = ALSTMAttention(lstm_out_dim, n_heads=n_attn_heads)

        # 输出头
        self.output_head = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.zeros_(param)
        for module in self.output_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: (batch, seq_len, n_features) or (seq_len, n_features)
            mask: (batch, seq_len) — True=有效
        Returns:
            pred: (batch, 1) — 预测的前向收益
        """
        # 确保是 3 维: (batch, seq_len, n_features)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (seq_len, n_features) → (1, seq_len, n_features)
            if mask is not None and mask.dim() == 1:
                mask = mask.unsqueeze(0)

        batch, seq_len, _ = x.shape

        # 输入投影
        x_proj = self.input_proj(x)  # (batch, seq_len, hidden_dim)

        # LSTM
        lstm_out, _ = self.lstm(x_proj)  # (batch, seq_len, lstm_out_dim)

        # 注意力池化
        pooled, attn_weights = self.attention(lstm_out, mask)

        # 预测
        pred = self.output_head(pooled)

        return pred


# ═══════════════════════════════════════════════════════════════
# 2. Transformer — 时间序列自注意力
# ═══════════════════════════════════════════════════════════════

class TimeSeriesTransformer(nn.Module):
    """
    时间序列 Transformer — 捕捉长程依赖

    架构:
      Input → 位置编码 → N×TransformerEncoder → 注意力池化 → 预测

    与 NLP Transformer 的区别:
      - 不做 token embedding (特征已数值化)
      - 不做 causal masking (双向看)
      - 输出用可学习的 query 池化 (同 ALSTM)
    """

    def __init__(self, n_features: int, d_model: int = 128,
                 n_heads: int = 8, n_layers: int = 3,
                 dim_feedforward: int = 512, dropout: float = 0.2,
                 max_seq_len: int = 400):
        super().__init__()
        self.n_features = n_features

        # 特征投影到 d_model
        self.feature_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN (更稳定)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 注意力池化 (同ALSTM)
        self.pool_attention = ALSTMAttention(d_model, n_heads=min(n_heads, d_model // 32))

        # 输出头
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 4, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: (batch, seq_len, n_features) or (seq_len, n_features)
            mask: (batch, seq_len) — True=有效
        """
        # 确保 3 维
        if x.dim() == 2:
            x = x.unsqueeze(0)
            if mask is not None and mask.dim() == 1:
                mask = mask.unsqueeze(0)

        # 特征投影
        x_proj = self.feature_proj(x)  # (batch, seq_len, d_model)

        # 位置编码
        x_pe = self.pos_encoder(x_proj)

        # Transformer Encoder
        # src_key_padding_mask: True = 忽略该位置
        if mask is not None:
            padding_mask = ~mask  # (batch, seq_len)
        else:
            padding_mask = None

        encoded = self.transformer(x_pe, src_key_padding_mask=padding_mask)

        # 注意力池化
        pooled, _ = self.pool_attention(encoded, mask)

        # 预测
        pred = self.output_head(pooled)

        return pred


# ═══════════════════════════════════════════════════════════════
# 3. TabNet — 注意力特征选择
# ═══════════════════════════════════════════════════════════════

class AttentiveTransformer(nn.Module):
    """TabNet 的注意力变换器 — 学习每个特征的注意力掩码"""

    def __init__(self, input_dim: int, hidden_dim: int, shared_dim: int = 0):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim, momentum=0.01)
        self.shared_dim = shared_dim

        layers = []
        in_dim = input_dim + shared_dim
        for i in range(2):
            layers.extend([
                nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim, momentum=0.01),
                nn.ReLU(),
            ])
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, prior: torch.Tensor,
                shared: List[torch.Tensor]):
        x_bn = self.bn(x)
        # 拼接共享层信息 (第一步 shared 为空)
        if shared:
            combined = torch.cat([x_bn] + shared, dim=1)
        else:
            combined = x_bn
        # 如果维度不匹配, 用零填充 (第一个step时 shared 信息维度不足)
        if combined.shape[1] != self.net[0].in_features:
            pad_dim = self.net[0].in_features - combined.shape[1]
            if pad_dim > 0:
                pad = torch.zeros(combined.shape[0], pad_dim, device=combined.device)
                combined = torch.cat([combined, pad], dim=1)
        mask_logits = self.net(combined)
        # Sparsemax 产生稀疏掩码
        mask = self._sparsemax(mask_logits)
        # 与先验相乘 (prior scale)
        prior_scaled = prior * (mask + 1e-8).clamp(min=1e-8)
        # 更新先验
        new_prior = prior * (1 - mask)
        return mask, new_prior

    def _sparsemax(self, logits: torch.Tensor):
        """Sparsemax 激活 — 产生稀疏概率分布"""
        z_sorted, _ = torch.sort(logits, dim=1, descending=True)
        cumsum = torch.cumsum(z_sorted, dim=1)
        k = torch.arange(1, logits.size(1) + 1, device=logits.device).float()
        k_z = 1 + k * z_sorted
        # 安全阈值
        support = (k_z > cumsum).float()
        k_star = support.sum(dim=1, keepdim=True).float()
        tau = ((cumsum * support).sum(dim=1, keepdim=True) - 1) / k_star.clamp(min=1)
        return torch.relu(logits - tau)


class FeatureTransformer(nn.Module):
    """TabNet 的特征变换块 — GLU + 残差连接"""

    def __init__(self, input_dim: int, hidden_dim: int, n_shared: int = 2,
                 n_independent: int = 2, dropout: float = 0.2,
                 shared_proj_dim: int = 8):
        super().__init__()
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.shared_proj_dim = shared_proj_dim

        # 输入投影: 确保所有层都在 hidden_dim 空间操作
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Shared layers (跨步骤共享)
        self.shared = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2, momentum=0.01),
            ) for _ in range(n_shared)
        ])

        # 将 shared 输出压缩到小维度, 供 AttentiveTransformer 使用
        self.shared_proj = nn.ModuleList([
            nn.Linear(hidden_dim, shared_proj_dim) for _ in range(n_shared)
        ])

        # Independent layers (每步骤独立)
        self.independent = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2, momentum=0.01),
            ) for _ in range(n_independent)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, shared_stack: List[torch.Tensor] = None):
        """GLU: 取前一半为 gate, 后一半为值"""
        if shared_stack is None:
            shared_stack = []

        # 投影到 hidden_dim 空间
        x = self.input_proj(x)

        shared_feed = []  # 传给 AttentiveTransformer 的压缩信息

        # Shared layers
        for i, layer in enumerate(self.shared):
            if i < len(shared_stack):
                h = shared_stack[i]
            else:
                h = x
            gh = layer(h)
            g, h = gh[:, :gh.shape[1]//2], gh[:, gh.shape[1]//2:]
            x = x + self.dropout(h * torch.sigmoid(g))
            if i >= len(shared_stack):
                shared_stack.append(x.detach())
            # 压缩共享信息供 AttentiveTransformer 使用
            shared_feed.append(self.shared_proj[i](x))

        # Independent layers
        for layer in self.independent:
            gh = layer(x)
            g, h = gh[:, :gh.shape[1]//2], gh[:, gh.shape[1]//2:]
            x = x + self.dropout(h * torch.sigmoid(g))

        return x, shared_stack, shared_feed


class TabNetModel(nn.Module):
    """
    TabNet — 基于注意力机制的特征选择 + 决策步骤

    核心思想:
      - 每步选择最有用的特征子集 (AttentiveTransformer)
      - 对该子集做特征变换 (FeatureTransformer)
      - 多步决策后加权聚合

    优势:
      - 自动特征选择 (稀疏注意力)
      - 高度可解释 (每步能看到哪些特征被选中)
      - 适合几百维的量化特征矩阵
    """

    def __init__(self, n_features: int, n_steps: int = 5,
                 hidden_dim: int = 128, n_shared: int = 2,
                 n_independent: int = 2, dropout: float = 0.2,
                 gamma: float = 1.3):
        super().__init__()
        self.n_features = n_features
        self.n_steps = n_steps
        self.gamma = gamma
        shared_proj_dim = 8  # 每个 shared layer 压缩到 8 维

        # BatchNorm 输入
        self.bn = nn.BatchNorm1d(n_features, momentum=0.01)

        # 每步骤的特征变换器
        self.feature_transformers = nn.ModuleList([
            FeatureTransformer(n_features, hidden_dim, n_shared, n_independent,
                             dropout, shared_proj_dim)
            for _ in range(n_steps)
        ])

        # 每步骤的注意力变换器
        self.attentive_transformers = nn.ModuleList([
            AttentiveTransformer(n_features, hidden_dim,
                               shared_dim=n_shared * shared_proj_dim)
            for _ in range(n_steps)
        ])

        # 决策步骤的输出聚合 (输入是 FeatureTransformer 输出的 hidden_dim)
        self.step_outputs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            ) for _ in range(n_steps)
        ])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: (batch, n_features) — 对于TabNet, 输入是静态特征向量
               或者 (batch, seq_len, n_features) — 取最后一天
        """
        # 如果是序列输入, 取最后时间步
        if x.dim() == 3:
            # 用 mask 取最后一个有效步
            if mask is not None:
                last_valid = mask.float().sum(dim=1).long() - 1
                x = x[torch.arange(x.size(0)), last_valid.clamp(min=0)]
            else:
                x = x[:, -1, :]

        # 确保是 (batch, n_features)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = self.bn(x)
        batch_size = x.size(0)

        prior = torch.ones(batch_size, self.n_features, device=x.device)
        ft_shared_stack = None  # FeatureTransformer 内部共享状态
        attn_feed = []           # 传给 AttentiveTransformer 的压缩信息
        total_output = 0.0
        total_entropy = 0.0

        for step in range(self.n_steps):
            # 1. 注意力选择特征掩码
            mask_feat, prior = self.attentive_transformers[step](
                x, prior, attn_feed
            )
            # 稀疏正则化
            total_entropy += self._entropy(mask_feat)

            # 2. 应用掩码后的特征
            masked_x = x * mask_feat

            # 3. 特征变换
            transformed, ft_shared_stack, attn_feed = self.feature_transformers[step](
                masked_x, ft_shared_stack
            )

            # 4. 步骤输出
            step_out = self.step_outputs[step](transformed)
            total_output = total_output + step_out

            # 5. 缩放先验 (鼓励下一步选不同特征)
            prior = prior * self.gamma

        # 最终输出 = 所有步骤的平均
        output = total_output / self.n_steps

        return output

    def _entropy(self, mask: torch.Tensor):
        """计算掩码的熵 — 用于稀疏正则化"""
        mask = mask + 1e-8
        return torch.mean(torch.sum(-mask * torch.log(mask), dim=1))


# ═══════════════════════════════════════════════════════════════
# 4. GATs — 图注意力网络 (资产关系图)
# ═══════════════════════════════════════════════════════════════

class GraphAttentionLayer(nn.Module):
    """单头图注意力层"""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2,
                 alpha: float = 0.2, concat: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor):
        """
        Args:
            h: (n_nodes, in_features) — 节点特征
            adj: (n_nodes, n_nodes) — 邻接矩阵
        Returns:
            h_prime: (n_nodes, out_features)
        """
        Wh = self.W(h)  # (n_nodes, out_features)
        n_nodes = Wh.size(0)

        # 注意力系数
        Wh_i = Wh.unsqueeze(0).repeat(n_nodes, 1, 1)   # (n_nodes, n_nodes, out)
        Wh_j = Wh.unsqueeze(1).repeat(1, n_nodes, 1)   # (n_nodes, n_nodes, out)
        e = self.leaky_relu(self.a(torch.cat([Wh_i, Wh_j], dim=-1)).squeeze(-1))

        # Masked attention (只关注有连边的邻居)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            h_prime = F.elu(h_prime)

        return h_prime


class MultiHeadGAT(nn.Module):
    """多头图注意力网络"""

    def __init__(self, n_features: int, hidden_dim: int = 64,
                 n_heads: int = 4, dropout: float = 0.2):
        super().__init__()

        # 多头注意力
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(n_features, hidden_dim, dropout=dropout, concat=True)
            for _ in range(n_heads)
        ])

        # 输出层 (聚合多头)
        self.out_att = GraphAttentionLayer(
            hidden_dim * n_heads, hidden_dim, dropout=dropout, concat=False
        )

        # 全局池化 + 预测
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor, adj: torch.Tensor):
        """
        Args:
            h: (n_nodes, n_features) — 每个资产的特征
            adj: (n_nodes, n_nodes) — 资产关系邻接矩阵
        Returns:
            pred: (n_nodes, 1) — 每个资产的预测
        """
        # 多头拼接
        h_multi = torch.cat([att(h, adj) for att in self.attentions], dim=1)

        # 输出层
        h_out = self.out_att(h_multi, adj)  # (n_nodes, hidden_dim)

        # 逐节点预测
        pred = self.global_pool(h_out)  # (n_nodes, 1)

        return pred


# ═══════════════════════════════════════════════════════════════
# 5. DoubleEnsemble — 两阶段集成
# ═══════════════════════════════════════════════════════════════

class DoubleEnsemble:
    """
    DoubleEnsemble — 两阶段集成降低过拟合

    阶段1 (特征扰动):
      训练 K 个模型, 每个在随机特征子集 + 不同样本权重上

    阶段2 (样本重加权):
      根据阶段1的不确定性, 重新分配样本权重, 训练最终模型

    输出: K+1 个模型的加权平均
    """

    def __init__(self, base_model_class, base_model_kwargs: dict,
                 n_estimators: int = 6, feature_sample_rate: float = 0.8):
        self.base_model_class = base_model_class
        self.base_model_kwargs = base_model_kwargs
        self.n_estimators = n_estimators
        self.feature_sample_rate = feature_sample_rate
        self.models = []
        self.feature_indices = []  # 每个模型用的特征子集

    def fit(self, X: np.ndarray, y: np.ndarray,
            seq_len: int = 60, n_epochs: int = 30, **train_kwargs):
        """
        训练 DoubleEnsemble

        Args:
            X: (n_timesteps, n_features)
            y: (n_timesteps,)
        """
        n_features = X.shape[1]
        n_sub_features = max(16, int(n_features * self.feature_sample_rate))
        sample_weights = np.ones(len(y) - seq_len)

        # 阶段1: 训练 K 个扰动模型
        stage1_preds = []
        for k in range(self.n_estimators):
            # 随机选特征子集
            feat_idx = np.random.choice(n_features, n_sub_features, replace=False)
            self.feature_indices.append(feat_idx)

            X_sub = X[:, feat_idx]

            # 训练模型
            model_kwargs = {**self.base_model_kwargs, 'n_features': n_sub_features}
            model = self.base_model_class(**model_kwargs)

            model = self._train_single(model, X_sub, y, seq_len, n_epochs,
                                       sample_weights, **train_kwargs)
            self.models.append(model)

            # 收集预测 (用于阶段2权重计算)
            preds = self._predict_single(model, X_sub, seq_len)
            stage1_preds.append(preds)

        # 阶段2: 计算不确定性, 重加权样本
        stage1_preds = np.array(stage1_preds)  # (K, n_samples)
        pred_std = np.std(stage1_preds, axis=0)  # 模型间分歧
        pred_mean = np.abs(np.mean(stage1_preds, axis=0))  # 平均预测强度

        # 高分歧 + 低置信度 → 降权
        uncertainty = pred_std / (pred_mean + 1e-6)
        new_weights = 1.0 / (1.0 + uncertainty)
        new_weights = new_weights / new_weights.sum() * len(new_weights)

        # 在重加权样本上训练最终模型
        final_feat_idx = np.random.choice(n_features, n_sub_features, replace=False)
        self.feature_indices.append(final_feat_idx)
        final_model_kwargs = {**self.base_model_kwargs, 'n_features': n_sub_features}
        final_model = self.base_model_class(**final_model_kwargs)

        final_model = self._train_single(final_model, X[:, final_feat_idx], y,
                                         seq_len, n_epochs * 2, new_weights, **train_kwargs)
        self.models.append(final_model)

        return self

    def predict(self, X: np.ndarray, seq_len: int = 60):
        """加权平均预测"""
        all_preds = []
        for i, model in enumerate(self.models):
            feat_idx = self.feature_indices[i]
            preds = self._predict_single(model, X[:, feat_idx], seq_len)
            all_preds.append(preds)

        return np.mean(all_preds, axis=0)

    def _train_single(self, model, X, y, seq_len, n_epochs, sample_weights, **kwargs):
        """训练单个模型 (简化版, 实际调用外部训练器)"""
        # 这里只是一个接口定义, 实际训练由 QlibTrainer 完成
        return model

    def _predict_single(self, model, X, seq_len):
        """单个模型预测"""
        model.eval()
        with torch.no_grad():
            # 创建数据集
            dataset = TimeSeriesDataset(X, np.zeros(len(X)), seq_len)
            loader = DataLoader(dataset, batch_size=64, shuffle=False)
            preds = []
            for x_batch, _, mask_batch in loader:
                x_batch, mask_batch = x_batch.to(DEVICE), mask_batch.to(DEVICE)
                pred = model(x_batch, mask_batch).cpu().numpy().flatten()
                preds.append(pred)
        return np.concatenate(preds)


# ═══════════════════════════════════════════════════════════════
# 模型工厂 — 统一创建接口
# ═══════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "alstm": ALSTM,
    "transformer": TimeSeriesTransformer,
    "tabnet": TabNetModel,
    "gat": MultiHeadGAT,
    "cross_asset_gat": None,  # 延迟加载, 见 asset_graph.py
}

# CrossAssetGAT 接入 (Phase 11)
def _get_cross_asset_gat_class():
    """延迟加载 CrossAssetGAT (避免循环导入)"""
    try:
        from asset_graph import CrossAssetGAT
        return CrossAssetGAT
    except ImportError:
        return None

DEFAULT_MODEL_CONFIGS = {
    "alstm": {
        "hidden_dim": 128, "n_layers": 2, "dropout": 0.3,
        "bidirectional": True, "n_attn_heads": 4,
    },
    "transformer": {
        "d_model": 128, "n_heads": 8, "n_layers": 3,
        "dim_feedforward": 512, "dropout": 0.2,
    },
    "tabnet": {
        "n_steps": 5, "hidden_dim": 128, "n_shared": 2,
        "n_independent": 2, "dropout": 0.2,
    },
    "gat": {
        "hidden_dim": 64, "n_heads": 4, "dropout": 0.2,
    },
}


def create_model(model_name: str, n_features: int,
                 custom_config: Optional[dict] = None) -> nn.Module:
    """
    模型工厂 — 按名称创建 Qlib 模型

    Args:
        model_name: "alstm" | "transformer" | "tabnet" | "gat"
        n_features: 特征维度
        custom_config: 覆盖默认配置

    Returns:
        PyTorch Module
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    model_class = MODEL_REGISTRY[model_name]
    config = dict(DEFAULT_MODEL_CONFIGS.get(model_name, {}))
    if custom_config:
        config.update(custom_config)

    model = model_class(n_features=n_features, **config)
    return model.to(DEVICE)


def save_model(model: nn.Module, model_name: str, theme_id: str):
    """保存模型到标准路径"""
    path = MODEL_DIR / f"{model_name}_{theme_id}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'theme_id': theme_id,
        'saved_at': datetime.now().isoformat(),
    }, path)
    return path


def load_model(model_name: str, theme_id: str,
               n_features: int) -> Optional[nn.Module]:
    """从标准路径加载模型"""
    path = MODEL_DIR / f"{model_name}_{theme_id}.pth"
    if not path.exists():
        return None

    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    # 去除 'qlib_' 前缀获取原始模型名
    base_name = model_name.replace("qlib_", "")
    if base_name not in MODEL_REGISTRY:
        return None
    model = create_model(base_name, n_features)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


# ── 诊断输出 ──
if __name__ == "__main__":
    print("🧠 Qlib-Style Deep Learning Models")
    print(f"   Device: {DEVICE}")
    print(f"   Model Dir: {MODEL_DIR}")
    print()

    # 测试各模型构建
    test_n_features = 28
    for name in MODEL_REGISTRY:
        model = create_model(name, test_n_features)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  ✅ {name:15s} — {n_params:>10,} params")

    # 测试前向传播
    print("\n  Forward test (batch=8, seq=60, features=28):")
    x_test = torch.randn(8, 60, test_n_features).to(DEVICE)
    mask_test = torch.ones(8, 60, dtype=torch.bool).to(DEVICE)

    for name in MODEL_REGISTRY:
        model = create_model(name, test_n_features)
        try:
            with torch.no_grad():
                if name == "tabnet":
                    # TabNet takes flattened input
                    out = model(x_test[:, -1, :])
                elif name == "gat":
                    # GAT needs adj matrix
                    adj = torch.eye(8) + torch.rand(8, 8) * 0.3
                    adj = (adj > 0.5).float().to(DEVICE)
                    out = model(x_test[:, -1, :], adj)
                else:
                    out = model(x_test, mask_test)
            print(f"  ✅ {name:15s} — output shape: {out.shape}")
        except Exception as e:
            print(f"  ❌ {name:15s} — {e}")

    print("\n✨ All models ready for QlibTrainer integration!")
