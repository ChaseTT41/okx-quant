"""
Chase量化策略 — 订单执行优化引擎
拆单算法: TWAP / VWAP / Adaptive / Iceberg / Smart Router

Phase 13: 基于 Almgren-Chriss 框架改编的加密货币执行优化
  使用方式:
    python3 execution.py --estimate 0.5 BTC/USDT              # 预交易成本估算
    python3 execution.py --simulate BTC/USDT --side buy --qty 0.1 --strategy twap
    python3 execution.py --compare                             # 策略对比
    python3 execution.py --stats                               # 执行质量历史
"""
from __future__ import annotations
import sys
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np

DATA_DIR = Path(__file__).parent / "data"


# ═══════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════

@dataclass
class ExecutionConfig:
    """全局执行参数"""
    strategy: str = "smart"            # twap | vwap | adaptive | iceberg | smart
    horizon_minutes: int = 60          # 总执行时间
    n_slices: int = 0                  # 子订单数 (0=自动决定)
    participation_rate: float = 0.05   # 最大成交量占比
    max_spread_cost_bps: float = 10.0  # 最大可接受点差成本 (bps)
    max_market_impact_bps: float = 15.0 # 最大可接受冲击成本 (bps)
    urgency: float = 0.5               # 0=被动(省成本), 1=激进(抢成交)
    randomize_slices: bool = True      # 加随机抖动防侦测
    min_slice_interval_sec: int = 30   # 最小切片间隔
    risk_aversion: float = 1e-6        # Almgren-Chriss 风险厌恶系数


# ═══════════════════════════════════════════
# 市场冲击模型 (Almgren-Chriss 改编版)
# ═══════════════════════════════════════════

@dataclass
class MarketImpactModel:
    """
    Almgren-Chriss 框架改编 — 适用于加密货币

    原始模型:
      temporary_impact = eta * sigma * (X / (V * T)) ^ beta   (暂时冲击, 成交后恢复)
      permanent_impact = gamma * sigma * (X / V) ^ alpha       (永久冲击, 改变均衡价)

    改编说明 (crypto 特性):
      - 24/7 交易 → 无离散交易时段, 使用 rolling 24h volume
      - 高波动 → beta 偏低 (流动性提供者更积极)
      - 无最小报价单位 → 连续价格, 简化离散化

    Args:
        eta: 暂时冲击系数 (default 0.142, 校准自 BTC/ETH)
        gamma: 永久冲击系数 (default 0.035)
        beta_temp: 暂时冲击指数 (default 0.6, 股票通常 0.5-0.8)
        alpha_perm: 永久冲击指数 (default 0.891)
        sigma_annual: 年化波动率 (default 0.8 for crypto)
    """
    eta: float = 0.142
    gamma: float = 0.035
    beta_temp: float = 0.6
    alpha_perm: float = 0.891
    sigma_annual: float = 0.8

    def estimate_impact(self, order_size: float, avg_daily_volume: float,
                        volatility: float = None, spread: float = 0.001,
                        horizon_hours: float = 1.0) -> dict:
        """
        估算订单冲击成本

        Args:
            order_size: 订单规模 (base currency units, e.g. BTC)
            avg_daily_volume: 日均成交量 (base currency)
            volatility: 周期波动率 (默认从年化推算)
            spread: 买卖价差 (decimal, e.g. 0.001 = 10bps)
            horizon_hours: 执行时间窗口 (小时)

        Returns:
            {temporary_bps, permanent_bps, spread_bps, total_bps,
             slippage_pct, warning}
        """
        if avg_daily_volume <= 0:
            return {"temporary_bps": 999, "permanent_bps": 999,
                    "spread_bps": spread * 10000, "total_bps": 999,
                    "slippage_pct": 9.99, "warning": "成交量数据异常"}

        # 周期波动率
        if volatility is None:
            # 从年化推算到 horizon 窗口
            periods_per_year = 365 * 24 / horizon_hours
            volatility = self.sigma_annual / np.sqrt(periods_per_year)

        # 成交量占比
        volume_in_horizon = avg_daily_volume * (horizon_hours / 24)
        participation = order_size / max(volume_in_horizon, order_size * 0.01)
        daily_participation = order_size / avg_daily_volume

        # 暂时冲击 (成交时产生, 随流动性恢复而消退)
        temp_impact = self.eta * volatility * (participation ** self.beta_temp)

        # 永久冲击 (信息泄露, 改变市场均衡价)
        perm_impact = self.gamma * volatility * (daily_participation ** self.alpha_perm)

        # 点差成本 (一半点差, 假设被动成交)
        spread_cost = spread * 0.5

        # 转换为 bps
        temp_bps = temp_impact * 10000
        perm_bps = perm_impact * 10000
        spread_bps = spread_cost * 10000
        total_bps = temp_bps + perm_bps + spread_bps

        # 警告判断
        warning = None
        if total_bps > 50:
            warning = f"⚠️ 预估总成本 {total_bps:.1f}bps (>50bps), 建议增加切片数"
        elif total_bps > 20:
            warning = f"⚡ 预估总成本 {total_bps:.1f}bps (>20bps), 建议使用 Iceberg"
        elif participation > 0.1:
            warning = f"📊 参与率 {participation:.1%} (>10%), 建议延长执行时间"

        return {
            "temporary_bps": round(temp_bps, 2),
            "permanent_bps": round(perm_bps, 2),
            "spread_bps": round(spread_bps, 2),
            "total_bps": round(total_bps, 2),
            "slippage_pct": round(total_bps / 10000 * 100, 4),
            "participation_rate": round(participation, 4),
            "warning": warning,
        }

    def optimal_slices(self, order_size: float, avg_daily_volume: float,
                       volatility: float, risk_aversion: float = 1e-6,
                       horizon_hours: float = 1.0) -> int:
        """
        Almgren-Chriss 闭式解 → 最优切片数

        推导: 总成本 = 冲击成本 + 风险成本
          C(N) = eta*sigma*X*(X/(V*T))^beta * N^(beta-1)  [冲击 — 随N增大而减小]
               + lambda*sigma^2*X^2*T/(3*N^2)              [风险 — 随N增大而增大]

        最优 N = argmin C(N), 整数解

        Returns:
            最优切片数 (capped at 3-100)
        """
        if avg_daily_volume <= 0:
            return 5

        T = horizon_hours / 24  # 转换为天
        V = avg_daily_volume
        X = order_size
        sigma = volatility

        # 参与率决定最大切片数 (小单不需要太多切片)
        participation = X / V
        if participation < 0.001:
            max_n = 5    # 微小单: 最多5片
        elif participation < 0.01:
            max_n = 15   # 小单: 最多15片
        elif participation < 0.05:
            max_n = 40   # 中单
        else:
            max_n = 100  # 大单

        min_n = 1 if participation < 0.001 else 3

        # 数值搜索最优 N
        best_n = max(3, min_n + 1)
        best_cost = float("inf")

        for n in range(min_n, max_n + 1):
            # 每切片的成交量占比
            slice_participation = (X / n) / (V * T / n) if (V * T / n) > 0 else 0

            # 冲击成本
            impact = self.eta * sigma * X * (slice_participation ** self.beta_temp)

            # 风险成本 (未执行部分的价格风险)
            risk = 0.0
            for i in range(n):
                remaining = X * (1 - i / n)
                time_left = T * (1 - i / n)
                risk += remaining * remaining * sigma * sigma * time_left / n

            total_cost = impact + risk_aversion * risk

            if total_cost < best_cost:
                best_cost = total_cost
                best_n = n

        return max(min_n, min(max_n, best_n))


# ═══════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════

@dataclass
class OrderSlice:
    """单个子订单"""
    slice_id: int
    parent_id: str
    quantity: float
    price_limit: Optional[float] = None    # None = 市价单
    scheduled_at: float = 0.0              # 相对开始时间的秒数
    status: str = "pending"                # pending | filled | partial | skipped
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    slippage_bps: float = 0.0
    fill_time: Optional[str] = None


@dataclass
class ExecutionReport:
    """执行质量报告"""
    parent_id: str
    symbol: str
    side: str                              # buy | sell
    total_quantity: float
    arrival_price: float                   # 决策时中间价
    avg_execution_price: float             # 成交量加权平均成交价
    implementation_shortfall_bps: float    # IS = (avg_px - arrival) / arrival * 1e4
    vwap_slippage_bps: float               # vs 区间 VWAP
    market_impact_bps: float               # 估算冲击
    spread_cost_bps: float                 # 点差成本
    delay_cost_bps: float                  # 延迟成本
    fill_rate: float                       # 成交率
    duration_seconds: float                # 实际执行时间
    n_slices_total: int
    n_slices_filled: int
    slice_details: List[dict] = field(default_factory=list)
    strategy_used: str = "unknown"
    timestamp: str = ""
    notes: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ═══════════════════════════════════════════
# 拆单策略
# ═══════════════════════════════════════════

class TWAPSlicer:
    """
    TWAP — 时间加权平均价格

    将订单等分为 N 份, 在时间 T 内均匀执行
    最简单、最可预测, 适合:
      - 中小订单 (<1% ADV)
      - 低波动环境
      - 不想暴露意图的场景 (可加随机抖动)
    """

    @staticmethod
    def generate_schedule(total_qty: float, n_slices: int, horizon_min: float,
                          start_time: float = 0.0, randomize: bool = True,
                          parent_id: str = "") -> List[OrderSlice]:
        """
        生成 TWAP 切片计划

        Args:
            total_qty: 总数量
            n_slices: 切片数
            horizon_min: 总执行时间 (分钟)
            start_time: 开始时间偏移 (秒)
            randomize: 是否加随机抖动
            parent_id: 父订单 ID

        Returns:
            切片列表
        """
        if n_slices <= 0:
            n_slices = 5

        total_seconds = horizon_min * 60
        base_interval = total_seconds / n_slices
        base_qty = total_qty / n_slices

        slices = []
        for i in range(n_slices):
            # 随机抖动 (±20% 时间偏移, ±10% 数量偏移)
            if randomize:
                jitter_time = np.random.uniform(-0.2, 0.2) * base_interval
                jitter_qty = np.random.uniform(-0.1, 0.1) * base_qty
            else:
                jitter_time = 0.0
                jitter_qty = 0.0

            scheduled = start_time + i * base_interval + base_interval / 2 + jitter_time
            scheduled = max(0, scheduled)

            qty = max(0, base_qty + jitter_qty)

            slices.append(OrderSlice(
                slice_id=i + 1,
                parent_id=parent_id,
                quantity=qty,
                scheduled_at=scheduled,
            ))

        # 重新归一化数量 (确保总量精确)
        total_scheduled = sum(s.quantity for s in slices)
        if total_scheduled > 0:
            scale = total_qty / total_scheduled
            for s in slices:
                s.quantity *= scale

        return slices


class VWAPSlicer:
    """
    VWAP — 成交量加权平均价格

    按历史成交量曲线分配切片权重, 在流动性好的时段多成交
    适合:
      - 中等订单 (1-5% ADV)
      - 减少滑点
      - 跟市场节奏走

    Crypto 成交量特征 (UTC):
      - 01:00-08:00: 亚洲时段, 中等
      - 08:00-12:00: 欧洲开盘, 上升
      - 12:00-16:00: 欧美重叠, 最高
      - 16:00-20:00: 美国独市, 高
      - 20:00-01:00: 亚洲凌晨, 最低
    """

    # 标准 Crypto 24h 成交量分布曲线 (小时桶, 归一化)
    # 基于 BTC/ETH 实际数据拟合
    DEFAULT_VOLUME_PROFILE = np.array([
        0.65, 0.55, 0.50, 0.48,  # 00-04 UTC
        0.50, 0.55, 0.65, 0.80,  # 04-08 UTC
        1.00, 1.15, 1.25, 1.30,  # 08-12 UTC (欧洲开盘)
        1.40, 1.50, 1.45, 1.35,  # 12-16 UTC (欧美重叠, 峰值)
        1.30, 1.20, 1.10, 0.95,  # 16-20 UTC (美国独市)
        0.85, 0.75, 0.65, 0.60,  # 20-24 UTC (亚洲凌晨)
    ], dtype=np.float64)

    @classmethod
    def estimate_volume_profile(cls, symbol: str = None) -> np.ndarray:
        """
        返回 24 小时成交量分布曲线 (每小时一个桶)

        未来可扩展: 从交易所 API 拉取实际成交量曲线
        """
        profile = cls.DEFAULT_VOLUME_PROFILE.copy()

        # 特定币种微调
        if symbol:
            sym_upper = symbol.upper()
            if "BTC" in sym_upper:
                # BTC 全天交易活跃, 曲线更平
                profile = 0.7 + 0.3 * profile / profile.max()
            elif "ETH" in sym_upper:
                # ETH DeFi 时段 (UTC 12-20) 更活跃
                profile[12:20] *= 1.1

        # 归一化
        return profile / profile.sum()

    @classmethod
    def generate_schedule(cls, total_qty: float, n_slices: int, horizon_min: float,
                          volume_profile: np.ndarray = None, start_time: float = 0.0,
                          parent_id: str = "") -> List[OrderSlice]:
        """
        生成 VWAP 切片计划

        Args:
            total_qty: 总数量
            n_slices: 切片数
            horizon_min: 总执行时间 (分钟)
            volume_profile: 成交量分布 (如果为 None 则用默认曲线)
            start_time: 开始时间偏移 (秒)
            parent_id: 父订单 ID

        Returns:
            切片列表
        """
        if n_slices <= 0:
            n_slices = 10

        total_seconds = horizon_min * 60

        # 获取成交量曲线 (按切片插值)
        if volume_profile is None:
            volume_profile = cls.estimate_volume_profile()

        # 将 24h 成交量曲线映射到执行窗口
        # 假设执行从当前小时开始
        current_hour = datetime.now(timezone.utc).hour
        n_hours = int(np.ceil(horizon_min / 60))

        # 构建窗口内的成交量权重
        window_weights = []
        for i in range(n_slices):
            hour_in_window = (i / n_slices) * (horizon_min / 60)
            hour_of_day = (current_hour + hour_in_window) % 24
            # 线性插值
            h_floor = int(np.floor(hour_of_day)) % 24
            h_ceil = (h_floor + 1) % 24
            frac = hour_of_day - np.floor(hour_of_day)
            weight = (1 - frac) * volume_profile[h_floor] + frac * volume_profile[h_ceil]
            window_weights.append(weight)

        window_weights = np.array(window_weights)
        if window_weights.sum() > 0:
            window_weights = window_weights / window_weights.sum()
        else:
            window_weights = np.ones(n_slices) / n_slices

        slices = []
        for i in range(n_slices):
            qty = total_qty * window_weights[i]
            # 时间均匀分布 + 量随成交量
            scheduled = start_time + (i + 0.5) * (total_seconds / n_slices)

            slices.append(OrderSlice(
                slice_id=i + 1,
                parent_id=parent_id,
                quantity=float(qty),
                scheduled_at=scheduled,
            ))

        # 归一化
        total_scheduled = sum(s.quantity for s in slices)
        if total_scheduled > 0:
            scale = total_qty / total_scheduled
            for s in slices:
                s.quantity *= scale

        return slices


class AdaptiveSlicer:
    """
    Adaptive — 实现缺口最小化 (Implementation Shortfall)

    根据市场条件动态调整:
      - 高 urgency → 前面多切 (抢成交, 减少延迟风险)
      - 低 urgency → 后面多切 (省冲击成本)
      - 高波动 → 前面多切 (减少波动暴露)
      - 高 spread → 后面多切 (等流动性改善)

    适用: 大单 (>5% ADV), 高波动环境
    """

    @staticmethod
    def generate_schedule(total_qty: float, n_slices: int, horizon_min: float,
                          urgency: float = 0.5, volatility: float = 0.02,
                          spread: float = 0.0005, avg_daily_vol: float = 1e8,
                          parent_id: str = "") -> List[OrderSlice]:
        """
        生成自适应切片计划

        权重分配逻辑:
          slice_weight[i] = base_weight[i] * urgency_bias * vol_bias * spread_bias

        base_weight: 指数衰减 (urgency > 0.5) 或 指数递增 (urgency < 0.5)
        urgency_bias: 前半段额外加权
        vol_bias: 高波动时前半段多切
        spread_bias: 高 spread 时后半段多切
        """
        if n_slices <= 0:
            n_slices = 10

        total_seconds = horizon_min * 60

        # 基础权重: 根据 urgency 决定前后分配
        i = np.arange(1, n_slices + 1)
        if urgency > 0.5:
            # 前多后少 (指数衰减)
            decay_rate = urgency * 3
            base_weights = np.exp(-decay_rate * (i - 1) / n_slices)
        elif urgency < 0.3:
            # 前少后多 (指数递增)
            growth_rate = (1 - urgency) * 3
            base_weights = np.exp(growth_rate * (i - 1) / n_slices)
        else:
            # 均匀
            base_weights = np.ones(n_slices)

        # 波动率调整: 高波动时前面多切 (减少不确定性暴露)
        vol_factor = 1.0 + max(0, (volatility - 0.01) / 0.05)  # >1% vol 开始前移
        vol_bias = np.ones(n_slices)
        for j in range(n_slices):
            vol_bias[j] = 1.0 + (vol_factor - 1.0) * (1 - j / n_slices)

        # 点差调整: 高 spread 时后面多切 (等流动性)
        spread_factor = 1.0 + max(0, (spread - 0.0005) / 0.005)
        spread_bias = np.ones(n_slices)
        for j in range(n_slices):
            spread_bias[j] = 1.0 + (spread_factor - 1.0) * (j / n_slices)

        # 合成权重
        weights = base_weights * vol_bias * spread_bias
        weights = weights / weights.sum()

        # 生成切片
        slices = []
        for j in range(n_slices):
            qty = total_qty * weights[j]
            scheduled = (j + 0.5) * (total_seconds / n_slices)

            slices.append(OrderSlice(
                slice_id=j + 1,
                parent_id=parent_id,
                quantity=float(qty),
                scheduled_at=scheduled,
            ))

        return slices


class IcebergSlicer:
    """
    Iceberg — 冰山订单

    只暴露很小的可见量, 隐藏真实意图
    每一片成交后再暴露下一片

    适合:
      - 超大单 (>10% ADV)
      - 薄盘 (高 spread, 低深度)
      - 不想被侦测的订单
    """

    @staticmethod
    def generate_schedule(total_qty: float, visible_qty: float = None,
                          horizon_min: float = 60, n_slices: int = 0,
                          parent_id: str = "") -> List[OrderSlice]:
        """
        生成冰山切片计划

        Args:
            total_qty: 总数量
            visible_qty: 每次暴露的数量 (None = 自动, 默认 2% ADV)
            horizon_min: 最长执行时间
            n_slices: 最大切片数 (0 = 自动)
            parent_id: 父订单 ID

        Returns:
            切片列表 (每片 = visible_qty, 直到总量)
        """
        if visible_qty is None or visible_qty <= 0:
            visible_qty = total_qty * 0.1  # 默认每次暴露 10%

        if n_slices <= 0:
            n_slices = max(3, int(np.ceil(total_qty / visible_qty)))
            n_slices = min(n_slices, 100)  # cap

        total_seconds = horizon_min * 60

        # 冰山切片: 每片大小 = visible_qty, 最后一片 = 余量
        slices = []
        remaining = total_qty
        slice_id = 0

        # 实际需要的切片数
        actual_n = min(n_slices, int(np.ceil(total_qty / visible_qty)))

        for i in range(actual_n):
            slice_id += 1
            qty = min(visible_qty, remaining)
            remaining -= qty

            if qty <= 0:
                break

            # 时间间隔不均匀 — 模拟"等待成交→暴露下一片"
            # 实际中每片等待时间取决于市场流动性
            if actual_n > 1:
                # 前面的片快一些 (信号还没泄露)
                time_frac = i / (actual_n - 1) if actual_n > 1 else 0.5
                scheduled = time_frac * total_seconds
                # 加随机等待 (模拟成交不确定性)
                if i > 0:
                    scheduled += np.random.uniform(0, total_seconds / actual_n * 0.5)
            else:
                scheduled = total_seconds / 2

            slices.append(OrderSlice(
                slice_id=slice_id,
                parent_id=parent_id,
                quantity=qty,
                scheduled_at=min(scheduled, total_seconds),
            ))

        if remaining > 0:
            # 最后一点零头并到最后一片
            slices[-1].quantity += remaining

        return slices


class SmartOrderRouter:
    """
    智能路由 — 自动选择最优策略

    决策逻辑:
      订单大小 vs ADV        → 决定基础策略
      价差 (spread)          → 高 spread → Iceberg
      波动率 (volatility)    → 高波动 → Adaptive
      紧急度 (urgency)       → 高 urgency → Adaptive 前载

    规则表:
      | 订单/ADV  | 默认策略   | 高spread   | 高波动     | 高urgency  |
      |-----------|-----------|-----------|-----------|-----------|
      | <0.1%     | 单笔市价   | 单笔      | TWAP 3片  | 单笔      |
      | 0.1-1%    | TWAP 5片  | Iceberg   | Adaptive  | Adaptive  |
      | 1-5%      | VWAP 10片 | Iceberg   | Adaptive  | Adaptive  |
      | >5%       | Adaptive  | Iceberg   | Adaptive  | Iceberg   |
    """

    @staticmethod
    def route(total_qty: float, avg_daily_volume: float, spread: float,
              volatility: float, urgency: float = 0.5,
              symbol: str = "") -> Tuple[str, dict]:
        """
        自动选择策略

        Returns:
            (strategy_name, strategy_params)
        """
        if avg_daily_volume <= 0:
            return ("twap", {"n_slices": 5})

        participation = total_qty / avg_daily_volume

        high_spread = spread > 0.002  # >20bps
        high_vol = volatility > 0.03   # >3% daily
        high_urgency = urgency > 0.7

        if participation < 0.001:
            # 微小单: 直接成交
            if high_spread or high_vol:
                return ("twap", {"n_slices": 3})
            return ("single", {"n_slices": 1})

        elif participation < 0.01:
            # 小单
            if high_spread:
                return ("iceberg", {"visible_pct": 0.05})
            elif high_urgency or high_vol:
                return ("adaptive", {"n_slices": 5})
            else:
                return ("twap", {"n_slices": 5})

        elif participation < 0.05:
            # 中单
            if high_spread:
                return ("iceberg", {"visible_pct": 0.03})
            elif high_urgency:
                return ("adaptive", {"n_slices": 15})
            else:
                return ("vwap", {"n_slices": 10})

        else:
            # 大单
            if high_urgency:
                return ("iceberg", {"visible_pct": 0.02, "n_slices": 20})
            elif high_spread:
                return ("iceberg", {"visible_pct": 0.02})
            else:
                return ("adaptive", {"n_slices": 20})

    @staticmethod
    def generate_schedule(total_qty: float, avg_daily_volume: float,
                          spread: float, volatility: float, urgency: float = 0.5,
                          symbol: str = "", horizon_min: float = 60,
                          parent_id: str = "") -> Tuple[str, List[OrderSlice]]:
        """
        路由 + 生成完整切片计划

        Returns:
            (strategy_name, slice_schedule)
        """
        strategy, params = SmartOrderRouter.route(
            total_qty, avg_daily_volume, spread, volatility, urgency, symbol
        )

        n_slices = params.get("n_slices", 10)

        if strategy == "single":
            slices = [OrderSlice(
                slice_id=1, parent_id=parent_id,
                quantity=total_qty, scheduled_at=0.0
            )]
        elif strategy == "twap":
            slices = TWAPSlicer.generate_schedule(
                total_qty, n_slices, horizon_min, parent_id=parent_id
            )
        elif strategy == "vwap":
            slices = VWAPSlicer.generate_schedule(
                total_qty, n_slices, horizon_min, parent_id=parent_id
            )
        elif strategy == "adaptive":
            slices = AdaptiveSlicer.generate_schedule(
                total_qty, n_slices, horizon_min, urgency=urgency,
                volatility=volatility, spread=spread,
                avg_daily_vol=avg_daily_volume, parent_id=parent_id
            )
        elif strategy == "iceberg":
            visible_pct = params.get("visible_pct", 0.05)
            visible_qty = total_qty * visible_pct
            slices = IcebergSlicer.generate_schedule(
                total_qty, visible_qty=visible_qty,
                horizon_min=horizon_min, n_slices=n_slices,
                parent_id=parent_id
            )
        else:
            slices = TWAPSlicer.generate_schedule(
                total_qty, n_slices, horizon_min, parent_id=parent_id
            )

        return strategy, slices


# ═══════════════════════════════════════════
# 执行引擎
# ═══════════════════════════════════════════

class ExecutionEngine:
    """
    订单执行总控

    在纸交易模式下, 模拟切片执行过程:
      1. 预交易分析 → 选策略 + 生成切片
      2. 按时间顺序模拟每片成交 (含仿真滑点)
      3. 汇总执行质量报告
    """

    def __init__(self, config: ExecutionConfig = None,
                 impact_model: MarketImpactModel = None):
        self.config = config or ExecutionConfig()
        self.impact_model = impact_model or MarketImpactModel()

    def pre_trade_estimate(self, order_size: float, symbol: str,
                           market_data: dict = None) -> dict:
        """
        预交易成本估算

        Args:
            order_size: 订单量 (base units)
            symbol: 交易对
            market_data: {price, avg_daily_volume, volatility, spread}

        Returns:
            {optimal_strategy, recommended_slices, est_costs, warning_flags}
        """
        if market_data is None:
            market_data = {}

        price = market_data.get("price", 0)
        adv = market_data.get("avg_daily_volume", 1e6)
        vol = market_data.get("volatility", 0.02)
        spread = market_data.get("spread", 0.001)

        # 冲击估算
        impact = self.impact_model.estimate_impact(
            order_size, adv, vol, spread,
            horizon_hours=self.config.horizon_minutes / 60
        )

        # 最优切片数
        optimal_n = self.impact_model.optimal_slices(
            order_size, adv, vol,
            risk_aversion=self.config.risk_aversion,
            horizon_hours=self.config.horizon_minutes / 60
        )

        # 策略推荐
        strategy, params = SmartOrderRouter.route(
            order_size, adv, spread, vol, self.config.urgency, symbol
        )

        # 费用估算
        fee_rate = 0.001  # 0.1% taker fee
        notional = order_size * price if price > 0 else 0
        est_fee = notional * fee_rate
        est_slippage = notional * impact["total_bps"] / 10000
        est_total_cost = est_fee + est_slippage

        return {
            "symbol": symbol,
            "order_size": order_size,
            "notional_value": round(notional, 2),
            "avg_daily_volume": adv,
            "optimal_strategy": strategy,
            "recommended_slices": optimal_n,
            "strategy_params": params,
            "impact_estimate": impact,
            "est_fee": round(est_fee, 2),
            "est_slippage": round(est_slippage, 2),
            "est_total_cost": round(est_total_cost, 2),
            "est_total_cost_bps": round(impact["total_bps"], 2),
            "warning_flags": impact.get("warning"),
        }

    def execute_paper(self, symbol: str, side: str, quantity: float,
                      price: float, market_data: dict,
                      pf=None) -> ExecutionReport:
        """
        纸交易模拟执行

        Args:
            symbol: 交易对 (e.g. BTC/USDT)
            side: buy | sell
            quantity: 总数量
            price: 当前价格 (arrival price)
            market_data: {avg_daily_volume, volatility, spread, ...}
            pf: PortfolioManager (可选, 传入则自动更新持仓)

        Returns:
            ExecutionReport
        """
        parent_id = f"exec_{uuid.uuid4().hex[:12]}"
        adv = market_data.get("avg_daily_volume", 1e6)
        vol = market_data.get("volatility", 0.02)
        spread = market_data.get("spread", 0.001)

        # 1. 选择策略 + 生成切片
        if self.config.strategy == "smart":
            strategy_name, slices = SmartOrderRouter.generate_schedule(
                quantity, adv, spread, vol, self.config.urgency,
                symbol, self.config.horizon_minutes, parent_id
            )
        elif self.config.strategy == "twap":
            n = self.config.n_slices or 10
            slices = TWAPSlicer.generate_schedule(
                quantity, n, self.config.horizon_minutes,
                randomize=self.config.randomize_slices, parent_id=parent_id
            )
            strategy_name = "twap"
        elif self.config.strategy == "vwap":
            n = self.config.n_slices or 10
            slices = VWAPSlicer.generate_schedule(
                quantity, n, self.config.horizon_minutes, parent_id=parent_id
            )
            strategy_name = "vwap"
        elif self.config.strategy == "adaptive":
            n = self.config.n_slices or 15
            slices = AdaptiveSlicer.generate_schedule(
                quantity, n, self.config.horizon_minutes,
                urgency=self.config.urgency, volatility=vol,
                spread=spread, avg_daily_vol=adv, parent_id=parent_id
            )
            strategy_name = "adaptive"
        elif self.config.strategy == "iceberg":
            n = self.config.n_slices or 20
            slices = IcebergSlicer.generate_schedule(
                quantity, horizon_min=self.config.horizon_minutes,
                n_slices=n, parent_id=parent_id
            )
            strategy_name = "iceberg"
        else:
            n = self.config.n_slices or 10
            slices = TWAPSlicer.generate_schedule(
                quantity, n, self.config.horizon_minutes, parent_id=parent_id
            )
            strategy_name = "twap"

        # 2. 模拟每片成交
        total_filled = 0.0
        total_notional = 0.0
        filled_slices = 0
        slice_details = []
        total_slippage_bps = 0.0

        # 预交易冲击估算
        impact_est = self.impact_model.estimate_impact(
            quantity, adv, vol, spread,
            horizon_hours=self.config.horizon_minutes / 60
        )

        for sl in slices:
            # 模拟成交 (在当前价格上加仿真滑点)
            fill_price, filled_qty, sl_bps = self._simulate_slice_fill(
                sl, price, side, vol, spread
            )

            sl.filled_qty = filled_qty
            sl.avg_fill_price = fill_price
            sl.slippage_bps = sl_bps
            sl.status = "filled" if filled_qty > 0 else "skipped"
            sl.fill_time = datetime.now(timezone.utc).isoformat()

            total_filled += filled_qty
            total_notional += fill_price * filled_qty
            total_slippage_bps += sl_bps * filled_qty  # weighted

            if filled_qty > 0:
                filled_slices += 1

            slice_details.append({
                "slice_id": sl.slice_id,
                "scheduled_at": sl.scheduled_at,
                "quantity": sl.quantity,
                "filled_qty": filled_qty,
                "fill_price": round(fill_price, 2),
                "slippage_bps": round(sl_bps, 2),
                "status": sl.status,
            })

            # 更新 Portfolio (如果传入)
            if pf is not None and filled_qty > 0:
                self._update_portfolio(pf, symbol, side, filled_qty, fill_price, parent_id)

        # 3. 计算执行质量
        avg_px = total_notional / total_filled if total_filled > 0 else price

        # Implementation Shortfall
        if side == "buy":
            is_bps = (avg_px / price - 1) * 10000 if price > 0 else 0
        else:
            is_bps = (1 - avg_px / price) * 10000 if price > 0 else 0

        # VWAP slippage (简化: 用 arrival price 近似)
        vwap_slip = is_bps * 0.8  # 通常 IS < VWAP slippage

        # 成本分解
        spread_cost = spread * 0.5 * 10000  # 半spread
        market_impact = impact_est["total_bps"] * 0.6
        delay_cost = max(0, abs(is_bps) - spread_cost - market_impact)

        fill_rate = total_filled / quantity if quantity > 0 else 1.0
        duration = slices[-1].scheduled_at if slices else 0

        # 4. 构建报告
        report = ExecutionReport(
            parent_id=parent_id,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            arrival_price=price,
            avg_execution_price=round(avg_px, 4),
            implementation_shortfall_bps=round(is_bps, 2),
            vwap_slippage_bps=round(vwap_slip, 2),
            market_impact_bps=round(market_impact, 2),
            spread_cost_bps=round(spread_cost, 2),
            delay_cost_bps=round(delay_cost, 2),
            fill_rate=round(fill_rate, 4),
            duration_seconds=duration,
            n_slices_total=len(slices),
            n_slices_filled=filled_slices,
            slice_details=slice_details,
            strategy_used=strategy_name,
        )

        # 5. 保存
        store = ExecutionStore()
        store.save(report)

        return report

    def execute_live(self, symbol: str, side: str, quantity: float,
                     price: float, market_data: dict,
                     exchange_trader) -> ExecutionReport:
        """
        实盘执行 — 通过交易所 API 真实下单 (Binance / OKX)

        与 execute_paper() 同签名，复用:
          - 预交易分析（策略选择、拆片）
          - 执行层风控检查
          - ExecutionReport 格式

        Args:
            symbol: 交易对 (e.g. BTC/USDT)
            side: buy | sell
            quantity: 总数量
            price: 当前价格 (arrival price)
            market_data: {avg_daily_volume, volatility, spread, ...}
            exchange_trader: BinanceLiveTrader 实例 (支持 Binance/OKX)

        Returns:
            ExecutionReport
        """
        from binance_live import LiveTradingBlockedError, LiveAPIError

        parent_id = f"live_{uuid.uuid4().hex[:12]}"
        adv = market_data.get("avg_daily_volume", 1e6)
        vol = market_data.get("volatility", 0.02)
        spread = market_data.get("spread", 0.001)

        # 1. 预交易分析（复用现有逻辑）
        impact_est = self.impact_model.estimate_impact(
            quantity, adv, vol, spread,
            horizon_hours=self.config.horizon_minutes / 60
        )

        # 2. 执行层风控检查
        from risk import RiskController
        rc = RiskController(None)  # 实盘模式不需要 paper portfolio
        exec_check = rc.execution_risk_check(
            quantity, adv, spread, vol,
            n_slices=self.config.n_slices or 5
        )
        if not exec_check.passed:
            # 风控不通过 → 拒绝执行
            report = ExecutionReport(
                parent_id=parent_id, symbol=symbol, side=side,
                total_quantity=quantity, arrival_price=price,
                avg_execution_price=price,
                implementation_shortfall_bps=0, vwap_slippage_bps=0,
                market_impact_bps=impact_est.get("total_bps", 0),
                spread_cost_bps=spread * 0.5 * 10000,
                delay_cost_bps=0, fill_rate=0,
                duration_seconds=0, n_slices_total=0, n_slices_filled=0,
                strategy_used="rejected",
                notes=f"执行层风控拦截: {exec_check.reason}"
            )
            return report

        # 3. 实盘下单
        start_time = datetime.now(timezone.utc)
        try:
            # 计算 USDT 金额
            amount_usdt = quantity * price if price > 0 else 0

            if amount_usdt < 10:
                # 低于最小交易额
                report = ExecutionReport(
                    parent_id=parent_id, symbol=symbol, side=side,
                    total_quantity=quantity, arrival_price=price,
                    avg_execution_price=price,
                    implementation_shortfall_bps=0, vwap_slippage_bps=0,
                    market_impact_bps=0, spread_cost_bps=0,
                    delay_cost_bps=0, fill_rate=0,
                    duration_seconds=0, n_slices_total=0, n_slices_filled=0,
                    strategy_used="rejected",
                    notes=f"金额 ${amount_usdt:.2f} 低于最小交易额 $10"
                )
                return report

            # 执行市价单
            if side == "buy":
                result = exchange_trader.market_buy(
                    symbol, amount_usdt,
                    note=f"[{parent_id}] {self.config.strategy.upper()} 实盘执行"
                )
            else:
                result = exchange_trader.market_sell(
                    symbol, quantity,
                    note=f"[{parent_id}] {self.config.strategy.upper()} 实盘执行"
                )

            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            avg_px = result.price if result.price > 0 else price
            fill_rate = 1.0 if result.is_filled else 0.0

            # Implementation Shortfall
            if side == "buy":
                is_bps = (avg_px / price - 1) * 10000 if price > 0 else 0
            else:
                is_bps = (1 - avg_px / price) * 10000 if price > 0 else 0

            spread_cost = spread * 0.5 * 10000
            market_impact = max(0, abs(is_bps) - spread_cost)

            report = ExecutionReport(
                parent_id=parent_id, symbol=symbol, side=side,
                total_quantity=quantity, arrival_price=price,
                avg_execution_price=round(avg_px, 4),
                implementation_shortfall_bps=round(is_bps, 2),
                vwap_slippage_bps=round(is_bps * 0.9, 2),
                market_impact_bps=round(market_impact, 2),
                spread_cost_bps=round(spread_cost, 2),
                delay_cost_bps=0,
                fill_rate=round(fill_rate, 4),
                duration_seconds=elapsed,
                n_slices_total=1, n_slices_filled=1 if fill_rate > 0 else 0,
                slice_details=[{
                    "slice_id": 1, "quantity": quantity,
                    "filled_qty": result.quantity if fill_rate > 0 else 0,
                    "fill_price": round(avg_px, 2),
                    "status": "filled" if fill_rate > 0 else "failed",
                    "order_id": result.order_id,
                }],
                strategy_used=f"live_{self.config.strategy}",
                notes=f"订单ID: {result.order_id} | 手续费: {result.fee:.4f} {result.fee_currency}"
            )

            # 保存执行记录
            store = ExecutionStore()
            store.save(report)

            return report

        except LiveTradingBlockedError as e:
            # 紧急停止/熔断 — 返回拒绝报告
            report = ExecutionReport(
                parent_id=parent_id, symbol=symbol, side=side,
                total_quantity=quantity, arrival_price=price,
                avg_execution_price=price,
                implementation_shortfall_bps=0, vwap_slippage_bps=0,
                market_impact_bps=0, spread_cost_bps=0,
                delay_cost_bps=0, fill_rate=0,
                duration_seconds=0, n_slices_total=0, n_slices_filled=0,
                strategy_used="blocked",
                notes=f"交易被阻止: {e}"
            )
            return report

        except LiveAPIError as e:
            # API 错误 — 返回失败报告
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            report = ExecutionReport(
                parent_id=parent_id, symbol=symbol, side=side,
                total_quantity=quantity, arrival_price=price,
                avg_execution_price=price,
                implementation_shortfall_bps=0, vwap_slippage_bps=0,
                market_impact_bps=0, spread_cost_bps=0,
                delay_cost_bps=0, fill_rate=0,
                duration_seconds=elapsed, n_slices_total=0, n_slices_filled=0,
                strategy_used="failed",
                notes=f"API错误: {e}"
            )
            return report

        except Exception as e:
            # 未知错误
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            print(f"  ❌ 实盘执行异常: {e}")
            report = ExecutionReport(
                parent_id=parent_id, symbol=symbol, side=side,
                total_quantity=quantity, arrival_price=price,
                avg_execution_price=price,
                implementation_shortfall_bps=0, vwap_slippage_bps=0,
                market_impact_bps=0, spread_cost_bps=0,
                delay_cost_bps=0, fill_rate=0,
                duration_seconds=elapsed, n_slices_total=0, n_slices_filled=0,
                strategy_used="error",
                notes=f"未知错误: {e}"
            )
            return report

    def execute_swap_live(self, symbol: str, side: str, quantity_contracts: float,
                           price: float, leverage: int, market_data: dict,
                           leverage_engine, stop_loss_price: float = None,
                           take_profit_price: float = None) -> dict:
        """
        ⚡ 实盘合约执行 — 通过 OKX API 下永续合约单。

        与 execute_live() 共享相同的预交易分析和风控，但:
          - 使用 swaps 端点 (set_leverage + 合约下单)
          - 杠杆风险检查
          - 止盈止损作为 algo order

        Returns: dict with order_id, status, margin, notional
        """
        parent_id = f"swap_{uuid.uuid4().hex[:12]}"
        adv = market_data.get("avg_daily_volume", 1e6)

        try:
            # 1. 杠杆风控
            from risk import RiskController
            rc = RiskController(None)
            lev_check = rc.leverage_pre_trade_check(
                symbol=symbol,
                margin_usdt=quantity_contracts * price / leverage,
                leverage=leverage,
                total_equity=market_data.get("total_equity", 1000),
                funding_rate=market_data.get("funding_rate", 0.0),
            )
            if not lev_check.passed:
                return {
                    "ok": False, "error": f"杠杆风控: {lev_check.reason}",
                    "parent_id": parent_id,
                }

            # 2. 下合约单
            order = leverage_engine.create_swap_market_order(
                symbol=symbol,
                side=side,
                quantity_contracts=quantity_contracts,
                leverage=leverage,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                note=parent_id[:20],
            )

            if not order:
                return {
                    "ok": False, "error": "合约下单返回空",
                    "parent_id": parent_id,
                }

            return {
                "ok": True,
                "parent_id": parent_id,
                "order_id": order.get("id", ""),
                "symbol": symbol,
                "side": side,
                "quantity_contracts": quantity_contracts,
                "leverage": leverage,
                "notional_usdt": round(quantity_contracts * price, 2),
                "margin_usdt": round(quantity_contracts * price / leverage, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            return {
                "ok": False, "error": str(e),
                "parent_id": parent_id,
            }

    def _simulate_slice_fill(self, slice_obj: OrderSlice, current_price: float,
                             side: str, volatility: float, spread: float) -> Tuple[float, float, float]:
        """
        模拟单切片成交

        Returns:
            (fill_price, filled_qty, slippage_bps)
        """
        # 价格随机游走 (微结构噪声)
        micro_noise = np.random.normal(0, volatility * current_price * 0.1)

        # 买卖方向偏差
        if side == "buy":
            # 买入: 价格向上偏移 (流动性消耗)
            direction_bias = abs(np.random.normal(0, spread * current_price * 0.3))
            fill_price = current_price + direction_bias + micro_noise * 0.5
        else:
            direction_bias = abs(np.random.normal(0, spread * current_price * 0.3))
            fill_price = current_price - direction_bias + micro_noise * 0.5

        fill_price = max(fill_price, current_price * 0.95)
        fill_price = min(fill_price, current_price * 1.05)

        # 成交量: 假设大部分全成交
        fill_prob = 0.95  # 95% 概率全成交
        if np.random.random() < fill_prob:
            filled_qty = slice_obj.quantity
        else:
            filled_qty = slice_obj.quantity * np.random.uniform(0.5, 0.9)

        # 滑点 (bps)
        if side == "buy":
            slippage_bps = (fill_price / current_price - 1) * 10000
        else:
            slippage_bps = (1 - fill_price / current_price) * 10000

        return fill_price, filled_qty, slippage_bps

    def _update_portfolio(self, pf, symbol: str, side: str,
                          quantity: float, price: float, parent_id: str):
        """更新虚拟盘持仓 (分笔)"""
        try:
            from portfolio import PortfolioManager
            if not hasattr(pf, 'buy_partial'):
                # 降级: 用标准 buy
                if side == "buy":
                    name = symbol.replace("/USDT", "")
                    pf.buy(
                        market="crypto", symbol=symbol, name=name,
                        price=price, quantity=quantity,
                        reason=f"[{parent_id}] sliced fill"
                    )
        except Exception:
            pass  # 静默失败, 纸交易模式


# ═══════════════════════════════════════════
# 执行记录存储
# ═══════════════════════════════════════════

class ExecutionStore:
    """执行质量分析存储"""

    def __init__(self, store_dir: Path = None):
        self.store_dir = store_dir or (DATA_DIR / "executions")
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.store_dir / "index.json"

    def save(self, report: ExecutionReport) -> Path:
        """保存执行报告"""
        filepath = self.store_dir / f"{report.parent_id}.json"

        data = {
            "parent_id": report.parent_id,
            "symbol": report.symbol,
            "side": report.side,
            "total_quantity": report.total_quantity,
            "arrival_price": report.arrival_price,
            "avg_execution_price": report.avg_execution_price,
            "implementation_shortfall_bps": report.implementation_shortfall_bps,
            "vwap_slippage_bps": report.vwap_slippage_bps,
            "market_impact_bps": report.market_impact_bps,
            "spread_cost_bps": report.spread_cost_bps,
            "delay_cost_bps": report.delay_cost_bps,
            "fill_rate": report.fill_rate,
            "duration_seconds": report.duration_seconds,
            "n_slices_total": report.n_slices_total,
            "n_slices_filled": report.n_slices_filled,
            "strategy_used": report.strategy_used,
            "timestamp": report.timestamp,
            "notes": report.notes,
            "slice_details": report.slice_details,
        }
        filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False))

        # 更新索引
        self._update_index(report)

        return filepath

    def _update_index(self, report: ExecutionReport):
        """更新执行记录索引"""
        index = []
        if self._index_file.exists():
            try:
                index = json.loads(self._index_file.read_text())
            except (json.JSONDecodeError, FileNotFoundError):
                index = []

        index.append({
            "parent_id": report.parent_id,
            "symbol": report.symbol,
            "side": report.side,
            "strategy_used": report.strategy_used,
            "implementation_shortfall_bps": report.implementation_shortfall_bps,
            "fill_rate": report.fill_rate,
            "timestamp": report.timestamp,
        })

        # 只保留最近 500 条
        if len(index) > 500:
            index = index[-500:]

        self._index_file.write_text(json.dumps(index, indent=2, ensure_ascii=False))

    def load(self, n: int = 50) -> List[dict]:
        """加载最近 N 条执行记录"""
        reports = []
        if not self._index_file.exists():
            return reports

        try:
            index = json.loads(self._index_file.read_text())
            for entry in index[-n:]:
                filepath = self.store_dir / f"{entry['parent_id']}.json"
                if filepath.exists():
                    reports.append(json.loads(filepath.read_text()))
        except (json.JSONDecodeError, FileNotFoundError):
            pass

        return reports

    def get_stats(self, symbol: str = None) -> dict:
        """汇总执行质量统计"""
        reports = self.load(100)
        if symbol:
            reports = [r for r in reports if r["symbol"] == symbol]

        if not reports:
            return {"n_reports": 0, "message": "暂无执行记录"}

        shortfalls = [r["implementation_shortfall_bps"] for r in reports]
        fill_rates = [r["fill_rate"] for r in reports]
        durations = [r["duration_seconds"] for r in reports]

        return {
            "n_reports": len(reports),
            "avg_shortfall_bps": round(np.mean(shortfalls), 2),
            "median_shortfall_bps": round(np.median(shortfalls), 2),
            "max_shortfall_bps": round(max(shortfalls), 2),
            "avg_fill_rate": round(np.mean(fill_rates), 4),
            "avg_duration_sec": round(np.mean(durations), 1),
            "best_execution_bps": round(min(shortfalls), 2),
            "worst_execution_bps": round(max(shortfalls), 2),
        }

    def get_strategy_comparison(self) -> List[dict]:
        """各策略执行质量对比"""
        reports = self.load(200)
        if not reports:
            return []

        strategies = {}
        for r in reports:
            s = r["strategy_used"]
            if s not in strategies:
                strategies[s] = []
            strategies[s].append(r)

        comparison = []
        for strategy, reps in strategies.items():
            shortfalls = [rp["implementation_shortfall_bps"] for rp in reps]
            fill_rates = [rp["fill_rate"] for rp in reps]
            comparison.append({
                "strategy": strategy,
                "n_trades": len(reps),
                "avg_shortfall_bps": round(np.mean(shortfalls), 2),
                "median_shortfall_bps": round(np.median(shortfalls), 2),
                "std_shortfall_bps": round(np.std(shortfalls), 2),
                "avg_fill_rate": round(np.mean(fill_rates), 4),
            })

        # 按平均 shortfall 排序 (越小越好)
        comparison.sort(key=lambda x: x["avg_shortfall_bps"])
        return comparison


# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════

def _format_bps(bps: float) -> str:
    """格式化 bps 显示"""
    if bps > 50:
        return f"🔴 {bps:.1f} bps"
    elif bps > 20:
        return f"🟡 {bps:.1f} bps"
    else:
        return f"🟢 {bps:.1f} bps"


def _print_separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chase量化策略 · 订单执行优化")
    parser.add_argument("--estimate", type=str, metavar="QTY SYMBOL",
                       nargs=2, help="预交易成本估算 (e.g. --estimate 0.5 BTC/USDT)")
    parser.add_argument("--simulate", type=str, metavar="SYMBOL",
                       help="模拟执行订单")
    parser.add_argument("--side", type=str, default="buy", choices=["buy", "sell"])
    parser.add_argument("--qty", type=float, default=0.1, help="订单数量")
    parser.add_argument("--strategy", type=str, default="smart",
                       choices=["twap", "vwap", "adaptive", "iceberg", "smart"])
    parser.add_argument("--horizon", type=int, default=60, help="执行时间窗口 (分钟)")
    parser.add_argument("--slices", type=int, default=0, help="切片数 (0=自动)")
    parser.add_argument("--urgency", type=float, default=0.5, help="紧急度 0-1")
    parser.add_argument("--price", type=float, default=0, help="当前价格 (0=使用模拟价)")
    parser.add_argument("--compare", action="store_true", help="策略执行质量对比")
    parser.add_argument("--stats", action="store_true", help="执行质量统计")
    parser.add_argument("--list", action="store_true", help="最近执行记录")

    args = parser.parse_args()

    # ── 预交易估算 ──
    if args.estimate:
        qty_str, symbol = args.estimate
        try:
            qty = float(qty_str)
        except ValueError:
            print(f"❌ 无效数量: {qty_str}")
            sys.exit(1)

        # 模拟市场数据 (实际应从交易所获取)
        mock_prices = {
            "BTC/USDT": {"price": 87000, "adv": 35000, "vol": 0.025, "spread": 0.0002},
            "ETH/USDT": {"price": 3200, "adv": 500000, "vol": 0.030, "spread": 0.0003},
            "SOL/USDT": {"price": 180, "adv": 8000000, "vol": 0.045, "spread": 0.0005},
            "BNB/USDT": {"price": 620, "adv": 500000, "vol": 0.028, "spread": 0.0004},
            "XRP/USDT": {"price": 2.5, "adv": 200000000, "vol": 0.035, "spread": 0.0008},
        }
        mdata = mock_prices.get(symbol, {"price": 100, "adv": 1e6, "vol": 0.03, "spread": 0.001})

        engine = ExecutionEngine()
        est = engine.pre_trade_estimate(qty, symbol, mdata)

        _print_separator(f"📊 预交易成本估算 — {qty} {symbol}")
        print(f"  名义价值:        ${est['notional_value']:,.0f}")
        print(f"  推荐策略:        {est['optimal_strategy'].upper()}")
        print(f"  推荐切片数:      {est['recommended_slices']}")
        print(f"  预估冲击成本:    {_format_bps(est['impact_estimate']['total_bps'])}")
        print(f"    - 暂时冲击:    {est['impact_estimate']['temporary_bps']:.1f} bps")
        print(f"    - 永久冲击:    {est['impact_estimate']['permanent_bps']:.1f} bps")
        print(f"    - 点差成本:    {est['impact_estimate']['spread_bps']:.1f} bps")
        print(f"  预估手续费:      ${est['est_fee']:.2f}")
        print(f"  预估滑点损失:    ${est['est_slippage']:.2f}")
        print(f"  预估总成本:      ${est['est_total_cost']:.2f} ({est['est_total_cost_bps']:.1f} bps)")
        if est["warning_flags"]:
            print(f"  ⚠️  {est['warning_flags']}")

    # ── 模拟执行 ──
    elif args.simulate:
        symbol = args.simulate
        mock_prices = {
            "BTC/USDT": {"price": 87000, "adv": 35000, "vol": 0.025, "spread": 0.0002},
            "ETH/USDT": {"price": 3200, "adv": 500000, "vol": 0.030, "spread": 0.0003},
            "SOL/USDT": {"price": 180, "adv": 8000000, "vol": 0.045, "spread": 0.0005},
            "BNB/USDT": {"price": 620, "adv": 500000, "vol": 0.028, "spread": 0.0004},
            "XRP/USDT": {"price": 2.5, "adv": 200000000, "vol": 0.035, "spread": 0.0008},
        }
        mdata = mock_prices.get(symbol, {"price": 100, "adv": 1e6, "vol": 0.03, "spread": 0.001})
        price = args.price if args.price > 0 else mdata["price"]

        config = ExecutionConfig(
            strategy=args.strategy,
            horizon_minutes=args.horizon,
            n_slices=args.slices,
            urgency=args.urgency,
        )
        engine = ExecutionEngine(config=config)

        _print_separator(f"🔪 模拟执行 — {args.qty} {symbol} [{args.strategy.upper()}]")

        report = engine.execute_paper(
            symbol=symbol, side=args.side, quantity=args.qty,
            price=price, market_data=mdata
        )

        print(f"  父订单 ID:       {report.parent_id}")
        print(f"  策略:            {report.strategy_used.upper()}")
        print(f"  切片:            {report.n_slices_filled}/{report.n_slices_total} 成交")
        print(f"  成交率:          {report.fill_rate:.1%}")
        print(f"  Arrival Price:   ${report.arrival_price:,.2f}")
        print(f"  Avg Exec Price:  ${report.avg_execution_price:,.2f}")
        print(f"  ─────────────────────────────────────────")
        print(f"  IS (缺口):       {_format_bps(report.implementation_shortfall_bps)}")
        print(f"  VWAP 滑点:       {_format_bps(report.vwap_slippage_bps)}")
        print(f"  冲击成本:        {report.market_impact_bps:.1f} bps")
        print(f"  点差成本:        {report.spread_cost_bps:.1f} bps")
        print(f"  延迟成本:        {report.delay_cost_bps:.1f} bps")
        print(f"  ─────────────────────────────────────────")
        print(f"  执行时间:        {report.duration_seconds:.0f}s")

        # 切片明细
        if report.slice_details:
            print(f"\n  📋 切片明细:")
            print(f"  {'#':<4} {'Qty':>10} {'Fill Qty':>10} {'Fill Px':>12} {'Slip':>8}")
            for sl in report.slice_details[:10]:
                print(f"  {sl['slice_id']:<4} {sl['quantity']:>10.4f} {sl['filled_qty']:>10.4f} "
                      f"${sl['fill_price']:>11.2f} {sl['slippage_bps']:>7.1f}bps")

    # ── 策略对比 ──
    elif args.compare:
        store = ExecutionStore()
        comparison = store.get_strategy_comparison()

        _print_separator("📊 策略执行质量对比")
        if not comparison:
            print("  📭 暂无执行记录, 请先运行 --simulate")
        else:
            print(f"  {'策略':<12} {'交易数':>6} {'Avg IS':>10} {'Med IS':>10} {'Std IS':>10} {'Fill%':>8}")
            print(f"  {'─'*60}")
            for c in comparison:
                print(f"  {c['strategy']:<12} {c['n_trades']:>6} {c['avg_shortfall_bps']:>9.1f}bps "
                      f"{c['median_shortfall_bps']:>9.1f}bps {c['std_shortfall_bps']:>9.1f}bps "
                      f"{c['avg_fill_rate']:>7.1%}")

    # ── 执行统计 ──
    elif args.stats:
        store = ExecutionStore()
        stats = store.get_stats()

        _print_separator("📈 执行质量统计")
        if stats.get("n_reports", 0) == 0:
            print("  📭 暂无执行记录")
        else:
            print(f"  总执行次数:      {stats['n_reports']}")
            print(f"  平均 IS:         {_format_bps(stats['avg_shortfall_bps'])}")
            print(f"  中位 IS:         {_format_bps(stats['median_shortfall_bps'])}")
            print(f"  最大 IS:         {_format_bps(stats['max_shortfall_bps'])}")
            print(f"  最佳执行:        {_format_bps(stats['best_execution_bps'])}")
            print(f"  平均成交率:      {stats['avg_fill_rate']:.1%}")
            print(f"  平均执行时间:    {stats['avg_duration_sec']:.0f}s")

    # ── 最近记录 ──
    elif args.list:
        store = ExecutionStore()
        reports = store.load(20)

        _print_separator("📋 最近执行记录")
        if not reports:
            print("  📭 暂无执行记录")
        else:
            for r in reversed(reports):
                is_str = _format_bps(r["implementation_shortfall_bps"])
                print(f"  {r['timestamp'][:19]} | {r['symbol']:>10} | {r['side']:<4} | "
                      f"{r['strategy_used']:<10} | IS={is_str} | "
                      f"Fill={r['fill_rate']:.0%} | {r['n_slices_filled']}/{r['n_slices_total']} slices")

    # ── 默认: demo ──
    else:
        print("🐾 Chase量化策略 · 订单执行优化引擎 v1.0")
        print()
        print("用法:")
        print("  python3 execution.py --estimate 0.5 BTC/USDT    预交易成本估算")
        print("  python3 execution.py --simulate BTC/USDT --qty 0.1 --strategy twap")
        print("  python3 execution.py --simulate BTC/USDT --qty 0.5 --strategy smart")
        print("  python3 execution.py --compare                   策略对比")
        print("  python3 execution.py --stats                     执行质量统计")
        print("  python3 execution.py --list                      最近记录")
        print()
        print("策略: twap | vwap | adaptive | iceberg | smart")
        print()

        # 跑一个快速 demo
        print("🎬 快速 Demo: 估算 0.1 BTC 买入成本...")
        print()
        engine = ExecutionEngine()
        est = engine.pre_trade_estimate(0.1, "BTC/USDT", {
            "price": 87000, "avg_daily_volume": 35000,
            "volatility": 0.025, "spread": 0.0002,
        })
        print(f"  推荐策略:   {est['optimal_strategy'].upper()}")
        print(f"  推荐切片:   {est['recommended_slices']} 片")
        print(f"  预估成本:   {est['est_total_cost_bps']:.1f} bps (~${est['est_total_cost']:.2f})")
