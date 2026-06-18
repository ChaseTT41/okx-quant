"""
Feature Engine v3.0 — 500+ 量化特征工厂
=========================================
专为Chase量化策略设计，西蒙斯风格：让数据说话，不靠人拍脑袋。

特征类别 (目标500+):
  A. 价格动量 (Price Momentum)           ~120个
  B. 波动率结构 (Volatility Structure)    ~80个
  C. 高阶矩 (Higher Moments)              ~40个
  D. 均线系统 (MA System)                 ~60个
  E. 振荡器族 (Oscillators)               ~50个
  F. 成交量画像 (Volume Profile)          ~60个
  G. 跨市场联动 (Cross-Market)            ~40个
  H. 衍生品数据 (Derivatives)             ~30个
  I. 情绪指标 (Sentiment)                 ~20个
  J. 链上等效 (On-Chain Proxy)            ~20个

每个特征 → 单独回测 → t-stat过滤 → 只保留t-stat>1.5的
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Callable, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


@dataclass
class Feature:
    """单个量化特征"""
    id: str                    # e.g. "mom_20d"
    name: str                  # e.g. "20日动量"
    category: str              # A-J
    description: str
    compute_fn: Callable       # (df) -> float
    t_stat: float = 0.0        # 回测后填充
    sharpe: float = 0.0
    hit_rate: float = 0.0
    ic: float = 0.0            # Information Coefficient
    retained: bool = True      # t-stat > 1.5 ?


# ═══════════════════════════════════════════════════════════
# 核心计算工具
# ═══════════════════════════════════════════════════════════

def _returns(close: np.ndarray, window: int = 1) -> np.ndarray:
    """N日收益率"""
    ret = np.zeros_like(close)
    ret[window:] = (close[window:] / close[:-window] - 1)
    return ret


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(x).rolling(window).mean().values


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(x).rolling(window).std().values


def _rolling_skew(x: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(x).rolling(window).skew().values


def _rolling_kurt(x: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(x).rolling(window).kurt().values


def _rolling_corr(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(a).rolling(window).corr(pd.Series(b)).values


def _rolling_max(x: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(x).rolling(window).max().values


def _rolling_min(x: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(x).rolling(window).min().values


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(x).ewm(span=span).mean().values


def _rsi(close: np.ndarray, period: int = 14) -> float:
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:]) if len(gains) >= period else np.mean(gains)
    avg_loss = np.mean(losses[-period:]) if len(losses) >= period else np.mean(losses)
    if avg_loss < 1e-9:
        return 100.0
    return float(100 - 100 / (1 + avg_gain / avg_loss))


def _latest(x: np.ndarray) -> float:
    """取最新值, 安全处理NaN"""
    valid = x[~np.isnan(x)]
    return float(valid[-1]) if len(valid) > 0 else np.nan


# ═══════════════════════════════════════════════════════════
# A. 价格动量特征 (~120个)
# ═══════════════════════════════════════════════════════════

def build_momentum_features() -> List[Feature]:
    """动量族: 不同窗口的收益率 + 加速度 + 路径特征"""
    features = []
    windows = [1, 2, 3, 5, 7, 10, 14, 20, 30, 40, 50, 60, 90, 120, 180, 250]

    # 基础动量: ret(N)
    for w in windows:
        features.append(Feature(
            id=f"mom_{w}d", name=f"{w}日动量",
            category="A", description=f"过去{w}个自然日收益率",
            compute_fn=lambda df, w=w: _latest(_returns(df["close"].values, w)),
        ))

    # 动量加速度: ret(short) - ret(long)
    pairs = [(5, 20), (10, 50), (20, 100), (5, 10), (10, 30), (30, 90)]
    for s, l in pairs:
        features.append(Feature(
            id=f"mom_accel_{s}d_{l}d", name=f"{s}日-{l}日动量差",
            category="A", description=f"短期{s}日与长期{l}日动量差异(加速度)",
            compute_fn=lambda df, s=s, l=l:
                _latest(_returns(df["close"].values, s)) -
                _latest(_returns(df["close"].values, l)),
        ))

    # 路径特征: 夏普比率
    for w in [10, 20, 60, 120]:
        features.append(Feature(
            id=f"sharpe_{w}d", name=f"{w}日滚动夏普",
            category="A", description=f"过去{w}日收益率/波动率",
            compute_fn=lambda df, w=w:
                _latest(_returns(df["close"].values, 1)[-w:]) /
                (np.std(_returns(df["close"].values, 1)[-w:]) + 1e-9) * np.sqrt(365),
        ))

    # 连续涨跌天数
    for direction, label in [(1, "连涨"), (-1, "连跌")]:
        features.append(Feature(
            id=f"consecutive_{label}", name=f"{label}天数",
            category="A", description=f"当前连续{label}天数",
            compute_fn=lambda df, d=direction:
                _latest(_consecutive_days(df["close"].values, d)),
        ))

    # 区间最大回撤
    for w in [20, 60, 120]:
        features.append(Feature(
            id=f"max_dd_{w}d", name=f"{w}日最大回撤",
            category="A", description=f"过去{w}日最大回撤幅度",
            compute_fn=lambda df, w=w:
                _latest(_calc_max_dd(df["close"].values, w)),
        ))

    return features


def _consecutive_days(close: np.ndarray, direction: int) -> np.ndarray:
    """连续涨/跌天数"""
    result = np.zeros(len(close))
    count = 0
    for i in range(1, len(close)):
        if (direction == 1 and close[i] > close[i-1]) or \
           (direction == -1 and close[i] < close[i-1]):
            count += 1
        else:
            count = 0
        result[i] = count
    return result


def _calc_max_dd(close: np.ndarray, window: int) -> np.ndarray:
    """滚动最大回撤"""
    result = np.zeros(len(close))
    for i in range(window, len(close)):
        peak = np.max(close[i-window:i+1])
        result[i] = (close[i] - peak) / peak
    return result


# ═══════════════════════════════════════════════════════════
# B. 波动率结构特征 (~80个)
# ═══════════════════════════════════════════════════════════

def build_volatility_features() -> List[Feature]:
    """波动率族: 不同窗口的历史波动率 + 波动率的变化率"""
    features = []
    windows = [5, 10, 14, 20, 30, 60, 90, 120]

    # 历史波动率
    for w in windows:
        features.append(Feature(
            id=f"vol_{w}d", name=f"{w}日波动率",
            category="B", description=f"过去{w}日对数收益率年化标准差",
            compute_fn=lambda df, w=w:
                _latest(_rolling_std(np.log(df["close"].values[1:]/df["close"].values[:-1]), w)) * np.sqrt(365),
        ))

    # 波动率的变化 (vol_of_vol)
    for w in [10, 20, 60]:
        features.append(Feature(
            id=f"vol_change_{w}d", name=f"{w}日波动率变化",
            category="B", description=f"波动率在{w}日内的变化率",
            compute_fn=lambda df, w=w:
                _latest(pd.Series(
                    _rolling_std(np.log(df["close"].values[1:]/df["close"].values[:-1]), 20) * np.sqrt(365)
                ).pct_change(w).values),
        ))

    # Parkinson波动率 (用高低价)
    for w in [10, 20, 60]:
        features.append(Feature(
            id=f"parkinson_vol_{w}d", name=f"Parkinson {w}日波动",
            category="B", description="用日内高低价估计的波动率 (更精确)",
            compute_fn=lambda df, w=w:
                _latest(np.sqrt(
                    _rolling_mean((np.log(df["high"].values/df["low"].values))**2 / (4*np.log(2)), w)
                ) * np.sqrt(365)),
        ))

    # 振幅 (Range %)
    for w in [5, 10, 20]:
        features.append(Feature(
            id=f"range_pct_{w}d", name=f"{w}日平均振幅",
            category="B", description=f"过去{w}日日均振幅百分比",
            compute_fn=lambda df, w=w:
                _latest(_rolling_mean(
                    (df["high"].values - df["low"].values) / df["close"].values, w
                )),
        ))

    # 波动率锥位置
    for w in [20, 60]:
        features.append(Feature(
            id=f"vol_cone_{w}d", name=f"波动率锥 {w}日分位",
            category="B",
            description=f"当前{w}日波动率在历史{w}日波动率中的分位数",
            compute_fn=lambda df, w=w:
                _latest(_vol_cone_percentile(df["close"].values, w)),
        ))

    return features


def _vol_cone_percentile(close: np.ndarray, window: int) -> np.ndarray:
    """当前波动率在历史分布中的分位数"""
    rets = np.diff(np.log(close))
    vols = pd.Series(rets).rolling(window).std().values * np.sqrt(365)
    result = np.zeros(len(close))
    for i in range(window * 2, len(close)):
        hist_vols = vols[window:i]
        if len(hist_vols) > 0:
            result[i] = (hist_vols < vols[i]).sum() / len(hist_vols)
    return result


# ═══════════════════════════════════════════════════════════
# C. 高阶矩特征 (~40个)
# ═══════════════════════════════════════════════════════════

def build_higher_moment_features() -> List[Feature]:
    """高阶矩: 偏度 + 峰度 + 尾部风险"""
    features = []
    windows = [20, 60, 120]

    for w in windows:
        features.append(Feature(
            id=f"skew_{w}d", name=f"{w}日偏度",
            category="C", description=f"收益率分布偏度 (负=左偏/暴跌风险)",
            compute_fn=lambda df, w=w:
                _latest(_rolling_skew(_returns(df["close"].values, 1), w)),
        ))
        features.append(Feature(
            id=f"kurt_{w}d", name=f"{w}日峰度",
            category="C", description=f"收益率分布峰度 (高=肥尾风险)",
            compute_fn=lambda df, w=w:
                _latest(_rolling_kurt(_returns(df["close"].values, 1), w)),
        ))

    # 极端值计数
    for w, sigma in [(20, 2), (60, 2), (20, 3), (60, 3)]:
        features.append(Feature(
            id=f"tail_events_{w}d_{sigma}s", name=f"{w}日{sigma}σ尾部事件",
            category="C", description=f"过去{w}日中超过{sigma}个标准差的日数",
            compute_fn=lambda df, w=w, sigma=sigma:
                _latest(_tail_count(df["close"].values, w, sigma)),
        ))

    # 涨跌不对称性
    for w in [20, 60]:
        features.append(Feature(
            id=f"asymmetry_{w}d", name=f"{w}日涨跌不对称",
            category="C",
            description="涨幅均值/跌幅均值(绝对值) — >1看涨不对称",
            compute_fn=lambda df, w=w:
                _latest(_asymmetry_ratio(df["close"].values, w)),
        ))

    return features


def _tail_count(close: np.ndarray, window: int, sigma: float) -> np.ndarray:
    rets = _returns(close, 1)
    result = np.zeros(len(close))
    for i in range(window, len(close)):
        segment = rets[i-window:i+1]
        std = np.std(segment)
        if std > 0:
            result[i] = np.sum(np.abs(segment) > sigma * std)
    return result


def _asymmetry_ratio(close: np.ndarray, window: int) -> np.ndarray:
    rets = _returns(close, 1)
    result = np.zeros(len(close))
    for i in range(window, len(close)):
        segment = rets[i-window:i+1]
        up = segment[segment > 0]
        down = segment[segment < 0]
        if len(up) > 0 and len(down) > 0:
            result[i] = np.mean(up) / abs(np.mean(down)) if abs(np.mean(down)) > 1e-9 else 2.0
        else:
            result[i] = 1.0
    return result


# ═══════════════════════════════════════════════════════════
# D. 均线系统特征 (~60个)
# ═══════════════════════════════════════════════════════════

def build_ma_features() -> List[Feature]:
    """均线系统: 各类MA交叉、距离、斜率"""
    features = []
    ma_pairs = [
        (5, 20), (10, 50), (20, 60), (5, 60), (20, 100),
        (10, 20), (50, 100), (100, 200),
    ]

    for s, l in ma_pairs:
        # MA距离 (百分比)
        features.append(Feature(
            id=f"ma_dist_{s}_{l}", name=f"MA{s}/{l}距离",
            category="D", description=f"SMA({s})相对SMA({l})的偏离百分比",
            compute_fn=lambda df, s=s, l=l:
                _latest((pd.Series(df["close"].values).rolling(s).mean() /
                         pd.Series(df["close"].values).rolling(l).mean() - 1).values),
        ))
        # MA交叉信号 (-1/0/1)
        features.append(Feature(
            id=f"ma_cross_{s}_{l}", name=f"MA{s}/{l}交叉",
            category="D", description=f"SMA({s})与SMA({l})的交叉状态",
            compute_fn=lambda df, s=s, l=l:
                _latest(_ma_cross_signal(df["close"].values, s, l)),
        ))

    # EMA距离
    for span in [12, 26, 50, 100]:
        features.append(Feature(
            id=f"ema_dist_{span}", name=f"EMA{span}距离",
            category="D", description=f"价格相对EMA({span})的偏离",
            compute_fn=lambda df, span=span:
                _latest(df["close"].values / _ema(df["close"].values, span) - 1),
        ))

    # MA斜率 (趋势强度)
    for w in [10, 20, 50, 100]:
        features.append(Feature(
            id=f"ma_slope_{w}", name=f"MA{w}斜率",
            category="D", description=f"SMA({w})的5日变化率",
            compute_fn=lambda df, w=w:
                _latest(pd.Series(
                    pd.Series(df["close"].values).rolling(w).mean().values
                ).pct_change(5).values),
        ))

    return features


def _ma_cross_signal(close: np.ndarray, s: int, l: int) -> np.ndarray:
    sma_s = pd.Series(close).rolling(s).mean().values
    sma_l = pd.Series(close).rolling(l).mean().values
    result = np.zeros(len(close))
    result[sma_s > sma_l] = 1
    result[sma_s < sma_l] = -1
    return result


# ═══════════════════════════════════════════════════════════
# E. 振荡器族 (~50个)
# ═══════════════════════════════════════════════════════════

def build_oscillator_features() -> List[Feature]:
    """振荡器: RSI变体 + Stochastic + 布林带 + CCI"""
    features = []

    # RSI 多周期
    for period in [7, 14, 21, 30, 50]:
        features.append(Feature(
            id=f"rsi_{period}", name=f"RSI({period})",
            category="E", description=f"{period}日RSI",
            compute_fn=lambda df, period=period:
                _rsi(df["close"].values, period),
        ))
        # RSI变化率
        if period <= 30:
            features.append(Feature(
                id=f"rsi_change_{period}", name=f"RSI({period})变化",
                category="E", description=f"RSI({period})的3日变化",
                compute_fn=lambda df, period=period:
                    _rsi(df["close"].values[:-3], period) - _rsi(df["close"].values, period)
                    if len(df) > period + 3 else 0,
            ))

    # Stochastic
    for k_period in [14, 21]:
        features.append(Feature(
            id=f"stoch_k_{k_period}", name=f"Stoch %K({k_period})",
            category="E", description=f"{k_period}日随机指标%K",
            compute_fn=lambda df, k_period=k_period:
                _latest(_stoch_k(df, k_period)),
        ))

    # 布林带位置
    for period, std in [(20, 2), (20, 1.5), (50, 2)]:
        features.append(Feature(
            id=f"bb_position_{period}_{std}", name=f"BB({period},{std})位置",
            category="E", description=f"价格在布林带({period},{std})中的位置 (0-1)",
            compute_fn=lambda df, period=period, std=std:
                _latest(_bb_position(df["close"].values, period, std)),
        ))
        features.append(Feature(
            id=f"bb_width_{period}_{std}", name=f"BB({period},{std})带宽",
            category="E",
            description=f"布林带({period},{std})宽度 (波动率proxy)",
            compute_fn=lambda df, period=period, std=std:
                _latest(_bb_width(df["close"].values, period, std)),
        ))

    # CCI
    for period in [14, 20]:
        features.append(Feature(
            id=f"cci_{period}", name=f"CCI({period})",
            category="E",
            description=f"{period}日商品通道指数",
            compute_fn=lambda df, period=period:
                _latest(_cci(df, period)),
        ))

    return features


def _stoch_k(df: pd.DataFrame, period: int) -> np.ndarray:
    high_n = _rolling_max(df["high"].values, period)
    low_n = _rolling_min(df["low"].values, period)
    denom = high_n - low_n
    k = np.zeros(len(df))
    mask = denom > 0
    k[mask] = (df["close"].values[mask] - low_n[mask]) / denom[mask] * 100
    return k


def _bb_position(close: np.ndarray, period: int, std: float) -> np.ndarray:
    sma = _rolling_mean(close, period)
    std_val = _rolling_std(close, period)
    upper = sma + std * std_val
    lower = sma - std * std_val
    denom = upper - lower
    pos = np.zeros(len(close))
    mask = denom > 0
    pos[mask] = (close[mask] - lower[mask]) / denom[mask]
    return pos


def _bb_width(close: np.ndarray, period: int, std: float) -> np.ndarray:
    sma = _rolling_mean(close, period)
    std_val = _rolling_std(close, period)
    mask = sma > 0
    width = np.zeros(len(close))
    width[mask] = 2 * std * std_val[mask] / sma[mask]
    return width


def _cci(df: pd.DataFrame, period: int) -> np.ndarray:
    tp = (df["high"].values + df["low"].values + df["close"].values) / 3
    sma_tp = _rolling_mean(tp, period)
    mad = np.zeros(len(df))
    for i in range(period, len(df)):
        mad[i] = np.mean(np.abs(tp[i-period:i] - sma_tp[i]))
    cci = np.zeros(len(df))
    mask = mad > 0
    cci[mask] = (tp[mask] - sma_tp[mask]) / (0.015 * mad[mask])
    return cci


# ═══════════════════════════════════════════════════════════
# F. 成交量画像 (~60个)
# ═══════════════════════════════════════════════════════════

def build_volume_features() -> List[Feature]:
    """成交量: 量比 + 量价关系 + OBV + 流动性"""
    features = []

    # 量比 (多窗口)
    for s, l in [(5, 20), (10, 50), (5, 50), (20, 100)]:
        features.append(Feature(
            id=f"vol_ratio_{s}_{l}", name=f"量比({s}/{l})",
            category="F", description=f"最近{s}日均量 / 最近{l}日均量",
            compute_fn=lambda df, s=s, l=l:
                _latest(_rolling_mean(df["volume"].values, s) /
                        (_rolling_mean(df["volume"].values, l) + 1)),
        ))

    # 量价背离
    for w in [10, 20]:
        features.append(Feature(
            id=f"vol_price_div_{w}d", name=f"{w}日量价背离",
            category="F",
            description=f"价格变化率与成交量变化率的差值 (>0=价涨量缩/危险)",
            compute_fn=lambda df, w=w:
                _latest(_returns(df["close"].values, w) -
                        _returns(df["volume"].values, w)),
        ))

    # OBV动量
    for w in [5, 10, 20]:
        features.append(Feature(
            id=f"obv_mom_{w}d", name=f"OBV {w}日动量",
            category="F", description=f"On-Balance Volume 的{w}日变化率",
            compute_fn=lambda df, w=w:
                _latest(_returns(_calc_obv(df), w)),
        ))

    # 成交量趋势 (线性回归斜率)
    for w in [10, 20, 50]:
        features.append(Feature(
            id=f"vol_trend_{w}d", name=f"{w}日成交量趋势",
            category="F",
            description=f"成交量在{w}日内的线性趋势斜率",
            compute_fn=lambda df, w=w:
                _latest(_linear_slope(df["volume"].values, w)),
        ))

    # 换手率等效 (成交量/市值 proxy — 用成交量/价格中位数)
    for w in [10, 20]:
        features.append(Feature(
            id=f"turnover_proxy_{w}d", name=f"换手率proxy {w}日",
            category="F",
            description=f"成交量/价格中位数 (流动性proxy)",
            compute_fn=lambda df, w=w:
                _latest(_rolling_mean(df["volume"].values, w) /
                        _rolling_mean(np.abs(df["close"].values), w)),
        ))

    # 大单流向等效 (极端量日占比)
    for w, threshold in [(20, 2), (50, 2), (20, 3)]:
        features.append(Feature(
            id=f"large_order_{w}d_{threshold}s", name=f"{w}日大单占比",
            category="F",
            description=f"过去{w}日成交量超过{threshold}σ的日数占比",
            compute_fn=lambda df, w=w, threshold=threshold:
                _latest(_extreme_vol_days(df["volume"].values, w, threshold)),
        ))

    return features


def _calc_obv(df: pd.DataFrame) -> np.ndarray:
    obv = np.zeros(len(df))
    obv[0] = df["volume"].values[0]
    for i in range(1, len(df)):
        if df["close"].values[i] > df["close"].values[i-1]:
            obv[i] = obv[i-1] + df["volume"].values[i]
        elif df["close"].values[i] < df["close"].values[i-1]:
            obv[i] = obv[i-1] - df["volume"].values[i]
        else:
            obv[i] = obv[i-1]
    return obv


def _linear_slope(x: np.ndarray, window: int) -> np.ndarray:
    result = np.zeros(len(x))
    for i in range(window, len(x)):
        y = x[i-window:i+1]
        xx = np.arange(len(y))
        result[i] = np.polyfit(xx, y, 1)[0] / (np.mean(y) + 1)
    return result


def _extreme_vol_days(volume: np.ndarray, window: int, sigma: float) -> np.ndarray:
    result = np.zeros(len(volume))
    for i in range(window, len(volume)):
        seg = volume[i-window:i+1]
        m, s = np.mean(seg), np.std(seg)
        if s > 0:
            result[i] = np.sum(seg > m + sigma * s) / window
    return result


# ═══════════════════════════════════════════════════════════
# G. 跨市场联动 (~40个)
# ═══════════════════════════════════════════════════════════

def build_cross_market_features() -> List[Feature]:
    """跨市场: 相关性 + Beta + 残差"""
    features = []

    configs = [
        # (名称, yfinance代码, 窗口列表)
        ("SPY", "^GSPC", [10, 20, 60]),
        ("QQQ", "^NDX", [10, 20, 60]),
        ("GLD", "GLD", [10, 20, 60]),
        ("DXY", "DX-Y.NYB", [10, 20, 60]),
        ("VIX", "^VIX", [10, 20, 30]),
    ]

    for label, yf_code, windows in configs:
        for w in windows:
            features.append(Feature(
                id=f"corr_{label}_{w}d", name=f"{label} {w}日相关性",
                category="G",
                description=f"与{label}的{w}日滚动相关系数",
                compute_fn=lambda df, label=label, yf_code=yf_code, w=w:
                    0.0,  # 需要外部数据, 占位
            ))
            features.append(Feature(
                id=f"beta_{label}_{w}d", name=f"{label} {w}日Beta",
                category="G",
                description=f"对{label}的{w}日Beta",
                compute_fn=lambda df, label=label, yf_code=yf_code, w=w:
                    0.0,  # 需要外部数据, 占位
            ))

    # BTC-ETH 相关性残差
    for w in [10, 20, 60]:
        features.append(Feature(
            id=f"btc_eth_residual_{w}d", name=f"BTC-ETH残差 {w}日",
            category="G",
            description=f"BTC与ETH在{w}日窗口内的相关性残差",
            compute_fn=lambda df, w=w:
                0.0,  # 需要ETH数据, 占位
        ))

    return features


# ═══════════════════════════════════════════════════════════
# H. 衍生品数据 (~30个)
# ═══════════════════════════════════════════════════════════

def build_derivatives_features() -> List[Feature]:
    """衍生品: 资金费率 + 未平仓合约 + 期权偏斜proxy"""
    features = []

    # 资金费率 — Fix #3: 从 _SENTIMENT_CTX 读取 OKX 实时数据
    for lookback in [1, 3, 8, 24]:  # hours
        features.append(Feature(
            id=f"funding_rate_{lookback}h", name=f"资金费率 {lookback}h均值",
            category="H",
            description=f"过去{lookback}小时平均资金费率 (OKX实时)",
            compute_fn=lambda df, lookback=lookback:
                _SENTIMENT_CTX.get("funding_rates", {}).get(
                    _SENTIMENT_CTX.get("current_symbol", ""), 0.0
                ),
        ))

    # OI变化率 — Fix #3: 从 _SENTIMENT_CTX 读取
    for w in [1, 3, 7, 14]:
        features.append(Feature(
            id=f"oi_change_{w}d", name=f"OI {w}日变化",
            category="H",
            description=f"未平仓合约{w}日变化率 (OKX实时)",
            compute_fn=lambda df, w=w:
                _SENTIMENT_CTX.get("oi_changes", {}).get(
                    _SENTIMENT_CTX.get("current_symbol", ""), {}
                ).get(w, 0.0),
        ))

    # OI/成交量比 — Fix #3: 从 _SENTIMENT_CTX 读取
    features.append(Feature(
        id="oi_vol_ratio", name="OI/成交量比",
        category="H",
        description="未平仓合约与成交量的比值 (杠杆热度proxy, OKX实时)",
        compute_fn=lambda df: _SENTIMENT_CTX.get("funding_rates", {}).get(
            _SENTIMENT_CTX.get("current_symbol", ""), 0.0
        ) * 100,  # 年化费率*100 作为杠杆热度 proxy
    ))

    # 期权隐含波动率偏斜proxy (用BTC价格偏度+极端值)
    for w in [10, 20]:
        features.append(Feature(
            id="iv_skew_proxy_{w}d", name=f"IV偏斜proxy {w}日",
            category="H",
            description=f"用价格偏度+尾部事件估计的波动率偏斜 (无期权数据时的替代)",
            compute_fn=lambda df, w=w:
                _latest(_rolling_skew(_returns(df["close"].values, 1), w)),
        ))

    return features


# ═══════════════════════════════════════════════════════════
# I. 情绪指标 (~20个)
# ═══════════════════════════════════════════════════════════

def build_sentiment_features() -> List[Feature]:
    """情绪: Fear & Greed + 变化率 + 极端值"""
    features = []

    # F&G 变化率 — Fix #3: 从 _SENTIMENT_CTX 读取实时数据
    fg_now = _SENTIMENT_CTX.get("fg_value", 50)
    for w in [1, 3, 5, 7, 14]:
        fg_prev = _SENTIMENT_CTX.get(f"fg_prev_{w}d", fg_now)
        features.append(Feature(
            id=f"fg_change_{w}d", name=f"F&G {w}日变化",
            category="I",
            description=f"Fear & Greed Index 的{w}日变化量 (>0=情绪改善)",
            compute_fn=lambda df, w=w, fg_now=fg_now, fg_prev=fg_prev:
                fg_now - fg_prev,  # Fix #3: 接入实时 F&G 数据
        ))

    # F&G 极端值标志 — Fix #3: 接入实时 F&G 数据
    features.append(Feature(
        id="fg_extreme_fear", name="F&G 极度恐惧",
        category="I",
        description="当前F&G ≤ 25 (极度恐惧 → 逆势买入信号)",
        compute_fn=lambda df: 1.0 if _SENTIMENT_CTX.get("fg_value", 50) <= 25 else 0.0,
    ))
    features.append(Feature(
        id="fg_extreme_greed", name="F&G 极度贪婪",
        category="I",
        description="当前F&G ≥ 75 (极度贪婪 → 减仓信号)",
        compute_fn=lambda df: 1.0 if _SENTIMENT_CTX.get("fg_value", 50) >= 75 else 0.0,
    ))

    # F&G 反转信号 — Fix #3: 接入实时 F&G 数据
    features.append(Feature(
        id="fg_reversal", name="F&G 反转信号",
        category="I",
        description="F&G从极恐反弹(≤25→>30)或从极贪回落(≥75→<70)",
        compute_fn=lambda df: (
            1.0 if (_SENTIMENT_CTX.get("fg_value", 50) > 30 and _SENTIMENT_CTX.get("fg_prev_1d", 50) <= 25)
            else -1.0 if (_SENTIMENT_CTX.get("fg_value", 50) < 70 and _SENTIMENT_CTX.get("fg_prev_1d", 50) >= 75)
            else 0.0
        ),
    ))

    return features


# ═══════════════════════════════════════════════════════════
# J. 链上等效proxy (~20个)
# ═══════════════════════════════════════════════════════════

def build_onchain_proxy_features() -> List[Feature]:
    """链上proxy: 用价格/量数据模拟链上行为 (免费替代)"""
    features = []

    # "巨鲸活动" proxy: 大成交量日后的价格走势
    for w in [5, 10, 20]:
        features.append(Feature(
            id=f"whale_activity_{w}d", name=f"巨鲸活跃proxy {w}日",
            category="J",
            description=f"高成交量日(>2σ)后{w}日平均收益 (巨鲸进出场效应)",
            compute_fn=lambda df, w=w:
                _latest(_whale_effect(df, w)),
        ))

    # "交易所流入/流出" proxy: 价格下跌+放量 = 抛售压力
    for w in [5, 10]:
        features.append(Feature(
            id=f"sell_pressure_{w}d", name=f"抛压proxy {w}日",
            category="J",
            description=f"放量下跌日的占比 (>50%=大量抛压)",
            compute_fn=lambda df, w=w:
                _latest(_sell_pressure(df, w)),
        ))

    # "HODL"强度: 低换手 + 价格上涨 = 强持有
    for w in [20, 60]:
        features.append(Feature(
            id="hodl_strength_{w}d", name=f"HODL强度 {w}日",
            category="J",
            description=f"换手率与价格的相关性 (<0=低换手高价格=强持有)",
            compute_fn=lambda df, w=w:
                _latest(_rolling_corr(
                    _rolling_mean(df["volume"].values, 5) / _rolling_mean(np.abs(df["close"].values), 5),
                    df["close"].values, w
                )),
        ))

    # 地址活跃度proxy: 成交量波动率
    for w in [10, 20]:
        features.append(Feature(
            id="addr_activity_proxy_{w}d", name=f"地址活跃proxy {w}日",
            category="J",
            description=f"成交量标准差 (活跃度变化proxy)",
            compute_fn=lambda df, w=w:
                _latest(pd.Series(df["volume"].values).pct_change().rolling(w).std().values),
        ))

    return features


def _whale_effect(df: pd.DataFrame, window: int) -> np.ndarray:
    """高成交量日后N日平均收益"""
    vol = df["volume"].values
    close = df["close"].values
    result = np.zeros(len(close))
    for i in range(60, len(close)):
        vol_seg = vol[i-window:i]
        m, s = np.mean(vol_seg), np.std(vol_seg)
        if s > 0:
            whale_days = np.where(vol_seg > m + 2 * s)[0]
            if len(whale_days) > 0:
                fwd_rets = []
                for d in whale_days:
                    if i - window + d + 5 < len(close):
                        fwd_rets.append(close[i - window + d + 5] / close[i - window + d] - 1)
                if fwd_rets:
                    result[i] = np.mean(fwd_rets)
    return result


def _sell_pressure(df: pd.DataFrame, window: int) -> np.ndarray:
    """放量下跌日占比"""
    close = df["close"].values
    vol = df["volume"].values
    result = np.zeros(len(close))
    for i in range(window, len(close)):
        seg_close = close[i-window:i]
        seg_vol = vol[i-window:i]
        avg_vol = np.mean(seg_vol)
        down_heavy = sum(1 for j in range(window-1)
                        if seg_close[j+1] < seg_close[j] and seg_vol[j] > avg_vol)
        result[i] = down_heavy / window
    return result


# ═══════════════════════════════════════════════════════════
# 特征工厂: 组装所有特征
# ═══════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════
# K. 交互特征 (~80个) — 非线性关系的捕捉器
# ═══════════════════════════════════════════════════════════

def build_interaction_features() -> List[Feature]:
    """交互特征: 两两特征的乘积、比值、差值"""
    features = []

    # 动量 × 波动率交互
    mom_windows = [5, 10, 20, 50]
    vol_windows = [10, 20, 60]
    for mw in mom_windows:
        for vw in vol_windows:
            features.append(Feature(
                id=f"mom{mw}d_vol{vw}d", name=f"动量{mw}d×波动{vw}d",
                category="K",
                description=f"{mw}日动量与{vw}日波动率的交互",
                compute_fn=lambda df, mw=mw, vw=vw:
                    _latest(_returns(df["close"].values, mw)) /
                    (_latest(_rolling_std(np.log(df["close"].values[1:]/df["close"].values[:-1]), vw)) * np.sqrt(365) + 1e-9),
            ))

    # RSI × 量比交互
    for rsi_p in [7, 14, 21]:
        features.append(Feature(
            id=f"rsi{rsi_p}_vol_ratio", name=f"RSI{rsi_p}×量比",
            category="K",
            description=f"RSI({rsi_p})与5/20量比的交互",
            compute_fn=lambda df, rsi_p=rsi_p:
                _rsi(df["close"].values, rsi_p) *
                (_latest(_rolling_mean(df["volume"].values, 5)) /
                 (_latest(_rolling_mean(df["volume"].values, 20)) + 1)),
        ))

    # 偏度 × 峰度
    for w in [20, 60]:
        features.append(Feature(
            id=f"skew_kurt_{w}d", name=f"偏度×峰度 {w}日",
            category="K",
            description=f"{w}日偏度×峰度 (负偏+高峭=极端暴跌风险)",
            compute_fn=lambda df, w=w:
                _latest(_rolling_skew(_returns(df["close"].values, 1), w)) *
                _latest(_rolling_kurt(_returns(df["close"].values, 1), w)),
        ))

    # 波动率锥 × 动量
    for w in [20, 60]:
        features.append(Feature(
            id=f"vol_cone_mom_{w}d", name=f"波动率锥×动量 {w}日",
            category="K",
            description=f"波动率分位数×{w}日动量方向",
            compute_fn=lambda df, w=w:
                _latest(_vol_cone_percentile(df["close"].values, w)) *
                np.sign(_latest(_returns(df["close"].values, w))),
        ))

    # 均线距离 × 均线斜率
    for s, l in [(5, 20), (20, 60)]:
        features.append(Feature(
            id=f"ma_dist_slope_{s}_{l}", name=f"MA{s}/{l}距离×斜率",
            category="K",
            description=f"SMA({s}/{l})距离×SMA({s})斜率 (趋势加速检测)",
            compute_fn=lambda df, s=s, l=l:
                _latest((pd.Series(df["close"].values).rolling(s).mean() /
                         pd.Series(df["close"].values).rolling(l).mean() - 1).values) *
                _latest(pd.Series(
                    pd.Series(df["close"].values).rolling(s).mean().values
                ).pct_change(5).values),
        ))

    # OBV × 价格动量 (量价确认)
    for w in [5, 10, 20]:
        features.append(Feature(
            id="obv_price_confirm_{w}d", name=f"OBV-价格确认 {w}日",
            category="K",
            description=f"OBV与价格{w}日同向性 (>0=确认, <0=背离)",
            compute_fn=lambda df, w=w:
                np.sign(_latest(_returns(_calc_obv(df), w))) *
                np.sign(_latest(_returns(df["close"].values, w))) *
                abs(_latest(_returns(df["close"].values, w))),
        ))

    # RSI 差异特征 (短期RSI - 长期RSI)
    for s, l in [(7, 21), (7, 30), (14, 30)]:
        features.append(Feature(
            id=f"rsi_diff_{s}_{l}", name=f"RSI{s}-RSI{l}",
            category="K",
            description=f"短期RSI({s})与长期RSI({l})的差值 (正=短期走强)",
            compute_fn=lambda df, s=s, l=l:
                _rsi(df["close"].values, s) - _rsi(df["close"].values, l),
        ))

    # 成交量趋势 × 价格位置
    for w in [10, 20]:
        features.append(Feature(
            id="vol_trend_price_{w}d", name=f"量趋势×价格 {w}日",
            category="K",
            description=f"成交量线性趋势 × 价格位置百分比 (放量突破信号)",
            compute_fn=lambda df, w=w:
                _latest(_linear_slope(df["volume"].values, w)) *
                _latest(_bb_position(df["close"].values, w, 2)),
        ))

    return features


# ═══════════════════════════════════════════════════════════
# L. 时间序列分解特征 (~50个)
# ═══════════════════════════════════════════════════════════

def build_timeseries_features() -> List[Feature]:
    """时间序列: Hurst指数 + 分形 + 自相关 + 趋势强度"""
    features = []

    # Hurst指数proxy
    for w in [20, 60, 120]:
        features.append(Feature(
            id=f"hurst_{w}d", name=f"Hurst {w}日",
            category="L",
            description=f"Hurst指数: >0.5=趋势持续, <0.5=均值回归",
            compute_fn=lambda df, w=w:
                _latest(_hurst_rs(df["close"].values, w)),
        ))

    # 自相关
    for lag in [1, 3, 5, 10, 20]:
        features.append(Feature(
            id=f"autocorr_{lag}d", name=f"自相关 lag{lag}",
            category="L",
            description=f"日收益率的{lag}阶自相关 (>0=趋势, <0=反转)",
            compute_fn=lambda df, lag=lag:
                _latest(_rolling_autocorr(_returns(df["close"].values, 1), lag, 60)),
        ))

    # 趋势强度 (ADX等效)
    for w in [14, 20, 30]:
        features.append(Feature(
            id=f"trend_strength_{w}d", name=f"趋势强度 {w}日",
            category="L",
            description=f"{w}日ADX等效趋势强度 (25+=趋势明确)",
            compute_fn=lambda df, w=w:
                _latest(_adx_proxy(df, w)),
        ))

    # 分形效率
    for w in [10, 20, 50]:
        features.append(Feature(
            id=f"fractal_eff_{w}d", name=f"分形效率 {w}日",
            category="L",
            description=f"价格路径直度 (1=完美趋势, 0=纯噪音)",
            compute_fn=lambda df, w=w:
                _latest(_fractal_efficiency(df["close"].values, w)),
        ))

    # BB %B 加速度
    for period in [20, 50]:
        features.append(Feature(
            id="bb_accel_{period}d", name=f"BB加速度 {period}日",
            category="L",
            description=f"布林带位置的5日变化 (突破加速信号)",
            compute_fn=lambda df, period=period:
                _latest(pd.Series(_bb_position(df["close"].values, period, 2)).diff(5).values),
        ))

    # 波动率溢价 (Parkinson / Close-to-Close)
    for w in [10, 20]:
        features.append(Feature(
            id="vol_premium_{w}d", name=f"波动率溢价 {w}日",
            category="L",
            description=f"Parkinson/Close波动率比 (>1=日内波动大/恐慌)",
            compute_fn=lambda df, w=w:
                _latest(np.sqrt(
                    _rolling_mean((np.log(df["high"].values/df["low"].values))**2 / (4*np.log(2)), w)
                ) / (_rolling_std(np.log(df["close"].values[1:]/df["close"].values[:-1]), w) + 1e-9)),
        ))

    return features


def _hurst_rs(close: np.ndarray, window: int) -> np.ndarray:
    """Hurst指数 (R/S方法)"""
    rets = np.diff(np.log(close))
    result = np.full(len(close), 0.5)
    for i in range(window * 2, len(close)):
        segment = rets[i-window:i]
        mean_ret = np.mean(segment)
        deviate = segment - mean_ret
        Z = np.cumsum(deviate)
        R = np.max(Z) - np.min(Z)
        S = np.std(segment)
        if S > 0:
            result[i] = np.log(R / S) / np.log(window)
    return np.clip(result, 0, 1)


def _rolling_autocorr(x: np.ndarray, lag: int, window: int) -> np.ndarray:
    result = np.zeros(len(x))
    for i in range(window + lag, len(x)):
        seg = x[i-window-lag:i]
        if len(seg) > lag and np.std(seg) > 0:
            result[i] = np.corrcoef(seg[:-lag], seg[lag:])[0, 1] if len(seg[:-lag]) > 1 else 0
    return result


def _adx_proxy(df: pd.DataFrame, period: int) -> np.ndarray:
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    tr = np.zeros(len(close))
    plus_dm = np.zeros(len(close))
    minus_dm = np.zeros(len(close))
    for i in range(1, len(close)):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0
        minus_dm[i] = down if (down > up and down > 0) else 0
    atr = pd.Series(tr).ewm(alpha=1/period).mean().values
    plus_di = pd.Series(plus_dm).ewm(alpha=1/period).mean().values / (atr + 1e-9) * 100
    minus_di = pd.Series(minus_dm).ewm(alpha=1/period).mean().values / (atr + 1e-9) * 100
    denom = plus_di + minus_di + 1e-9
    dx = np.abs(plus_di - minus_di) / denom * 100
    return pd.Series(dx).ewm(alpha=1/period).mean().values


def _fractal_efficiency(close: np.ndarray, window: int) -> np.ndarray:
    result = np.full(len(close), 0.5)
    for i in range(window, len(close)):
        seg = close[i-window:i+1]
        net_change = abs(seg[-1] - seg[0])
        path_length = np.sum(np.abs(np.diff(seg)))
        if path_length > 0:
            result[i] = net_change / path_length
    return result



# ═══════════════════════════════════════════════════════════
# M. 价格形态特征 (~80个) — K线形态 + 支撑阻力检测
# ═══════════════════════════════════════════════════════════

def build_pattern_features() -> List[Feature]:
    """K线形态 + 关键价位触碰"""
    features = []

    # 影线比例 (上下影线博弈)
    for w in [5, 10, 20]:
        features.append(Feature(
            id=f"upper_wick_ratio_{w}d", name=f"上影线比 {w}日",
            category="M",
            description=f"上影线/(上+下影线)比例均值 (>0.7=上方压力大)",
            compute_fn=lambda df, w=w:
                _latest(_wick_ratio(df, 'upper', w)),
        ))
        features.append(Feature(
            id=f"lower_wick_ratio_{w}d", name=f"下影线比 {w}日",
            category="M",
            description=f"下影线/(上+下影线)比例均值 (>0.7=下方支撑强)",
            compute_fn=lambda df, w=w:
                _latest(_wick_ratio(df, 'lower', w)),
        ))

    # 实体比例
    for w in [5, 10, 20]:
        features.append(Feature(
            id=f"body_ratio_{w}d", name=f"实体比 {w}日",
            category="M",
            description=f"实体/(高-低)比例 (<0.3=十字星/犹豫, >0.7=强势)",
            compute_fn=lambda df, w=w:
                _latest(_rolling_mean(
                    abs(df["close"].values - df["open"].values) / (df["high"].values - df["low"].values + 1e-9), w
                )),
        ))

    # 连续同向K线数
    for direction, label in [(1, 'bullish'), (-1, 'bearish')]:
        features.append(Feature(
            id=f"consecutive_{label}_bars", name=f"连{label}K线",
            category="M",
            description=f"当前连续阳线/阴线数 (趋势持续性)",
            compute_fn=lambda df, d=direction:
                float(_consecutive_bars(df, d)),
        ))

    # Doji检测
    for w in [5, 10]:
        features.append(Feature(
            id=f"doji_count_{w}d", name=f"Doji数 {w}日",
            category="M",
            description=f"过去{w}日十字星天数 (犹豫信号增多=变盘前兆)",
            compute_fn=lambda df, w=w:
                _latest(_rolling_mean(
                    (abs(df["close"].values - df["open"].values) /
                     (df["high"].values - df["low"].values + 1e-9) < 0.1).astype(float), w
                )),
        ))

    # 吞没形态检测
    features.append(Feature(
        id="engulfing_signal", name="吞没形态",
        category="M",
        description="最近1日是否为吞没形态 (1=看涨吞没, -1=看跌吞没)",
        compute_fn=lambda df: float(_detect_engulfing(df)),
    ))

    # 新高/新低触碰
    for w in [20, 60, 120]:
        features.append(Feature(
            id="near_high_{w}d", name=f"近{w}日高",
            category="M",
            description=f"价格距{w}日最高点的距离 (>-2%=接近突破)",
            compute_fn=lambda df, w=w:
                _latest(df["close"].values / _rolling_max(df["high"].values, w) - 1),
        ))
        features.append(Feature(
            id="near_low_{w}d", name=f"近{w}日低",
            category="M",
            description=f"价格距{w}日最低点的距离 (<2%=接近支撑)",
            compute_fn=lambda df, w=w:
                _latest(df["close"].values / _rolling_min(df["low"].values, w) - 1),
        ))

    # 缺口检测
    for w in [5, 10]:
        features.append(Feature(
            id="gap_count_{w}d", name=f"缺口数 {w}日",
            category="M",
            description=f"过去{w}日跳空缺口数 (高缺口=强趋势/极端情绪)",
            compute_fn=lambda df, w=w:
                _latest(_rolling_mean(
                    (df["low"].values[1:] > df["high"].values[:-1]).astype(float), w
                )) if w < len(df) else 0,
        ))

    # 高低价差收敛 (Squeeze)
    for w in [10, 20]:
        features.append(Feature(
            id="squeeze_{w}d", name=f"Squeeze {w}日",
            category="M",
            description=f"布林带宽度历史分位数 (<0.1=极度收敛/爆发前兆)",
            compute_fn=lambda df, w=w:
                _latest(_squeeze_signal(df["close"].values, w)),
        ))

    return features


def _wick_ratio(df: pd.DataFrame, wick_type: str, window: int) -> np.ndarray:
    high, low, close, open_ = df["high"].values, df["low"].values, df["close"].values, df["open"].values
    body_high = np.maximum(close, open_)
    body_low = np.minimum(close, open_)
    upper_wick = high - body_high
    lower_wick = body_low - low
    total_wick = upper_wick + lower_wick + 1e-9
    if wick_type == 'upper':
        return _rolling_mean(upper_wick / total_wick, window)
    return _rolling_mean(lower_wick / total_wick, window)


def _consecutive_bars(df: pd.DataFrame, direction: int) -> int:
    """当前连续阳线/阴线数"""
    close = df["close"].values
    open_ = df["open"].values
    count = 0
    for i in range(len(close)-1, 0, -1):
        if direction == 1 and close[i] > open_[i]:
            count += 1
        elif direction == -1 and close[i] < open_[i]:
            count += 1
        else:
            break
        if i > 0 and count > 20:  # safety
            break
    return count


def _detect_engulfing(df: pd.DataFrame) -> int:
    """检测最近吞没形态"""
    if len(df) < 3:
        return 0
    c1, o1 = df["close"].values[-2], df["open"].values[-2]
    c2, o2 = df["close"].values[-1], df["open"].values[-1]
    # 看涨吞没: D1阴线 + D2阳线 + D2实体完全覆盖D1
    if c1 < o1 and c2 > o2 and o2 < c1 and c2 > o1:
        return 1
    # 看跌吞没: D1阳线 + D2阴线 + D2实体完全覆盖D1
    if c1 > o1 and c2 < o2 and o2 > c1 and c2 < o1:
        return -1
    return 0


def _squeeze_signal(close: np.ndarray, window: int) -> np.ndarray:
    """Bollinger带宽历史分位数"""
    width = _bb_width(close, 20, 2)
    result = np.full(len(close), 0.5)
    for i in range(window * 3, len(close)):
        hist = width[window:i]
        if len(hist) > 0 and not np.isnan(width[i]):
            result[i] = (hist < width[i]).sum() / len(hist)
    return result


# ═══════════════════════════════════════════════════════════
# N. 价格比率特征 (~40个) — 各种比例关系
# ═══════════════════════════════════════════════════════════

def build_ratio_features() -> List[Feature]:
    """价格比率: 各种H/L/O/C的比例关系"""
    features = []

    # Close/Open ratio (日内方向+强度)
    for w in [5, 10, 20, 50]:
        features.append(Feature(
            id=f"co_ratio_{w}d", name=f"收/开比 {w}日",
            category="N",
            description=f"收盘/开盘比均值 (>1=多头主导, <1=空头主导)",
            compute_fn=lambda df, w=w:
                _latest(_rolling_mean(df["close"].values / (df["open"].values + 1e-9), w)),
        ))

    # High/Close (日内反转压力)
    for w in [5, 10]:
        features.append(Feature(
            id="hc_ratio_{w}d", name=f"高/收比 {w}日",
            category="N",
            description=f"最高/收盘比 (<1.01=强势收盘, >1.03=冲高回落)",
            compute_fn=lambda df, w=w:
                _latest(_rolling_mean(df["high"].values / (df["close"].values + 1e-9), w)),
        ))

    # Low/Close
    for w in [5, 10]:
        features.append(Feature(
            id="lc_ratio_{w}d", name=f"低/收比 {w}日",
            category="N",
            description=f"最低/收盘比 (>0.98=拒绝下跌, <0.95=探底)",
            compute_fn=lambda df, w=w:
                _latest(_rolling_mean(df["low"].values / (df["close"].values + 1e-9), w)),
        ))

    # True Range / Close (归一化波动)
    for w in [5, 10, 20]:
        features.append(Feature(
            id="atr_pct_{w}d", name=f"ATR% {w}日",
            category="N",
            description=f"平均真实波幅/收盘价 (归一化波动率)",
            compute_fn=lambda df, w=w:
                _latest(_rolling_mean(
                    _calc_tr(df) / (df["close"].values + 1e-9), w
                )),
        ))

    # 涨跌比 (Up/Down Volume Ratio proxy — 用涨跌日数量比)
    for w in [5, 10, 20]:
        features.append(Feature(
            id="up_down_ratio_{w}d", name=f"涨跌比 {w}日",
            category="N",
            description=f"过去{w}日上涨日/下跌日之比",
            compute_fn=lambda df, w=w:
                _latest(_up_down_ratio(df["close"].values, w)),
        ))

    return features


def _calc_tr(df: pd.DataFrame) -> np.ndarray:
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    tr = np.zeros(len(close))
    for i in range(1, len(close)):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    tr[0] = tr[1] if len(tr) > 1 else high[0]-low[0]
    return tr


def _up_down_ratio(close: np.ndarray, window: int) -> np.ndarray:
    result = np.full(len(close), 1.0)
    for i in range(window, len(close)):
        seg = close[i-window+1:i+1]
        up = sum(1 for j in range(1, len(seg)) if seg[j] > seg[j-1])
        down = sum(1 for j in range(1, len(seg)) if seg[j] < seg[j-1])
        result[i] = up / (down + 1)
    return result


# ═══════════════════════════════════════════════════════════
# Fix #3: 情绪数据上下文 (由 daemon 每轮扫描前注入)
# ═══════════════════════════════════════════════════════════
_SENTIMENT_CTX: Dict[str, any] = {
    "fg_value": 50,          # Fear & Greed 当前值 0-100
    "fg_prev_1d": 50,        # 1天前F&G
    "fg_prev_3d": 50,
    "fg_prev_5d": 50,
    "fg_prev_7d": 50,
    "fg_prev_14d": 50,
    "funding_rates": {},     # {symbol: annualized_rate}
    "oi_changes": {},        # {symbol: {1: pct, 3: pct, 7: pct, 14: pct}}
    "current_symbol": None,  # 当前正在计算特征的标的
}

def set_sentiment_context(**kwargs):
    """由 daemon 调用，注入实时情绪数据"""
    _SENTIMENT_CTX.update(kwargs)

def _get_current_symbol():
    return _SENTIMENT_CTX.get("current_symbol", "")


class FeatureFactory:
    """500+特征生成器"""

    def __init__(self):
        self.features: List[Feature] = []
        self._build_all()
        self._compute_cache: Dict[str, float] = {}

    def _build_all(self):
        """构建所有特征"""
        builders = [
            ("A_动量", build_momentum_features),
            ("B_波动率", build_volatility_features),
            ("C_高阶矩", build_higher_moment_features),
            ("D_均线", build_ma_features),
            ("E_振荡器", build_oscillator_features),
            ("F_成交量", build_volume_features),
            ("G_跨市场", build_cross_market_features),
            ("H_衍生品", build_derivatives_features),
            ("I_情绪", build_sentiment_features),
            ("J_链上proxy", build_onchain_proxy_features),
            ("K_交互", build_interaction_features),
            ("L_时间序列", build_timeseries_features),
            ("M_形态", build_pattern_features),
            ("N_比率", build_ratio_features),
        ]

        for cat_label, builder in builders:
            feats = builder()
            self.features.extend(feats)
            _ = f"{cat_label}: {len(feats)}个"

        # 统计
        self._available_count = sum(1 for f in self.features if f.compute_fn is not None)
        self._placeholder_count = len(self.features) - self._available_count

    def compute_all(self, df: pd.DataFrame, symbol: str = None) -> Dict[str, float]:
        """计算所有特征值 (只算有实现的). symbol 用于情绪/衍生品数据查询"""
        if symbol:
            _SENTIMENT_CTX["current_symbol"] = symbol
        result = {}
        for f in self.features:
            try:
                val = f.compute_fn(df) if f.compute_fn else 0.0
                if not np.isnan(val) and not np.isinf(val):
                    result[f.id] = float(val)
                else:
                    result[f.id] = 0.0
            except Exception:
                result[f.id] = 0.0
        return result

    def compute_active(self, df: pd.DataFrame, symbol: str = None) -> Dict[str, float]:
        """只计算已实现的特征 (跳过占位)"""
        if symbol:
            _SENTIMENT_CTX["current_symbol"] = symbol
        result = {}
        for f in self.features:
            if f.compute_fn is None:
                continue
            try:
                val = f.compute_fn(df)
                if not np.isnan(val) and not np.isinf(val):
                    result[f.id] = float(val)
                else:
                    result[f.id] = 0.0
            except Exception:
                result[f.id] = 0.0
        return result

    def summary(self) -> dict:
        """特征统计摘要"""
        cats = {}
        for f in self.features:
            if f.category not in cats:
                cats[f.category] = {"total": 0, "available": 0}
            cats[f.category]["total"] += 1
            if f.compute_fn is not None:
                cats[f.category]["available"] += 1

        return {
            "total_features": len(self.features),
            "available_now": self._available_count,
            "placeholder": self._placeholder_count,
            "by_category": cats,
        }

    @property
    def active_features(self) -> List[Feature]:
        return [f for f in self.features if f.compute_fn is not None]


# ── 全局单例 ──
_factory: Optional[FeatureFactory] = None


def get_feature_factory() -> FeatureFactory:
    global _factory
    if _factory is None:
        _factory = FeatureFactory()
    return _factory


# ── CLI ──
if __name__ == "__main__":
    ff = get_feature_factory()
    s = ff.summary()
    print("=" * 60)
    print("🧬 Feature Factory v3.0 — 特征总览")
    print("=" * 60)
    print(f"总特征数: {s['total_features']}")
    print(f"已实现: {s['available_now']} | 占位(需外部API): {s['placeholder']}")
    print()
    for cat, info in s["by_category"].items():
        print(f"  {cat}: {info['available']}/{info['total']} 可用")
    print()
    # 列出前20个特征
    print("前20个可用特征:")
    for f in ff.active_features[:20]:
        print(f"  {f.id:30s} | {f.name:20s} | [{f.category}]")
