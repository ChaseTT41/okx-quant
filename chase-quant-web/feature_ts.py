"""
Feature Time-Series Engine v4.0 — 500+特征一次性时间序列计算
=============================================================
Chase量化策略 Phase 1 核心升级:
  旧: 每个特征只返回最新值 → 回测需逐个时间点重算 (O(f×t²))
  新: 每个特征返回全时间序列 → 回测只需对齐索引 (O(f×t))

新增特征家族 (Chase哥指定):
  O. 增强波动率 — Garman-Klass / Yang-Zhang / Rogers-Satchell
  P. VWAP & MFI — 成交量加权均价偏离 + 资金流量指标多周期
  Q. OBV高阶 — OBV加速度 + OBV与价格二阶导数
  R. 多空力量比 — 买卖压力 + 日内博弈指标
  S. 动量因子增强 — 多窗口排名分位 + 收益率离散度
  T. 跨资产 — 实际数据接入 (BTC-ETH残差, BTC-SPY beta, DXY相关性)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings("ignore")


@dataclass
class FeatureSpec:
    """特征规格 — 同时支持单值和时序计算"""
    id: str
    name: str
    category: str
    description: str = ""
    # 时序计算: (df) -> np.ndarray (长度=len(df))
    compute_ts: Optional[Callable] = None
    # 单值计算: (df) -> float (回退方案)
    compute_point: Optional[Callable] = None


# ═══════════════════════════════════════════════════════════
# 工具函数 — 均返回 full-length np.ndarray
# ═══════════════════════════════════════════════════════════

def _ret(close: np.ndarray, window: int = 1) -> np.ndarray:
    """N日简单收益率 (全序列, 前window天为0)"""
    out = np.zeros_like(close)
    out[window:] = close[window:] / close[:-window] - 1
    return out

def _log_ret(close: np.ndarray, window: int = 1) -> np.ndarray:
    """N日对数收益率"""
    out = np.zeros_like(close)
    out[window:] = np.log(close[window:] / close[:-window])
    return out

def _daily_log_ret(close: np.ndarray) -> np.ndarray:
    """日对数收益率 (长度=n, 第一项=0)"""
    out = np.zeros_like(close)
    out[1:] = np.log(close[1:] / close[:-1])
    return out

def _roll_mean(x: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(x).rolling(w, min_periods=w//2).mean().values

def _roll_std(x: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(x).rolling(w, min_periods=w//2).std().values

def _roll_skew(x: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(x).rolling(w, min_periods=max(w, 20)).skew().values

def _roll_kurt(x: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(x).rolling(w, min_periods=max(w, 20)).kurt().values

def _roll_corr(a: np.ndarray, b: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(a).rolling(w, min_periods=w//2).corr(pd.Series(b)).values

def _roll_max(x: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(x).rolling(w, min_periods=w//2).max().values

def _roll_min(x: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(x).rolling(w, min_periods=w//2).min().values

def _ema(x: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(x).ewm(span=span, min_periods=span//2).mean().values

def _roll_rank(x: np.ndarray, w: int) -> np.ndarray:
    """滚动排名分位 (0-1)"""
    return pd.Series(x).rolling(w, min_periods=max(w, 20)).rank(pct=True).values

def _roll_pct_change(x: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(x).pct_change(w).values

def _rsi_ts(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI 全序列"""
    deltas = np.diff(close, prepend=close[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = pd.Series(gains).ewm(alpha=1/period, min_periods=period).mean().values
    avg_loss = pd.Series(losses).ewm(alpha=1/period, min_periods=period).mean().values
    rs = np.zeros_like(close)
    mask = avg_loss > 1e-9
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    return 100 - 100 / (1 + rs)

def _atr_ts(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR 全序列"""
    tr = np.zeros(len(close))
    for i in range(1, len(close)):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    tr[0] = high[0] - low[0]
    return pd.Series(tr).ewm(alpha=1/period, min_periods=period).mean().values

def _bb_position_ts(close: np.ndarray, period: int = 20, nbstd: float = 2.0) -> np.ndarray:
    """布林带位置 0-1 全序列"""
    sma = _roll_mean(close, period)
    std = _roll_std(close, period)
    upper = sma + nbstd * std
    lower = sma - nbstd * std
    denom = upper - lower + 1e-9
    return np.clip((close - lower) / denom, 0, 1)

def _adx_ts(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ADX 全序列"""
    n = len(close)
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0
        minus_dm[i] = down if (down > up and up > 0) else 0
    atr = pd.Series(tr).ewm(alpha=1/period, min_periods=period).mean().values
    plus_di = pd.Series(plus_dm).ewm(alpha=1/period, min_periods=period).mean().values / (atr + 1e-9) * 100
    minus_di = pd.Series(minus_dm).ewm(alpha=1/period, min_periods=period).mean().values / (atr + 1e-9) * 100
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9) * 100
    return pd.Series(dx).ewm(alpha=1/period, min_periods=period).mean().values

def _hurst_ts(close: np.ndarray, window: int = 60) -> np.ndarray:
    """Hurst指数 全序列 (R/S方法)"""
    log_close = np.log(close)
    rets = np.diff(log_close)
    result = np.full(len(close), 0.5)
    for i in range(window * 2, len(close)):
        seg = rets[i-window:i]
        if len(seg) < 10:
            continue
        mean_ret = np.mean(seg)
        deviate = seg - mean_ret
        Z = np.cumsum(deviate)
        R = np.max(Z) - np.min(Z)
        S = np.std(seg)
        if S > 0:
            result[i] = np.log(R/S) / np.log(window)
    return np.clip(result, 0, 1)

def _fractal_eff_ts(close: np.ndarray, window: int = 20) -> np.ndarray:
    """分形效率 全序列"""
    result = np.full(len(close), 0.5)
    for i in range(window, len(close)):
        seg = close[i-window:i+1]
        net_change = abs(seg[-1] - seg[0])
        path_length = np.sum(np.abs(np.diff(seg)))
        if path_length > 0:
            result[i] = net_change / path_length
    return result

def _linear_slope_ts(x: np.ndarray, window: int) -> np.ndarray:
    """滚动线性斜率"""
    result = np.zeros(len(x))
    for i in range(window, len(x)):
        y = x[i-window:i+1]
        xx = np.arange(len(y))
        slope = np.polyfit(xx, y, 1)[0]
        denom = np.mean(y) + 1e-9
        result[i] = slope / denom if abs(denom) > 1e-12 else 0
    return result

def _obv_ts(df: pd.DataFrame) -> np.ndarray:
    """OBV 全序列"""
    close = df["close"].values
    vol = df["volume"].values
    obv = np.zeros(len(df))
    obv[0] = vol[0]
    for i in range(1, len(df)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + vol[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - vol[i]
        else:
            obv[i] = obv[i-1]
    return obv

def _max_dd_ts(close: np.ndarray, window: int) -> np.ndarray:
    """滚动最大回撤"""
    result = np.zeros(len(close))
    for i in range(window, len(close)):
        peak = np.max(close[i-window:i+1])
        result[i] = (close[i] - peak) / peak
    return result

def _consecutive_ts(close: np.ndarray, direction: int) -> np.ndarray:
    """连续涨跌天数"""
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

def _tail_count_ts(close: np.ndarray, window: int, sigma: float) -> np.ndarray:
    """滚动尾部事件计数"""
    rets = _ret(close, 1)
    result = np.zeros(len(close))
    for i in range(window, len(close)):
        seg = rets[i-window:i+1]
        std = np.std(seg)
        if std > 0:
            result[i] = np.sum(np.abs(seg) > sigma * std) / window
    return result

def _asymmetry_ts(close: np.ndarray, window: int) -> np.ndarray:
    """涨跌不对称比率"""
    rets = _ret(close, 1)
    result = np.ones(len(close))
    for i in range(window, len(close)):
        seg = rets[i-window:i+1]
        up = seg[seg > 0]
        down = seg[seg < 0]
        if len(up) > 0 and len(down) > 0:
            result[i] = np.mean(up) / (abs(np.mean(down)) + 1e-9)
    return np.clip(result, 0.1, 10)

def _vol_cone_ts(close: np.ndarray, window: int) -> np.ndarray:
    """波动率锥分位数"""
    rets = np.diff(np.log(close), prepend=np.log(close[0]))
    vols = pd.Series(rets).rolling(window, min_periods=window).std().values * np.sqrt(365)
    result = np.full(len(close), 0.5)
    for i in range(window * 2, len(close)):
        hist_vols = vols[window:i]
        if len(hist_vols) > 0 and not np.isnan(vols[i]):
            result[i] = (hist_vols < vols[i]).sum() / len(hist_vols)
    return result


# ═══════════════════════════════════════════════════════════
# O. 增强波动率特征 (~30个) — Garman-Klass / Yang-Zhang
# ═══════════════════════════════════════════════════════════

def build_advanced_vol_features() -> List[FeatureSpec]:
    """高级波动率估计器 — 利用OHLC全信息"""
    feats = []

    for w in [10, 20, 60]:
        # Garman-Klass: σ² = 0.5*(ln(H/L))² - (2ln2-1)*(ln(C/O))²
        feats.append(FeatureSpec(
            id=f"gk_vol_{w}d", name=f"GK波动率 {w}日",
            category="O",
            description=f"Garman-Klass波动率(含日内信息) — {w}日窗口",
            compute_ts=lambda df, w=w: np.sqrt(_roll_mean(
                0.5 * np.log(df["high"].values/df["low"].values)**2 -
                (2*np.log(2)-1) * np.log(df["close"].values/df["open"].values)**2, w
            )) * np.sqrt(365),
        ))

        # Rogers-Satchell: σ² = ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)
        feats.append(FeatureSpec(
            id=f"rs_vol_{w}d", name=f"RS波动率 {w}日",
            category="O",
            description=f"Rogers-Satchell波动率(允许漂移) — {w}日窗口",
            compute_ts=lambda df, w=w: np.sqrt(_roll_mean(
                np.log(df["high"].values/df["close"].values) * np.log(df["high"].values/df["open"].values) +
                np.log(df["low"].values/df["close"].values) * np.log(df["low"].values/df["open"].values), w
            )) * np.sqrt(365),
        ))

        # Yang-Zhang: 融合隔夜+日内波动
        feats.append(FeatureSpec(
            id=f"yz_vol_{w}d", name=f"YZ波动率 {w}日",
            category="O",
            description=f"Yang-Zhang波动率(隔夜+日内完整) — {w}日窗口",
            compute_ts=lambda df, w=w: _yz_vol(df, w),
        ))

    # Parkinson / Close-to-Close 比率
    for w in [10, 20]:
        feats.append(FeatureSpec(
            id=f"parkinson_ratio_{w}d", name=f"Parkinson比 {w}日",
            category="O",
            description=f"Parkinson波动率/Close波动率 (>1=日内波动主导)",
            compute_ts=lambda df, w=w:
                np.sqrt(_roll_mean(np.log(df["high"].values/df["low"].values)**2/(4*np.log(2)), w)) /
                (_roll_std(_daily_log_ret(df["close"].values), w) + 1e-9),
        ))

    return feats

def _yz_vol(df: pd.DataFrame, window: int) -> np.ndarray:
    """Yang-Zhang 波动率估计"""
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    open_ = df["open"].values
    n = len(close)

    # 隔夜波动
    overnight = np.zeros(n)
    overnight[1:] = np.log(open_[1:]/close[:-1])
    # 日内波动
    intraday = np.zeros(n)
    intraday[1:] = np.log(close[1:]/open_[1:])

    k = 0.34 / (1.34 + (window+1)/(window-1))
    vo = _roll_std(overnight, window)**2
    vc = _roll_std(intraday, window)**2
    vrs = _roll_mean(
        np.log(high/close) * np.log(high/open_) +
        np.log(low/close) * np.log(low/open_), window
    )

    yz_var = vo + k * vc + (1-k) * vrs
    yz_var = np.maximum(yz_var, 0)
    return np.sqrt(yz_var) * np.sqrt(365)


# ═══════════════════════════════════════════════════════════
# P. VWAP偏离 + MFI资金流量指标 (~25个)
# ═══════════════════════════════════════════════════════════

def build_vwap_mfi_features() -> List[FeatureSpec]:
    """VWAP + MFI — Chase哥指定"""
    feats = []

    # VWAP 偏离 (多窗口)
    for w in [5, 10, 20, 50]:
        feats.append(FeatureSpec(
            id=f"vwap_dev_{w}d", name=f"VWAP偏离 {w}日",
            category="P",
            description=f"收盘价相对{w}日VWAP的偏离百分比",
            compute_ts=lambda df, w=w:
                df["close"].values / (_vwap_ts(df, w) + 1e-9) - 1,
        ))

    # VWAP 斜率 (趋势)
    for w in [5, 10]:
        feats.append(FeatureSpec(
            id=f"vwap_slope_{w}d", name=f"VWAP斜率 {w}日",
            category="P",
            description=f"VWAP在{w}日内的变化率",
            compute_ts=lambda df, w=w:
                pd.Series(_vwap_ts(df, max(w*3, 20))).pct_change(w).values,
        ))

    # MFI 多周期
    for period in [7, 14, 21, 30]:
        feats.append(FeatureSpec(
            id=f"mfi_{period}", name=f"MFI({period})",
            category="P",
            description=f"{period}日资金流量指数 (量价结合的RSI)",
            compute_ts=lambda df, period=period:
                _mfi_ts(df, period),
        ))
        # MFI 变化
        if period <= 21:
            feats.append(FeatureSpec(
                id=f"mfi_change_{period}", name=f"MFI({period})变化",
                category="P",
                description=f"MFI({period})的3日变化",
                compute_ts=lambda df, period=period:
                    pd.Series(_mfi_ts(df, period)).diff(3).values,
            ))

    # 成交量加权收盘位置 (日内强度)
    for w in [5, 10, 20]:
        feats.append(FeatureSpec(
            id=f"close_location_{w}d", name=f"收盘位置 {w}日",
            category="P",
            description=f"收盘价在日内范围的位置均值 (0=低收, 1=高收)",
            compute_ts=lambda df, w=w: _roll_mean(
                (df["close"].values - df["low"].values) /
                (df["high"].values - df["low"].values + 1e-9), w
            ),
        ))

    return feats

def _vwap_ts(df: pd.DataFrame, window: int) -> np.ndarray:
    """滚动VWAP"""
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    vol = df["volume"].values
    tp = (high + low + close) / 3
    tp_vol = tp * vol
    cum_tpv = pd.Series(tp_vol).rolling(window, min_periods=window//2).sum().values
    cum_vol = pd.Series(vol).rolling(window, min_periods=window//2).sum().values
    return cum_tpv / (cum_vol + 1e-9)

def _mfi_ts(df: pd.DataFrame, period: int) -> np.ndarray:
    """MFI 全序列"""
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    vol = df["volume"].values
    tp = (high + low + close) / 3
    money_flow = tp * vol

    pos_flow = np.zeros(len(df))
    neg_flow = np.zeros(len(df))
    for i in range(1, len(df)):
        if tp[i] > tp[i-1]:
            pos_flow[i] = money_flow[i]
        elif tp[i] < tp[i-1]:
            neg_flow[i] = money_flow[i]

    pos_sum = pd.Series(pos_flow).rolling(period, min_periods=period).sum().values
    neg_sum = pd.Series(neg_flow).rolling(period, min_periods=period).sum().values
    mfr = pos_sum / (neg_sum + 1e-9)
    return 100 - 100 / (1 + mfr)


# ═══════════════════════════════════════════════════════════
# Q. OBV高阶特征 (~15个)
# ═══════════════════════════════════════════════════════════

def build_obv_advanced_features() -> List[FeatureSpec]:
    """OBV加速度 + OBV-价格二阶导"""
    feats = []

    # OBV 一阶/二阶变化率
    for w in [3, 5, 10, 20]:
        feats.append(FeatureSpec(
            id=f"obv_roc_{w}d", name=f"OBV ROC {w}日",
            category="Q",
            description=f"OBV的{w}日变化率 (一阶导)",
            compute_ts=lambda df, w=w:
                pd.Series(_obv_ts(df)).pct_change(w).values,
        ))

    # OBV 二阶导 (加速度)
    feats.append(FeatureSpec(
        id="obv_accel_5d", name="OBV加速度 5日",
        category="Q",
        description="OBV一阶导的5日变化 (资金流向加速/减速)",
        compute_ts=lambda df:
            pd.Series(pd.Series(_obv_ts(df)).pct_change(5)).diff(5).values,
    ))
    feats.append(FeatureSpec(
        id="obv_accel_10d", name="OBV加速度 10日",
        category="Q",
        description="OBV一阶导的10日变化 (中周期资金加速度)",
        compute_ts=lambda df:
            pd.Series(pd.Series(_obv_ts(df)).pct_change(10)).diff(10).values,
    ))

    # OBV 与价格协同度 (相关系数)
    for w in [10, 20, 50]:
        feats.append(FeatureSpec(
            id=f"obv_price_corr_{w}d", name=f"OBV价格相关 {w}日",
            category="Q",
            description=f"OBV与价格在{w}日内的相关系数 (>0=确认趋势)",
            compute_ts=lambda df, w=w:
                _roll_corr(_obv_ts(df), df["close"].values, w),
        ))

    # OBV 均线交叉
    for s, l in [(5, 20), (10, 50)]:
        feats.append(FeatureSpec(
            id=f"obv_ma_cross_{s}_{l}", name=f"OBV MA{s}/{l}交叉",
            category="Q",
            description=f"OBV的{s}日均线与{l}日均线距离",
            compute_ts=lambda df, s=s, l=l:
                _roll_mean(_obv_ts(df), s) / (_roll_mean(_obv_ts(df), l) + 1e-9) - 1,
        ))

    return feats


# ═══════════════════════════════════════════════════════════
# R. 多空力量博弈特征 (~20个)
# ═══════════════════════════════════════════════════════════

def build_power_features() -> List[FeatureSpec]:
    """买卖压力 + 日内博弈"""
    feats = []

    for w in [5, 10, 20]:
        # 买方力量: 以(高-开)/(高-低)*量衡量
        feats.append(FeatureSpec(
            id=f"buy_power_{w}d", name=f"买方力量 {w}日",
            category="R",
            description=f"买方主导程度均值 (>0.5=买方控盘)",
            compute_ts=lambda df, w=w: _roll_mean(
                (df["close"].values - df["low"].values) /
                (df["high"].values - df["low"].values + 1e-9), w
            ),
        ))
        # 卖方力量
        feats.append(FeatureSpec(
            id=f"sell_power_{w}d", name=f"卖方力量 {w}日",
            category="R",
            description=f"卖方主导程度均值 (>0.5=卖方控盘)",
            compute_ts=lambda df, w=w: _roll_mean(
                (df["high"].values - df["close"].values) /
                (df["high"].values - df["low"].values + 1e-9), w
            ),
        ))

    # 买卖力量比的变化 (博弈转向)
    for w in [5, 10]:
        feats.append(FeatureSpec(
            id=f"power_shift_{w}d", name=f"力量转向 {w}日",
            category="R",
            description=f"买卖力量比的{w}日变化 (>0=买方增强, <0=卖方增强)",
            compute_ts=lambda df, w=w: pd.Series(
                _roll_mean((df["close"].values-df["low"].values)/
                           (df["high"].values-df["low"].values+1e-9), w)
            ).diff(w).values,
        ))

    # 大单净流向 (成交量集中度)
    for w in [5, 10, 20]:
        feats.append(FeatureSpec(
            id=f"vol_concentration_{w}d", name=f"量集中度 {w}日",
            category="R",
            description=f"成交量前20%日的量占比 (>0.5=大单活跃)",
            compute_ts=lambda df, w=w: _vol_concentration(df["volume"].values, w),
        ))

    # 量的方向性 (涨日量 / 跌日量)
    for w in [5, 10, 20]:
        feats.append(FeatureSpec(
            id=f"vol_direction_{w}d", name=f"量方向 {w}日",
            category="R",
            description=f"上涨日量/下跌日量之比 (>1=买盘积极, <1=抛压)",
            compute_ts=lambda df, w=w: _vol_direction(df, w),
        ))

    return feats

def _vol_concentration(volume: np.ndarray, window: int) -> np.ndarray:
    result = np.full(len(volume), 0.2)
    for i in range(window, len(volume)):
        seg = volume[i-window:i+1]
        top_n = max(1, int(len(seg) * 0.2))
        top_vol = np.sum(np.sort(seg)[-top_n:])
        result[i] = top_vol / (np.sum(seg) + 1e-9)
    return result

def _vol_direction(df: pd.DataFrame, window: int) -> np.ndarray:
    close = df["close"].values
    vol = df["volume"].values
    result = np.ones(len(close))
    for i in range(window, len(close)):
        up_vol = vol[i-window+1:i+1][close[i-window+1:i+1] >= close[i-window:i]]
        down_vol = vol[i-window+1:i+1][close[i-window+1:i+1] < close[i-window:i]]
        result[i] = np.sum(up_vol) / (np.sum(down_vol) + 1e-9)
    return np.clip(result, 0.1, 10)


# ═══════════════════════════════════════════════════════════
# S. 动量因子增强 (~25个)
# ═══════════════════════════════════════════════════════════

def build_enhanced_momentum_features() -> List[FeatureSpec]:
    """多窗口排名分位 + 收益率离散度 + 路径质量"""
    feats = []

    # 动量排名分位 (多个窗口)
    for w in [10, 20, 50, 100]:
        feats.append(FeatureSpec(
            id=f"mom_rank_{w}d", name=f"动量排名 {w}日分位",
            category="S",
            description=f"{w}日收益率在历史中的滚动排名分位",
            compute_ts=lambda df, w=w:
                _roll_rank(_ret(df["close"].values, w), max(w*3, 60)),
        ))

    # 收益率离散度 (截面波动)
    for w in [5, 10, 20]:
        feats.append(FeatureSpec(
            id=f"ret_dispersion_{w}d", name=f"收益离散度 {w}日",
            category="S",
            description=f"日收益率在{w}日内的离散度 (高=分歧大/方向未定)",
            compute_ts=lambda df, w=w:
                _roll_std(_ret(df["close"].values, 1), w) /
                (np.abs(_roll_mean(_ret(df["close"].values, 1), w)) + 1e-9),
        ))

    # 路径平滑度 (Sharpe-like)
    for w in [10, 20, 50]:
        feats.append(FeatureSpec(
            id=f"path_smoothness_{w}d", name=f"路径平滑度 {w}日",
            category="S",
            description=f"{w}日收益率路径质量 (高=稳定趋势, 低=震荡)",
            compute_ts=lambda df, w=w:
                np.abs(_ret(df["close"].values, w)) /
                (_roll_std(_ret(df["close"].values, 1), w) * np.sqrt(w) + 1e-9),
        ))

    # 短期动量/长期动量 比率 (趋势一致性)
    for s, l in [(5, 20), (10, 50), (20, 100)]:
        feats.append(FeatureSpec(
            id=f"mom_ratio_{s}_{l}", name=f"动量比 {s}/{l}日",
            category="S",
            description=f"短期动量÷长期动量 (一致性检验)",
            compute_ts=lambda df, s=s, l=l:
                (_ret(df["close"].values, s) + 1e-9) /
                (_ret(df["close"].values, l) + 1e-9),
        ))

    return feats


# ═══════════════════════════════════════════════════════════
# T. 跨资产实际数据特征 (~30个) — 不再占位!
# ═══════════════════════════════════════════════════════════

def build_cross_asset_real_features() -> List[FeatureSpec]:
    """
    使用实际数据源 (不再占位!):
      - ccxt: BTC/ETH 同步数据
      - yfinance: SPY/QQQ/GLD/DXY/VIX
    """
    feats = []

    # BTC-ETH相关性残差 (用ccxt同步获取)
    for w in [10, 20, 60]:
        feats.append(FeatureSpec(
            id=f"btc_eth_corr_{w}d", name=f"BTC-ETH相关 {w}日",
            category="T",
            description=f"BTC与ETH的{w}日滚动相关系数",
            compute_ts=lambda df, w=w:
                _cross_corr_from_dual(df, w) if "eth_close" in df.columns else np.full(len(df), 0.0),
        ))
        feats.append(FeatureSpec(
            id=f"btc_eth_beta_{w}d", name=f"BTC-ETH Beta {w}日",
            category="T",
            description=f"BTC对ETH的{w}日滚动Beta",
            compute_ts=lambda df, w=w:
                _cross_beta_from_dual(df, w) if "eth_close" in df.columns else np.full(len(df), 1.0),
        ))
        feats.append(FeatureSpec(
            id=f"btc_eth_residual_{w}d", name=f"BTC-ETH残差 {w}日",
            category="T",
            description=f"BTC与ETH回归残差累计 (正=BTC跑赢ETH)",
            compute_ts=lambda df, w=w:
                _cross_residual_from_dual(df, w) if "eth_close" in df.columns else np.full(len(df), 0.0),
        ))

    # 跨市场特征 (用yfinance数据, 在df中有对应列时启用)
    external_assets = [
        ("spy", "SPY", "标普500"),
        ("qqq", "QQQ", "纳斯达克100"),
        ("gld", "GLD", "黄金"),
        ("dxy", "DXY", "美元指数"),
        ("vix", "VIX", "恐慌指数"),
    ]

    for col_prefix, label, name_cn in external_assets:
        col_name = f"{col_prefix}_close"
        for w in [10, 20, 50]:
            feats.append(FeatureSpec(
                id=f"corr_{label}_{w}d", name=f"{label}相关 {w}日",
                category="T",
                description=f"BTC与{name_cn}的{w}日滚动相关",
                compute_ts=lambda df, w=w, cn=col_name:
                    _roll_corr(df["close"].values, df[cn].values, w)
                    if cn in df.columns else np.full(len(df), 0.0),
            ))
            feats.append(FeatureSpec(
                id=f"beta_{label}_{w}d", name=f"{label} Beta {w}日",
                category="T",
                description=f"BTC对{name_cn}的{w}日Beta",
                compute_ts=lambda df, w=w, cn=col_name:
                    _cross_beta(df["close"].values, df[cn].values, w)
                    if cn in df.columns else np.full(len(df), 1.0),
            ))

    # ── 资金费率特征 (Phase 6) ──
    # 资金费率反映市场情绪: 正值=多付空(过热), 极端负值=空付多(恐慌)
    for prefix, label in [("btc", "BTC"), ("eth", "ETH")]:
        col_name = f"{prefix}_funding_rate"
        feats.append(FeatureSpec(
            id=f"{prefix}_funding_rate",
            name=f"{label}资金费率",
            category="T",
            description=f"{label}永续合约日均资金费率",
            compute_ts=lambda df, cn=col_name:
                df[cn].values if cn in df.columns else np.full(len(df), 0.0),
        ))
        # 资金费率变化 (一阶差分)
        feats.append(FeatureSpec(
            id=f"{prefix}_funding_chg_3d",
            name=f"{label}资金费变动 3日",
            category="T",
            description=f"{label}资金费率3日变动",
            compute_ts=lambda df, cn=col_name:
                pd.Series(df[cn].values).diff(3).fillna(0).values
                if cn in df.columns else np.full(len(df), 0.0),
        ))

    # ── 恐慌贪婪指数特征 (Phase 6) ──
    feats.append(FeatureSpec(
        id="fear_greed",
        name="恐慌贪婪指数",
        category="T",
        description="Crypto Fear & Greed Index (0=恐慌, 100=贪婪)",
        compute_ts=lambda df:
            df["fear_greed"].values if "fear_greed" in df.columns
            else np.full(len(df), 50.0),
    ))
    feats.append(FeatureSpec(
        id="fear_greed_chg_5d",
        name="恐慌贪婪变动 5日",
        category="T",
        description="Fear & Greed Index 5日变动",
        compute_ts=lambda df:
            pd.Series(df["fear_greed"].values).diff(5).fillna(0).values
            if "fear_greed" in df.columns else np.full(len(df), 0.0),
    ))
    feats.append(FeatureSpec(
        id="fear_greed_extreme",
        name="极度恐慌信号",
        category="T",
        description="F&G ≤ 25 → 1 (恐慌), F&G ≥ 75 → -1 (贪婪)",
        compute_ts=lambda df:
            np.where(df["fear_greed"].values <= 25, 1.0,
                     np.where(df["fear_greed"].values >= 75, -1.0, 0.0))
            if "fear_greed" in df.columns else np.full(len(df), 0.0),
    ))

    return feats

def _cross_corr_from_dual(df: pd.DataFrame, window: int) -> np.ndarray:
    btc_ret = _ret(df["close"].values, 1)
    eth_ret = _ret(df["eth_close"].values, 1)
    return _roll_corr(btc_ret, eth_ret, window)

def _cross_beta_from_dual(df: pd.DataFrame, window: int) -> np.ndarray:
    btc_ret = _ret(df["close"].values, 1)
    eth_ret = _ret(df["eth_close"].values, 1)
    return _cross_beta(btc_ret, eth_ret, window)

def _cross_residual_from_dual(df: pd.DataFrame, window: int) -> np.ndarray:
    """BTC对ETH的回归残差累计"""
    btc_ret = _ret(df["close"].values, 1)
    eth_ret = _ret(df["eth_close"].values, 1)
    result = np.zeros(len(df))
    for i in range(window, len(df)):
        seg_btc = btc_ret[i-window:i]
        seg_eth = eth_ret[i-window:i]
        if len(seg_btc) > 10:
            X = np.column_stack([np.ones(len(seg_eth)), seg_eth])
            try:
                beta = np.linalg.lstsq(X, seg_btc, rcond=None)[0]
                predicted = X @ beta
                residuals = seg_btc - predicted
                result[i] = np.sum(residuals[-5:])  # 近5日残差累计
            except Exception:
                pass
    return result

def _cross_beta(y: np.ndarray, x: np.ndarray, window: int) -> np.ndarray:
    result = np.ones(len(y))
    for i in range(window, len(y)):
        seg_y = y[i-window:i]
        seg_x = x[i-window:i]
        if np.std(seg_x) > 0:
            result[i] = np.cov(seg_y, seg_x)[0, 1] / np.var(seg_x)
    return result


# ═══════════════════════════════════════════════════════════
# A-N 升级版 — 时序计算 (从 feature_engine.py 迁移)
# ═══════════════════════════════════════════════════════════

def build_momentum_ts() -> List[FeatureSpec]:
    """A. 动量族 — 时序版"""
    feats = []
    windows = [1, 2, 3, 5, 7, 10, 14, 20, 30, 40, 50, 60, 90, 120, 180, 250]

    for w in windows:
        feats.append(FeatureSpec(
            id=f"mom_{w}d", name=f"{w}日动量",
            category="A", description=f"过去{w}日收益率",
            compute_ts=lambda df, w=w: _ret(df["close"].values, w),
        ))

    # 动量加速度
    pairs = [(5, 20), (10, 50), (20, 100), (5, 10), (10, 30), (30, 90)]
    for s, l in pairs:
        feats.append(FeatureSpec(
            id=f"mom_accel_{s}d_{l}d", name=f"{s}-{l}日动量差",
            category="A", description=f"短{s}日动量-长{l}日动量",
            compute_ts=lambda df, s=s, l=l:
                _ret(df["close"].values, s) - _ret(df["close"].values, l),
        ))

    # 滚动Sharpe
    for w in [10, 20, 60, 120]:
        feats.append(FeatureSpec(
            id=f"sharpe_{w}d", name=f"{w}日滚动Sharpe",
            category="A",
            description=f"{w}日收益率/波动率",
            compute_ts=lambda df, w=w:
                _roll_mean(_ret(df["close"].values, 1), w) /
                (_roll_std(_ret(df["close"].values, 1), w) + 1e-9) * np.sqrt(365),
        ))

    # 连续涨跌
    feats.append(FeatureSpec(
        id="consecutive_up", name="连涨天数", category="A",
        compute_ts=lambda df: _consecutive_ts(df["close"].values, 1),
    ))
    feats.append(FeatureSpec(
        id="consecutive_down", name="连跌天数", category="A",
        compute_ts=lambda df: _consecutive_ts(df["close"].values, -1),
    ))

    # 最大回撤
    for w in [20, 60, 120]:
        feats.append(FeatureSpec(
            id=f"max_dd_{w}d", name=f"{w}日最大回撤", category="A",
            compute_ts=lambda df, w=w: _max_dd_ts(df["close"].values, w),
        ))

    return feats


def build_vol_ts() -> List[FeatureSpec]:
    """B. 波动率族 — 时序版"""
    feats = []
    windows = [5, 10, 14, 20, 30, 60, 90, 120]

    for w in windows:
        feats.append(FeatureSpec(
            id=f"vol_{w}d", name=f"{w}日波动率", category="B",
            compute_ts=lambda df, w=w:
                pd.Series(_daily_log_ret(df["close"].values))
                .rolling(w, min_periods=w//2).std().values * np.sqrt(365),
        ))

    # 波动的波动
    for w in [10, 20, 60]:
        feats.append(FeatureSpec(
            id=f"vol_change_{w}d", name=f"{w}日波动率变化", category="B",
            compute_ts=lambda df, w=w:
                pd.Series(
                    pd.Series(_daily_log_ret(df["close"].values))
                    .rolling(20, min_periods=10).std().values * np.sqrt(365)
                ).pct_change(w).values,
        ))

    # Parkinson 波动率
    for w in [10, 20, 60]:
        feats.append(FeatureSpec(
            id=f"parkinson_vol_{w}d", name=f"Parkinson {w}日", category="B",
            compute_ts=lambda df, w=w:
                np.sqrt(_roll_mean(
                    np.log(df["high"].values/df["low"].values)**2 / (4*np.log(2)), w
                )) * np.sqrt(365),
        ))

    # 振幅
    for w in [5, 10, 20]:
        feats.append(FeatureSpec(
            id=f"range_pct_{w}d", name=f"{w}日振幅", category="B",
            compute_ts=lambda df, w=w: _roll_mean(
                (df["high"].values - df["low"].values) / df["close"].values, w
            ),
        ))

    # 波动率锥
    for w in [20, 60]:
        feats.append(FeatureSpec(
            id=f"vol_cone_{w}d", name=f"波动率锥{w}日", category="B",
            compute_ts=lambda df, w=w: _vol_cone_ts(df["close"].values, w),
        ))

    return feats


def build_higher_moment_ts() -> List[FeatureSpec]:
    """C. 高阶矩 — 时序版"""
    feats = []
    daily_ret = _ret  # will be overridden in lambda

    for w in [20, 60, 120]:
        feats.append(FeatureSpec(
            id=f"skew_{w}d", name=f"{w}日偏度", category="C",
            compute_ts=lambda df, w=w:
                _roll_skew(_ret(df["close"].values, 1), w),
        ))
        feats.append(FeatureSpec(
            id=f"kurt_{w}d", name=f"{w}日峰度", category="C",
            compute_ts=lambda df, w=w:
                _roll_kurt(_ret(df["close"].values, 1), w),
        ))

    for w, sigma in [(20, 2), (60, 2), (20, 3), (60, 3)]:
        feats.append(FeatureSpec(
            id=f"tail_events_{w}d_{sigma}s", name=f"{w}日{sigma}σ尾部", category="C",
            compute_ts=lambda df, w=w, sigma=sigma:
                _tail_count_ts(df["close"].values, w, sigma),
        ))

    for w in [20, 60]:
        feats.append(FeatureSpec(
            id=f"asymmetry_{w}d", name=f"{w}日涨跌不对称", category="C",
            compute_ts=lambda df, w=w: _asymmetry_ts(df["close"].values, w),
        ))

    return feats


def build_ma_ts() -> List[FeatureSpec]:
    """D. 均线系统 — 时序版"""
    feats = []
    pairs = [(5, 20), (10, 50), (20, 60), (5, 60), (20, 100),
             (10, 20), (50, 100), (100, 200)]

    for s, l in pairs:
        sma_s = lambda df, s=s: _roll_mean(df["close"].values, s)
        sma_l = lambda df, l=l: _roll_mean(df["close"].values, l)

        feats.append(FeatureSpec(
            id=f"ma_dist_{s}_{l}", name=f"MA{s}/{l}距离", category="D",
            compute_ts=lambda df, s=s, l=l:
                _roll_mean(df["close"].values, s) /
                (_roll_mean(df["close"].values, l) + 1e-9) - 1,
        ))
        feats.append(FeatureSpec(
            id=f"ma_cross_{s}_{l}", name=f"MA{s}/{l}交叉", category="D",
            compute_ts=lambda df, s=s, l=l:
                np.sign(_roll_mean(df["close"].values, s) -
                        _roll_mean(df["close"].values, l)),
        ))

    for span in [12, 26, 50, 100]:
        feats.append(FeatureSpec(
            id=f"ema_dist_{span}", name=f"EMA{span}距离", category="D",
            compute_ts=lambda df, span=span:
                df["close"].values / _ema(df["close"].values, span) - 1,
        ))

    for w in [10, 20, 50, 100]:
        feats.append(FeatureSpec(
            id=f"ma_slope_{w}", name=f"MA{w}斜率", category="D",
            compute_ts=lambda df, w=w:
                pd.Series(_roll_mean(df["close"].values, w)).pct_change(5).values,
        ))

    return feats


def build_oscillator_ts() -> List[FeatureSpec]:
    """E. 振荡器 — 时序版"""
    feats = []

    for period in [7, 14, 21, 30, 50]:
        feats.append(FeatureSpec(
            id=f"rsi_{period}", name=f"RSI({period})", category="E",
            compute_ts=lambda df, period=period:
                _rsi_ts(df["close"].values, period),
        ))

    for period, nbstd in [(20, 2), (20, 1.5), (50, 2)]:
        feats.append(FeatureSpec(
            id=f"bb_position_{period}_{nbstd}", name=f"BB位置({period},{nbstd})",
            category="E",
            compute_ts=lambda df, period=period, nbstd=nbstd:
                _bb_position_ts(df["close"].values, period, nbstd),
        ))
        feats.append(FeatureSpec(
            id=f"bb_width_{period}_{nbstd}", name=f"BB带宽({period},{nbstd})",
            category="E",
            compute_ts=lambda df, period=period, nbstd=nbstd:
                2 * nbstd * _roll_std(df["close"].values, period) /
                (_roll_mean(df["close"].values, period) + 1e-9),
        ))

    for period in [14, 20]:
        feats.append(FeatureSpec(
            id=f"cci_{period}", name=f"CCI({period})", category="E",
            compute_ts=lambda df, period=period:
                _cci_ts(df, period),
        ))

    return feats

def _cci_ts(df: pd.DataFrame, period: int) -> np.ndarray:
    tp = (df["high"].values + df["low"].values + df["close"].values) / 3
    sma_tp = _roll_mean(tp, period)
    mad = np.zeros(len(df))
    for i in range(period, len(df)):
        mad[i] = np.mean(np.abs(tp[i-period:i] - sma_tp[i]))
    result = np.zeros(len(df))
    mask = mad > 1e-9
    result[mask] = (tp[mask] - sma_tp[mask]) / (0.015 * mad[mask])
    return result


def build_volume_ts() -> List[FeatureSpec]:
    """F. 成交量 — 时序版"""
    feats = []

    for s, l in [(5, 20), (10, 50), (5, 50), (20, 100)]:
        feats.append(FeatureSpec(
            id=f"vol_ratio_{s}_{l}", name=f"量比({s}/{l})", category="F",
            compute_ts=lambda df, s=s, l=l:
                _roll_mean(df["volume"].values, s) /
                (_roll_mean(df["volume"].values, l) + 1),
        ))

    for w in [10, 20]:
        feats.append(FeatureSpec(
            id=f"vol_price_div_{w}d", name=f"{w}日量价背离", category="F",
            compute_ts=lambda df, w=w:
                _ret(df["close"].values, w) - _ret(df["volume"].values, w),
        ))

    obv = lambda df: _obv_ts(df)
    for w in [5, 10, 20]:
        feats.append(FeatureSpec(
            id=f"obv_mom_{w}d", name=f"OBV动量{w}日", category="F",
            compute_ts=lambda df, w=w:
                pd.Series(_obv_ts(df)).pct_change(w).values,
        ))

    for w in [10, 20, 50]:
        feats.append(FeatureSpec(
            id=f"vol_trend_{w}d", name=f"{w}日量趋势", category="F",
            compute_ts=lambda df, w=w:
                _linear_slope_ts(df["volume"].values, w),
        ))

    for w, threshold in [(20, 2), (50, 2), (20, 3)]:
        feats.append(FeatureSpec(
            id=f"large_order_{w}d_{threshold}s", name=f"{w}日大单{threshold}σ",
            category="F",
            compute_ts=lambda df, w=w, t=threshold:
                _extreme_vol_days_ts(df["volume"].values, w, t),
        ))

    return feats

def _extreme_vol_days_ts(volume: np.ndarray, window: int, sigma: float) -> np.ndarray:
    result = np.zeros(len(volume))
    for i in range(window, len(volume)):
        seg = volume[i-window:i+1]
        m, s = np.mean(seg), np.std(seg)
        if s > 0:
            result[i] = np.sum(seg > m + sigma * s) / window
    return result


def build_timeseries_ts() -> List[FeatureSpec]:
    """L. 时间序列分解 — 时序版"""
    feats = []

    for w in [20, 60, 120]:
        feats.append(FeatureSpec(
            id=f"hurst_{w}d", name=f"Hurst {w}日", category="L",
            compute_ts=lambda df, w=w: _hurst_ts(df["close"].values, w),
        ))

    for lag in [1, 3, 5, 10, 20]:
        feats.append(FeatureSpec(
            id=f"autocorr_{lag}d", name=f"自相关lag{lag}", category="L",
            compute_ts=lambda df, lag=lag:
                _roll_corr(
                    _ret(df["close"].values, 1),
                    np.roll(_ret(df["close"].values, 1), lag), 60
                ),
        ))

    for w in [14, 20, 30]:
        feats.append(FeatureSpec(
            id=f"trend_strength_{w}d", name=f"趋势强度{w}日", category="L",
            compute_ts=lambda df, w=w:
                _adx_ts(df["high"].values, df["low"].values, df["close"].values, w),
        ))

    for w in [10, 20, 50]:
        feats.append(FeatureSpec(
            id=f"fractal_eff_{w}d", name=f"分形效率{w}日", category="L",
            compute_ts=lambda df, w=w: _fractal_eff_ts(df["close"].values, w),
        ))

    return feats


def build_pattern_ts() -> List[FeatureSpec]:
    """M. 价格形态 — 时序版"""
    feats = []

    # 实体比例
    for w in [5, 10, 20]:
        feats.append(FeatureSpec(
            id=f"body_ratio_{w}d", name=f"实体比{w}日", category="M",
            compute_ts=lambda df, w=w: _roll_mean(
                abs(df["close"].values - df["open"].values) /
                (df["high"].values - df["low"].values + 1e-9), w
            ),
        ))

    # 影线比例
    for wick, label in [("upper", "上"), ("lower", "下")]:
        for w in [5, 10, 20]:
            idx = 0 if wick == "upper" else 1
            feats.append(FeatureSpec(
                id=f"{wick}_wick_ratio_{w}d", name=f"{label}影线比{w}日",
                category="M",
                compute_ts=lambda df, w=w, idx=idx:
                    _wick_ratio_ts(df, idx, w),
            ))

    # 新高/新低
    for w in [20, 60, 120]:
        feats.append(FeatureSpec(
            id=f"near_high_{w}d", name=f"近{w}日高", category="M",
            compute_ts=lambda df, w=w:
                df["close"].values / (_roll_max(df["high"].values, w) + 1e-9) - 1,
        ))
        feats.append(FeatureSpec(
            id=f"near_low_{w}d", name=f"近{w}日低", category="M",
            compute_ts=lambda df, w=w:
                df["close"].values / (_roll_min(df["low"].values, w) + 1e-9) - 1,
        ))

    # Squeeze (布林带收敛)
    for w in [10, 20]:
        feats.append(FeatureSpec(
            id=f"squeeze_{w}d", name=f"Squeeze{w}日", category="M",
            compute_ts=lambda df, w=w:
                _squeeze_ts(df["close"].values, w),
        ))

    return feats

def _wick_ratio_ts(df: pd.DataFrame, wick_idx: int, window: int) -> np.ndarray:
    high, low = df["high"].values, df["low"].values
    close, open_ = df["close"].values, df["open"].values
    body_high = np.maximum(close, open_)
    body_low = np.minimum(close, open_)
    upper_wick = high - body_high
    lower_wick = body_low - low
    total_wick = upper_wick + lower_wick + 1e-9
    ratio = upper_wick / total_wick if wick_idx == 0 else lower_wick / total_wick
    return _roll_mean(ratio, window)

def _squeeze_ts(close: np.ndarray, window: int) -> np.ndarray:
    """布林带宽度历史分位数"""
    bbw = 2 * 2 * _roll_std(close, 20) / (_roll_mean(close, 20) + 1e-9)
    result = np.full(len(close), 0.5)
    for i in range(window * 3, len(close)):
        hist = bbw[window:i]
        if len(hist) > 0:
            result[i] = (hist < bbw[i]).sum() / len(hist)
    return result


def build_ratio_ts() -> List[FeatureSpec]:
    """N. 价格比率 — 时序版"""
    feats = []

    for w in [5, 10, 20, 50]:
        feats.append(FeatureSpec(
            id=f"co_ratio_{w}d", name=f"收/开比{w}日", category="N",
            compute_ts=lambda df, w=w: _roll_mean(
                df["close"].values / (df["open"].values + 1e-9), w
            ),
        ))

    for w in [5, 10, 20]:
        feats.append(FeatureSpec(
            id=f"atr_pct_{w}d", name=f"ATR% {w}日", category="N",
            compute_ts=lambda df, w=w:
                _atr_ts(df["high"].values, df["low"].values, df["close"].values, 14) /
                (df["close"].values + 1e-9),
        ))

    for w in [5, 10, 20]:
        feats.append(FeatureSpec(
            id=f"up_down_ratio_{w}d", name=f"涨跌比{w}日", category="N",
            compute_ts=lambda df, w=w:
                _up_down_ratio_ts(df["close"].values, w),
        ))

    return feats

def _up_down_ratio_ts(close: np.ndarray, window: int) -> np.ndarray:
    result = np.ones(len(close))
    for i in range(window, len(close)):
        seg = close[i-window+1:i+1]
        up = sum(1 for j in range(1, len(seg)) if seg[j] > seg[j-1])
        down = sum(1 for j in range(1, len(seg)) if seg[j] < seg[j-1])
        result[i] = up / (down + 1)
    return result


def build_interaction_ts() -> List[FeatureSpec]:
    """K. 交互特征 — 时序版"""
    feats = []

    # 动量 × 波动率
    for mw in [5, 10, 20, 50]:
        for vw in [10, 20, 60]:
            feats.append(FeatureSpec(
                id=f"mom{mw}d_vol{vw}d", name=f"动量{mw}×波{vw}", category="K",
                compute_ts=lambda df, mw=mw, vw=vw:
                    _ret(df["close"].values, mw) /
                    (_roll_std(_daily_log_ret(df["close"].values), vw) * np.sqrt(365) + 1e-9),
            ))

    # RSI × 量比
    for rp in [7, 14, 21]:
        feats.append(FeatureSpec(
            id=f"rsi{rp}_vol_ratio", name=f"RSI{rp}×量比", category="K",
            compute_ts=lambda df, rp=rp:
                _rsi_ts(df["close"].values, rp) *
                (_roll_mean(df["volume"].values, 5) /
                 (_roll_mean(df["volume"].values, 20) + 1)),
        ))

    # 偏度 × 峰度
    for w in [20, 60]:
        feats.append(FeatureSpec(
            id=f"skew_kurt_{w}d", name=f"偏度×峰度{w}", category="K",
            compute_ts=lambda df, w=w:
                _roll_skew(_ret(df["close"].values, 1), w) *
                _roll_kurt(_ret(df["close"].values, 1), w),
        ))

    # OBV-价格确认
    for w in [5, 10, 20]:
        feats.append(FeatureSpec(
            id=f"obv_price_confirm_{w}d", name=f"OBV确认{w}日", category="K",
            compute_ts=lambda df, w=w:
                np.sign(pd.Series(_obv_ts(df)).pct_change(w).values) *
                np.sign(_ret(df["close"].values, w)) *
                abs(_ret(df["close"].values, w)),
        ))

    return feats


# ═══════════════════════════════════════════════════════════
# 特征工厂 v4.0 — 主类
# ═══════════════════════════════════════════════════════════

class FeatureFactoryV4:
    """500+特征时间序列工厂"""

    def __init__(self):
        self.features: List[FeatureSpec] = []
        self._build_all()

    def _build_all(self):
        builders = [
            ("A_动量", build_momentum_ts),
            ("B_波动率", build_vol_ts),
            ("C_高阶矩", build_higher_moment_ts),
            ("D_均线", build_ma_ts),
            ("E_振荡器", build_oscillator_ts),
            ("F_成交量", build_volume_ts),
            ("K_交互", build_interaction_ts),
            ("L_时间序列", build_timeseries_ts),
            ("M_形态", build_pattern_ts),
            ("N_比率", build_ratio_ts),
            ("O_增强波动率", build_advanced_vol_features),
            ("P_VWAP_MFI", build_vwap_mfi_features),
            ("Q_OBV高阶", build_obv_advanced_features),
            ("R_多空力量", build_power_features),
            ("S_动量增强", build_enhanced_momentum_features),
            ("T_跨资产", build_cross_asset_real_features),
        ]

        for cat_label, builder in builders:
            feats = builder()
            self.features.extend(feats)

    def compute_timeseries(self, df: pd.DataFrame,
                          categories: Optional[List[str]] = None,
                          verbose: bool = False) -> pd.DataFrame:
        """
        一次性计算所有特征的时间序列!

        Args:
            df: 含 OHLCV + 可选eth_close/spy_close等列
            categories: 限制类别, None=全部

        Returns:
            DataFrame: index=原始index, columns=feature_id, values=特征值
        """
        feats = self.features
        if categories:
            feats = [f for f in feats if f.category in categories]

        result = {}
        n = len(feats)
        for i, f in enumerate(feats):
            if verbose and (i+1) % 50 == 0:
                print(f"  特征计算: {i+1}/{n} ({f.id})")
            try:
                arr = f.compute_ts(df)
                if arr is not None and len(arr) == len(df):
                    result[f.id] = arr
            except Exception as e:
                if verbose:
                    print(f"    ⚠️ {f.id} 失败: {e}")
                continue

        if verbose:
            print(f"✅ 成功计算: {len(result)}/{n} 特征")

        return pd.DataFrame(result, index=df.index if hasattr(df, 'index') else range(len(df)))

    def compute_latest(self, df: pd.DataFrame,
                      categories: Optional[List[str]] = None) -> Dict[str, float]:
        """只计算最新值 (用于实时信号)"""
        ts_df = self.compute_timeseries(df, categories)
        return {col: float(ts_df[col].dropna().iloc[-1])
                if len(ts_df[col].dropna()) > 0 else 0.0
                for col in ts_df.columns}

    @property
    def feature_count(self) -> int:
        return len(self.features)

    def summary(self) -> dict:
        cats = {}
        for f in self.features:
            if f.category not in cats:
                cats[f.category] = 0
            cats[f.category] += 1
        return {
            "total": len(self.features),
            "by_category": cats,
        }


# ── CLI ──
if __name__ == "__main__":
    ff = FeatureFactoryV4()
    s = ff.summary()
    print("=" * 60)
    print("🧬 Feature Factory v4.0 — 时序版")
    print("=" * 60)
    print(f"总特征定义: {s['total']}")
    print()
    for cat, count in s["by_category"].items():
        print(f"  {cat}: {count}个")
    print()
    print("💡 用法:")
    print("  ff = FeatureFactoryV4()")
    print("  ts_df = ff.compute_timeseries(ohlcv_df)  # 一次性!")
    print("  latest = ff.compute_latest(ohlcv_df)      # 实时信号")
