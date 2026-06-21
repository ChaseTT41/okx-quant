#!/usr/bin/env python3
"""
🐼 熊猫教练交易策略引擎 — 三指标共振 + 多时间框架 + 背离确认
Chase哥 2026-06-21 要求：10u保证金起步，可变杠杆，按企微推送频率

核心理念（蒸馏自熊猫交易学社250+期视频）:
  1. 背离 > 零轴位置 > 金叉死叉 > 柱状图
  2. "没有任何一个技术指标可以独立用于实盘交易"
  3. 三指标共振缺一不可: MACD定方向 + KDJ找时机 + RSI验力度
  4. "第一次机会往往是陷阱" — 等复杂回调
  5. "趋势一旦破坏，无条件空仓"
"""

import sys
import os
import json
import time
import hashlib
import hmac
import base64
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import requests
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════

# 交易参数 (Chase哥指定)
MARGIN_PER_TRADE = 10.0       # 每笔保证金 10 USDT
MIN_CONFIDENCE = 65           # 最低信心分才开仓 (满分100)
MAX_POSITIONS = 3             # 最多同时持有3个熊猫策略仓位

# 杠杆映射: 信心分 → (杠杆倍数, 仓位占比)
# 信心越高 → 杠杆越高 → 同样10u保证金能开更大名义价值
LEVERAGE_TABLE = [
    (90, 15),   # ≥90分 → 15x (三指标共振+多TF一致+背离确认)
    (80, 10),   # ≥80分 → 10x (三指标共振+多TF一致)
    (70, 7),    # ≥70分 → 7x  (三指标共振，TF部分分歧)
    (65, 5),    # ≥65分 → 5x  (两指标共振+TF基本一致)
]

# 止损: ATR倍数 (熊猫教练: 止损在结构破坏位)
STOP_LOSS_ATR_MULT = 1.5
# 止盈: 风险比 (盈亏比≥2:1才出手)
TAKE_PROFIT_RISK_RATIO = 2.0

# 扫描币种
WATCH_COINS = ["BTC", "ETH", "SOL"]
# 时间框架 (熊猫教练: 顺大逆小 — 日线定方向，4H找结构，1H找入场)
TIMEFRAMES = ["1h", "4h", "1d"]

# OKX API 配置 (从环境变量或 leverage_engine 获取)
OKX_HOST = "www.okx.cab"

# 企业微信 Webhook
WECOM_WEBHOOK = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=2c602b48-5da2-4989-9193-30c0e226c769"


# ═══════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════

@dataclass
class TimeframeSignal:
    """单个时间框架的信号"""
    tf: str
    # MACD
    macd_dif: float
    macd_dea: float
    macd_hist: float
    macd_above_zero: bool       # DIF在零轴上?
    macd_golden_cross: bool     # 刚金叉?
    macd_death_cross: bool      # 刚死叉?
    macd_bull: bool             # 多方(DIF>DEA)?
    macd_top_divergence: bool   # 顶背离
    macd_bottom_divergence: bool # 底背离

    # KDJ
    kdj_k: float
    kdj_d: float
    kdj_j: float
    kdj_golden_cross: bool
    kdj_death_cross: bool
    kdj_bull: bool              # K>D?
    kdj_oversold: bool          # K<20
    kdj_overbought: bool        # K>80

    # RSI
    rsi: float
    rsi_above_50: bool
    rsi_oversold: bool          # <30
    rsi_overbought: bool        # >70

    # EMA
    ema_bull: bool              # 价格在EMA9/20上方?
    ema9_20_sticky: bool        # EMA9/20粘合(<0.5%)

    # 布林
    bb_position_pct: float
    bb_squeeze: bool            # 带宽收窄?

    # 量价
    vol_ratio: float
    vol_price_healthy: bool     # 价涨量增?

    # ADX
    adx: float
    adx_trending: bool          # >25?

    # 综合
    three_resonance_bull: bool  # 三指标共振做多
    three_resonance_bear: bool  # 三指标共振做空
    resonance_count: int        # 共振指标数(0-3)


@dataclass
class PandaSignal:
    """熊猫策略综合信号"""
    symbol: str
    timestamp: str

    # 多时间框架信号
    tf_signals: Dict[str, TimeframeSignal] = field(default_factory=dict)

    # 综合评分
    direction: str = "neutral"   # "long" / "short" / "neutral"
    confidence: float = 0.0      # 0-100
    suggested_leverage: int = 5

    # 多TF对齐
    tf_alignment_score: int = 0  # 0-12 (3 TF × 4分满分)
    tf_alignment_detail: str = ""

    # 关键信号
    has_divergence: bool = False
    divergence_type: str = ""    # "top" / "bottom" / ""
    entry_ready: bool = False    # 可以入场?

    # 风险
    stop_loss_pct: float = 0.0
    risk_warnings: List[str] = field(default_factory=list)

    # 摘要
    summary: str = ""


# ═══════════════════════════════════════════════════════════
# K线数据获取
# ═══════════════════════════════════════════════════════════

def fetch_klines(symbol: str, timeframe: str = "1h", limit: int = 100) -> pd.DataFrame:
    """从 Binance 获取K线"""
    tf_map = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
    interval = tf_map.get(timeframe, "1h")

    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": f"{symbol}USDT", "interval": interval, "limit": limit}

    resp = requests.get(url, params=params, timeout=15)
    data = resp.json()

    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_vol', 'trades', 'taker_buy_vol',
        'taker_buy_quote_vol', 'ignore'
    ])

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    return df


# ═══════════════════════════════════════════════════════════
# 指标计算 (与 chart_vision_analyzer.py 保持一致)
# ═══════════════════════════════════════════════════════════

def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算全部技术指标"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    vol = df['volume'].values

    # EMA
    for p in [9, 20, 50, 200]:
        df[f'ema_{p}'] = pd.Series(close).ewm(span=p, adjust=False).mean().values

    # MACD (12/26/9)
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
    df['macd_dif'] = (ema12 - ema26).values
    df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean().values
    df['macd_hist'] = 2 * (df['macd_dif'] - df['macd_dea']).values

    # RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, 100), where=avg_loss != 0)
    df['rsi'] = 100 - (100 / (1 + rs))

    # KDJ (9,3,3)
    n = 9
    low_n = pd.Series(low).rolling(window=n).min().values
    high_n = pd.Series(high).rolling(window=n).max().values
    rsv = np.where(high_n != low_n, (close - low_n) / (high_n - low_n) * 100, 50)

    k = np.full_like(rsv, 50.0)
    d = np.full_like(rsv, 50.0)
    first_valid = next((i for i in range(len(rsv)) if not np.isnan(rsv[i])), 0)
    for i in range(first_valid, len(rsv)):
        rsv_val = rsv[i] if not np.isnan(rsv[i]) else 50.0
        k[i] = 2/3 * k[i-1] + 1/3 * rsv_val
        d[i] = 2/3 * d[i-1] + 1/3 * k[i]
    j = 3 * k - 2 * d
    df['kdj_k'] = k
    df['kdj_d'] = d
    df['kdj_j'] = j

    # 布林带 (20,2)
    sma20 = pd.Series(close).rolling(window=20).mean().values
    std20 = pd.Series(close).rolling(window=20).std().values
    df['bb_mid'] = sma20
    df['bb_up'] = sma20 + 2 * std20
    df['bb_low'] = sma20 - 2 * std20
    df['bb_width'] = (df['bb_up'] - df['bb_low']) / df['bb_mid'] * 100

    # ATR (14)
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    df['atr'] = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values

    # ADX (14)
    n_adx = 14
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low),
                        np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)),
                         np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = dm_minus[0] = 0
    atr_smooth = pd.Series(tr).ewm(alpha=1/n_adx, adjust=False).mean().values
    di_plus = pd.Series(dm_plus).ewm(alpha=1/n_adx, adjust=False).mean().values / atr_smooth * 100
    di_minus = pd.Series(dm_minus).ewm(alpha=1/n_adx, adjust=False).mean().values / atr_smooth * 100
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10) * 100
    df['adx'] = pd.Series(dx).ewm(alpha=1/n_adx, adjust=False).mean().values
    df['di_plus'] = di_plus
    df['di_minus'] = di_minus

    # 成交量
    df['vol_ma20'] = pd.Series(vol).rolling(20).mean().values

    return df


# ═══════════════════════════════════════════════════════════
# 单时间框架信号检测
# ═══════════════════════════════════════════════════════════

def detect_timeframe_signal(df: pd.DataFrame, tf: str) -> TimeframeSignal:
    """检测单个时间框架的完整信号"""
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # ── MACD 检测 ──
    dif_now = last['macd_dif']
    dif_prev = prev['macd_dif']
    dea_now = last['macd_dea']
    dea_prev = prev['macd_dea']

    macd_above_zero = dif_now > 0
    macd_golden = dif_prev <= dea_prev and dif_now > dea_now
    macd_death = dif_prev >= dea_prev and dif_now < dea_now
    macd_bull = dif_now > dea_now

    # MACD 背离检测
    lookback = min(30, len(df) - 1)
    recent_close = df['close'].iloc[-lookback:].values
    recent_dif = df['macd_dif'].iloc[-lookback:].values

    macd_top_div = False
    macd_bottom_div = False

    price_high_idx = np.argmax(recent_close)
    dif_at_price_high = recent_dif[price_high_idx]
    if price_high_idx > lookback * 0.5 and recent_dif[-1] < dif_at_price_high * 0.8:
        macd_top_div = True

    price_low_idx = np.argmin(recent_close)
    dif_at_price_low = recent_dif[price_low_idx]
    if price_low_idx > lookback * 0.5 and recent_dif[-1] > dif_at_price_low * 1.2:
        macd_bottom_div = True

    # ── KDJ 检测 ──
    k_now = last['kdj_k']
    k_prev = prev['kdj_k']
    d_now = last['kdj_d']
    d_prev = prev['kdj_d']

    kdj_golden = k_prev <= d_prev and k_now > d_now
    kdj_death = k_prev >= d_prev and k_now < d_now
    kdj_bull = k_now > d_now
    kdj_oversold = k_now < 20
    kdj_overbought = k_now > 80

    # ── RSI ──
    rsi_now = last['rsi']
    rsi_above_50 = rsi_now > 50
    rsi_oversold = rsi_now < 30
    rsi_overbought = rsi_now > 70

    # ── EMA ──
    ema_bull = last['close'] > last['ema_9'] and last['close'] > last['ema_20']
    ema9_20_diff = abs(last['ema_9'] - last['ema_20']) / last['ema_20'] * 100
    ema9_20_sticky = ema9_20_diff < 0.5

    # ── 布林 ──
    bb_pos = (last['close'] - last['bb_low']) / (last['bb_up'] - last['bb_low']) * 100
    bb_squeeze = df['bb_width'].iloc[-1] < df['bb_width'].iloc[-5]

    # ── 量价 ──
    vol_ratio = last['volume'] / last['vol_ma20'] if last['vol_ma20'] > 0 else 1
    price_up = last['close'] > prev['close']
    vol_healthy = price_up and vol_ratio > 1.0

    # ── ADX ──
    adx_now = last['adx']
    adx_trending = adx_now > 25

    # ── 三指标共振 ──
    # 做多共振: MACD多方 + KDJ多方(K>D且非超买) + RSI中性(30-70)
    macd_bull_signal = macd_bull and not macd_top_div
    kdj_bull_signal = kdj_bull and not kdj_overbought
    rsi_neutral = 30 < rsi_now < 70

    resonance_bull = macd_bull_signal and kdj_bull_signal and rsi_neutral

    # 做空共振: MACD空方 + KDJ空方(K<D且非超卖) + RSI中性(30-70)
    macd_bear_signal = not macd_bull and not macd_bottom_div
    kdj_bear_signal = not kdj_bull and not kdj_oversold

    resonance_bear = macd_bear_signal and kdj_bear_signal and rsi_neutral

    resonance_count = sum([macd_bull_signal, kdj_bull_signal, rsi_neutral])

    return TimeframeSignal(
        tf=tf,
        macd_dif=dif_now, macd_dea=dea_now, macd_hist=last['macd_hist'],
        macd_above_zero=macd_above_zero, macd_golden_cross=macd_golden,
        macd_death_cross=macd_death, macd_bull=macd_bull,
        macd_top_divergence=macd_top_div, macd_bottom_divergence=macd_bottom_div,
        kdj_k=k_now, kdj_d=d_now, kdj_j=last['kdj_j'],
        kdj_golden_cross=kdj_golden, kdj_death_cross=kdj_death, kdj_bull=kdj_bull,
        kdj_oversold=kdj_oversold, kdj_overbought=kdj_overbought,
        rsi=rsi_now, rsi_above_50=rsi_above_50,
        rsi_oversold=rsi_oversold, rsi_overbought=rsi_overbought,
        ema_bull=ema_bull, ema9_20_sticky=ema9_20_sticky,
        bb_position_pct=bb_pos, bb_squeeze=bb_squeeze,
        vol_ratio=vol_ratio, vol_price_healthy=vol_healthy,
        adx=adx_now, adx_trending=adx_trending,
        three_resonance_bull=resonance_bull, three_resonance_bear=resonance_bear,
        resonance_count=resonance_count,
    )


# ═══════════════════════════════════════════════════════════
# 多时间框架综合评分
# ═══════════════════════════════════════════════════════════

def score_panda_signal(tf_signals: Dict[str, TimeframeSignal], symbol: str) -> PandaSignal:
    """
    熊猫策略综合评分引擎

    评分维度 (总分100):
      1. 三指标共振 (0-35分) — 每个TF共振得10-12分
      2. MACD背离 (0-20分) — 底背离+20(做多)，顶背离-20(做空)
      3. 多TF对齐 (0-20分) — TF方向一致性
      4. KDJ/RSI位置质量 (0-15分) — 超卖区金叉>中性区金叉
      5. 量价确认 (0-10分) — 放量上涨/缩量下跌
    """
    sig_1h = tf_signals.get("1h")
    sig_4h = tf_signals.get("4h")
    sig_1d = tf_signals.get("1d")

    if not all([sig_1h, sig_4h, sig_1d]):
        return PandaSignal(symbol=symbol, timestamp=datetime.now().isoformat(),
                          summary="数据不完整")

    score = 0.0
    details = []
    risk_warnings = []

    # ═══ 维度1: 三指标共振 (0-35) ═══
    # 1D共振最重(×1.5), 4H次之(×1.2), 1H最轻(×1.0)
    resonance_scores = {
        "1d": 15 if sig_1d.three_resonance_bull else (8 if sig_1d.resonance_count >= 2 else 0),
        "4h": 12 if sig_4h.three_resonance_bull else (6 if sig_4h.resonance_count >= 2 else 0),
        "1h": 8 if sig_1h.three_resonance_bull else (4 if sig_1h.resonance_count >= 2 else 0),
    }
    resonance_total = sum(resonance_scores.values())
    score += resonance_total
    details.append(f"共振: 1D={resonance_scores['1d']} + 4H={resonance_scores['4h']} + 1H={resonance_scores['1h']} = {resonance_total}/35")

    # ═══ 维度2: MACD背离 (0-20) ═══
    divergence_score = 0
    div_type = ""
    # 底背离=做多信号，周期越大权重越高
    if sig_1d.macd_bottom_divergence:
        divergence_score += 20
        div_type = "bottom"
        details.append("🔥 日线MACD底背离! +20分 — 空头动能衰竭，反转在即")
    elif sig_4h.macd_bottom_divergence:
        divergence_score += 15
        div_type = "bottom"
        details.append("💡 4H MACD底背离! +15分")
    elif sig_1h.macd_bottom_divergence:
        divergence_score += 10
        div_type = "bottom"
        details.append("💡 1H MACD底背离! +10分")

    # 顶背离=做空信号或减仓警告
    if sig_1d.macd_top_divergence:
        divergence_score -= 20
        div_type = "top"
        risk_warnings.append("⚠️ 日线MACD顶背离! 多头动能衰竭，不宜追多")
        details.append("⚠️ 日线MACD顶背离! -20分")
    elif sig_4h.macd_top_divergence:
        divergence_score -= 12
        div_type = "top" if not div_type else div_type
        risk_warnings.append("⚠️ 4H MACD顶背离! 谨慎做多")
        details.append("⚠️ 4H MACD顶背离! -12分")

    score += divergence_score
    score = max(0, min(100, score))  # clamp

    # ═══ 维度3: 多TF对齐 (0-20) ═══
    # 检查MACD方向在3个时间框架是否一致
    macd_directions = []
    for sig, name in [(sig_1d, "1D"), (sig_4h, "4H"), (sig_1h, "1H")]:
        if sig.macd_bull:
            macd_directions.append("多")
        else:
            macd_directions.append("空")

    tf_alignment = 0
    if macd_directions.count("多") == 3:
        tf_alignment = 20
        details.append("✅ 3/3 TF MACD一致做多 +20分")
    elif macd_directions.count("多") == 2 and macd_directions[0] == "多":  # 日线是多方
        tf_alignment = 14
        details.append("🟡 2/3 TF MACD做多(日线多方) +14分")
    elif macd_directions.count("多") == 2 and macd_directions[0] != "多":  # 日线空方
        tf_alignment = 8
        details.append(f"🟡 2/3 TF MACD做多但日线空方 +8分 (顺大逆小?)")
        risk_warnings.append("⚠️ 日线空方但短周期多方 — 可能是反弹不是反转")
    elif macd_directions.count("空") == 3:
        tf_alignment = 0  # 做空得分，但不计入做多评分
        details.append("🔴 3/3 TF MACD一致做空 → 做多评分0")
        risk_warnings.append("🔴 所有时间框架MACD空方排列")

    score += tf_alignment

    # ═══ 维度4: KDJ/RSI位置质量 (0-15) ═══
    position_score = 0

    # KDJ超卖区金叉 = 最佳买点
    if sig_1h.kdj_golden_cross and sig_1h.kdj_oversold:
        position_score += 8
        details.append("🎯 1H KDJ超卖区金叉! +8分 — 熊猫教练⭐⭐⭐⭐⭐信号")
    elif sig_1h.kdj_golden_cross:
        position_score += 4
        details.append("✅ 1H KDJ金叉 +4分")
    elif sig_4h.kdj_golden_cross and sig_4h.kdj_oversold:
        position_score += 10
        details.append("🎯 4H KDJ超卖区金叉! +10分 — 大周期最佳买点")
    elif sig_1d.kdj_golden_cross:
        position_score += 6
        details.append("✅ 日线KDJ金叉 +6分")

    # RSI位置
    if 40 < sig_1h.rsi < 55 and sig_1h.rsi > sig_4h.rsi:
        position_score += 5
        details.append("✅ RSI回调到位(40-55)且回升 +5分")
    elif sig_1h.rsi < 35:
        position_score += 3
        details.append("🟡 RSI<35超卖反弹区 +3分")
    elif sig_1h.rsi > 65:
        position_score += 1
        details.append("⚠️ RSI>65追高风险 — 仅+1分")
        risk_warnings.append("⚠️ RSI偏高(>65)，追多需谨慎")

    score += position_score
    score = max(0, min(100, score))

    # ═══ 维度5: 量价确认 (0-10) ═══
    vol_score = 0
    if sig_1h.vol_price_healthy:
        vol_score += 5
        details.append("✅ 1H价涨量增 +5分")
    if sig_4h.vol_price_healthy:
        vol_score += 3
        details.append("✅ 4H价涨量增 +3分")
    if sig_1h.vol_ratio > 1.5:
        vol_score += 2
        details.append(f"🔥 1H显著放量(量比{sig_1h.vol_ratio:.1f}) +2分")

    score += vol_score
    score = max(0, min(100, score))

    # ═══ 确定方向 ═══
    direction = "neutral"
    if tf_alignment >= 8 and resonance_total >= 10 and divergence_score > -5:
        direction = "long"
    elif tf_alignment <= 5 and sig_1d.macd_top_divergence:
        direction = "short"

    # 如果有顶背离+MACD空方排列 = 做空
    if sig_1d.macd_top_divergence and all(not s.macd_bull for s in [sig_1d, sig_4h]):
        direction = "short"

    # ═══ 杠杆映射 ═══
    leverage = 5  # 默认
    for threshold, lev in LEVERAGE_TABLE:
        if score >= threshold:
            leverage = lev
            break

    # ═══ 止损计算 (基于ATR) ═══
    # 使用最近ATR值
    atr_4h = sig_4h.adx / 30 * 0.02  # 近似ATR%（如果有直接ATR数据更好）
    stop_loss_pct = 2.0  # 默认2%

    # ═══ 入场条件 ═══
    entry_ready = (
        direction != "neutral"
        and score >= MIN_CONFIDENCE
        and not sig_1d.macd_top_divergence  # 日线顶背离禁止做多
        and len(risk_warnings) <= 2
    )

    # ═══ 生成摘要 ═══
    summary_lines = [
        f"🐼 {symbol} 熊猫策略评分: {score:.0f}/100",
        f"方向: {'🟢 做多' if direction == 'long' else '🔴 做空' if direction == 'short' else '🟡 观望'}",
        f"建议杠杆: {leverage}x | 保证金: {MARGIN_PER_TRADE}U",
        f"止损: {stop_loss_pct}% | {'✅ 可入场' if entry_ready else '❌ 等待'}",
    ]

    return PandaSignal(
        symbol=symbol,
        timestamp=datetime.now().isoformat(),
        tf_signals=tf_signals,
        direction=direction,
        confidence=score,
        suggested_leverage=leverage,
        tf_alignment_score=tf_alignment,
        tf_alignment_detail=" → ".join(macd_directions),
        has_divergence=divergence_score != 0,
        divergence_type=div_type,
        entry_ready=entry_ready,
        stop_loss_pct=stop_loss_pct,
        risk_warnings=risk_warnings,
        summary="\n".join(summary_lines) + "\n" + "\n".join(f"  {d}" for d in details[-6:]),
    )


# ═══════════════════════════════════════════════════════════
# 主分析函数
# ═══════════════════════════════════════════════════════════

def analyze_coin(symbol: str) -> PandaSignal:
    """对单个币种执行完整熊猫策略分析"""
    tf_signals = {}

    for tf in TIMEFRAMES:
        try:
            df = fetch_klines(symbol, tf, limit=100)
            df = calc_indicators(df)
            sig = detect_timeframe_signal(df, tf)
            tf_signals[tf] = sig
        except Exception as e:
            print(f"  ⚠️ {symbol} {tf} 分析失败: {e}")

    return score_panda_signal(tf_signals, symbol)


def scan_all() -> List[PandaSignal]:
    """扫描所有关注币种"""
    results = []
    for sym in WATCH_COINS:
        print(f"🐼 分析 {sym}...")
        signal = analyze_coin(sym)
        results.append(signal)
        print(signal.summary)
        print("─" * 50)
    return results


# ═══════════════════════════════════════════════════════════
# 企业微信推送
# ═══════════════════════════════════════════════════════════

def push_to_wecom(signals: List[PandaSignal]):
    """推送熊猫策略信号到企业微信"""
    now = datetime.now().strftime("%m-%d %H:%M")
    lines = [f"🐼 熊猫策略扫描 — {now}", ""]

    # 排序: 信心分从高到低
    sorted_signals = sorted(signals, key=lambda s: s.confidence, reverse=True)

    for sig in sorted_signals:
        emoji = "🟢" if sig.direction == "long" else "🔴" if sig.direction == "short" else "🟡"
        lines.append(f"{emoji} **{sig.symbol}** — {sig.confidence:.0f}分 | {sig.suggested_leverage}x")
        r1d = sig.tf_signals.get('1d')
        r4h = sig.tf_signals.get('4h')
        r1h = sig.tf_signals.get('1h')
        r1d_count = r1d.resonance_count if r1d else 0
        r4h_count = r4h.resonance_count if r4h else 0
        r1h_count = r1h.resonance_count if r1h else 0
        lines.append(f"> 共振: 1D={r1d_count}/3 4H={r4h_count}/3 1H={r1h_count}/3")
        lines.append(f"> MACD: {sig.tf_alignment_detail}")

        if sig.has_divergence:
            lines.append(f"> ⚡ 背离: {sig.divergence_type}")
        if sig.risk_warnings:
            for w in sig.risk_warnings[:2]:
                lines.append(f"> ⚠️ {w}")
        if sig.entry_ready:
            lines.append(f"> ✅ 入场信号! 杠杆{sig.suggested_leverage}x 保证金{MARGIN_PER_TRADE}U")
        lines.append("")

    # 账户状态
    try:
        from leverage_engine import LeverageEngine
        le = LeverageEngine()
        eq = le.fetch_equity()
        positions = le.fetch_open_positions()
        lines.append(f"💰 账户权益: ${eq:.2f} | 持仓: {len(positions)}个")
    except:
        pass

    content = "\n".join(lines)

    try:
        resp = requests.post(WECOM_WEBHOOK, json={
            "msgtype": "markdown",
            "markdown": {"content": content}
        }, timeout=10)
        print(f"📱 企微推送: {'成功' if resp.status_code == 200 else f'失败 {resp.status_code}'}")
    except Exception as e:
        print(f"📱 企微推送异常: {e}")


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="🐼 熊猫教练交易策略引擎")
    p.add_argument("--coin", default="ALL", help="币种 (BTC/ETH/SOL/ALL)")
    p.add_argument("--push", action="store_true", help="推送到企业微信")
    p.add_argument("--once", action="store_true", help="单次运行")
    args = p.parse_args()

    if args.coin == "ALL":
        signals = scan_all()
    else:
        signal = analyze_coin(args.coin.upper())
        signals = [signal]
        print(signal.summary)

    if args.push:
        push_to_wecom(signals)
