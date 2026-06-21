#!/usr/bin/env python3
"""
🎻 图表视觉分析器 — 生成专业K线图 + 火山引擎识图分析
用途: 用AI视觉能力看图识别金叉死叉、拐头、形态，补充纯数据计算盲区

用法:
    python3 -u chart_vision_analyzer.py              # 分析 BTC/ETH/SOL
    python3 -u chart_vision_analyzer.py --coin BTC   # 单币种
    python3 -u chart_vision_analyzer.py --no-vision  # 只生成图不用AI分析
"""

import sys
import os
import json
import time
import base64
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker

# ── 火山引擎配置 ──
ARK_API_KEY = "ark-63f9a7e7-183a-4064-8842-83a683935150-56301"
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
VISION_MODEL = "doubao-seed-2-0-pro-260215"

OUTPUT_DIR = Path("/Users/chasett/yina-app/chase-quant-web/chart_images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── matplotlib 中文支持 ──
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang HK', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


# ═══════════════════════════════════════════════════════════
# 数据获取
# ═══════════════════════════════════════════════════════════

def fetch_klines(symbol: str, timeframe: str = "1h", limit: int = 100) -> pd.DataFrame:
    """从 Binance 获取K线数据"""
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
# 指标计算
# ═══════════════════════════════════════════════════════════

def calc_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算全部技术指标"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    vol = df['volume'].values

    # ── EMA 矩阵 ──
    for p in [9, 20, 50, 200]:
        df[f'ema_{p}'] = pd.Series(close).ewm(span=p, adjust=False).mean().values

    # ── MACD (12/26/9) ──
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
    df['macd_dif'] = (ema12 - ema26).values
    df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean().values  # DEA = EMA9 of DIF
    df['macd_hist'] = 2 * (df['macd_dif'] - df['macd_dea']).values

    # ── RSI (14) ──
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, 100), where=avg_loss != 0)
    df['rsi'] = 100 - (100 / (1 + rs))

    # ── KDJ (9,3,3) ──
    n = 9
    low_n = pd.Series(low).rolling(window=n).min().values
    high_n = pd.Series(high).rolling(window=n).max().values
    rsv = np.where(high_n != low_n, (close - low_n) / (high_n - low_n) * 100, 50)

    k = np.full_like(rsv, 50.0)
    d = np.full_like(rsv, 50.0)
    # 找到第一个有效RSV的索引
    first_valid = 0
    for i in range(len(rsv)):
        if not np.isnan(rsv[i]):
            first_valid = i
            break
    for i in range(first_valid, len(rsv)):
        rsv_val = rsv[i] if not np.isnan(rsv[i]) else 50.0
        k[i] = 2/3 * k[i-1] + 1/3 * rsv_val
        d[i] = 2/3 * d[i-1] + 1/3 * k[i]
    j = 3 * k - 2 * d
    df['kdj_k'] = k
    df['kdj_d'] = d
    df['kdj_j'] = j

    # ── 布林带 (20,2) ──
    sma20 = pd.Series(close).rolling(window=20).mean().values
    std20 = pd.Series(close).rolling(window=20).std().values
    df['bb_mid'] = sma20
    df['bb_up'] = sma20 + 2 * std20
    df['bb_low'] = sma20 - 2 * std20
    df['bb_width'] = (df['bb_up'] - df['bb_low']) / df['bb_mid'] * 100

    # ── ATR (14) ──
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    df['atr'] = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values

    # ── OBV ──
    obv = np.zeros_like(close)
    obv[0] = vol[0]
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + vol[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - vol[i]
        else:
            obv[i] = obv[i-1]
    df['obv'] = obv

    # ── Stochastic RSI ──
    rsi_series = df['rsi'].values
    n_stoch = 14
    stoch_k = np.zeros_like(rsi_series)
    stoch_d = np.zeros_like(rsi_series)
    for i in range(n_stoch, len(rsi_series)):
        window = rsi_series[i-n_stoch:i+1]
        rsi_high = np.max(window)
        rsi_low = np.min(window)
        if rsi_high != rsi_low:
            stoch_k[i] = (rsi_series[i] - rsi_low) / (rsi_high - rsi_low) * 100
        else:
            stoch_k[i] = stoch_k[i-1] if i > 0 else 50
    # SMA 3 of stoch_k
    for i in range(3, len(stoch_k)):
        stoch_d[i] = np.mean(stoch_k[i-3:i+1])
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d

    # ── ADX (14) ──
    n_adx = 14
    tr_adx = tr.copy()
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low),
                        np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)),
                         np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = dm_minus[0] = 0

    atr_smooth = pd.Series(tr_adx).ewm(alpha=1/n_adx, adjust=False).mean().values
    di_plus = pd.Series(dm_plus).ewm(alpha=1/n_adx, adjust=False).mean().values / atr_smooth * 100
    di_minus = pd.Series(dm_minus).ewm(alpha=1/n_adx, adjust=False).mean().values / atr_smooth * 100
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10) * 100
    adx = pd.Series(dx).ewm(alpha=1/n_adx, adjust=False).mean().values
    df['adx'] = adx
    df['di_plus'] = di_plus
    df['di_minus'] = di_minus

    # ── MFI (14) ──
    n_mfi = 14
    tp = (high + low + close) / 3
    raw_mf = tp * vol
    pos_flow = np.zeros_like(tp)
    neg_flow = np.zeros_like(tp)
    for i in range(1, len(tp)):
        if tp[i] > tp[i-1]:
            pos_flow[i] = raw_mf[i]
        elif tp[i] < tp[i-1]:
            neg_flow[i] = raw_mf[i]
        else:
            pos_flow[i] = raw_mf[i] / 2
            neg_flow[i] = raw_mf[i] / 2
    pos_sum = pd.Series(pos_flow).rolling(n_mfi).sum().values
    neg_sum = pd.Series(neg_flow).rolling(n_mfi).sum().values
    df['mfi'] = 100 - 100 / (1 + pos_sum / (neg_sum + 1e-10))

    # ── 成交量均线 ──
    df['vol_ma20'] = pd.Series(vol).rolling(20).mean().values

    # ── K线形态标记 ──
    df['body'] = close - df['open'].values
    df['upper_wick'] = high - np.maximum(close, df['open'].values)
    df['lower_wick'] = np.minimum(close, df['open'].values) - low

    return df


# ═══════════════════════════════════════════════════════════
# 数据文本分析 (提取金叉死叉拐头等)
# ═══════════════════════════════════════════════════════════

def detect_signals_text(df: pd.DataFrame, symbol: str, tf: str) -> str:
    """用数据计算检测关键信号，生成文本描述"""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev5 = df.iloc[-6:-1]

    signals = []

    # KDJ 金叉死叉
    k_now = last['kdj_k']; d_now = last['kdj_d']
    k_prev = prev['kdj_k']; d_prev = prev['kdj_d']

    if k_prev <= d_prev and k_now > d_now:
        signals.append(f"🔔 KDJ金叉! K上穿D (K={k_now:.1f}, D={d_now:.1f})")
    elif k_prev >= d_prev and k_now < d_now:
        signals.append(f"🔔 KDJ死叉! K下穿D (K={k_now:.1f}, D={d_now:.1f})")
    else:
        k_trend = "↑" if k_now > k_prev else "↓"
        d_trend = "↑" if d_now > d_prev else "↓"
        signals.append(f"KDJ: K={k_now:.1f}{k_trend} D={d_now:.1f}{d_trend} J={last['kdj_j']:.1f}")

    # MACD 金叉死叉
    dif_now = last['macd_dif']; dif_prev = prev['macd_dif']
    dea_now = last['macd_dea']; dea_prev = prev['macd_dea']

    if dif_prev <= dea_prev and dif_now > dea_now:
        signals.append(f"🔔 MACD金叉! DIF上穿DEA")
    elif dif_prev >= dea_prev and dif_now < dea_now:
        signals.append(f"🔔 MACD死叉! DIF下穿DEA")
    else:
        hist_trend = "扩张" if abs(last['macd_hist']) > abs(prev['macd_hist']) else "缩窄"
        signals.append(f"MACD: DIF={dif_now:.4f} DEA={dea_now:.4f} 柱{hist_trend}")

    # RSI
    rsi_now = last['rsi']
    rsi_trend = "↑" if rsi_now > prev['rsi'] else "↓"
    if rsi_now < 30:
        signals.append(f"RSI={rsi_now:.1f}{rsi_trend} ⚠️超卖区")
    elif rsi_now > 70:
        signals.append(f"RSI={rsi_now:.1f}{rsi_trend} ⚠️超买区")
    else:
        signals.append(f"RSI={rsi_now:.1f}{rsi_trend}")

    # 布林带
    bb_pos = (last['close'] - last['bb_low']) / (last['bb_up'] - last['bb_low']) * 100
    signals.append(f"布林: 位置{bb_pos:.0f}% 带宽{last['bb_width']:.1f}%")

    # EMA 排列
    emas = [last[f'ema_{p}'] for p in [9, 20, 50, 200]]
    above = sum(1 for e in emas if last['close'] > e)
    signals.append(f"EMA排列: 价在{above}/4均线上 {'🟢多头' if above >= 3 else '🔴空头' if above <= 1 else '🟡交织'}")

    # 量价关系
    vol_ratio = last['volume'] / last['vol_ma20'] if last['vol_ma20'] > 0 else 1
    price_up = last['close'] > prev['close']
    if price_up and vol_ratio > 1.3:
        signals.append(f"量价: 放量上涨✅ (量比{vol_ratio:.1f})")
    elif price_up and vol_ratio < 0.7:
        signals.append(f"量价: 缩量上涨⚠️ (量比{vol_ratio:.1f})")
    elif not price_up and vol_ratio > 1.3:
        signals.append(f"量价: 放量下跌❌ (量比{vol_ratio:.1f})")
    else:
        signals.append(f"量价: 量比{vol_ratio:.1f}")

    # ADX
    adx_now = last['adx']
    if adx_now > 40:
        signals.append(f"ADX={adx_now:.1f} 极强趋势 {'多头' if last['di_plus'] > last['di_minus'] else '空头'}")
    elif adx_now > 25:
        signals.append(f"ADX={adx_now:.1f} 趋势中 {'多头' if last['di_plus'] > last['di_minus'] else '空头'}")
    else:
        signals.append(f"ADX={adx_now:.1f} 无趋势/震荡")

    header = f"📊 {symbol}/USDT {tf} — 数据检测信号\n"
    return header + "\n".join(f"  {s}" for s in signals)


# ═══════════════════════════════════════════════════════════
# 火山引擎识图
# ═══════════════════════════════════════════════════════════

ANALYSIS_PROMPT = """你是一位顶级加密货币技术分析师，师从熊猫教练。请仔细分析这张K线图，包含以下指标面板:

**上图**: K线 + EMA均线(9/20/50/200) + 布林带 + 成交量
**第二图**: MACD (DIF/DEA/柱状图)
**第三图**: KDJ (K/D/J三线)
**第四图**: RSI (14) + Stochastic RSI

请逐项回答（必须具体，用数据说话）:

1. **K线形态**: 最近5根K线是什么形态？（大阳/大阴/十字星/锤子/倒锤/吞没）有哪些关键信号K？

2. **EMA排列**: 均线是多头排列(9>20>50>200)还是空头？价格在哪些均线之上/之下？EMA9和EMA20是否即将交叉？

3. **MACD**: DIF和DEA在零轴上方还是下方？是金叉还是死叉状态？柱状图是在扩张还是缩窄？有没有即将金叉/死叉的迹象（两线收敛）？

4. **KDJ**: K/D/J三线各自的方向（向上/向下/拐头）？K和D是否刚发生金叉或死叉？三线是否在超买(>80)或超卖(<20)区域？

5. **RSI**: RSI数值范围？方向是向上还是向下？是否有拐头迹象？是否在超买超卖边界？

6. **布林带**: 价格在布林带什么位置（上轨/中轨/下轨）？带宽是在扩张还是收缩？是否有收窄后即将突破的迹象？

7. **成交量**: 最近几根量柱是放量还是缩量？上涨时量是否配合？有没有量价背离？

8. **综合判断**:
   - 当前是多方控盘还是空方控盘？
   - 短期(数小时)方向判断
   - 是否有入场信号？（需同时满足: KDJ金叉+MACD多方+RSI不超买+K线信号K）
   - 风险提示

请用中文回答，每个判断都要说明你在图上看到了什么具体特征。"""


def analyze_chart_vision(image_path: str, prompt: str = None) -> dict:
    """使用火山引擎 Seed-2.0-Pro 识图分析"""
    if prompt is None:
        prompt = ANALYSIS_PROMPT

    # 读取图片编码
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    ext = Path(image_path).suffix.lower().replace(".", "")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(ext, "png")
    data_url = f"data:image/{mime};base64,{img_b64}"

    print(f"  👁️  调用火山引擎 Seed-2.0-Pro 识图...", end=" ", flush=True)

    t0 = time.time()

    # 使用 OpenAI 兼容协议
    from openai import OpenAI
    client = OpenAI(api_key=ARK_API_KEY, base_url=ARK_BASE_URL)

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ]
        }],
        max_tokens=4096,
        temperature=0.3,
    )

    latency = (time.time() - t0) * 1000
    text = response.choices[0].message.content

    print(f"✅ {latency:.0f}ms")

    return {
        "text": text.strip(),
        "model": VISION_MODEL,
        "latency_ms": round(latency),
        "image": image_path,
    }


# ═══════════════════════════════════════════════════════════
# 图表生成
# ═══════════════════════════════════════════════════════════

def generate_chart(df: pd.DataFrame, symbol: str, timeframe: str) -> str:
    """生成包含全部指标的专业K线图，返回保存路径"""

    # 色彩方案 — 暗色专业主题
    bg_color = '#1a1a2e'
    grid_color = '#2d2d44'
    text_color = '#c9d1d9'
    green_color = '#00ff88'
    red_color = '#ff4757'
    blue_color = '#4da6ff'
    yellow_color = '#ffd700'
    purple_color = '#b366ff'
    white_color = '#ffffff'

    n = len(df)

    # 创建4面板图
    fig = plt.figure(figsize=(20, 14), facecolor=bg_color)

    # ── 面板1: K线 + EMA + 布林带 + 成交量 ──
    gs = fig.add_gridspec(4, 1, height_ratios=[2.5, 1, 1, 1], hspace=0.05)

    ax1 = fig.add_subplot(gs[0], facecolor=bg_color)
    ax1_vol = ax1.twinx()

    # 布林带填充
    ax1.fill_between(range(n), df['bb_low'], df['bb_up'], alpha=0.08, color='#4da6ff')
    ax1.plot(df['bb_up'].values, color='#4da6ff', alpha=0.3, linewidth=0.8, linestyle='--')
    ax1.plot(df['bb_mid'].values, color='#4da6ff', alpha=0.5, linewidth=0.8, linestyle='--')
    ax1.plot(df['bb_low'].values, color='#4da6ff', alpha=0.3, linewidth=0.8, linestyle='--')

    # K线 — 用红绿柱
    colors = [green_color if df['close'].iloc[i] >= df['open'].iloc[i] else red_color
              for i in range(n)]
    body_width = 0.6

    for i in range(n):
        # 影线
        ax1.plot([i, i], [df['low'].iloc[i], df['high'].iloc[i]],
                 color=colors[i], linewidth=1, alpha=0.8)
        # 实体
        body_bottom = min(df['open'].iloc[i], df['close'].iloc[i])
        body_height = abs(df['close'].iloc[i] - df['open'].iloc[i])
        body_height = max(body_height, 1e-8)  # 十字星也显示一条线
        ax1.add_patch(plt.Rectangle((i - body_width/2, body_bottom), body_width, body_height,
                                     facecolor=colors[i], edgecolor=colors[i], alpha=0.9))

    # EMA均线
    for p, color, alpha in [(9, '#ff6b6b', 0.8), (20, '#ffd93d', 0.8),
                              (50, '#6bcb77', 0.7), (200, '#4d96ff', 0.6)]:
        ema_vals = df[f'ema_{p}'].values
        ax1.plot(ema_vals, color=color, linewidth=1.2, alpha=alpha, label=f'EMA{p}')

    # 成交量柱
    vol_colors = [green_color if df['close'].iloc[i] >= df['open'].iloc[i] else red_color
                  for i in range(n)]
    vol_max = df['volume'].max()
    ax1_vol.bar(range(n), df['volume'].values, width=0.5, color=vol_colors, alpha=0.3)
    ax1_vol.set_ylim(0, vol_max * 6)  # 压缩成交量
    ax1_vol.set_ylabel('')
    ax1_vol.tick_params(colors=text_color, labelsize=6)

    ax1.legend(loc='upper left', fontsize=7, facecolor=bg_color, edgecolor=grid_color,
               labelcolor=text_color)
    ax1.set_ylabel(f'{symbol}/USDT', color=text_color, fontsize=10, fontweight='bold')
    ax1.tick_params(colors=text_color, labelsize=7)
    ax1.grid(True, alpha=0.15, color=grid_color)
    ax1.set_xlim(-1, n+1)

    # 最新价标注
    last_price = df['close'].iloc[-1]
    ax1.axhline(y=last_price, color='white', alpha=0.2, linewidth=0.5, linestyle=':')
    ax1.annotate(f'${last_price:.4f}', xy=(n-1, last_price), xytext=(n+2, last_price),
                color='white', fontsize=8, fontweight='bold', va='center')

    # 标题
    title = f'{symbol}/USDT  {timeframe}  —  {datetime.now().strftime("%Y-%m-%d %H:%M")}'
    ax1.set_title(title, color=white_color, fontsize=12, fontweight='bold', pad=10)

    # ── 面板2: MACD ──
    ax2 = fig.add_subplot(gs[1], facecolor=bg_color)
    macd_colors = [green_color if df['macd_hist'].iloc[i] >= df['macd_hist'].iloc[i-1]
                   else red_color for i in range(1, n)]
    macd_colors.insert(0, green_color)

    ax2.bar(range(n), df['macd_hist'].values, width=0.5, color=macd_colors, alpha=0.7)
    ax2.plot(df['macd_dif'].values, color='white', linewidth=1.2, label='DIF')
    ax2.plot(df['macd_dea'].values, color=yellow_color, linewidth=1.2, label='DEA')
    ax2.axhline(y=0, color='white', alpha=0.2, linewidth=0.5)

    # 标注金叉死叉
    for i in range(1, n):
        if df['macd_dif'].iloc[i-1] <= df['macd_dea'].iloc[i-1] and df['macd_dif'].iloc[i] > df['macd_dea'].iloc[i]:
            ax2.annotate('金叉', xy=(i, df['macd_dif'].iloc[i]), fontsize=6, color=green_color,
                        ha='center', va='bottom', fontweight='bold')
        elif df['macd_dif'].iloc[i-1] >= df['macd_dea'].iloc[i-1] and df['macd_dif'].iloc[i] < df['macd_dea'].iloc[i]:
            ax2.annotate('死叉', xy=(i, df['macd_dif'].iloc[i]), fontsize=6, color=red_color,
                        ha='center', va='top', fontweight='bold')

    ax2.legend(loc='upper left', fontsize=7, facecolor=bg_color, edgecolor=grid_color, labelcolor=text_color)
    ax2.set_ylabel('MACD', color=text_color, fontsize=9, fontweight='bold')
    ax2.tick_params(colors=text_color, labelsize=7)
    ax2.grid(True, alpha=0.15, color=grid_color)
    ax2.set_xlim(-1, n+1)

    # ── 面板3: KDJ ──
    ax3 = fig.add_subplot(gs[2], facecolor=bg_color)
    ax3.plot(df['kdj_k'].values, color=white_color, linewidth=1.2, label='K')
    ax3.plot(df['kdj_d'].values, color=yellow_color, linewidth=1.2, label='D')
    ax3.plot(df['kdj_j'].values, color=purple_color, linewidth=0.8, alpha=0.7, label='J')
    ax3.axhline(y=80, color=red_color, alpha=0.3, linewidth=0.5, linestyle='--')
    ax3.axhline(y=20, color=green_color, alpha=0.3, linewidth=0.5, linestyle='--')
    ax3.fill_between(range(n), 80, 100, alpha=0.05, color=red_color)
    ax3.fill_between(range(n), 0, 20, alpha=0.05, color=green_color)

    # 标注KDJ金叉死叉
    for i in range(1, n):
        if df['kdj_k'].iloc[i-1] <= df['kdj_d'].iloc[i-1] and df['kdj_k'].iloc[i] > df['kdj_d'].iloc[i]:
            ax3.annotate('金叉', xy=(i, df['kdj_k'].iloc[i]), fontsize=6, color=green_color,
                        ha='center', va='bottom', fontweight='bold')
        elif df['kdj_k'].iloc[i-1] >= df['kdj_d'].iloc[i-1] and df['kdj_k'].iloc[i] < df['kdj_d'].iloc[i]:
            ax3.annotate('死叉', xy=(i, df['kdj_k'].iloc[i]), fontsize=6, color=red_color,
                        ha='center', va='top', fontweight='bold')

    ax3.legend(loc='upper left', fontsize=7, facecolor=bg_color, edgecolor=grid_color, labelcolor=text_color)
    ax3.set_ylabel('KDJ', color=text_color, fontsize=9, fontweight='bold')
    ax3.tick_params(colors=text_color, labelsize=7)
    ax3.grid(True, alpha=0.15, color=grid_color)
    ax3.set_xlim(-1, n+1)
    ax3.set_ylim(-5, 105)

    # ── 面板4: RSI + StochRSI ──
    ax4 = fig.add_subplot(gs[3], facecolor=bg_color)
    ax4.plot(df['rsi'].values, color=blue_color, linewidth=1.5, label='RSI(14)')
    ax4.plot(df['stoch_k'].values, color=white_color, linewidth=0.8, alpha=0.6, label='Stoch K')
    ax4.plot(df['stoch_d'].values, color=yellow_color, linewidth=0.8, alpha=0.6, label='Stoch D')
    ax4.axhline(y=70, color=red_color, alpha=0.3, linewidth=0.5, linestyle='--')
    ax4.axhline(y=30, color=green_color, alpha=0.3, linewidth=0.5, linestyle='--')
    ax4.axhline(y=50, color='white', alpha=0.15, linewidth=0.5)
    ax4.fill_between(range(n), 70, 100, alpha=0.05, color=red_color)
    ax4.fill_between(range(n), 0, 30, alpha=0.05, color=green_color)

    ax4.legend(loc='upper left', fontsize=7, facecolor=bg_color, edgecolor=grid_color, labelcolor=text_color)
    ax4.set_ylabel('RSI / Stoch', color=text_color, fontsize=9, fontweight='bold')
    ax4.set_xlabel(f'K线序号 (最新→右, 共{n}根)', color=text_color, fontsize=8)
    ax4.tick_params(colors=text_color, labelsize=7)
    ax4.grid(True, alpha=0.15, color=grid_color)
    ax4.set_xlim(-1, n+1)
    ax4.set_ylim(-5, 105)

    # 保存
    plt.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{symbol}_{timeframe}_{ts}.png"
    fpath = OUTPUT_DIR / fname
    fig.savefig(fpath, dpi=150, facecolor=bg_color, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

    print(f"  📈 图表已保存: {fpath} ({Path(fpath).stat().st_size // 1024}KB)")
    return str(fpath)


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def analyze_coin(symbol: str, timeframes: list = None, use_vision: bool = True):
    """对一个币种执行完整的图表+视觉分析"""
    if timeframes is None:
        timeframes = ["1h", "4h", "1d"]

    print(f"\n{'='*70}")
    print(f"🎻 {symbol}/USDT 图表视觉分析")
    print(f"{'='*70}")

    for tf in timeframes:
        print(f"\n── {symbol} {tf} ──")

        # 取K线
        limit = {"15m": 80, "1h": 100, "4h": 80, "1d": 60}.get(tf, 80)
        df = fetch_klines(symbol, tf, limit)
        if df.empty:
            print(f"  ❌ 获取{symbol} {tf}数据失败")
            continue

        # 计算指标
        df = calc_all_indicators(df)

        # 数据信号检测
        print()
        print(detect_signals_text(df, symbol, tf))

        # 生成图表
        chart_path = generate_chart(df, symbol, tf)

        # AI视觉分析
        if use_vision:
            print()
            try:
                result = analyze_chart_vision(chart_path)
                print(f"\n{'─'*60}")
                print(f"🤖 火山引擎 Seed-2.0-Pro 视觉分析结果:")
                print(f"{'─'*60}")
                print(result['text'])
                print(f"{'─'*60}")
                print(f"⏱️  耗时: {result['latency_ms']}ms | 模型: {result['model']}")
            except Exception as e:
                print(f"  ❌ 视觉分析失败: {e}")

        print()
        time.sleep(1)  # API限速间隔


def main():
    parser = argparse.ArgumentParser(description="图表视觉分析器")
    parser.add_argument("--coin", "-c", default="BTC", help="币种 (BTC/ETH/SOL/ALL)")
    parser.add_argument("--timeframe", "-t", default="1h,4h,1d", help="时间框架")
    parser.add_argument("--no-vision", action="store_true", help="只生成图，不用AI分析")
    parser.add_argument("--prompt", "-p", help="自定义分析提示词")
    args = parser.parse_args()

    coins = ["BTC", "ETH", "SOL"] if args.coin == "ALL" else [args.coin.upper()]
    tfs = args.timeframe.split(",")
    use_vision = not args.no_vision

    print("🎻 Yina 图表视觉分析器")
    print(f"📊 币种: {coins}  时间框架: {tfs}  AI识图: {'✅' if use_vision else '❌'}")
    print()

    for coin in coins:
        analyze_coin(coin, tfs, use_vision)


if __name__ == "__main__":
    main()
