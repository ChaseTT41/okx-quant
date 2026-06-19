#!/usr/bin/env python3
"""
🕐 30分钟自动分析循环 — Yina 主动策略引擎 v2.0
每30分钟: K线扫描 + 情绪引擎 + 五维评分卡 + MPT组合检查 → 决策建议 → 企业微信推送

v2.0 新增:
  - 五维评分卡 (趋势/超买超卖/支撑阻力/基本面/风险度)
  - 多空方向冲突检测
  - MPT板块集中度检查
  - 全哥"飞天猪"模型

用法:
  python3 auto_30min_analyst.py --daemon   # 后台持续运行
  python3 auto_30min_analyst.py --once     # 单次运行
"""

import os, sys, json, time, argparse, traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests
import ccxt

# ── 配置 ──
PROJECT_DIR = Path(__file__).parent
WECHAT_KEY = "2c602b48-5da2-4989-9193-30c0e226c769"
WECHAT_URL = f"https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={WECHAT_KEY}"

# 关注列表
WATCHLIST = [
    'INTC-USDT-SWAP', 'ARM-USDT-SWAP', 'TSM-USDT-SWAP',
    'QCOM-USDT-SWAP', 'MRVL-USDT-SWAP', 'AVGO-USDT-SWAP',
    'NVDA-USDT-SWAP', 'TSLA-USDT-SWAP', 'SOL-USDT-SWAP',
    'BTC-USDT-SWAP', 'ETH-USDT-SWAP', 'DOGE-USDT-SWAP',
]

def load_env():
    """加载环境变量"""
    env_file = PROJECT_DIR / '.env'
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    sys.path.insert(0, str(PROJECT_DIR))


def init_exchange():
    """初始化OKX连接"""
    return ccxt.okx({
        'apiKey': os.environ.get('OKX_API_KEY', ''),
        'secret': os.environ.get('OKX_SECRET_KEY', ''),
        'password': os.environ.get('OKX_PASSPHRASE', ''),
        'hostname': 'www.okx.cab',
        'options': {'defaultType': 'swap'},
    })


def calc_indicators(df: pd.DataFrame) -> dict:
    """计算技术指标"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    vol = df['volume'].values

    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean().values
    avg_loss = pd.Series(loss).rolling(14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss > 0)
    rsi = 100 - (100 / (1 + rs))

    # EMAs
    ema20 = pd.Series(close).ewm(span=20, adjust=False).mean().values
    ema50 = pd.Series(close).ewm(span=50, adjust=False).mean().values

    # MACD
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
    macd_line = ema12 - ema26
    macd_signal = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values

    # ATR(14)
    tr = np.maximum(high - low, np.maximum(
        abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(14).mean().values

    # Bollinger
    bb_mid = pd.Series(close).rolling(20).mean().values
    bb_std = pd.Series(close).rolling(20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    last = -1
    bb_pos = ((close[last] - bb_lower[last]) /
              (bb_upper[last] - bb_lower[last])) if bb_upper[last] != bb_lower[last] else 0.5

    # Vol ratio
    vol_ratio = vol[-5:].mean() / vol[-20:].mean() if vol[-20:].mean() > 0 else 1.0

    return {
        'rsi': round(float(rsi[last]), 1),
        'price': round(float(close[last]), 4),
        'ema20': round(float(ema20[last]), 4),
        'ema50': round(float(ema50[last]), 4),
        'vs_ema20': round((close[last] / ema20[last] - 1) * 100, 2),
        'vs_ema50': round((close[last] / ema50[last] - 1) * 100, 2),
        'macd_line': round(float(macd_line[last]), 4),
        'macd_signal': round(float(macd_signal[last]), 4),
        'macd_hist': round(float(macd_line[last] - macd_signal[last]), 4),
        'atr_pct': round(float(atr[last] / close[last] * 100), 2),
        'bb_position': round(float(bb_pos), 2),
        'vol_ratio': round(float(vol_ratio), 2),
    }


def five_dimension_score(df: pd.DataFrame, side: str = 'long') -> dict:
    """🆕 v2.0 五维评分卡 — 趋势/超买超卖/支撑阻力/基本面/风险度"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    vol = df['volume'].values

    # EMA 计算
    ema5 = pd.Series(close).ewm(span=5, adjust=False).mean().values
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema20 = pd.Series(close).ewm(span=20, adjust=False).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
    ema60 = pd.Series(close).ewm(span=60, adjust=False).mean().values

    # RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean().values
    avg_loss = pd.Series(loss).rolling(14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss > 0)
    rsi_val = float(100 - (100 / (1 + rs[-1])))

    # MACD
    macd_line = ema12 - ema26
    macd_signal = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values

    # ATR
    tr_arr = np.maximum(high[1:] - low[1:], np.maximum(
        abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
    atr_val = float(np.mean(tr_arr[-14:])) if len(tr_arr) >= 14 else 0

    # ADX
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    atr_smooth = pd.Series(tr_arr).ewm(span=14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False).mean().values[-1] / atr_val if atr_val > 0 else 0
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False).mean().values[-1] / atr_val if atr_val > 0 else 0
    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0

    # 波动率
    returns = np.diff(close) / close[:-1]
    hv_30d = float(np.std(returns[-30:]) * np.sqrt(365) * 100) if len(returns) >= 30 else 50
    max_dd_20d = float(abs(np.min((close[-20:] - np.maximum.accumulate(close[-20:])) / np.maximum.accumulate(close[-20:]) * 100))) if len(close) >= 20 else 0

    vol_ratio = float(vol[-5:].mean() / vol[-20:].mean()) if vol[-20:].mean() > 0 else 1.0

    # ── 维度1: 趋势强度 (25%) ──
    trend = 0
    if macd_line[-1] > macd_signal[-1]: trend += 20
    if ema5[-1] > ema20[-1]: trend += 15
    if ema20[-1] > ema60[-1]: trend += 15
    if dx > 25: trend += 20
    if close[-1] > ema20[-1]: trend += 15
    if len(macd_line) >= 3 and macd_line[-1] > macd_signal[-1] and macd_line[-2] <= macd_signal[-2]:
        trend += 15  # 金叉
    trend = min(trend, 100)

    # ── 维度2: 超买超卖 (15%) ──
    if 40 <= rsi_val <= 60: ob_os = 80
    elif 30 <= rsi_val < 40: ob_os = 60
    elif 60 < rsi_val <= 70: ob_os = 50
    elif rsi_val < 30: ob_os = 30
    elif rsi_val > 70: ob_os = 20
    else: ob_os = 10

    # ── 维度3: 支撑阻力 (20%) ──
    sr = 50
    recent_high = float(np.max(high[-20:]))
    recent_low = float(np.min(low[-20:]))
    dist_resist = (recent_high - close[-1]) / close[-1] * 100
    dist_support = (close[-1] - recent_low) / close[-1] * 100
    if dist_support < 2: sr += 30
    elif dist_support < 5: sr += 15
    if dist_resist > 10: sr += 20
    elif dist_resist > 5: sr += 10
    if dist_resist < 2: sr -= 20
    sr = max(0, min(sr, 100))

    # ── 维度4: 基本面 (25%) 用成交量和价格行为替代 ──
    fund = 50
    if vol_ratio > 1.3: fund += 15
    elif vol_ratio > 1.1: fund += 8
    elif vol_ratio < 0.7: fund -= 10
    if close[-1] > ema20[-1]: fund += 10
    if hv_30d < 30: fund += 10
    fund = max(0, min(fund, 100))

    # ── 维度5: 风险度 (15%) ──
    risk = 70
    if hv_30d > 80: risk -= 30
    elif hv_30d > 60: risk -= 20
    elif hv_30d < 30: risk += 20
    if max_dd_20d > 15: risk -= 20
    elif max_dd_20d > 10: risk -= 10
    risk = max(0, min(risk, 100))

    # 加权综合
    weights = {'trend': 0.25, 'ob_os': 0.15, 'sr': 0.20, 'fundamental': 0.25, 'risk': 0.15}
    composite = trend * 0.25 + ob_os * 0.15 + sr * 0.20 + fund * 0.25 + risk * 0.15

    # 行动映射
    if composite >= 80: action = 'STRONG_BUY'
    elif composite >= 65: action = 'BUY'
    elif composite >= 50: action = 'LIGHT_BUY'
    elif composite >= 35: action = 'WATCH'
    elif composite >= 20: action = 'LIGHT_SHORT'
    else: action = 'SELL'

    stars = min(5, max(1, round(composite / 20)))

    return {
        'scores': {'trend': trend, 'ob_os': ob_os, 'sr': sr, 'fundamental': fund, 'risk': risk},
        'composite': round(composite, 1),
        'action': action,
        'stars': stars,
        'rsi': round(rsi_val, 1),
        'volatility': round(hv_30d, 1),
        'max_dd_20d': round(max_dd_20d, 1),
        'vol_ratio': round(vol_ratio, 2),
        'dx': round(float(dx), 1),
        'dist_support': round(float(dist_support), 1),
        'dist_resist': round(float(dist_resist), 1),
    }


def scan_klines(exchange, symbols: List[str]) -> dict:
    """多时间框架裸K扫描 + 指标 + 五维评分卡"""
    from naked_k_scanner import NakedKScanner

    results = {}
    for sym in symbols:
        sym_results = {}
        try:
            for tf in ['1h', '4h', '1d']:
                ohlcv = exchange.fetch_ohlcv(sym, tf, limit=100)
                if not ohlcv or len(ohlcv) < 30:
                    continue
                df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
                ind = calc_indicators(df)

                try:
                    scanner = NakedKScanner(df, symbol=sym, timeframe=tf, higher_tf_df=None)
                    result = scanner.scan()
                    d = result.to_dict()
                    bias = d.get('market_bias', '?')
                    trend = d.get('structure', {}).get('trend_strength', 0)
                    # 🆕 新特征
                    complex_pbs = d.get('complex_pullbacks', [])
                    fvgs = d.get('fvgs', [])
                    order_blocks = d.get('order_blocks', [])
                    channel_outcome = d.get('channel_outcome')
                    low_highs = d.get('low_high_counts', [])       # 🆕 Vision
                    sig_pairs = d.get('signal_entry_pairs', [])   # 🆕 Vision
                    mom_warnings = d.get('momentum_warnings', []) # 🆕 Vision
                except Exception:
                    bias = 'ERR'
                    trend = 0
                    complex_pbs, fvgs, order_blocks = [], [], []
                    channel_outcome = None
                    low_highs, sig_pairs, mom_warnings = [], [], []

                # 综合方向打分
                score = 0
                if ind['vs_ema20'] > 0: score += 1
                if ind['vs_ema50'] > 0: score += 1
                if ind['rsi'] > 50: score += 1
                if ind['macd_hist'] > 0: score += 1
                if bias == 'BULLISH': score += 2
                elif bias == 'CHANNEL_UP': score += 1
                elif bias == 'BEARISH': score -= 2
                elif bias == 'CHANNEL_DOWN': score -= 1

                direction = 'BULL' if score >= 3 else ('BEAR' if score <= -1 else 'NEUT')

                sym_results[tf] = {
                    'price': ind['price'],
                    'rsi': ind['rsi'],
                    'vs_ema20': ind['vs_ema20'],
                    'macd_hist': ind['macd_hist'],
                    'bb_pos': ind['bb_position'],
                    'bias': bias,
                    'trend': trend,
                    'score': score,
                    'direction': direction,
                    # 🆕 裸K新特征
                    'complex_pullbacks': complex_pbs,
                    'fvgs': fvgs[:5],
                    'order_blocks': order_blocks[:5],
                    'channel_outcome': channel_outcome,
                    # 🆕 Vision识图新特征
                    'low_high_counts': low_highs[:6],       # Low1/Low2/High1/High2
                    'signal_entry_pairs': sig_pairs[:5],    # 信号K+入场K对
                    'momentum_warnings': mom_warnings[:3],  # 强动能追单警告
                }

                # 🆕 日线额外计算五维评分卡
                if tf == '1d':
                    try:
                        fds = five_dimension_score(df)
                        sym_results['five_dim'] = fds
                    except Exception:
                        sym_results['five_dim'] = None
        except Exception as e:
            sym_results['error'] = str(e)

        results[sym] = sym_results

    return results


def get_sentiment():
    """获取市场情绪"""
    try:
        from market_sentiment import MarketSentimentEngine
        mse = MarketSentimentEngine()
        fg = mse.fetch_fear_greed()
        overlays = {}
        for sym in ['BTC', 'ETH', 'SOL']:
            try:
                overlays[sym] = mse.get_sentiment_overlay(sym)
            except Exception:
                overlays[sym] = {}
        return {
            'fear_greed': fg.current_value if hasattr(fg, 'current_value') else '?',
            'classification': fg.classification if hasattr(fg, 'classification') else '?',
            'overlays': overlays,
        }
    except Exception as e:
        return {'error': str(e)}


def get_positions(exchange) -> Tuple[List[dict], dict]:
    """获取持仓和余额"""
    positions = []
    try:
        bal = exchange.fetch_balance()
        equity = bal['total'].get('USDT', 0)
        free = bal['free'].get('USDT', 0)
    except Exception:
        equity, free = 0, 0

    try:
        pos = exchange.fetch_positions()
        for p in pos:
            if float(p.get('contracts', 0)) > 0:
                positions.append({
                    'symbol': p['symbol'],
                    'side': p['side'],
                    'entry': float(p['entryPrice']),
                    'mark': float(p['markPrice']),
                    'size': float(p['contracts']),
                    'pnl': float(p['unrealizedPnl']),
                    'leverage': int(p.get('leverage', 0)),
                    'margin': float(p['initialMargin']),
                })
    except Exception:
        pass

    return positions, {'equity': equity, 'free': free}


def generate_decisions(positions: list, klines: dict, sentiment: dict) -> List[str]:
    """基于多维数据+五维评分卡生成决策建议"""
    decisions = []

    for p in positions:
        sym = p['symbol'].replace('/USDT:USDT', '-USDT-SWAP')
        sym_short = p['symbol'].split('/')[0]
        k = klines.get(sym, {})
        d1 = k.get('1d', {})
        d4 = k.get('4h', {})
        d1h = k.get('1h', {})
        fds = k.get('five_dim')  # 🆕 五维评分

        pnl_pct = (p['pnl'] / p['margin'] * 100) if p['margin'] > 0 else 0

        # 止损检查
        if pnl_pct <= -15:
            decisions.append(f"🚨 止损! {sym_short} {p['side']} 亏损{pnl_pct:.0f}%: 立即平仓")
            continue

        # 盈利保护
        if pnl_pct >= 15:
            decisions.append(f"💰 盈利保护 {sym_short} +{pnl_pct:.0f}%: 建议推止盈到成本价")

        # 🆕 五维评分冲突检测
        if fds:
            score_action = fds.get('action', '')
            score_composite = fds.get('composite', 50)
            stars = fds.get('stars', 3)

            # 多空冲突
            if p['side'] == 'long' and score_action in ('SELL', 'LIGHT_SHORT'):
                decisions.append(f"🚨 {sym_short} 做多但五维评分建议{score_action}({score_composite:.0f}/100): 三框架矛盾，立即平仓!")
            elif p['side'] == 'short' and score_action in ('STRONG_BUY', 'BUY', 'LIGHT_BUY'):
                decisions.append(f"🚨 {sym_short} 做空但五维评分建议{score_action}({score_composite:.0f}/100): 三框架矛盾，立即平仓!")

            # 超买减仓
            rsi_val = fds.get('rsi', 50)
            if p['side'] == 'long' and rsi_val >= 75 and fds['scores']['trend'] >= 70:
                decisions.append(f"🔻 {sym_short} 五维趋势强但RSI={rsi_val}超买: 全哥建议减仓1/3锁定利润")
            elif p['side'] == 'long' and rsi_val >= 70 and fds['scores']['trend'] >= 50:
                decisions.append(f"🟡 {sym_short} RSI={rsi_val}偏热: 注意回调风险")

            # 飞天猪检测
            if fds['scores']['trend'] >= 70 and rsi_val < 70:
                decisions.append(f"🐷 {sym_short} 飞天猪! 趋势强且未过热，全哥说继续拿")
            elif fds['scores']['trend'] <= 30:
                decisions.append(f"🥓 {sym_short} 猪已宰 — 趋势很弱，全哥建议考虑退出")
        else:
            # 无五维数据时用原来的裸K判断
            daily_bias = d1.get('bias', '?')
            daily_score = d1.get('score', 0)
            if p['side'] == 'long' and daily_score <= -1:
                decisions.append(f"⚠️ {sym_short} 做多但日线偏空(score={daily_score}): 考虑减仓或设紧止损")
            elif p['side'] == 'short' and daily_score >= 3:
                decisions.append(f"⚠️ {sym_short} 做空但日线偏多(score={daily_score}): 平空! 别对抗趋势")

        # 超买超卖（用4h数据作为备选）
        rsi_4h = d4.get('rsi', 50)
        if isinstance(rsi_4h, (int, float)):
            if p['side'] == 'long' and rsi_4h >= 80:
                decisions.append(f"🔻 {sym_short} 4h RSI={rsi_4h} 严重超买: 考虑减仓1/3")
            elif p['side'] == 'short' and rsi_4h <= 20:
                decisions.append(f"🔺 {sym_short} 4h RSI={rsi_4h} 严重超卖: 空单注意风险")

    # 加密机会检测
    btc_k = klines.get('BTC-USDT-SWAP', {})
    btc_4h = btc_k.get('4h', {})
    btc_rsi = btc_4h.get('rsi', 50)
    fg = sentiment.get('fear_greed', 50)

    if isinstance(fg, (int, float)) and fg <= 15 and btc_rsi <= 30:
        decisions.append(f"🎯 加密抄底机会! 恐慌={fg} BTC 4h RSI={btc_rsi}: 等BTC确认$60K支撑后考虑多单")

    return decisions


def build_report(positions: list, klines: dict, sentiment: dict, decisions: list) -> str:
    """构建白话策略报告"""
    now = datetime.now(timezone.utc).strftime('%m-%d %H:%M UTC')

    # 盈亏汇总
    total_pnl = sum(p['pnl'] for p in positions)
    longs = [p for p in positions if p['side'] == 'long']
    shorts = [p for p in positions if p['side'] == 'short']

    fg = sentiment.get('fear_greed', '?')
    fg_class = sentiment.get('classification', '?')

    report = f"""🐾 Yina 30min策略速报 | {now}

━━━━━━━━━━━━━━━━
📊 账户总览
━━━━━━━━━━━━━━━━
💰 持仓 {len(positions)} 个 (多{len(longs)}/空{len(shorts)}) | 浮动盈亏: {total_pnl:+.2f} USDT
😱 恐慌指数: {fg} ({fg_class})

"""
    # 持仓明细
    if positions:
        report += "━━━━━━━━━━━━━━━━\n📋 持仓状态\n━━━━━━━━━━━━━━━━\n"
        for p in positions:
            sym = p['symbol'].split('/')[0]
            pnl_pct = (p['pnl'] / p['margin'] * 100) if p['margin'] > 0 else 0
            emoji = '🟢' if pnl_pct > 5 else ('🟡' if pnl_pct > -5 else '🔴')

            # K线信号
            k_sym = p['symbol'].replace('/USDT:USDT', '-USDT-SWAP')
            k = klines.get(k_sym, {})
            d1 = k.get('1d', {})
            d4 = k.get('4h', {})
            bias_1d = d1.get('bias', '?')
            rsi_4h = d4.get('rsi', '?')

            rsi_str = f"{rsi_4h:.1f}" if isinstance(rsi_4h, (int, float)) else str(rsi_4h)
            report += f"{emoji} {sym:6s} {p['side']:5s} | 入场:{p['entry']:.2f} 现价:{p['mark']:.2f} | PnL:{p['pnl']:+.2f} ({pnl_pct:+.1f}%) | 日线:{str(bias_1d)} 4hRSI:{rsi_str}\n"

    # K线快照（关键品种）
    report += "\n━━━━━━━━━━━━━━━━\n🔍 关键品种K线快照\n━━━━━━━━━━━━━━━━\n"
    key_pairs = ['BTC-USDT-SWAP', 'ETH-USDT-SWAP', 'INTC-USDT-SWAP', 'SOL-USDT-SWAP', 'TSLA-USDT-SWAP']
    for sym in key_pairs:
        k = klines.get(sym, {})
        if not k or 'error' in k:
            continue
        name = sym.replace('-USDT-SWAP', '')
        d1 = k.get('1d', {})
        d4 = k.get('4h', {})
        d1h = k.get('1h', {})
        rsi4 = d4.get('rsi','?')
        rsi1 = d1h.get('rsi','?')
        rsi4_str = f"{rsi4:.1f}" if isinstance(rsi4, (int, float)) else str(rsi4)
        rsi1_str = f"{rsi1:.1f}" if isinstance(rsi1, (int, float)) else str(rsi1)
        report += f"  {name:6s} 日线:{str(d1.get('bias','?')):12s} trend:{d1.get('trend',0):.2f} | 4h RSI:{rsi4_str:>5s} | 1h RSI:{rsi1_str:>5s}\n"

    # 🆕 Vision识图: Low2/信号K+入场K 高胜率信号汇总
    vision_highlights = []
    for sym in key_pairs:
        k = klines.get(sym, {})
        for tf in ['1h', '4h']:
            tf_data = k.get(tf, {})
            pairs = tf_data.get('signal_entry_pairs', [])
            lh = tf_data.get('low_high_counts', [])
            mom_w = tf_data.get('momentum_warnings', [])
            # 高胜率Low2/High2信号
            for l in lh:
                if l.get('confirmed') and l.get('quality', 0) >= 0.8:
                    vision_highlights.append(f"{sym.replace('-USDT-SWAP','')}{tf} {l['kind']}✅")
                    break
            # 强动能追单警告
            if mom_w:
                vision_highlights.append(f"{sym.replace('-USDT-SWAP','')}{tf} 追单⚠️")

    if vision_highlights:
        report += "\n🎓 Vision信号: " + " | ".join(vision_highlights[:8]) + "\n"

    # 决策
    if decisions:
        report += "\n━━━━━━━━━━━━━━━━\n🎯 策略建议\n━━━━━━━━━━━━━━━━\n"
        for i, d in enumerate(decisions, 1):
            report += f"{i}. {d}\n"

    if not decisions:
        report += "\n✅ 当前持仓无异常信号，维持现有策略\n"

    report += f"\n🐶 下次分析: 30分钟后 | 自动循环运行中~"
    return report


def push_wechat(content: str):
    """推送到企业微信"""
    try:
        # 分段发送（企微限4096字符）
        max_len = 3500
        if len(content) <= max_len:
            r = requests.post(WECHAT_URL, json={
                "msgtype": "markdown",
                "markdown": {"content": content}
            }, timeout=10)
            return r.json()
        else:
            parts = []
            remaining = content
            while len(remaining) > max_len:
                split_at = remaining.rfind('\n', 0, max_len)
                if split_at == -1:
                    split_at = max_len
                parts.append(remaining[:split_at])
                remaining = remaining[split_at:]
            parts.append(remaining)

            results = []
            for part in parts:
                r = requests.post(WECHAT_URL, json={
                    "msgtype": "markdown",
                    "markdown": {"content": part}
                }, timeout=10)
                results.append(r.json())
                time.sleep(0.5)
            return results
    except Exception as e:
        return {'error': str(e)}


def run_cycle(quiet: bool = False):
    """执行一次完整分析循环"""
    load_env()

    if not quiet:
        print(f"🕐 [{datetime.now().strftime('%H:%M:%S')}] 开始30分钟分析循环...")

    exchange = init_exchange()

    # 1. 获取持仓
    positions, balance = get_positions(exchange)
    if not quiet:
        print(f"   持仓: {len(positions)}个 | 权益: ${balance['equity']:.2f} | 可用: ${balance['free']:.2f}")

    # 2. 获取情绪
    sentiment = get_sentiment()
    if not quiet:
        fg = sentiment.get('fear_greed', '?')
        print(f"   恐慌指数: {fg}")

    # 3. K线扫描（持仓+关注列表）
    all_syms = list(set(
        [p['symbol'].replace('/USDT:USDT', '-USDT-SWAP') for p in positions] +
        ['BTC-USDT-SWAP', 'ETH-USDT-SWAP']
    ))
    klines = scan_klines(exchange, all_syms)
    if not quiet:
        print(f"   K线扫描: {len([k for k in klines if 'error' not in k])}个完成")

    # 4. 生成决策
    decisions = generate_decisions(positions, klines, sentiment)
    if not quiet:
        print(f"   决策: {len(decisions)}条建议")

    # 5. 构建报告
    report = build_report(positions, klines, sentiment, decisions)

    # 6. 推送到企微
    result = push_wechat(report)
    if not quiet:
        print(f"   推送: {result}")

    return {
        'positions': len(positions),
        'balance': balance,
        'sentiment': sentiment,
        'decisions': decisions,
    }


def run_daemon(interval_minutes: int = 30):
    """后台持续运行"""
    print(f"🤖 Yina 30分钟自动分析引擎启动")
    print(f"   间隔: {interval_minutes}分钟")
    print(f"   推送: 企业微信「金融监控」群")
    print(f"   PID: {os.getpid()}")
    print()

    # 启动时立即跑一次
    print("⚡ 首次分析...")
    try:
        run_cycle(quiet=False)
    except Exception as e:
        print(f"   ⚠️ 首次分析失败: {e}")
        traceback.print_exc()

    # 然后定期循环
    cycle = 1
    while True:
        wait = interval_minutes * 60
        print(f"\n⏰ 下次分析: {interval_minutes}分钟后...")
        time.sleep(wait)

        cycle += 1
        print(f"\n🔄 第{cycle}次分析 [{datetime.now().strftime('%H:%M:%S')}]")
        try:
            run_cycle(quiet=False)
        except Exception as e:
            print(f"   ⚠️ 分析失败: {e}")
            traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Yina 30分钟自动分析引擎')
    parser.add_argument('--daemon', action='store_true', help='后台持续运行')
    parser.add_argument('--once', action='store_true', help='单次运行')
    parser.add_argument('--interval', type=int, default=30, help='分析间隔(分钟)')
    args = parser.parse_args()

    if args.daemon:
        run_daemon(args.interval)
    else:
        # 默认单次
        result = run_cycle(quiet=False)
        print("\n✅ 分析完成" if result else "\n❌ 分析失败")
