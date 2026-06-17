"""
🐾 多策略并行运行器 — Strategy Runner
=====================================
统一管理所有加密市场策略的初始化、数据拉取、信号生成、去重。
为 daemon 提供干净的信号列表，不直接操作持仓。

当前策略:
  1. ML融合动量 (EMA三确认+MACD+RSI投票)
  2. 均值回归网格 (布林带+RSI+网格交易)
  3. 跨市场Alpha套利 (多空配对+动量轮动)
  4. 激进交易 (多时间框架RSI+MACD+布林带)

使用:
  from strategy_runner import run_all_strategies
  signals = run_all_strategies()
"""

from __future__ import annotations
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict

# ── 策略开关 ──
STRATEGY_CONFIG = {
    "ml_momentum": True,           # ML融合动量策略
    "mean_reversion_grid": True,   # 均值回归网格策略
    "cross_market_alpha": True,    # 跨市场Alpha套利
    "aggressive": False,           # 激进交易 (默认关闭，5分钟扫描+大仓位)
}

# ── 需要拉取数据的币种 (所有策略覆盖的币种并集) ──
# 从统一配置中心加载 Tier1+2 (~80个币种)
try:
    from symbol_config import get_all_crypto_symbols
    ALL_CRYPTO_SYMBOLS = get_all_crypto_symbols(tiers=[1, 2])
except ImportError:
    ALL_CRYPTO_SYMBOLS = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
        "XRP/USDT", "AVAX/USDT", "ADA/USDT", "DOGE/USDT",
        "DOT/USDT", "LINK/USDT",
    ]

# ── ccxt 交易所实例 (懒加载) ──
_exchange = None


def _get_exchange():
    global _exchange
    if _exchange is None:
        try:
            import ccxt
            _exchange = ccxt.binance({"enableRateLimit": True, "timeout": 15000})
        except Exception:
            _exchange = None
    return _exchange


def fetch_all_ohlcv(symbols: list = None, limit: int = 200) -> Dict[str, pd.DataFrame]:
    """
    拉取所有币种的日线OHLCV数据

    Returns:
        {symbol: DataFrame with columns [open, high, low, close, volume]}
    """
    if symbols is None:
        symbols = ALL_CRYPTO_SYMBOLS

    ex = _get_exchange()
    if ex is None:
        print("⚠️ [StrategyRunner] 无法连接交易所")
        return {}

    market_data = {}
    for sym in symbols:
        try:
            ohlcv = ex.fetch_ohlcv(sym, "1d", limit=limit)
            df = pd.DataFrame(ohlcv, columns=["date", "open", "high", "low", "close", "volume"])
            df["date"] = pd.to_datetime(df["date"], unit="ms")
            if len(df) >= 50:
                market_data[sym] = df
            else:
                print(f"  ⚠️ [StrategyRunner] {sym} 数据不足 ({len(df)}行)")
        except Exception as e:
            print(f"  ⚠️ [StrategyRunner] {sym} 拉取失败: {e}")

    return market_data


def _coin_name(symbol: str) -> str:
    """BTC/USDT → Bitcoin"""
    names = {
        "BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana",
        "BNB": "BNB Chain", "XRP": "XRP", "AVAX": "Avalanche",
    }
    ticker = symbol.split("/")[0]
    return names.get(ticker, ticker)


def _normalize_signal(sig: dict, strategy_name: str) -> dict:
    """
    将策略返回的信号统一为标准格式

    标准格式:
      {symbol, name, action, price, score, confidence,
       reasons, risk_level, suggested_size, stop_loss, take_profit, strategy_name}
    """
    # 兼容 strategies.py 的信号格式 (已经比较标准)
    # 兼容 aggressive_trader 的 TradeSignal
    symbol = sig.get("symbol", "")
    if not symbol and hasattr(sig, "symbol"):
        symbol = sig.symbol

    return {
        "symbol": symbol,
        "name": sig.get("name", _coin_name(symbol)),
        "action": sig.get("action", "HOLD"),
        "price": sig.get("price", 0),
        "score": sig.get("score", 50),
        "confidence": sig.get("confidence", 0.5),
        "reasons": sig.get("reasons", []) if isinstance(sig.get("reasons"), list) else [sig.get("reason", "")],
        "risk_level": sig.get("risk_level", "medium"),
        "suggested_size": sig.get("suggested_size", 0.05),
        "stop_loss": sig.get("stop_loss", 0),
        "take_profit": sig.get("take_profit", 0),
        "strategy_name": strategy_name,
    }


def _deduplicate_signals(all_signals: List[dict]) -> List[dict]:
    """
    跨策略去重: 同 symbol+action → 保留 confidence 最高的
    不同 action 的信号都保留
    """
    # 按 (symbol, action) 分组，每组保留最高 confidence
    best = {}
    for s in all_signals:
        key = (s["symbol"], s["action"])
        if key not in best or s["confidence"] > best[key]["confidence"]:
            best[key] = s

    # 按 score 降序排列
    return sorted(best.values(), key=lambda s: s["score"], reverse=True)


# ═══════════════════════════════════════════════════════════
# 策略 1: ML融合动量
# ═══════════════════════════════════════════════════════════

def _run_ml_momentum(market_data: Dict[str, pd.DataFrame]) -> List[dict]:
    """运行 ML融合动量策略"""
    from strategies import MLMomentumStrategy
    strategy = MLMomentumStrategy.create()

    # 过滤出策略需要的标的
    strategy_data = {s: market_data[s] for s in strategy.config.symbols if s in market_data}
    if not strategy_data:
        return []

    try:
        raw_signals = strategy.generate_signals(strategy_data)
    except Exception as e:
        print(f"  ⚠️ [ML融合动量] 信号生成失败: {e}")
        return []

    return [_normalize_signal(s, "ML融合动量") for s in raw_signals]


# ═══════════════════════════════════════════════════════════
# 策略 2: 均值回归网格
# ═══════════════════════════════════════════════════════════

def _run_mean_reversion(market_data: Dict[str, pd.DataFrame]) -> List[dict]:
    """运行 均值回归网格策略"""
    from strategies import MeanReversionGridStrategy
    strategy = MeanReversionGridStrategy.create()

    strategy_data = {s: market_data[s] for s in strategy.config.symbols if s in market_data}
    if not strategy_data:
        return []

    try:
        raw_signals = strategy.generate_signals(strategy_data)
    except Exception as e:
        print(f"  ⚠️ [均值回归网格] 信号生成失败: {e}")
        return []

    return [_normalize_signal(s, "均值回归网格") for s in raw_signals]


# ═══════════════════════════════════════════════════════════
# 策略 3: 跨市场Alpha套利
# ═══════════════════════════════════════════════════════════

def _run_cross_market_alpha(market_data: Dict[str, pd.DataFrame]) -> List[dict]:
    """运行 跨市场Alpha套利策略"""
    from strategies import CrossMarketAlphaStrategy
    strategy = CrossMarketAlphaStrategy.create()

    strategy_data = {s: market_data[s] for s in strategy.config.symbols if s in market_data}
    if not strategy_data:
        return []

    try:
        raw_signals = strategy.generate_signals(strategy_data)
    except Exception as e:
        print(f"  ⚠️ [Alpha套利] 信号生成失败: {e}")
        return []

    return [_normalize_signal(s, "跨市场Alpha套利") for s in raw_signals]


# ═══════════════════════════════════════════════════════════
# 策略 4: 激进交易
# ═══════════════════════════════════════════════════════════

def _run_aggressive(market_data: Dict[str, pd.DataFrame]) -> List[dict]:
    """
    运行 激进交易策略
    注意: aggressive_trader 用自己的 fetch_ohlcv + compute_indicators，
    这里只取其 scan_symbol 逻辑。但因为 aggressive_trader 用 1h/4h 数据，
    而非日线，所以需要单独处理。
    """
    try:
        from aggressive_trader import scan_symbol, SYMBOLS
    except ImportError as e:
        print(f"  ⚠️ [激进交易] 导入失败: {e}")
        return []

    signals = []
    for sym in SYMBOLS:
        try:
            sig = scan_symbol(sym)
            if sig is None:
                continue
            signals.append(_normalize_signal({
                "symbol": sig.symbol,
                "name": sig.name,
                "action": sig.action,
                "price": sig.price,
                "score": sig.score,
                "confidence": max(0.3, min(0.85, sig.score / 100)),
                "reasons": [sig.reason],
                "risk_level": "high",
                "suggested_size": 0.10,  # 激进=更大的仓位比例
                "stop_loss": sig.stop_loss,
                "take_profit": sig.take_profit,
            }, "激进交易"))
        except Exception as e:
            print(f"  ⚠️ [激进交易] {sym} 扫描失败: {e}")

    return signals


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

STRATEGY_RUNNERS = {
    "ml_momentum": _run_ml_momentum,
    "mean_reversion_grid": _run_mean_reversion,
    "cross_market_alpha": _run_cross_market_alpha,
    "aggressive": _run_aggressive,
}

STRATEGY_NAMES = {
    "ml_momentum": "ML融合动量",
    "mean_reversion_grid": "均值回归网格",
    "cross_market_alpha": "跨市场Alpha套利",
    "aggressive": "激进交易",
}


def run_all_strategies(market_data: Dict[str, pd.DataFrame] = None,
                        sentiment_engine=None) -> List[dict]:
    """
    运行所有启用的策略，返回去重后的信号列表

    Args:
        market_data: 预先拉取的市场数据（可选）。如果为None，自动拉取。
        sentiment_engine: 市场情绪引擎 (MarketSentimentEngine, 可选)

    Returns:
        [{symbol, name, action, price, score, confidence,
          reasons, risk_level, suggested_size, stop_loss, take_profit, strategy_name}, ...]
    """
    # 1. 拉取数据
    if market_data is None:
        market_data = fetch_all_ohlcv()

    if not market_data:
        print("⚠️ [StrategyRunner] 无市场数据，跳过所有策略")
        return []

    # 2. 运行各策略
    all_signals = []

    for key, runner in STRATEGY_RUNNERS.items():
        if not STRATEGY_CONFIG.get(key, False):
            continue

        name = STRATEGY_NAMES.get(key, key)
        try:
            signals = runner(market_data)
            buy_count = sum(1 for s in signals if s["action"] == "BUY")
            sell_count = sum(1 for s in signals if s["action"] == "SELL")

            if buy_count or sell_count:
                print(f"  🧠 [{name}] {len(signals)} 信号 ({buy_count} BUY, {sell_count} SELL)")
                for s in signals:
                    icon = "🟢" if s["action"] == "BUY" else "🔴"
                    print(f"     {icon} {s['symbol']:12s} {s['action']:4s} | "
                          f"评分{s['score']:3.0f} | 置信{s['confidence']:.0%}")
            else:
                print(f"  💤 [{name}] 无信号")

            all_signals.extend(signals)
        except Exception as e:
            print(f"  ❌ [{name}] 运行失败: {e}")
            import traceback
            traceback.print_exc()

    # 3. 去重
    deduped = _deduplicate_signals(all_signals)

    if len(deduped) < len(all_signals):
        print(f"  🔄 去重: {len(all_signals)} → {len(deduped)} 信号")

    # 4. 🎭 情绪叠加 (如果传入 sentiment_engine)
    if sentiment_engine:
        for sig in deduped:
            try:
                overlay = sentiment_engine.get_sentiment_overlay(sig["symbol"])
                adj = overlay.get("composite_sentiment", 0.0) * 0.10
                sig["confidence"] = min(0.95, max(0.15, sig["confidence"] + adj))
                sig["sentiment_overlay"] = overlay
            except Exception:
                pass

    return deduped


# ═══════════════════════════════════════════════════════════
# CLI 测试入口
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🐾 策略运行器 — 测试模式\n")
    print(f"策略开关: {STRATEGY_CONFIG}\n")

    print("📡 拉取市场数据...")
    data = fetch_all_ohlcv()
    print(f"   已获取: {list(data.keys())}\n")

    print("🧠 运行所有策略...\n")
    signals = run_all_strategies(market_data=data)

    print(f"\n{'='*60}")
    print(f"📊 最终信号汇总: {len(signals)} 条")
    print(f"{'='*60}")

    for s in signals:
        icon = "🟢" if s["action"] == "BUY" else "🔴" if s["action"] == "SELL" else "⚪"
        print(f"\n  {icon} [{s['strategy_name']}] {s['symbol']} {s['action']}")
        print(f"     价格: ${s['price']:.2f} | 评分: {s['score']:.0f} | 置信: {s['confidence']:.0%}")
        print(f"     止损: ${s['stop_loss']:.2f} | 止盈: ${s['take_profit']:.2f}")
        reasons = s.get('reasons', [])
        if reasons:
            for r in reasons[:3]:
                print(f"     · {r}")
