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
    "aggressive": True,            # 激进交易 (Fix #1: 开启，多时间框架RSI+MACD+布林带)
    "naked_k": True,               # 🕯️ 裸K价格行为策略 (优先)
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

# ── OKX REST 数据源 (零 ccxt 依赖) ──
_okx_provider = None


def _get_okx_provider():
    """懒加载 OKX REST 数据提供器"""
    global _okx_provider
    if _okx_provider is None:
        from okx_rest_data import OKXDataProvider
        _okx_provider = OKXDataProvider()
    return _okx_provider


def fetch_all_ohlcv(symbols: list = None, limit: int = 200) -> Dict[str, pd.DataFrame]:
    """
    拉取所有币种的日线OHLCV数据 (纯 OKX REST API，零 ccxt 依赖)

    Returns:
        {symbol: DataFrame with columns [date, open, high, low, close, volume]}
    """
    if symbols is None:
        symbols = ALL_CRYPTO_SYMBOLS

    okx = _get_okx_provider()

    # 转换格式: "BTC/USDT" → "BTC-USDT"
    okx_symbols = [s.replace("/", "-") for s in symbols]

    market_data = {}
    raw_data = okx.fetch_all_ohlcv(okx_symbols, "1D", limit=limit)
    for sym, df in raw_data.items():
        if df is not None and len(df) >= 50:
            # 转回 ccxt 兼容格式: "BTC-USDT" → "BTC/USDT"
            orig_sym = sym.replace("-", "/")
            market_data[orig_sym] = df

    if not market_data:
        print("⚠️ [StrategyRunner] 无市场数据 (OKX REST)")

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

    # 🔴 归一化 stop_loss: 策略信号可能是价格也可能是百分比
    raw_sl = sig.get("stop_loss", 0)
    raw_price = sig.get("price", 0)
    if raw_sl and raw_sl > 1.0 and raw_price > 0:
        # 价格格式 → 百分比 (e.g., stop_loss=2.12, price=2.30 → SL=7.8%)
        stop_loss_pct = abs(raw_sl / raw_price - 1)
    elif raw_sl and 0 < abs(raw_sl) <= 1.0:
        stop_loss_pct = abs(raw_sl)
    else:
        stop_loss_pct = 0

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
        "stop_loss": stop_loss_pct,
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


def _merge_with_priority(all_signals: List[dict]) -> List[dict]:
    """
    🕯️ 优先级合并: 裸K信号 (score >= 7/10) 覆盖同symbol的ML信号

    规则:
      1. 裸K信号 (kline_priority=True 且 kline_score_3step >= 7)
         → 覆盖同symbol的所有ML信号 (不管ML给BUY还是SELL)
      2. 裸K SCORE < 7 的信号与ML并列显示但不覆盖
      3. 裸K HOLD 信号不阻塞ML
      4. 不同symbol的ML信号不受影响
    """
    # 分离裸K和ML信号
    kline_sigs = [s for s in all_signals if s.get("kline_priority")]
    ml_sigs = [s for s in all_signals if not s.get("kline_priority")]

    # 裸K优先覆盖的symbol集合 (score_3step >= 7)
    override_symbols = set()
    for s in kline_sigs:
        if s.get("kline_score_3step", 0) >= 7:
            override_symbols.add(s["symbol"])

    # 合并: 裸K信号全保留 + ML信号中未被覆盖的
    merged = list(kline_sigs)
    overridden_count = 0
    for s in ml_sigs:
        if s["symbol"] not in override_symbols:
            merged.append(s)
        else:
            overridden_count += 1

    if overridden_count > 0:
        print(f"  🔝 [优先覆盖] 裸K信号覆盖了 {overridden_count} 条ML信号 "
              f"(symbols: {', '.join(sorted(override_symbols))})")

    # 按 score 降序排列，裸K优先标记置顶
    return sorted(merged, key=lambda s: (s.get("kline_priority", False), s["score"]), reverse=True)


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
# 策略 5: 裸K价格行为策略 🕯️ 优先
# ═══════════════════════════════════════════════════════════

def _run_naked_k(market_data: Dict[str, pd.DataFrame]) -> List[dict]:
    """
    运行裸K价格行为策略 — 熊猫教练「三步读图法」体系

    独立拉取 1h/4h/1d 数据 (不同于ML的日线),
    多时间框架扫描 (1d → 4h → 1h 逐级确认),
    三步评分过滤 + Low2/High2门禁 + 动量追单拦截。

    使用 OKX REST API 直接获取数据 (零 ccxt 依赖)。
    """
    from strategies import KLineStrategy
    from naked_k_scanner import scan_multi_timeframe, scan_symbol
    from okx_rest_data import fetch_okx_ohlcv

    strategy = KLineStrategy.create()
    if strategy.status != "running":
        return []

    # 裸K深度分析聚焦Tier1币种 (15个, 太多会超API限制)
    try:
        from symbol_config import TIER1_ML_HEAVY
        scan_symbols = [s for s in TIER1_ML_HEAVY if "/USDT" in s][:20]
    except ImportError:
        scan_symbols = [
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
            "XRP/USDT", "ADA/USDT", "DOGE/USDT", "AVAX/USDT",
            "DOT/USDT", "LINK/USDT", "UNI/USDT", "ATOM/USDT",
            "APT/USDT", "NEAR/USDT", "LTC/USDT",
        ]

    pass_score = strategy.get_param("pass_score", 7)
    min_rr = strategy.get_param("min_risk_reward", 1.2)
    require_low2 = strategy.get_param("require_low2", True)
    block_chase = strategy.get_param("block_momentum_chase", True)
    timeframes = ['1H', '4H', '1D']  # OKX REST 格式
    limit = 200

    signals = []

    for sym in scan_symbols:
        try:
            # 拉取多时间框架OHLCV (OKX REST)
            dfs = {}
            okx_sym = sym.replace("/", "-")
            for tf in timeframes:
                try:
                    df = fetch_okx_ohlcv(okx_sym, tf, limit=limit)
                    if df is not None and len(df) >= 50:
                        dfs[tf.lower()] = df
                except Exception:
                    pass

            # 至少需要1h数据
            if '1h' not in dfs or '4h' not in dfs:
                continue

            # 尝试多时间框架扫描 (1d → 4h → 1h)
            results = None
            try:
                if '1d' in dfs:
                    results = scan_multi_timeframe(dfs, symbol=sym)
            except Exception:
                pass

            # Fallback: 单时间框架扫描
            if results is None:
                try:
                    result_4h = scan_symbol(dfs.get('4h'), sym, '4h',
                                            dfs.get('1d'))
                    result_1h = scan_symbol(dfs.get('1h'), sym, '1h',
                                            dfs.get('4h'))
                    results = {'4h': result_4h, '1h': result_1h}
                except Exception:
                    continue

            # 处理每个时间框架
            for tf in ['4h', '1h']:
                if tf not in results or results[tf] is None:
                    continue

                result = results[tf]

                for scalp_sig in result.signals:
                    # ── 三步评分过滤 ──
                    if scalp_sig.score_3step < pass_score:
                        continue
                    if scalp_sig.risk_reward < min_rr:
                        continue

                    # ── Low2/High2 门禁 ──
                    matching_lh = None
                    if require_low2 and result.low_high_counts:
                        for lh_dict in result.low_high_counts:
                            if (lh_dict.get('confirmed') and
                                lh_dict.get('quality', 0) >= 0.6):
                                # 匹配方向: 做多需要Low2, 做空需要High2
                                if (scalp_sig.action.value == 'BUY' and
                                    lh_dict.get('type', '').startswith('Low')):
                                    matching_lh = lh_dict
                                    break
                                elif (scalp_sig.action.value == 'SELL' and
                                      lh_dict.get('type', '').startswith('High')):
                                    matching_lh = lh_dict
                                    break
                        if matching_lh is None:
                            continue  # 跳过无Low2/High2确认的信号

                    # ── 动量追单拦截 ──
                    if block_chase and result.momentum_warnings:
                        sig_bar = getattr(scalp_sig, 'signal_bar_idx', -1)
                        has_warning = any(
                            w.get('bar_idx', -1) >= sig_bar - 3
                            for w in result.momentum_warnings
                            if isinstance(w, dict)
                        )
                        if has_warning:
                            continue

                    # ── 找匹配的信号K+入场K对 ──
                    matching_pair = None
                    if result.signal_entry_pairs:
                        sig_bar = getattr(scalp_sig, 'signal_bar_idx', -1)
                        for sp_dict in result.signal_entry_pairs:
                            if isinstance(sp_dict, dict):
                                if (sp_dict.get('signal_idx') == sig_bar or
                                    sp_dict.get('entry_idx') == sig_bar):
                                    matching_pair = sp_dict
                                    break

                    # ── 获取高级别趋势 ──
                    higher_bias = None
                    larger_tf = '4h' if tf == '1h' else '1d'
                    if larger_tf in results and results[larger_tf]:
                        higher_bias = results[larger_tf].market_bias.value if hasattr(
                            results[larger_tf].market_bias, 'value') else str(
                            results[larger_tf].market_bias)

                    # ── 转换信号 ──
                    sig = strategy._convert_signal(
                        scalp_sig, sym,
                        low_high=matching_lh,
                        sig_pair=matching_pair,
                        higher_tf_bias=higher_bias,
                    )
                    signals.append(sig)

            # 币种间延迟 (避免API限速)
            import time as _time
            _time.sleep(0.5)

        except Exception as e:
            print(f"  ⚠️ [裸K] {sym} 扫描失败: {e}")

    return signals

STRATEGY_RUNNERS = {
    "ml_momentum": _run_ml_momentum,
    "mean_reversion_grid": _run_mean_reversion,
    "cross_market_alpha": _run_cross_market_alpha,
    "aggressive": _run_aggressive,
    "naked_k": _run_naked_k,  # 🕯️ 裸K策略 (优先)
}

STRATEGY_NAMES = {
    "ml_momentum": "ML融合动量",
    "mean_reversion_grid": "均值回归网格",
    "cross_market_alpha": "跨市场Alpha套利",
    "aggressive": "激进交易",
    "naked_k": "裸K价格行为",
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

    # 3. 🕯️ 优先级合并 (裸K信号>=7分覆盖ML)
    deduped = _merge_with_priority(all_signals)

    if len(deduped) < len(all_signals):
        merged_ml = sum(1 for s in all_signals if not s.get("kline_priority"))
        merged_kline = sum(1 for s in all_signals if s.get("kline_priority"))
        print(f"  🔄 优先级合并: {len(all_signals)} → {len(deduped)} 信号 "
              f"(裸K{merged_kline}条 + ML{merged_ml}条)")

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
