"""
Chase量化策略 — 自动交易脚本
由 Cron 触发, 扫描→决策→执行→通知

Phase 7 升级: 支持ML信号引擎 (MLSignalEngineV4) + 多币种扫描
  使用方式:
    python3 auto_trade.py --ml              # ML模式 (推荐)
    python3 auto_trade.py                   # 传统模式 (RSI/MACD)
    python3 auto_trade.py --markets crypto --ml  # 只扫加密货币
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from portfolio import PortfolioManager, ALLOCATION
from risk import RiskController
from signals import SignalEngine, CryptoSignals, AStockSignals, USStockSignals

# 偏差修正
try:
    from bias_correction import full_bias_audit, estimate_survival_bias
    BIAS_AWARE = True
except ImportError:
    BIAS_AWARE = False

# ML信号引擎 (Phase 7)
try:
    from ml_signal_v4 import MLSignalEngineV4, SubSignalBuilder, EnsembleSignal
    ML_SIGNAL_AVAILABLE = True
except ImportError:
    ML_SIGNAL_AVAILABLE = False


def is_market_open(market: str) -> bool:
    """判断市场是否交易时段"""
    now = datetime.now()
    weekday = now.weekday()  # 0=Mon, 6=Sun

    if weekday >= 5:  # 周末
        return market == "crypto"

    hour = now.hour + now.minute / 60

    if market == "a_stock":
        return (9.5 <= hour <= 11.5) or (13.0 <= hour <= 15.0)
    elif market == "us_stock":
        # 美股夏令时 21:30-04:00 CST
        return hour >= 21.5 or hour <= 4.0
    elif market == "crypto":
        return True
    return False


class MLAutoTrader:
    """
    ML增强自动交易器 (Phase 7)

    用 MLSignalEngineV4 生成信号 → 多币种扫描 → 自动入场/出场
    跨市场数据共享: 拉一次, 所有symbol复用
    """

    # 加密货币扫描列表 (流动性好 + 数据充足)
    CRYPTO_WATCHLIST = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    ]

    def __init__(self, use_lgbm: bool = True):
        if not ML_SIGNAL_AVAILABLE:
            raise ImportError("ML信号引擎不可用")
        self.use_lgbm = use_lgbm
        self.engine = MLSignalEngineV4()
        self._exchange = None

    @property
    def exchange(self):
        if self._exchange is None:
            try:
                import ccxt
                self._exchange = ccxt.binance({"enableRateLimit": True, "timeout": 15000})
            except Exception:
                self._exchange = None
        return self._exchange

    def fetch_ohlcv(self, symbol: str, limit: int = 400) -> pd.DataFrame | None:
        """拉取OHLCV数据"""
        try:
            ex = self.exchange
            if ex is None:
                return None
            ohlcv = ex.fetch_ohlcv(symbol, "1d", limit=limit)
            df = pd.DataFrame(ohlcv, columns=["date", "open", "high", "low", "close", "volume"])
            df["date"] = pd.to_datetime(df["date"], unit="ms")
            return df
        except Exception as e:
            print(f"  ⚠️ {symbol} 数据拉取失败: {e}")
            return None

    def scan(self, symbols: list = None) -> list[dict]:
        """
        扫描多币种，返回ML信号列表

        Returns:
            [{"symbol": "BTC/USDT", "signal": EnsembleSignal, "price": ..., "action": ...}, ...]
        """
        if symbols is None:
            symbols = self.CRYPTO_WATCHLIST

        results = []
        n = len(symbols)

        # 跨市场数据共享: 只拉一次 (用第一个symbol触发)
        cross_market_loaded = False

        for i, symbol in enumerate(symbols):
            print(f"  🔍 [{i+1}/{n}] {symbol}...", end=" ")

            df = self.fetch_ohlcv(symbol)
            if df is None or len(df) < 200:
                print("数据不足")
                continue

            try:
                # 第一个symbol触发跨市场数据加载
                # 后续symbol跳过 (跨市场数据已在engine中缓存)
                signal = self.engine.generate_signal(df, symbol, use_lgbm=self.use_lgbm)
                price = float(df["close"].values[-1])

                results.append({
                    "symbol": symbol,
                    "name": symbol.replace("/USDT", ""),
                    "price": price,
                    "signal": signal,
                    "action": signal.action,
                    "signal_val": signal.signal_lgbm if (self.use_lgbm and signal.lgbm_available) else signal.signal_ic,
                    "confidence": signal.confidence,
                    "consensus": signal.consensus,
                    "active_themes": signal.n_sub_signals_active,
                })
                icon = "🟢" if signal.action == "BUY" else "🔴" if signal.action == "SELL" else "⚪"
                print(f"{icon} {signal.action} | sig={results[-1]['signal_val']:+.2f} | "
                      f"置信={signal.confidence:.0%} | {signal.n_sub_signals_active}/7主题")

            except Exception as e:
                print(f"信号生成失败: {e}")

        # 按信号值排序 (绝对值越大越极端)
        results.sort(key=lambda r: abs(r["signal_val"]), reverse=True)
        return results

    def get_status_summary(self) -> str:
        """引擎状态摘要"""
        lines = [
            f"🧠 ML信号引擎 v4.0",
            f"  模型: {'✅ 已加载' if self.engine._lgbm_loaded else '⚠️ 未加载 (使用线性IC加权)'}",
            f"  跨市场: {'✅ 已激活' if self.engine.cross_market_available else '⚠️ 不可用'}",
            f"  扫描列表: {', '.join(self.CRYPTO_WATCHLIST)}",
        ]
        return "\n".join(lines)


def auto_scan_and_trade(markets: list = None, use_ml: bool = False):
    """
    自动扫描 → 决策 → 执行

    Args:
        markets: 市场列表 (默认["crypto"])
        use_ml: True=ML信号引擎, False=传统RSI/MACD引擎
    Returns:
        (results, pf)
    """
    if markets is None:
        markets = ["crypto"]  # 默认只扫24/7市场

    pf = PortfolioManager()
    rc = RiskController(pf)
    total_value = pf.total_value
    results = []

    # ── ML模式 ──
    if use_ml and "crypto" in markets:
        if not ML_SIGNAL_AVAILABLE:
            print("⚠️ ML信号引擎不可用, 回退到传统模式")
            use_ml = False
        else:
            print("🧠 ML自动交易模式")
            trader = MLAutoTrader(use_lgbm=True)
            print(trader.get_status_summary())

            ml_signals = trader.scan()
            buy_sigs = [s for s in ml_signals if s["action"] == "BUY"]
            sell_sigs = [s for s in ml_signals if s["action"] == "SELL"]

            # 第一步: 检查持仓止损/止盈
            for pos in pf.open_positions:
                if pos.market != "crypto":
                    continue

                # 更新价格
                for sig in ml_signals:
                    if sig["symbol"] == pos.symbol:
                        pf.update_price(pos.id, sig["price"])
                        break

                # 止损
                if pos.pnl_pct <= -8.0:
                    pf.sell(pos.id, pos.current_price, f"硬止损触发: {pos.pnl_pct:.1f}%")
                    results.append(f"🔴 止损 {pos.symbol}: {pos.pnl_pct:.1f}%")
                # 止盈
                elif pos.pnl_pct >= 15.0:
                    pf.sell(pos.id, pos.current_price, f"止盈触发: +{pos.pnl_pct:.1f}%")
                    results.append(f"🟢 止盈 {pos.symbol}: +{pos.pnl_pct:.1f}%")
                # ML出场信号
                else:
                    for sig in sell_sigs:
                        if sig["symbol"] == pos.symbol:
                            pf.sell(pos.id, pos.current_price,
                                   f"ML卖出信号: {sig['signal_val']:+.2f}")
                            results.append(f"🔴 ML卖出 {pos.symbol}: {pos.pnl_pct:+.1f}%")
                            break

            # 第二步: ML入场机会
            held_symbols = [p.symbol for p in pf.open_positions if p.market == "crypto"]
            for sig in buy_sigs[:3]:  # Top 3
                if sig["symbol"] in held_symbols:
                    continue

                cash = pf.cash.get("crypto", 0)
                size_pct = sig["signal"].suggested_size_pct / 100
                max_size = min(cash * size_pct, cash * 0.5)

                if max_size < 200:
                    continue

                # ML信号分数 (映射到0-100)
                ml_score = abs(sig["signal_val"]) * 30 + sig["confidence"] * 20 + sig["consensus"] * 30
                ml_score = min(100, ml_score)

                check = rc.pre_trade_check("crypto", max_size, ml_score, total_value)
                if not check.passed:
                    print(f"  ⚠️ {sig['symbol']} 风控拦截: {check.reason}")
                    continue

                quantity = max_size / sig["price"]
                reasoning = (
                    f"ML信号={sig['signal_val']:+.2f} | "
                    f"置信={sig['confidence']:.0%} | "
                    f"活跃={sig['active_themes']}/7主题"
                )
                trade = pf.buy(
                    market="crypto", symbol=sig["symbol"],
                    name=sig["name"],
                    price=sig["price"], quantity=quantity,
                    reason=reasoning,
                )
                if trade:
                    results.append(
                        f"🧠 买入 {sig['symbol']} ¥{max_size:.0f} | {reasoning}"
                    )

            pf.take_snapshot()
            return results, pf

    # ── 传统模式 (RSI/MACD) ──
    engine = SignalEngine()

    for market in markets:
        if not is_market_open(market):
            continue

        scanners = {
            "crypto": engine.crypto,
            "a_stock": engine.a_stock,
            "us_stock": engine.us_stock,
        }
        scanner = scanners.get(market)
        if not scanner:
            continue

        all_sigs = scanner.scan()
        buy_sigs = [s for s in all_sigs if s.action == "BUY" and s.score >= 65]
        buy_sigs.sort(key=lambda s: s.score, reverse=True)

        # ── 第一步: 检查持仓止损/止盈 ──
        for pos in pf.open_positions:
            if pos.market != market or pos.status != "closed":
                continue
            for sig in all_sigs:
                if sig.symbol == pos.symbol:
                    pf.update_price(pos.id, sig.price)
                    break

            if pos.pnl_pct <= -8.0:
                pf.sell(pos.id, pos.current_price, f"硬止损触发: {pos.pnl_pct:.1f}%")
                results.append(f"🔴 止损 {pos.symbol}: {pos.pnl_pct:.1f}%")
            elif pos.pnl_pct >= 15.0:
                pf.sell(pos.id, pos.current_price, f"止盈触发: +{pos.pnl_pct:.1f}%")
                results.append(f"🟢 止盈 {pos.symbol}: +{pos.pnl_pct:.1f}%")

        # ── 第二步: 寻找新的入场机会 ──
        for sig in buy_sigs[:3]:
            held_symbols = [p.symbol for p in pf.open_positions if p.market == market]
            if sig.symbol in held_symbols:
                continue

            cash = pf.cash.get(market, 0)
            max_size = min(sig.suggested_size, cash * 0.5)

            if max_size < 200:
                continue

            check = rc.pre_trade_check(market, max_size, sig.score, total_value)
            if not check.passed:
                continue

            quantity = max_size / sig.price
            trade = pf.buy(
                market=market, symbol=sig.symbol, name=sig.name,
                price=sig.price, quantity=quantity,
                reason=" | ".join(sig.reasons),
            )
            if trade:
                results.append(
                    f"🟢 买入 {sig.symbol} ¥{max_size:.0f} | "
                    f"评分{sig.score:.0f} | {sig.reasons[0]}"
                )

    pf.take_snapshot()

    return results, pf


def generate_status_report() -> str:
    """生成状态报告"""
    pf = PortfolioManager()
    rc = RiskController(pf)

    lines = [
        "=" * 50,
        f"🐾 Chase量化策略 · 状态报告",
        f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 50,
        f"",
        f"💰 总资产: ¥{pf.total_value:,.2f} | 盈亏: {pf.total_pnl_pct:+.2f}%",
        f"💵 现金: ¥{pf.total_cash:,.2f} | 持仓: ¥{pf.positions_value:,.2f}",
        f"",
    ]

    # 持仓
    open_pos = pf.open_positions
    if open_pos:
        lines.append("📦 当前持仓:")
        for p in open_pos:
            emoji = "🟢" if p.pnl_pct >= 0 else "🔴"
            lines.append(f"  {emoji} {p.symbol} | ¥{p.entry_price:,.2f}→¥{p.current_price:,.2f} | {p.pnl_pct:+.2f}% | ¥{p.value:,.0f}")
    else:
        lines.append("📦 无持仓")

    lines.append("")

    # 风控
    alerts = pf.check_risk()
    if alerts:
        lines.append("🚨 风控告警:")
        for a in alerts:
            lines.append(f"  {a}")
    else:
        lines.append("✅ 风控正常")

    lines.append("")
    lines.append(f"🎯 月度目标: {pf.total_pnl_pct:+.1f}%/30%")

    # 偏差修正信息
    if BIAS_AWARE:
        bias = estimate_survival_bias("crypto")
        lines.append(f"🔍 偏差修正: 回测×{bias['correction_factor']} (虚高{bias['estimated_overstatement_pct']}%)")

    # ML状态 (Phase 7)
    if ML_SIGNAL_AVAILABLE:
        lines.append("🧠 ML引擎: ✅ 可用 (使用 --ml 启动)")
    else:
        lines.append("🧠 ML引擎: ⚠️ 不可用")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chase量化策略 · 自动交易")
    parser.add_argument("--markets", nargs="+", default=["crypto"],
                       choices=["crypto", "a_stock", "us_stock", "all"])
    parser.add_argument("--report", action="store_true",
                       help="仅生成状态报告")
    parser.add_argument("--ml", action="store_true",
                       help="使用ML信号引擎 (Phase 7)")
    parser.add_argument("--ml-scan", action="store_true",
                       help="仅ML扫描, 不执行交易 (调试用)")
    args = parser.parse_args()

    if args.report:
        print(generate_status_report())
        sys.exit(0)

    # ML调试模式: 仅扫描
    if args.ml_scan:
        if not ML_SIGNAL_AVAILABLE:
            print("❌ ML信号引擎不可用")
            sys.exit(1)
        print("🧠 ML扫描模式 (仅查看信号)\n")
        trader = MLAutoTrader(use_lgbm=True)
        print(trader.get_status_summary())
        print()
        signals = trader.scan()
        print(f"\n📊 扫描结果 ({len(signals)}个):")
        for s in signals:
            icon = "🟢" if s["action"] == "BUY" else "🔴" if s["action"] == "SELL" else "⚪"
            print(f"  {icon} {s['symbol']:12s} | {s['action']:4s} | "
                  f"信号={s['signal_val']:+.3f} | "
                  f"置信={s['confidence']:.0%} | "
                  f"共识={s['consensus']:.0%} | "
                  f"{s['active_themes']}/7主题")
        sys.exit(0)

    markets = ["crypto", "a_stock", "us_stock"] if args.markets == ["all"] else args.markets
    results, pf = auto_scan_and_trade(markets, use_ml=args.ml)

    print(generate_status_report())

    if results:
        print("\n📋 本次操作:")
        for r in results:
            print(f"  {r}")
    else:
        print("\n💤 本次扫描无操作")
