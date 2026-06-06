"""
Chase量化策略 — 自动交易脚本
由 Cron 触发, 扫描→决策→执行→通知
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json

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


def auto_scan_and_trade(markets: list = None):
    """
    自动扫描 → 决策 → 执行
    返回: 执行结果摘要
    """
    if markets is None:
        markets = ["crypto"]  # 默认只扫24/7市场

    pf = PortfolioManager()
    rc = RiskController(pf)
    engine = SignalEngine()

    results = []
    total_value = pf.total_value

    for market in markets:
        if not is_market_open(market):
            continue

        # 扫描信号
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
            # 更新限价
            for sig in all_sigs:
                if sig.symbol == pos.symbol:
                    pf.update_price(pos.id, sig.price)
                    break

            # 止损检查
            if pos.pnl_pct <= -8.0:
                pf.sell(pos.id, pos.current_price, f"硬止损触发: {pos.pnl_pct:.1f}%")
                results.append(f"🔴 止损 {pos.symbol}: {pos.pnl_pct:.1f}%")

            # 止盈检查
            elif pos.pnl_pct >= 15.0:
                pf.sell(pos.id, pos.current_price, f"止盈触发: +{pos.pnl_pct:.1f}%")
                results.append(f"🟢 止盈 {pos.symbol}: +{pos.pnl_pct:.1f}%")

        # ── 第二步: 寻找新的入场机会 ──
        for sig in buy_sigs[:3]:  # Top 3 BUY signals
            # 跳过已持仓标的
            held_symbols = [p.symbol for p in pf.open_positions if p.market == market]
            if sig.symbol in held_symbols:
                continue

            # 仓位大小
            cash = pf.cash.get(market, 0)
            max_size = min(sig.suggested_size, cash * 0.5)  # 不超过市剩余50%

            if max_size < 200:  # 最低 ¥200
                continue

            # 风控检查
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

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", nargs="+", default=["crypto"],
                       choices=["crypto", "a_stock", "us_stock", "all"])
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    if args.report:
        print(generate_status_report())
        sys.exit(0)

    markets = ["crypto", "a_stock", "us_stock"] if args.markets == ["all"] else args.markets
    results, pf = auto_scan_and_trade(markets)

    print(generate_status_report())

    if results:
        print("\n📋 本次操作:")
        for r in results:
            print(f"  {r}")
    else:
        print("\n💤 本次扫描无操作")
