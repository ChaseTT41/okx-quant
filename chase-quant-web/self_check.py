#!/usr/bin/env python3
"""
Yina 量化启动自检脚本 🐾
每次"打开量化"后必须运行，确保五市场全部畅通

检查项:
  1. 五市场现金分配
  2. 各市场交易时段
  3. 信号生成 (每个市场是否有BUY信号)
  4. 风控拦截检查 (每市场独立持仓上限)
  5. 持仓分布概览
"""

import sys, os, warnings, json
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_DYNAMIC'] = 'FALSE'
# 抑制 yfinance/tqdm 进度条
os.environ['TQDM_DISABLE'] = '1'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

from signals import SignalEngine
from auto_trade import is_market_open, PortfolioManager, RiskController


MARKET_INFO = {
    "crypto":   {"label": "₿ 加密货币",  "icon": "🟠", "always_open": True},
    "a_stock":  {"label": "🇨🇳 A股",    "icon": "🔴", "always_open": False},
    "us_stock": {"label": "🇺🇸 美股",    "icon": "🔵", "always_open": False},
    "hk_stock": {"label": "🇭🇰 港股",    "icon": "🟣", "always_open": False},
    "b_stock":  {"label": "🏦 bStocks", "icon": "🟢", "always_open": True},
}


def header(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def check_mark():
    """Step 1: 检查各市场交易时段"""
    header("📡 Step 1: 市场交易时段")
    for market, info in MARKET_INFO.items():
        is_open = is_market_open(market)
        if info["always_open"]:
            status = "✅ 24/7 永续"
        elif is_open:
            status = "✅ 交易中"
        else:
            status = "⏸️  休市 (仅更新估值)"
        print(f"  {info['icon']} {info['label']:12s} → {status}")


def check_cash():
    """Step 2: 检查各市场资金分配"""
    header("💰 Step 2: 资金分配")
    pf = PortfolioManager()
    total = pf.total_value

    issues = []
    for market, info in MARKET_INFO.items():
        cash = pf.cash.get(market, 0)
        positions_value = sum(
            p.value for p in pf.open_positions if p.market == market
        )
        market_total = cash + positions_value
        pct = market_total / total * 100 if total > 0 else 0

        pos_count = len([p for p in pf.open_positions if p.market == market])

        bar = "█" * int(pct / 2) + "░" * (20 - int(pct / 2))
        print(f"  {info['icon']} {info['label']:12s} {bar} {pct:5.1f}%  "
              f"现金=¥{cash:.0f}  持仓={pos_count}只  市值=¥{market_total:.0f}")

        if cash < 200 and pos_count == 0:
            issues.append(f"{info['label']} 现金不足 ¥200，无法开仓")

    return pf, issues


def check_signals():
    """Step 3: 信号扫描 — 每个市场是否能生成买入信号"""
    header("📊 Step 3: 各市场信号扫描")
    engine = SignalEngine()

    scanners = {
        "crypto": engine.crypto,
        "a_stock": engine.a_stock,
        "us_stock": engine.us_stock,
        "hk_stock": engine.hk_stock,
        "b_stock": engine.b_stock,
    }

    all_results = {}
    warnings_list = []

    for market, scanner in scanners.items():
        info = MARKET_INFO[market]
        try:
            sigs = scanner.scan()
            buy_sigs = [s for s in sigs if s.action == "BUY"]
            buy_high = [s for s in buy_sigs if s.score >= 50]
            all_results[market] = {"total": len(sigs), "buy": len(buy_sigs), "buy_high": len(buy_high)}

            if buy_high:
                top3 = sorted(buy_high, key=lambda s: s.score, reverse=True)[:3]
                print(f"  {info['icon']} {info['label']:12s} ✅ {len(buy_high)} 个买入信号 (score≥50)")
                for s in top3:
                    print(f"       {s.symbol:10s} 评分{s.score:.0f}  ¥{s.price:.2f}  {s.reasons[0] if s.reasons else ''}")
            elif buy_sigs:
                print(f"  {info['icon']} {info['label']:12s} ⚠️  {len(buy_sigs)} 个弱信号 (score<50)")
                for s in buy_sigs[:2]:
                    print(f"       {s.symbol:10s} 评分{s.score:.0f}  需等待更佳时机")
            else:
                print(f"  {info['icon']} {info['label']:12s} ⚪ 无买入信号")
                if is_market_open(market):
                    warnings_list.append(f"{info['label']} 交易时段内无买入信号")
        except Exception as e:
            print(f"  {info['icon']} {info['label']:12s} ❌ 扫描失败: {e}")
            all_results[market] = {"total": 0, "buy": 0, "buy_high": 0, "error": str(e)}
            warnings_list.append(f"{info['label']} 信号扫描异常!")

    return all_results, warnings_list


def check_risk_per_market(pf):
    """Step 4: 风控检查 — 每个市场独立验证不会被全局持仓阻塞"""
    header("🛡️  Step 4: 风控验证 (每市场独立持仓上限)")

    rc = RiskController(pf)
    total_value = pf.total_value

    for market, info in MARKET_INFO.items():
        current_positions = len([p for p in pf.open_positions if p.market == market])
        cash = pf.cash.get(market, 0)

        # 用小额测试 (取可用现金的50%和200的较小值)
        test_amount = min(200, cash * 0.5) if cash > 50 else cash
        if test_amount < 50:
            # 现金太少但已有持仓 → 说明资金已部署，是健康状态
            if current_positions > 0:
                print(f"  {info['icon']} {info['label']:12s} ✅ 资金已部署 (持仓{current_positions}只, 剩余¥{cash:.0f})")
            else:
                print(f"  {info['icon']} {info['label']:12s} ⚠️  现金不足且无持仓 (¥{cash:.0f})")
            continue

        test_score = 70
        check = rc.pre_trade_check(market, test_amount, test_score, total_value)

        if check.passed:
            print(f"  {info['icon']} {info['label']:12s} ✅ 风控通过 (当前{current_positions}只持仓, 可开仓)")
        else:
            # 区分：全局持仓阻塞 vs 其他原因
            if "达上限" in check.reason and current_positions < 5:
                print(f"  {info['icon']} {info['label']:12s} ❌ 风控拦截: {check.reason} (BUG:非本市场)")
            elif "达上限" in check.reason:
                print(f"  {info['icon']} {info['label']:12s} ⚠️  {check.reason} (本市场已满)")
            else:
                print(f"  {info['icon']} {info['label']:12s} ⚠️  {check.reason}")


def generate_summary(pf, signal_results, warnings_list):
    """Step 5: 最终总结"""
    header("📋 总结")

    total_positions = len(pf.open_positions)
    markets_with_positions = set(p.market for p in pf.open_positions)
    markets_without = [m for m in MARKET_INFO if m not in markets_with_positions]

    print(f"  总资产: ¥{pf.total_value:.2f}")
    print(f"  总PnL: {pf.total_pnl_pct:+.2f}%")
    print(f"  总持仓: {total_positions} 只 (分布在 {len(markets_with_positions)} 个市场)")

    # 各市场信号汇总
    print(f"\n  📊 信号汇总:")
    for market, info in MARKET_INFO.items():
        r = signal_results.get(market, {})
        error = r.get("error", "")
        if error:
            print(f"     {info['icon']} {info['label']:12s} ❌ 错误: {error}")
        else:
            print(f"     {info['icon']} {info['label']:12s} "
                  f"总信号{r.get('total',0)} | BUY={r.get('buy',0)} | "
                  f"高分BUY={r.get('buy_high',0)}")

    # 无持仓市场提醒
    if markets_without:
        print(f"\n  ⚠️  无持仓市场: {', '.join(MARKET_INFO[m]['label'] for m in markets_without)}")

    # 警告汇总
    if warnings_list:
        print(f"\n  ⚠️  注意事项:")
        for w in warnings_list:
            print(f"     • {w}")
    else:
        print(f"\n  ✅ 四市场全部正常！")

    # 健康度打分
    score = 100
    if warnings_list:
        score -= len(warnings_list) * 10
    if len(markets_without) > 0:
        score -= len(markets_without) * 5
    health = "🟢" if score >= 80 else "🟡" if score >= 50 else "🔴"
    print(f"\n  {health} 健康度: {score}/100")

    return score


def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║     🐾 Yina 量化启动自检 — 五市场全扫描            ║")
    print("╚══════════════════════════════════════════════════════╝")

    # Step 1
    check_mark()

    # Step 2
    pf, cash_issues = check_cash()

    # Step 3
    signal_results, signal_warnings = check_signals()

    # Step 4
    check_risk_per_market(pf)

    # Step 5
    all_warnings = cash_issues + signal_warnings
    health = generate_summary(pf, signal_results, all_warnings)

    print(f"\n{'='*55}")
    if health >= 80:
        print("  🎉 量化系统一切正常，Chase哥放心！")
    elif health >= 50:
        print("  ⚠️  有轻微问题，但核心功能正常~")
    else:
        print("  🚨 需要关注！请检查上面标红的项目")

    return 0 if health >= 80 else 1


if __name__ == "__main__":
    sys.exit(main())
