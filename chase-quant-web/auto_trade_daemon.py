"""
Yina 自主交易守护进程 🐾
==============================
持续运行: 扫单 → 决策 → 执行 → 企微通知
每10分钟扫描一次 (加密永不收市), 有交易时推送企微简报, 每日22:00发送日报

使用:
  python3 auto_trade_daemon.py              # 前台运行
  python3 auto_trade_daemon.py --daemon     # 后台运行
  python3 auto_trade_daemon.py --once       # 单次运行+推送
"""
from __future__ import annotations
import sys
import os
os.environ.setdefault('TQDM_DISABLE', '1')  # 关闭akshare进度条, 保持日志清爽
import json
import time
import signal
import traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

# ── 配置 ──
SCAN_INTERVAL_MINUTES = 10  # 扫描间隔 (加密永不休市, 10分钟捕捉机会)
DATA_DIR = Path(__file__).parent / "data"
LOG_DIR = DATA_DIR / "daemon_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = DATA_DIR / "daemon_state.json"

# 夜间休眠: 北京时间 01:00-07:00 暂停扫描 (加密市场仍开盘但波动低)
SLEEP_START_HOUR = 4  # 美股04:00收盘后才休眠
SLEEP_END_HOUR = 7
BEIJING_TZ = timezone(timedelta(hours=8))


def beijing_now() -> datetime:
    return datetime.now(BEIJING_TZ)


def is_sleep_time() -> bool:
    """夜间休眠时段"""
    h = beijing_now().hour
    return SLEEP_START_HOUR <= h < SLEEP_END_HOUR


def log(msg: str):
    """带时间戳的日志"""
    now_str = beijing_now().strftime("%m-%d %H:%M:%S")
    line = f"[{now_str}] {msg}"
    print(line, flush=True)
    # 写入日志文件
    log_file = LOG_DIR / f"daemon_{beijing_now().strftime('%Y%m%d')}.log"
    try:
        with open(log_file, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def push_wechat(title: str, content: str) -> bool:
    """推送消息到企业微信「金融监控」群"""
    try:
        # 加载 webhook key
        webhook_key = os.environ.get("WECHAT_WEBHOOK_KEY", "")
        if not webhook_key:
            env_file = Path(__file__).parent / ".env"
            if env_file.exists():
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("WECHAT_WEBHOOK_KEY="):
                            webhook_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break

        if not webhook_key:
            log("⚠️ 未配置 WECHAT_WEBHOOK_KEY, 跳过推送")
            return False

        import urllib.request
        url = f"https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={webhook_key}"

        markdown_content = f"## {title}\n{content}"
        payload = json.dumps({
            "msgtype": "markdown",
            "markdown": {"content": markdown_content}
        }).encode("utf-8")

        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())
            if result.get("errcode") == 0:
                log("✅ 企微推送成功")
                return True
            else:
                log(f"⚠️ 企微推送失败: {result}")
                return False
    except Exception as e:
        log(f"⚠️ 企微推送异常: {e}")
        return False


def load_state() -> dict:
    """加载守护进程状态"""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"cycles": 0, "total_trades": 0, "started_at": beijing_now().isoformat(),
            "last_push_day": "", "trade_history": []}


def save_state(state: dict):
    """保存守护进程状态"""
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log(f"⚠️ 保存状态失败: {e}")


def build_trade_summary(results: list, pf, scan_time: str) -> str:
    """构建交易摘要 (Markdown)"""
    lines = [f"> 📅 {scan_time}", ""]

    # 本次操作
    if results:
        lines.append("**📋 本次操作:**")
        for r in results[:10]:
            lines.append(f"> {r}")
    else:
        lines.append("> 💤 本次扫描无操作")

    lines.append("")

    # 持仓状态
    open_pos = pf.open_positions
    if open_pos:
        lines.append("**📦 当前持仓:**")
        for p in open_pos[:8]:
            emoji = "🟢" if p.pnl_pct >= 0 else "🔴"
            lines.append(f"> {emoji} {p.symbol} | ¥{p.entry_price:,.0f}→¥{p.current_price:,.0f} | {p.pnl_pct:+.2f}%")
    else:
        lines.append("> 📦 无持仓")

    lines.append("")
    lines.append(f"> 💰 总资产: ¥{pf.total_value:,.2f} | 总盈亏: {pf.total_pnl_pct:+.2f}%")
    lines.append(f"> 🎯 月度目标进度: {pf.total_pnl_pct:+.1f}%/30%")

    return "\n".join(lines)


def build_daily_report(state: dict, pf) -> str:
    """构建日报"""
    lines = [
        f"> 📅 {beijing_now().strftime('%Y-%m-%d')} 交易日报",
        "",
        "**📊 今日统计:**",
        f"> 扫描周期: {state.get('today_cycles', state.get('cycles', 0))} 次",
        f"> 交易操作: {state.get('today_trades', 0)} 笔",
        "",
    ]

    # 风险检查
    alerts = pf.check_risk()
    if alerts:
        lines.append("**🚨 风控告警:**")
        for a in alerts:
            lines.append(f"> ⚠️ {a}")
    else:
        lines.append("> ✅ 风控正常")

    lines.append("")
    lines.append("---")
    lines.append(f"> 🐾 Yina自主交易 · {beijing_now().strftime('%Y-%m-%d %H:%M')}")

    return "\n".join(lines)


# ── 多策略并行配置 ──
STRATEGY_CONFIG = {
    "ml_momentum": True,           # ML融合动量策略
    "mean_reversion_grid": True,   # 均值回归网格策略
    "cross_market_alpha": True,    # 跨市场Alpha套利
    "aggressive": False,           # 激进交易 (默认关闭，波动大)
}


def execute_strategy_signals(signals: list, pf) -> list:
    """
    执行策略信号 (仅BUY, SELL记录日志不自动平仓)

    规则:
      - 只执行 BUY 信号
      - 已有持仓的 symbol 跳过
      - 每策略最多执行 1 个
      - 风控检查 + 最低 ¥200
      - entry_reason 标注策略名
    """
    from risk import RiskController
    rc = RiskController(pf)
    total_value = pf.total_value
    results = []

    held_symbols = [p.symbol for p in pf.open_positions if p.market == "crypto"]

    # 🛡️ Fix #2: 最大持仓上限 (小资金分散过度 → 集中火力)
    MAX_CRYPTO_POSITIONS = 6
    if len(held_symbols) >= MAX_CRYPTO_POSITIONS:
        log(f"  🛑 加密持仓已达上限 {MAX_CRYPTO_POSITIONS}，跳过新信号")
        return results

    # 按策略分组，每策略选最强的 BUY
    buy_by_strategy = {}
    for s in signals:
        if s["action"] != "BUY":
            continue
        if s["symbol"] in held_symbols:
            continue
        strat = s.get("strategy_name", "未知策略")
        if strat not in buy_by_strategy or s["score"] > buy_by_strategy[strat]["score"]:
            buy_by_strategy[strat] = s

    for strat_name, sig in buy_by_strategy.items():
        try:
            cash = pf.cash.get("crypto", 0)
            size_pct = sig.get("suggested_size", 0.05)
            if isinstance(size_pct, float) and size_pct > 1:
                size_pct = size_pct / 100
            max_size = min(cash * size_pct, cash * 0.3)

            if max_size < 200:
                log(f"  💤 [{strat_name}] {sig['symbol']} 资金不足 (可用{cash:.0f}, 需≥200)")
                continue

            score = sig.get("score", 50)
            check = rc.pre_trade_check("crypto", max_size, score, total_value)
            if not check.passed:
                log(f"  ⚠️ [{strat_name}] {sig['symbol']} 风控拦截: {check.reason}")
                continue

            quantity = max_size / sig["price"]
            reason = f"[{strat_name}] " + " | ".join(sig.get("reasons", ["信号触发"])[:3])

            trade = pf.buy(
                market="crypto", symbol=sig["symbol"],
                name=sig.get("name", sig["symbol"]),
                price=sig["price"], quantity=quantity,
                reason=reason,
            )
            if trade:
                result_msg = f"🧠 [{strat_name}] BUY {sig['symbol']} ¥{max_size:.0f} | 评分{sig['score']:.0f} | 置信{sig['confidence']:.0%}"
                results.append(result_msg)
                log(f"  ✅ {result_msg}")
                held_symbols.append(sig["symbol"])  # 防止同轮重复买入
        except Exception as e:
            log(f"  ❌ [{strat_name}] 执行失败: {e}")

    # 记录 SELL 信号到日志
    sell_sigs = [s for s in signals if s["action"] == "SELL"]
    if sell_sigs:
        log(f"  📋 {len(sell_sigs)} 条SELL信号 (仅记录，出场走全局止损/止盈):")
        for s in sell_sigs[:5]:
            log(f"     🔴 [{s.get('strategy_name', '?')}] {s['symbol']} | 评分{s['score']:.0f} | 置信{s['confidence']:.0%}")

    return results


def _run_leverage_decisions(signals: list, sentiment_engine, leverage_engine, pf, log_func) -> list:
    """对合约标的运行杠杆决策 (实盘模式)"""
    from symbol_config import OKX_ALL_NON_CRYPTO_SWAPS, OKX_AI_CONCEPT_SWAPS

    swap_symbols = set(OKX_ALL_NON_CRYPTO_SWAPS + OKX_AI_CONCEPT_SWAPS)
    # 加密合约: Tier1+2 的 swap 版本
    from symbol_config import TIER1_ML_HEAVY, TIER2_TECHNICAL_LIGHT
    for s in TIER1_ML_HEAVY + TIER2_TECHNICAL_LIGHT:
        swap_symbols.add(s)

    results = []

    # 🛡️ 获取 OKX 现有持仓，防止重复开仓
    existing_positions = leverage_engine.fetch_open_positions()
    existing_list = list(existing_positions) if isinstance(existing_positions, set) else existing_positions

    # 🛡️ Fix #2: 合约最大持仓上限 (小资金集中火力)
    MAX_SWAP_POSITIONS = 6
    if len(existing_positions) >= MAX_SWAP_POSITIONS:
        log_func(f"  🛑 合约持仓已达上限 {MAX_SWAP_POSITIONS}个，跳过新信号。当前持仓: {', '.join(existing_list[:8])}")
        return results
    for sig in signals:
        sym = sig.get("symbol", "")
        if sym not in swap_symbols:
            continue  # 非合约标的, 走现货流程
        if sig.get("action") != "BUY":
            continue
        if sym in existing_positions:
            log_func(f"  ⚡ 跳过 {sym}: 已有持仓")
            continue

        # 拉取情绪叠加
        overlay = None
        if sentiment_engine:
            try:
                overlay = sentiment_engine.get_sentiment_overlay(sym)
            except Exception:
                pass

        # 拉取资金费率
        funding_rate = 0.0
        if sentiment_engine:
            fr_data = sentiment_engine.fetch_funding_rates([sym], use_cache=True)
            if sym in fr_data:
                funding_rate = fr_data[sym].funding_rate

        decision = leverage_engine.determine_leverage(
            confidence=sig.get("confidence", 0.5),
            signal_weighted=sig.get("signal_val", sig.get("signal_weighted", 0)),
            symbol=sym,
            strategy_name=sig.get("strategy_name", sig.get("strategy", "unknown")),
            funding_rate=funding_rate,
            sentiment_overlay=overlay,
        )

        if decision.skip_reason:
            log_func(f"  ⚡ 跳过 {sym}: {decision.skip_reason}")
            continue

        # 仓位计算 — 使用真实 OKX 余额 (非模拟盘)
        total_equity = leverage_engine.fetch_equity()
        if total_equity <= 0:
            log_func(f"  ⚡ 跳过 {sym}: 无法获取余额")
            continue
        price = sig.get("price", 100.0)
        pos = leverage_engine.calculate_position(total_equity, price, decision)

        log_func(
            f"  ⚡ {sym}: {decision.recommended_leverage}x | "
            f"WR={decision.blended_win_rate:.0%} | "
            f"保证金=${pos['margin_usdt']} | "
            f"止损={decision.stop_loss_pct:+.1%} | "
            f"止盈={decision.take_profit_pct:+.0%}"
        )

        # 执行合约单
        try:
            order = leverage_engine.create_swap_market_order(
                symbol=sym,
                side="buy",
                quantity_contracts=pos["quantity_contracts"],
                leverage=decision.recommended_leverage,
                stop_loss_price=pos["stop_loss_price"],
                take_profit_price=pos["take_profit_price"],
                note=f"{decision.recommended_leverage}x_{sym.split('/')[0][:8]}",
            )
            if order:
                results.append({
                    "symbol": sym, "action": "BUY_SWAP",
                    "leverage": decision.recommended_leverage,
                    "margin": pos["margin_usdt"],
                    "notional": pos["notional_usdt"],
                    "order_id": order.get("id", ""),
                })
                leverage_engine.increment_daily_trades()
        except Exception as e:
            log_func(f"  ❌ 合约下单失败 {sym}: {e}")

    return results


def run_trade_cycle(state: dict, trading_mode=None) -> dict:
    """执行一次交易周期

    Args:
        state: 守护进程状态
        trading_mode: TradingMode (Phase 15), None=paper
    """
    from portfolio import PortfolioManager
    from execution import ExecutionConfig
    from trading_config import TradingMode

    if trading_mode is None:
        trading_mode = TradingMode.PAPER

    # 构建执行配置: SMART 自适应策略
    exec_cfg = ExecutionConfig(
        strategy="smart",
        horizon_minutes=60,
        n_slices=0,  # 自动最优
        urgency=0.5,
    )

    # ── 🧠 AI复盘策略叠加 ──
    review_overlay = None
    try:
        from review_strategy_bridge import load_overlay
        review_overlay = load_overlay()
        if review_overlay and not review_overlay.is_stale:
            log(f"🧠 AI复盘叠加: 置信乘数={review_overlay.confidence_multiplier:.2f} | "
                f"偏好={review_overlay.favor_assets[:3]} | 回避={review_overlay.avoid_assets[:3]}")
        elif review_overlay:
            log(f"⚠️ AI复盘数据陈旧(>{review_overlay.ttl_hours}h)，使用默认参数")
    except Exception as e:
        log(f"⚠️ AI复盘桥接不可用: {e}")

    # ── 🎭 市场情绪引擎 ──
    sentiment_engine = None
    leverage_engine = None
    try:
        from market_sentiment import MarketSentimentEngine
        sentiment_engine = MarketSentimentEngine()
        snapshot = sentiment_engine.refresh_all(force=False)
        fg = snapshot.fear_greed
        log(f"🎭 情绪: F&G={fg.current_value} ({fg.classification})")
        if snapshot.sector_flows:
            top = max(snapshot.sector_flows.values(), key=lambda s: s.flow_score)
            log(f"📊 资金流: {top.sector_name} 领先 (flow={top.flow_score:+.2f})")
        if snapshot.rotation_prediction:
            r = snapshot.rotation_prediction
            log(f"🔮 轮动: {r.current_leader} → {r.next_leader} (conf={r.rotation_confidence:.0%})")

        # 🎭 Fix #3: 注入情绪数据到 feature_engine (消除7个占位符)
        try:
            from feature_engine import set_sentiment_context
            fg = snapshot.fear_greed
            if fg:
                # F&G 当前值 + 历史
                fg_ctx = {
                    "fg_value": fg.current_value,
                    "fg_prev_1d": fg.value_1d_ago or fg.current_value,
                    "fg_prev_7d": fg.value_7d_ago or fg.current_value,
                    # 无3d/5d/14d历史, 用线性插值估计
                    "fg_prev_3d": int(fg.current_value + (fg.value_7d_ago - fg.current_value) * 4/7) if fg.value_7d_ago else fg.current_value,
                    "fg_prev_5d": int(fg.current_value + (fg.value_7d_ago - fg.current_value) * 2/7) if fg.value_7d_ago else fg.current_value,
                    "fg_prev_14d": max(0, min(100, fg.current_value * 2 - (fg.value_7d_ago or fg.current_value))),
                }
                # 资金费率 (从缓存读取)
                fr_cache = sentiment_engine._read_cache("funding_rates") or {}
                funding_rates = {}
                for sym, data in fr_cache.items():
                    if isinstance(data, dict):
                        funding_rates[sym] = data.get("funding_rate", 0.0)
                fg_ctx["funding_rates"] = funding_rates
                set_sentiment_context(**fg_ctx)
        except Exception:
            pass

        # ⚡ 杠杆引擎 (实盘时才启用)
        if trading_mode.is_real:
            from leverage_engine import LeverageEngine
            leverage_engine = LeverageEngine(sentiment_engine=sentiment_engine)
            log(f"⚡ 杠杆引擎就绪: 全局胜率={leverage_engine.get_global_win_rate():.0%}")
    except Exception as e:
        log(f"⚠️ 情绪引擎不可用: {e}")

    try:
        from auto_trade import RollingAwareAutoTrader, ROLLING_AVAILABLE

        if ROLLING_AVAILABLE:
            trader = RollingAwareAutoTrader(
                use_v5=True,
                use_rolling=True,
                auto_retrain=False,
                use_graph=True,
                use_alphas=True,
                execution_config=exec_cfg,
                trading_mode=trading_mode,
                sentiment_engine=sentiment_engine,  # 🎭
            )
            log(f"🚀 引擎: {trader.engine_label}")
            results, pf = trader.run(review_overlay=review_overlay)
        else:
            from auto_trade import MLAutoTrader, ML_SIGNAL_AVAILABLE
            if ML_SIGNAL_AVAILABLE:
                trader = MLAutoTrader(use_lgbm=True)
                from auto_trade import auto_scan_and_trade
                results, pf = auto_scan_and_trade(["crypto"], use_ml=True, use_rolling=False)
            else:
                from auto_trade import auto_scan_and_trade
                results, pf = auto_scan_and_trade(["crypto"], use_ml=False, use_rolling=False)

        # ── 🆕 多策略并行引擎 (strategies.py + aggressive_trader) ──
        try:
            from strategy_runner import run_all_strategies, STRATEGY_CONFIG as ST_CFG
            enabled = [k for k, v in ST_CFG.items() if v]
            if enabled:
                log(f"🧠 多策略引擎: {len(enabled)}条策略线 ({', '.join(enabled)})")
                strategy_signals = run_all_strategies(
                    sentiment_engine=sentiment_engine  # 🎭 传入情绪
                )
                if strategy_signals:
                    strat_results = execute_strategy_signals(strategy_signals, pf)
                    if strat_results:
                        results.extend(strat_results)
                        log(f"🧠 多策略引擎: {len(strat_results)} 笔执行")

                    # ── ⚡ 杠杆决策 (实盘 + 合约标的) ──
                    if leverage_engine and trading_mode.is_real:
                        swap_results = _run_leverage_decisions(
                            strategy_signals, sentiment_engine,
                            leverage_engine, pf, log
                        )
                        if swap_results:
                            results.extend(swap_results)
                else:
                    log("💤 多策略引擎: 无信号")
        except Exception as e:
            log(f"⚠️ 多策略引擎异常: {e}")

        # ── 🏭 股票合约扫描 (半导体/AI软件/太空等, ML策略不覆盖) ──
        if leverage_engine and trading_mode.is_real:
            try:
                from stock_swap_scanner import scan_stock_swaps
                existing_swap_positions = set()
                try:
                    okx_pos = leverage_engine.fetch_open_positions()
                    # fetch_open_positions 返回 set，直接使用
                    existing_swap_positions = okx_pos if isinstance(okx_pos, set) else set(okx_pos.keys() if hasattr(okx_pos, 'keys') else okx_pos)
                except Exception:
                    pass

                # 每3个周期全量扫描一次扩展列表
                force_full = (state.get('cycles', 0) % 3 == 0)
                stock_signals = scan_stock_swaps(
                    sentiment_engine=sentiment_engine,
                    existing_positions=existing_swap_positions,
                    force_full=force_full,
                )
                if stock_signals:
                    log(f"🏭 股票合约扫描: {len(stock_signals)} 条信号")
                    for ss in stock_signals:
                        log(f"  📈 [{ss['strategy_name']}] {ss['name']} | "
                            f"评分{ss['score']:.0f} | 置信{ss['confidence']:.1%} | "
                            f"{', '.join(ss['reasons'][:2])}")
                    stock_results = _run_leverage_decisions(
                        stock_signals, sentiment_engine,
                        leverage_engine, pf, log
                    )
                    if stock_results:
                        results.extend(stock_results)
                        log(f"🏭 股票合约: {len(stock_results)} 笔执行")
            except Exception as e:
                log(f"⚠️ 股票合约扫描异常: {e}")

        # ── A股 + 美股传统信号扫描 (ML引擎不覆盖) ──
        try:
            from auto_trade import auto_scan_and_trade
            trad_results, pf = auto_scan_and_trade(
                ["a_stock", "us_stock", "hk_stock", "b_stock"], use_ml=False, use_rolling=False
            )
            if trad_results:
                results.extend(trad_results)
                log(f"📊 A股/美股/港股/bStocks: {len(trad_results)} 笔信号")
        except Exception as e:
            log(f"⚠️ A股/美股/港股扫描异常: {e}")

        return {"ok": True, "results": results, "pf": pf}

    except Exception as e:
        log(f"❌ 交易周期异常: {e}")
        traceback.print_exc()
        return {"ok": False, "error": str(e), "results": [], "pf": None}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Yina 自主交易守护进程 🐾")
    parser.add_argument("--daemon", action="store_true", help="后台持续运行")
    parser.add_argument("--once", action="store_true", help="单次运行+推送")
    parser.add_argument("--testnet", action="store_true", help="交易所测试网/Demo模式 (Phase 15)")
    parser.add_argument("--live", action="store_true", help="交易所实盘模式 (Phase 15, 真金白银!)")
    args = parser.parse_args()

    # 确定交易模式 (Phase 15)
    from trading_config import TradingConfig, TradingMode
    config = TradingConfig.from_env()
    if args.live:
        trading_mode = TradingMode.LIVE
        mode_label = f"🔴 {config.exchange.label} 实盘"
    elif args.testnet:
        trading_mode = TradingMode.TESTNET
        mode_label = f"🧪 {config.exchange.label} 测试网/Demo"
    else:
        trading_mode = TradingMode.PAPER
        mode_label = "📝 模拟盘"

    log(f"🐾 Yina 自主交易守护进程 启动! [{mode_label}]")
    log(f"📡 API: http://localhost:8766")
    log(f"⏱️  扫描间隔: {SCAN_INTERVAL_MINUTES}分钟")
    log(f"🌙 夜间休眠: {SLEEP_START_HOUR:02d}:00-{SLEEP_END_HOUR:02d}:00 (北京时间)")

    # 实盘安全检查
    if trading_mode.is_real:
        issues = config.validate()
        if issues:
            log("🚨 实盘配置问题:")
            for i in issues:
                log(f"  - {i}")
            log("🛑 退出 (安全问题)")
            sys.exit(1)
        if config.is_kill_switch_active():
            log("🚨 紧急停止开关已激活! 退出")
            sys.exit(1)
        log(f"✅ 实盘配置检查通过")
        log(f"💰 最低余额保护: ${config.min_balance_usdt:.0f}")
        log(f"📊 每日交易上限: {config.max_daily_trades} 笔 (信号驱动)")
        log(f"🛡️ 最大回撤限制: {config.max_drawdown_pct:.0%}")

    state = load_state()
    log(f"📊 历史统计: {state['cycles']} 次扫描, {state['total_trades']} 笔交易")

    # 重置今日统计
    today_str = beijing_now().strftime("%Y%m%d")
    if state.get("last_push_day") != today_str:
        state["today_cycles"] = 0
        state["today_trades"] = 0
        state["last_push_day"] = today_str

    # 启动通知
    push_wechat(
        f"🐾 Yina自主交易已启动 [{mode_label}]",
        f"> 🚀 交易引擎已就绪\n"
        f"> 📡 [查看仪表板](https://runs-student-skill-seeds.trycloudflare.com)\n"
        f"> ⏱️ 扫描间隔: {SCAN_INTERVAL_MINUTES}分钟\n"
        f"> 🧠 引擎: v5 Qlib融合 + 图增强 + Alpha + SMART拆单\n"
        f"> 🎮 模式: {mode_label}\n"
        f"> ⏰ {beijing_now().strftime('%Y-%m-%d %H:%M')}"
    )

    def run_cycle():
        nonlocal state
        cycle_start = time.time()
        state["cycles"] += 1
        state["today_cycles"] = state.get("today_cycles", 0) + 1

        log(f"\n{'='*50}")
        log(f"🔄 第 {state['cycles']} 次扫描 [{beijing_now().strftime('%H:%M')}]")
        log(f"{'='*50}")

        result = run_trade_cycle(state, trading_mode)

        if result["ok"]:
            pf = result["pf"]
            results = result["results"]

            n_trades = len(results)
            if n_trades > 0:
                state["total_trades"] += n_trades
                state["today_trades"] = state.get("today_trades", 0) + n_trades

            # 有交易时推送
            if n_trades > 0:
                log(f"📋 {n_trades} 笔操作, 推送企微...")
                summary = build_trade_summary(results, pf, beijing_now().strftime("%Y-%m-%d %H:%M"))
                push_wechat("📊 Yina交易通知", summary)
            else:
                log("💤 本次无操作")

            # 每日晚报 (22:00-22:30 之间, 且今天还没发过)
            now = beijing_now()
            if 22 <= now.hour < 23 and state.get("last_daily_push_day") != today_str:
                log("🌙 发送每日晚报...")
                report = build_daily_report(state, pf)
                push_wechat("🌙 Yina日报", report)
                state["last_daily_push_day"] = today_str

            save_state(state)
        else:
            log(f"⚠️ 周期失败: {result.get('error', 'unknown')}")

        elapsed = time.time() - cycle_start
        log(f"⏱️  耗时: {elapsed:.1f}s")

    # ── 单次模式 ──
    if args.once:
        run_cycle()
        log("✅ 单次运行完成")
        return

    # ── 守护进程模式 ──
    def handle_signal(sig, frame):
        log(f"\n🛑 收到信号 {sig}, 优雅退出...")
        state["stopped_at"] = beijing_now().isoformat()
        save_state(state)
        push_wechat("🛑 Yina交易已停止",
                     f"> ⏰ {beijing_now().strftime('%Y-%m-%d %H:%M')}\n"
                     f"> 📊 共 {state['cycles']} 次扫描, {state['total_trades']} 笔交易")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    log("🔄 进入持续交易模式...")
    consecutive_errors = 0

    while True:
        try:
            if is_sleep_time():
                now = beijing_now()
                wake_hour = SLEEP_END_HOUR
                sleep_sec = (timedelta(hours=wake_hour) -
                            timedelta(hours=now.hour, minutes=now.minute, seconds=now.second)).total_seconds()
                if sleep_sec < 0:
                    sleep_sec += 86400
                log(f"🌙 夜间休眠中... ({SLEEP_START_HOUR:02d}:00-{SLEEP_END_HOUR:02d}:00), "
                    f"{sleep_sec/60:.0f}分钟后唤醒")
                time.sleep(min(sleep_sec, 3600))  # 最多睡1小时再检查
                continue

            run_cycle()
            consecutive_errors = 0

            # 等待下一次扫描
            wait = SCAN_INTERVAL_MINUTES * 60
            log(f"⏳ 等待 {SCAN_INTERVAL_MINUTES} 分钟后下次扫描...\n")
            time.sleep(wait)

        except KeyboardInterrupt:
            handle_signal(signal.SIGINT, None)
        except Exception as e:
            consecutive_errors += 1
            log(f"❌ 主循环异常: {e}")
            traceback.print_exc()
            if consecutive_errors >= 5:
                log("🚨 连续5次异常, 发送告警后退出...")
                push_wechat("🚨 Yina交易异常退出",
                             f"> 连续 {consecutive_errors} 次异常\n"
                             f"> 最后错误: {e}\n"
                             f"> ⏰ {beijing_now().strftime('%Y-%m-%d %H:%M')}")
                sys.exit(1)
            # 后退等待
            backoff = min(60 * consecutive_errors, 600)
            log(f"🔙 后退 {backoff}s 后重试...")
            time.sleep(backoff)


if __name__ == "__main__":
    main()
