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
SCAN_INTERVAL_MINUTES = 5  # 扫描间隔 (加密永不休市, 5分钟捕捉机会)
PUSH_INTERVAL_MINUTES = 30  # 🆕 企微推送间隔 (30分钟, 避免刷屏)
MIN_MARGIN_USDT = 20.0      # 🚨 硬性最低保证金: <$20不玩，撒胡椒面没有意义
STOPPED_OUT_COOLDOWN_CYCLES = 3  # 🆕 止损后冷却周期数 (防止反复追同一标的)
_stopped_out_this_cycle: set = set()  # 🆕 本周期被止损/强平的币种 (用于冷却追踪)
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
    "naked_k": True,               # 🕯️ 裸K价格行为策略 (优先)
}


def _enforce_hard_stop_loss(pf, log_func) -> list:
    """
    🛡️ 硬止损强制执行 + 碎片化清理 — 所有持仓通用

    规则:
      - 浮动亏损 >= 8% → 立即平仓
      - 持有 > 7天且亏损 → 僵尸仓清理
      - 同币种 > 3笔持仓 → 只保留最好的3笔
    """
    from datetime import datetime, timezone, timedelta
    from collections import Counter
    results = []
    now = datetime.now(timezone.utc)

    # ── 🆕 刷新 crypto 持仓实时价格 (OKX REST, 零 ccxt) ──
    crypto_positions = [p for p in pf.open_positions if p.market == "crypto"]
    if crypto_positions:
        try:
            from okx_rest_data import get_okx_provider
            okx = get_okx_provider()
            symbols = list(set(p.symbol for p in crypto_positions))
            tickers = okx.fetch_tickers([s.replace("/", "-") for s in symbols])
            for sym in symbols:
                okx_sym = sym.replace("/", "-")
                ticker = tickers.get(okx_sym) or tickers.get(sym)
                if ticker and ticker.get('last', 0) > 0:
                    for p in crypto_positions:
                        if p.symbol == sym:
                            p.current_price = ticker['last']
        except Exception:
            pass  # OKX REST 不可用时跳过价格刷新

    # ── 碎片化检测: 同币种 > 3笔 ──
    symbol_counts = Counter(p.symbol for p in pf.open_positions if p.market == "crypto")
    fragmented_symbols = {sym for sym, cnt in symbol_counts.items() if cnt > 3}

    for pos in list(pf.open_positions):
        # 计算浮动盈亏%
        if pos.side == "LONG":
            pnl_pct = (pos.current_price / pos.entry_price - 1) * 100 if pos.current_price > 0 else 0
        else:
            pnl_pct = (pos.entry_price / pos.current_price - 1) * 100 if pos.current_price > 0 else 0

        should_close = False
        reason = ""

        # 1. 硬止损 -8%
        if pnl_pct <= -8.0:
            should_close = True
            reason = f"🛑 硬止损触发: {pnl_pct:.1f}% (<= -8%)"

        # 2. 僵尸仓 (持有 > 7天且亏损)
        elif pnl_pct < 0:
            held_days = 0
            if hasattr(pos, 'entry_time') and pos.entry_time:
                try:
                    entry_dt = datetime.fromisoformat(pos.entry_time.replace('+00:00', '+0000'))
                    held_days = (now.replace(tzinfo=None) - entry_dt.replace(tzinfo=None)).days
                except Exception:
                    pass
            if held_days > 7:
                should_close = True
                reason = f"💀 僵尸仓: 持有{held_days}天, 亏损{pnl_pct:.1f}%"

        # 3. 碎片化清理: 同币种 > 3笔, 关最差的
        if not should_close and pos.symbol in fragmented_symbols and pos.market == "crypto":
            same_symbol = [p for p in pf.open_positions
                          if p.symbol == pos.symbol and p.market == "crypto"]
            same_symbol_sorted = sorted(same_symbol,
                                       key=lambda p: (p.current_price / p.entry_price - 1) * 100
                                       if p.side == "LONG" else (p.entry_price / p.current_price - 1) * 100,
                                       reverse=True)
            if len(same_symbol_sorted) > 3 and pos in same_symbol_sorted[3:]:
                should_close = True
                reason = f"🧹 碎片化清理: {pos.symbol} {len(same_symbol_sorted)}笔→保留3笔"

        if should_close:
            try:
                if pos.side == "LONG":
                    pf.sell(pos.id, pos.current_price, reason=reason)
                else:
                    pf.cover_short(pos.id, pos.current_price, reason=reason)
                log_func(f"  {reason} | {pos.symbol} | 入场{pos.entry_price}→现价{pos.current_price}")
                results.append(f"🛑 止损/清理: {pos.symbol} ({reason})")
                _stopped_out_this_cycle.add(pos.symbol)  # 🆕 冷却追踪
            except Exception as e:
                log_func(f"  ❌ 止损平仓失败 {pos.symbol}: {e}")

    return results


def _manage_kline_positions(kline_positions: list, pf, log_func) -> list:
    """
    🕯️ 裸K专属持仓管理 — 结构止盈止损

    裸K止损基于K线结构 (前低/前高), 比ML固定%更精准。
    裸K止盈基于盈亏比倍数 (stop_distance × take_profit_rr)。
    出现反向裸K信号 (score >= 6) 立即平仓。
    """
    results = []
    for pos in kline_positions:
        try:
            # 更新当前价格 (OKX REST, 零 ccxt)
            try:
                from okx_rest_data import fetch_okx_ticker
                okx_sym = pos.symbol.replace("/", "-")
                ticker = fetch_okx_ticker(okx_sym)
                if ticker:
                    pos.current_price = ticker.get('last', pos.current_price)
            except Exception:
                pass  # 使用已有价格

            current = pos.current_price
            if current <= 0:
                continue

            # 计算PnL
            if pos.side == "LONG":
                pnl_pct = (current / pos.entry_price - 1) * 100
            else:
                pnl_pct = (pos.entry_price / current - 1) * 100

            should_close = False
            close_reason = ""

            # ── 检查结构止损 ──
            kline_sl = getattr(pos, 'kline_stop_loss', 0)
            kline_tp = getattr(pos, 'kline_take_profit', 0)

            if pos.side == "LONG" and kline_sl > 0 and current <= kline_sl:
                should_close = True
                close_reason = f"🕯️ 裸K结构止损: {pnl_pct:.1f}%"
            elif pos.side == "SHORT" and kline_sl > 0 and current >= kline_sl:
                should_close = True
                close_reason = f"🕯️ 裸K结构止损(空): {pnl_pct:.1f}%"

            # ── 检查结构止盈 ──
            if pos.side == "LONG" and kline_tp > 0 and current >= kline_tp:
                should_close = True
                close_reason = f"🕯️ 裸K结构止盈: +{pnl_pct:.1f}%"
            elif pos.side == "SHORT" and kline_tp > 0 and current <= kline_tp:
                should_close = True
                close_reason = f"🕯️ 裸K结构止盈(空): +{pnl_pct:.1f}%"

            if should_close:
                if pos.side == "LONG":
                    pf.sell(pos.id, current, reason=close_reason)
                else:
                    pf.cover_short(pos.id, current, reason=close_reason)
                results.append(close_reason)
                log_func(f"  {close_reason}")
                continue

            # ── 移动止损 (保本损) ──
            if pos.side == "LONG" and pnl_pct > 2.0:
                new_sl = pos.entry_price * 1.005  # 入场价+0.5%
                if kline_sl > 0 and new_sl > kline_sl:
                    pos.kline_stop_loss = new_sl
                    log_func(f"  🕯️ 保本损上移 {pos.symbol}: {new_sl:.2f}")
            elif pos.side == "SHORT" and pnl_pct > 2.0:
                new_sl = pos.entry_price * 0.995
                if kline_sl > 0 and new_sl < kline_sl:
                    pos.kline_stop_loss = new_sl
                    log_func(f"  🕯️ 保本损下移 {pos.symbol}: {new_sl:.2f}")

        except Exception as e:
            log_func(f"  ⚠️ 裸K持仓管理异常 {pos.symbol}: {e}")

    return results


def execute_strategy_signals(signals: list, pf, leverage_engine=None, live_trader=None) -> list:
    """
    执行策略信号 🆕 v3: 实盘OKX现货 + 集中火力

    规则:
      - BUY → 开多, SELL → 开空
      - 已有持仓的 symbol 跳过
      - 🔥 集中火力: 只做top1-2个最强信号, 每笔投入40-50%资金
      - 风控检查 + 最低 ¥200
      - entry_reason 标注策略名
      - 🔴 实盘模式: 通过BinanceLiveTrader下OKX现货单
    """
    from risk import RiskController
    rc = RiskController(pf)
    results = []

    # 🔴 实盘模式: 用交易所真实持仓+余额，不用 PortfolioManager 虚拟仓位
    held_symbols = []
    cash = pf.cash.get("crypto", 0)  # fallback
    total_equity = pf.total_value     # fallback
    if leverage_engine is not None:
        try:
            real_positions = leverage_engine.fetch_open_positions()
            if real_positions:
                held_symbols = list(real_positions) if isinstance(real_positions, set) else list(real_positions.keys())
                log(f"  📡 实盘持仓({len(held_symbols)}): {', '.join(held_symbols) if held_symbols else '空仓'}")
            # 🔴 用OKX真实余额计算仓位，不用虚拟盘
            real_equity = leverage_engine.fetch_equity()
            if real_equity > 0:
                cash = real_equity
                total_equity = real_equity
                log(f"  💰 实盘可用: \${real_equity:.2f}")
        except Exception as e:
            log(f"  ⚠️ 获取实盘持仓失败({e})，回退到虚拟盘")
    if not held_symbols and leverage_engine is None:
        # 虚拟盘模式：用 PortfolioManager
        held_symbols = [p.symbol for p in pf.open_positions if p.market == "crypto"]

    # 🛡️ 动态持仓上限 (按账户规模集中火力)
    # $0-100: 最多 2 仓 | $100-200: 最多 3 仓 | $200-500: 最多 4 仓 | $500+: 8 仓
    if total_equity <= 100:
        MAX_CRYPTO_POSITIONS = 2
    elif total_equity <= 200:
        MAX_CRYPTO_POSITIONS = 3
    elif total_equity <= 500:
        MAX_CRYPTO_POSITIONS = 4
    else:
        MAX_CRYPTO_POSITIONS = 8
    if len(held_symbols) >= MAX_CRYPTO_POSITIONS:
        log(f"  🛑 持仓已达上限 {MAX_CRYPTO_POSITIONS}个 (权益${total_equity:.0f})，跳过新信号。当前: {', '.join(held_symbols)}")
        return results

    # 🔥 集中火力: 按评分排序 + 最低质量门槛
    MIN_COMPOSITE = 55   # 🆕 score*confidence 综合分最低 55 (约 70分×78% 或 80分×69%)
    MIN_CONFIDENCE = 0.65  # 🆕 单信号置信度最低 65%
    valid_signals = []
    for s in signals:
        action = s.get("action", "HOLD")
        if action not in ("BUY", "SELL"):
            continue
        if s["symbol"] in held_symbols:
            continue
        conf = s.get("confidence", 0.5)
        score = s.get("score", 50)
        composite = score * conf
        if composite < MIN_COMPOSITE or conf < MIN_CONFIDENCE:
            continue
        valid_signals.append((composite, s))

    valid_signals.sort(key=lambda x: x[0], reverse=True)

    # 🔥 集中火力: 每个方向只取 top 1，且总数受上限约束
    available_slots = MAX_CRYPTO_POSITIONS - len(held_symbols)
    best_buy = None
    best_sell = None
    for _, s in valid_signals:
        action = s.get("action", "HOLD")
        if action == "BUY" and best_buy is None:
            best_buy = s
        elif action == "SELL" and best_sell is None:
            best_sell = s
        if best_buy and best_sell:
            break

    candidates = []
    if best_buy and available_slots > 0:
        candidates.append(("BUY", best_buy))
        available_slots -= 1
    if best_sell and available_slots > 0:
        candidates.append(("SELL", best_sell))

    if not candidates:
        log(f"  💤 无高质量策略信号 (需综合≥{MIN_COMPOSITE}, 置信≥{MIN_CONFIDENCE:.0%})")
        return results

    # 🔥 集中火力: 大仓位 — 1仓=50%可用, 2仓=各40%
    n_candidates = len(candidates)
    FIRE_PCT = min(0.48, 0.45 + 0.05 * (2 - n_candidates))  # 1个=50%, 2个=45% each

    for idx, (action, sig) in enumerate(candidates):
        try:
            # 🔥 集中火力仓位: 用更大比例
            max_size = cash * FIRE_PCT
            if max_size < MIN_MARGIN_USDT:
                log(f"  💤 [{sig.get('strategy_name', '?')}] {sig['symbol']} 资金不足 (可用\${cash:.2f}, 需≥\${MIN_MARGIN_USDT})")
                continue

            score = sig.get("score", 50)
            check = rc.pre_trade_check("crypto", max_size, score, total_value)
            if not check.passed:
                log(f"  ⚠️ [{sig.get('strategy_name', '?')}] {sig['symbol']} 风控拦截: {check.reason}")
                continue

            quantity = max_size / sig["price"]
            # 🕯️ 裸K优先标记
            is_kline = sig.get("kline_priority", False)
            prefix = "🕯️[裸K] " if is_kline else ""
            reason = prefix + f"[{sig.get('strategy_name', '?')}] " + " | ".join(sig.get("reasons", ["信号触发"])[:3])

            if action == "BUY":
                icon = "🟢" if not is_kline else "🕯️"
                action_word = "BUY"
                # 🔴 实盘OKX现货买入
                if live_trader is not None:
                    try:
                        order = live_trader.market_buy(
                            symbol=sig["symbol"],
                            amount_usdt=max_size,
                            note=f"🔥集中火力 [{sig.get('strategy_name', '?')}] {reason[:80]}"
                        )
                        if order and order.status == "filled":
                            log(f"  🔥 OKX实盘: {icon} BUY {sig['symbol']} ${max_size:.0f} | 成交@{order.avg_price:.2f}")
                        else:
                            log(f"  ⚠️ OKX下单异常: {order}")
                    except Exception as e:
                        log(f"  ❌ OKX买入失败 {sig['symbol']}: {e}")
                        continue
                # 同时记录到 PortfolioManager
                trade = pf.buy(
                    market="crypto", symbol=sig["symbol"],
                    name=sig.get("name", sig["symbol"]),
                    price=sig["price"], quantity=quantity,
                    reason=reason,
                )
            else:  # SELL → 开空仓
                icon = "🔴" if not is_kline else "🕯️"
                action_word = "SHORT"
                # 🔴 实盘OKX合约做空
                if live_trader is not None:
                    try:
                        # 根据信心分选杠杆: 高信心→高杠杆, 低信心→低杠杆
                        conf = sig.get("confidence", 0.5)
                        if conf >= 0.75:
                            short_lev = 5
                        elif conf >= 0.60:
                            short_lev = 3
                        else:
                            short_lev = 2
                        # 止损价: 做空止损在入场价上方
                        sl_price = sig["price"] * 1.025 if short_lev >= 5 else sig["price"] * 1.05
                        order = live_trader.open_short_swap(
                            symbol=sig["symbol"],
                            amount_usdt=max_size,
                            leverage=short_lev,
                            stop_loss_price=sl_price,
                            note=f"🔥集中火力 [{sig.get('strategy_name', '?')}] {reason[:60]}"
                        )
                        if order and order.status == "filled":
                            log(f"  🔥 OKX实盘: {icon} SHORT {sig['symbol']} ${max_size:.0f} @{short_lev}x | 成交@{order.avg_price:.2f}")
                        else:
                            log(f"  ⚠️ OKX做空异常 {sig['symbol']}: {order}")
                    except Exception as e:
                        log(f"  ❌ OKX做空失败 {sig['symbol']}: {e}")
                        continue
                # 记录到模拟盘
                if hasattr(pf, 'open_short'):
                    trade = pf.open_short(
                        market="crypto", symbol=sig["symbol"],
                        name=sig.get("name", sig["symbol"]),
                        price=sig["price"], quantity=quantity,
                        margin_usdt=max_size, leverage=1,
                        reason=reason,
                        stop_loss=sig["price"] * 1.08,
                        take_profit=sig["price"] * 0.88,
                    )
                else:
                    trade = None

            if trade:
                # 🕯️ 存储裸K专属止损/止盈
                if is_kline:
                    try:
                        for pos in pf.open_positions:
                            if pos.symbol == sig["symbol"] and pos.market == "crypto":
                                if hasattr(pos, 'kline_stop_loss'):
                                    pos.kline_stop_loss = sig.get("stop_loss", 0)
                                if hasattr(pos, 'kline_take_profit'):
                                    pos.kline_take_profit = sig.get("take_profit", 0)
                                if hasattr(pos, 'kline_signal_score'):
                                    pos.kline_signal_score = sig.get("kline_score_3step", 0)
                                break
                    except Exception:
                        pass

                kline_tag = " [裸K]" if is_kline else ""
                fire_tag = "🔥" if live_trader is not None else ""
                result_msg = f"{icon}{fire_tag} [{sig.get('strategy_name', '?')}]{kline_tag} {action_word} {sig['symbol']} ¥{max_size:.0f} | 评分{sig['score']:.0f} | 置信{sig['confidence']:.0%}"
                results.append(result_msg)
                log(f"  ✅ {result_msg}")
                held_symbols.append(sig["symbol"])
        except Exception as e:
            log(f"  ❌ [{sig.get('strategy_name', '?')}] 执行失败: {e}")

    # 🆕 SELL/SHORT 信号汇总
    sell_sigs = [s for s in signals if s["action"] == "SELL"]
    if sell_sigs:
        log(f"  📋 {len(sell_sigs)} 条做空信号"
            f" ({'实盘合约执行' if live_trader else '模拟盘已执行'}):")
        for s in sell_sigs[:5]:
            log(f"     🔴 [{s.get('strategy_name', '?')}] {s['symbol']} | 评分{s['score']:.0f} | 置信{s['confidence']:.0%}")

    return results


def _enforce_min_margin(leverage_engine, log_func) -> list:
    """
    🚨 硬性规则: 任何保证金 < MIN_MARGIN_USDT 的仓位，立即平仓。

    这是最后一道防线 — 理论上代码的入场门槛($20) + 仓位计算(40-50%权益)
    已经杜绝了小仓位，但如果因为任何原因（bug/旧代码/边界条件）出现了，
    这里直接平掉，不商量。

    Returns: list of close results
    """
    MIN_HARD = MIN_MARGIN_USDT  # 使用全局常量
    results = []
    try:
        import requests as _req, hmac as _hmac, base64 as _b64, json as _json
        from datetime import datetime as _dt, timezone as _tz

        # 获取实盘持仓
        ts = _dt.now(_tz.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        path = "/api/v5/account/positions?instType=SWAP"
        sign_str = ts + "GET" + path + ""
        sign = _b64.b64encode(_hmac.new(
            leverage_engine._okx_trading.secret.encode(), sign_str.encode(), "sha256"
        ).digest()).decode()
        r = _req.get(
            f"https://{leverage_engine._okx_trading.hostname}{path}",
            headers={
                "OK-ACCESS-KEY": leverage_engine._okx_trading.apiKey,
                "OK-ACCESS-SIGN": sign,
                "OK-ACCESS-TIMESTAMP": ts,
                "OK-ACCESS-PASSPHRASE": leverage_engine._okx_trading.password,
            },
            timeout=10,
        )
        positions = [p for p in r.json().get('data', []) if float(p.get('pos', 0)) > 0]

        for p in positions:
            inst_id = p['instId']
            sym = inst_id.replace('-USDT-SWAP', '').replace('-', '/')
            pos_side = p['posSide']
            contracts = float(p.get('pos', 0))
            margin = float(p.get('margin', 0))

            if margin >= MIN_HARD:
                continue  # 达标，放行

            # 不达标 — 直接平仓，不商量
            upl = float(p.get('upl', 0))
            log_func(f"  🚨 {sym}: 保证金${margin:.2f} < ${MIN_HARD}底线! 强制平仓! (UPL=${upl:+.4f})")

            close_side = "sell" if pos_side == "long" else "buy"

            # 取消 algo 订单
            try:
                algo_ts2 = _dt.now(_tz.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                algo_path2 = f"/api/v5/trade/orders-algo-pending?instId={inst_id}"
                algo_sign_str2 = algo_ts2 + "GET" + algo_path2 + ""
                algo_sign2 = _b64.b64encode(_hmac.new(
                    leverage_engine._okx_trading.secret.encode(), algo_sign_str2.encode(), "sha256"
                ).digest()).decode()
                algo_r2 = _req.get(
                    f"https://{leverage_engine._okx_trading.hostname}{algo_path2}",
                    headers={
                        "OK-ACCESS-KEY": leverage_engine._okx_trading.apiKey,
                        "OK-ACCESS-SIGN": algo_sign2,
                        "OK-ACCESS-TIMESTAMP": algo_ts2,
                        "OK-ACCESS-PASSPHRASE": leverage_engine._okx_trading.password,
                    },
                    timeout=10,
                )
                for ao in algo_r2.json().get('data', []):
                    try:
                        cancel_ts = _dt.now(_tz.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                        cancel_path = f"/api/v5/trade/cancel-algo"
                        cancel_body = _json.dumps([{"algoId": ao['algoId'], "instId": inst_id}])
                        cancel_sign_str = cancel_ts + "POST" + cancel_path + cancel_body
                        cancel_sign = _b64.b64encode(_hmac.new(
                            leverage_engine._okx_trading.secret.encode(), cancel_sign_str.encode(), "sha256"
                        ).digest()).decode()
                        _req.post(
                            f"https://{leverage_engine._okx_trading.hostname}{cancel_path}",
                            headers={
                                "OK-ACCESS-KEY": leverage_engine._okx_trading.apiKey,
                                "OK-ACCESS-SIGN": cancel_sign,
                                "OK-ACCESS-TIMESTAMP": cancel_ts,
                                "OK-ACCESS-PASSPHRASE": leverage_engine._okx_trading.password,
                                "Content-Type": "application/json",
                            },
                            data=cancel_body, timeout=10,
                        )
                    except Exception:
                        pass
            except Exception:
                pass

            # 下平仓市价单
            try:
                order_ts = _dt.now(_tz.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                order_path = f"/api/v5/trade/order"
                order_body = _json.dumps({
                    "instId": inst_id,
                    "tdMode": "isolated",
                    "side": close_side,
                    "posSide": pos_side,
                    "ordType": "market",
                    "sz": str(contracts),
                })
                order_sign_str = order_ts + "POST" + order_path + order_body
                order_sign = _b64.b64encode(_hmac.new(
                    leverage_engine._okx_trading.secret.encode(), order_sign_str.encode(), "sha256"
                ).digest()).decode()
                order_r = _req.post(
                    f"https://{leverage_engine._okx_trading.hostname}{order_path}",
                    headers={
                        "OK-ACCESS-KEY": leverage_engine._okx_trading.apiKey,
                        "OK-ACCESS-SIGN": order_sign,
                        "OK-ACCESS-TIMESTAMP": order_ts,
                        "OK-ACCESS-PASSPHRASE": leverage_engine._okx_trading.password,
                        "Content-Type": "application/json",
                    },
                    data=order_body, timeout=10,
                )
                resp = order_r.json()
                if resp.get("code") == "0":
                    log_func(f"  ✅ 已强制平仓 {sym} (保证金${margin:.2f}不达标)")
                    results.append({"symbol": sym, "action": "ENFORCED_CLOSE", "margin": margin, "upl": upl})
                    _stopped_out_this_cycle.add(sym)  # 🆕 冷却追踪
                else:
                    log_func(f"  ❌ 强制平仓失败 {sym}: {resp}")
            except Exception as e:
                log_func(f"  ❌ 强制平仓异常 {sym}: {e}")

    except Exception as e:
        log_func(f"⚠️ 最低保证金检查异常: {e}")

    return results


def _clean_low_conviction_positions(leverage_engine, log_func) -> list:
    """
    🧹 清理低确信/低质量仓位：盈利超手续费就回收。

    用于处理策略升级前遗留的"撒胡椒面"仓位。
    不勉强赚大钱，但也不能亏。
    盈利 > 0.3% (覆盖手续费+滑点) 就平掉。

    Returns: list of close results
    """
    results = []
    try:
        import requests as _req, hmac as _hmac, base64 as _b64, json as _json
        from datetime import datetime as _dt, timezone as _tz

        # 获取实盘持仓
        ts = _dt.now(_tz.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        path = "/api/v5/account/positions?instType=SWAP"
        sign_str = ts + "GET" + path + ""
        sign = _b64.b64encode(_hmac.new(
            leverage_engine._okx_trading.secret.encode(), sign_str.encode(), "sha256"
        ).digest()).decode()
        r = _req.get(
            f"https://{leverage_engine._okx_trading.hostname}{path}",
            headers={
                "OK-ACCESS-KEY": leverage_engine._okx_trading.apiKey,
                "OK-ACCESS-SIGN": sign,
                "OK-ACCESS-TIMESTAMP": ts,
                "OK-ACCESS-PASSPHRASE": leverage_engine._okx_trading.password,
            },
            timeout=10,
        )
        positions = [p for p in r.json().get('data', []) if float(p.get('pos', 0)) > 0]

        if not positions:
            return results

        total_equity = leverage_engine.fetch_equity()

        for p in positions:
            inst_id = p['instId']
            sym = inst_id.replace('-USDT-SWAP', '').replace('-', '/')
            pos_side = p['posSide']
            contracts = float(p.get('pos', 0))
            upl = float(p.get('upl', 0))
            margin = float(p.get('margin', 0))
            upl_pct = (upl / margin * 100) if margin > 0 else 0

            # 🧹 判断是否低确信仓位（保守策略 — 宁少清不多清）:
            # 1. 保证金 ≤ $12  (真正撒胡椒面的产物 — 大仓 >$12 不碰)
            # 2. 浮盈在 0.3%~2% 之间 (微利没冲劲 → 回收；>2% 有动量 → 让它跑)
            # 3. 不是大盘币 (BTC/ETH 大仓不清理)
            # 4. 唯一空头不清理 (对冲价值)
            if margin > 12:
                continue
            if upl_pct < 0.3 or upl_pct > 2.0:
                continue  # 水下不割肉，强动量(>2%)不截断
            if sym in ("BTC/USDT", "ETH/USDT"):
                continue
            # 检查是不是唯一空头（保留对冲）
            long_count = sum(1 for pp in positions if pp['posSide'] == 'long' and float(pp.get('pos',0)) > 0)
            short_count = sum(1 for pp in positions if pp['posSide'] == 'short' and float(pp.get('pos',0)) > 0)
            if pos_side == 'short' and short_count <= 1:
                continue  # 保留唯一空头做对冲
            if pos_side == 'long' and long_count <= 1:
                continue  # 保留唯一多头

            log_func(f"  🧹 {sym}: 低确信仓位盈利+{upl_pct:.1f}%, 回收中...")

            close_side = "sell" if pos_side == "long" else "buy"

            # 取消这个币的 algo 订单
            try:
                algo_ts = _dt.now(_tz.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                algo_path = f"/api/v5/trade/orders-algo-pending?instId={inst_id}"
                algo_sign_str = algo_ts + "GET" + algo_path + ""
                algo_sign = _b64.b64encode(_hmac.new(
                    leverage_engine._okx_trading.secret.encode(), algo_sign_str.encode(), "sha256"
                ).digest()).decode()
                algo_r = _req.get(
                    f"https://{leverage_engine._okx_trading.hostname}{algo_path}",
                    headers={
                        "OK-ACCESS-KEY": leverage_engine._okx_trading.apiKey,
                        "OK-ACCESS-SIGN": algo_sign,
                        "OK-ACCESS-TIMESTAMP": algo_ts,
                        "OK-ACCESS-PASSPHRASE": leverage_engine._okx_trading.password,
                    },
                    timeout=10,
                )
                for a in algo_r.json().get('data', []):
                    cancel_body = _json.dumps([{"algoId": a['algoId'], "instId": inst_id}])
                    cancel_ts = _dt.now(_tz.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                    cancel_path = "/api/v5/trade/cancel-algos"
                    cancel_sign_str = cancel_ts + "POST" + cancel_path + cancel_body
                    cancel_sign = _b64.b64encode(_hmac.new(
                        leverage_engine._okx_trading.secret.encode(), cancel_sign_str.encode(), "sha256"
                    ).digest()).decode()
                    _req.post(
                        f"https://{leverage_engine._okx_trading.hostname}{cancel_path}",
                        headers={
                            "OK-ACCESS-KEY": leverage_engine._okx_trading.apiKey,
                            "OK-ACCESS-SIGN": cancel_sign,
                            "OK-ACCESS-TIMESTAMP": cancel_ts,
                            "OK-ACCESS-PASSPHRASE": leverage_engine._okx_trading.password,
                            "Content-Type": "application/json",
                        },
                        data=cancel_body,
                        timeout=10,
                    )
            except Exception:
                pass

            # 下市价平仓单
            import math
            lot_sz = float(p.get('lotSz', 1) or 1)
            sz = math.floor(contracts / lot_sz) * lot_sz if lot_sz > 0 else contracts
            if sz <= 0:
                continue

            body = _json.dumps({
                "instId": inst_id, "tdMode": "isolated",
                "side": close_side, "posSide": pos_side,
                "ordType": "market", "sz": str(sz),
            })
            order_ts = _dt.now(_tz.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            order_path = "/api/v5/trade/order"
            order_sign_str = order_ts + "POST" + order_path + body
            order_sign = _b64.b64encode(_hmac.new(
                leverage_engine._okx_trading.secret.encode(), order_sign_str.encode(), "sha256"
            ).digest()).decode()
            order_r = _req.post(
                f"https://{leverage_engine._okx_trading.hostname}{order_path}",
                headers={
                    "OK-ACCESS-KEY": leverage_engine._okx_trading.apiKey,
                    "OK-ACCESS-SIGN": order_sign,
                    "OK-ACCESS-TIMESTAMP": order_ts,
                    "OK-ACCESS-PASSPHRASE": leverage_engine._okx_trading.password,
                    "Content-Type": "application/json",
                },
                data=body,
                timeout=10,
            )
            resp = order_r.json()
            if resp.get('code') == '0':
                log_func(f"  ✅ 回收 {sym}: 盈利约${upl:+.2f}")
                results.append({"symbol": sym, "action": "CLEAN_CLOSE", "pnl": upl})
                if upl < 0:
                    _stopped_out_this_cycle.add(sym)  # 🆕 亏损平仓加入冷却
            else:
                log_func(f"  ⚠️ 回收失败 {sym}: {resp}")

    except Exception as e:
        log_func(f"  ⚠️ 低确信清理异常: {e}")

    return results


def _run_leverage_decisions(signals: list, sentiment_engine, leverage_engine, pf, log_func,
                           remaining_slots: int = None, stopped_out_symbols: set = None) -> list:
    """对合约标的运行杠杆决策 (实盘模式) — 🆕 v2: 支持做多+做空

    Args:
        remaining_slots: 剩余可用仓位槽数 (None=自动从持仓计算)
        stopped_out_symbols: 最近被止损的币种集合 (冷却期内跳过)
    """
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

    # 🛡️ 动态持仓上限 (按账户规模)
    total_equity = leverage_engine.fetch_equity()
    if total_equity <= 100:
        MAX_SWAP_POSITIONS = 2
    elif total_equity <= 200:
        MAX_SWAP_POSITIONS = 3
    elif total_equity <= 500:
        MAX_SWAP_POSITIONS = 4
    else:
        MAX_SWAP_POSITIONS = 5

    # 🆕 使用传入的剩余槽数 (用于跨调用协调)
    if remaining_slots is None:
        remaining_slots = MAX_SWAP_POSITIONS - len(existing_positions)
    else:
        remaining_slots = min(remaining_slots, MAX_SWAP_POSITIONS - len(existing_positions))

    if remaining_slots <= 0:
        log_func(f"  🛑 合约持仓已达上限 {MAX_SWAP_POSITIONS}个 (权益${total_equity:.0f})。当前: {', '.join(existing_list[:8])}")
        return results

    # 🔥 集中火力: 动态信号数 + 最低质量门槛
    MAX_SWAP_SIGNALS = min(remaining_slots, 1 if total_equity <= 100 else (2 if total_equity <= 200 else 3))
    MIN_COMPOSITE = 55    # 🆕 score*confidence 综合分最低 55
    MIN_CONFIDENCE = 0.65  # 🆕 单信号置信度最低 65%

    valid_swaps = []
    for sig in signals:
        sym = sig.get("symbol", "")
        if sym not in swap_symbols:
            continue
        action = sig.get("action", "HOLD")
        if action not in ("BUY", "SELL"):
            continue
        if sym in existing_positions:
            continue
        # 🆕 冷却期检查: 最近被止损的币种跳过
        if stopped_out_symbols and sym in stopped_out_symbols:
            log_func(f"  🧊 冷却期跳过 {sym}: 最近被止损，{STOPPED_OUT_COOLDOWN_CYCLES}周期后再考虑")
            continue
        conf = sig.get("confidence", 0.5)
        score = sig.get("score", 50)
        composite = score * conf
        if composite < MIN_COMPOSITE or conf < MIN_CONFIDENCE:
            continue
        valid_swaps.append((composite, sig))

    valid_swaps.sort(key=lambda x: x[0], reverse=True)
    selected_swaps = valid_swaps[:MAX_SWAP_SIGNALS]

    if len(valid_swaps) > len(selected_swaps):
        skipped_names = [s[1].get('symbol','?') for s in valid_swaps[len(selected_swaps):]]
        log_func(f"  🔥 集中火力: 只做top{len(selected_swaps)}信号，跳过: {', '.join(skipped_names[:5])}")

    for _, sig in selected_swaps:
        sym = sig.get("symbol", "")
        action = sig.get("action", "HOLD")

        # 🆕 v2: 确定方向
        if action == "BUY":
            side = "buy"
            pos_side = "long"
            action_label = "BUY_LONG"
        else:  # SELL
            side = "sell"
            pos_side = "short"
            action_label = "SELL_SHORT"

        # 拉取情绪叠加
        overlay = None
        if sentiment_engine:
            try:
                overlay = sentiment_engine.get_sentiment_overlay(sym)
            except Exception:
                pass

        # 拉取资金费率 (做空时尤其重要 — 负费率做空有利)
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

        # 🏛️ Regime 杠杆帽: 安检门设定的最大杠杆
        regime_max_lev = sig.get("_max_leverage")
        if regime_max_lev and decision.recommended_leverage > regime_max_lev:
            decision.recommended_leverage = regime_max_lev
            decision.leverage_label = f"CAPPED-{regime_max_lev}x"

        if decision.skip_reason:
            log_func(f"  ⚡ 跳过 {sym}: {decision.skip_reason}")
            continue

        # 仓位计算 — 使用已缓存的 OKX 余额
        if total_equity <= 0:
            log_func(f"  ⚡ 跳过 {sym}: 无法获取余额")
            continue
        price = sig.get("price", 100.0)
        pos = leverage_engine.calculate_position(total_equity, price, decision, side=side)

        # 🐜 最低保证金过滤: <$20不玩，撒胡椒面没有意义
        if pos['margin_usdt'] < MIN_MARGIN_USDT:
            log_func(f"  🐜 跳过 {sym}: 保证金${pos['margin_usdt']:.2f} < ${MIN_MARGIN_USDT} (不值得做)")
            continue

        icon = "🟢" if action == "BUY" else "🔴"
        log_func(
            f"  {icon} {sym}: {decision.recommended_leverage}x | "
            f"WR={decision.blended_win_rate:.0%} | "
            f"保证金=${pos['margin_usdt']} | "
            f"止损={decision.stop_loss_pct:+.1%} | "
            f"止盈={decision.take_profit_pct:+.0%}"
        )

        # 🛡️ 入场时机校验: 价格在24h区间位置
        price_check = leverage_engine.check_price_position(sym, side, price)
        if not price_check["safe"]:
            log_func(f"  ⚠️ [{sym}] {price_check['reason']}")
            continue  # 跳过，不开仓
        elif price_check["percentile"] != 50.0:  # 正常检查通过
            log_func(f"  ✅ [{sym}] 价格位置安全: {price_check['percentile']:.0f}%分位 "
                    f"(区间{price_check['low_24h']:.4f}-{price_check['high_24h']:.4f})")

        # 执行合约单
        try:
            order = leverage_engine.create_swap_market_order(
                symbol=sym,
                side=side,           # 🆕: "buy" for long, "sell" for short
                quantity_contracts=pos["quantity_contracts"],
                leverage=decision.recommended_leverage,
                stop_loss_price=pos["stop_loss_price"],
                take_profit_price=pos["take_profit_price"],
                note=f"{decision.recommended_leverage}x_{pos_side}_{sym.split('/')[0][:8]}",
            )
            if order:
                results.append({
                    "symbol": sym, "action": action_label,
                    "leverage": decision.recommended_leverage,
                    "margin": pos["margin_usdt"],
                    "notional": pos["notional_usdt"],
                    "order_id": order.get("id", ""),
                })
                leverage_engine.increment_daily_trades()

                # 🔴 下单后验证: 实际成交 vs 预期保证金 偏差>50%告警
                actual_cost = order.get("cost", 0) or 0
                actual_filled = order.get("filled", 0) or 0
                expected_notional = pos["notional_usdt"]
                if actual_cost > 0 and expected_notional > 0:
                    deviation = abs(actual_cost - expected_notional) / expected_notional
                    if deviation > 0.50:
                        log_func(f"  🚨 订单偏差! {sym}: 预期保证金=${pos['margin_usdt']} 实际成交=${actual_cost:.2f} (偏差{deviation:.0%}) — 可能ctVal计算错误!")
                    elif deviation > 0.20:
                        log_func(f"  ⚠️ 订单偏差 {sym}: 预期=${pos['margin_usdt']} 实际成交=${actual_cost:.2f} (偏差{deviation:.0%})")
                elif actual_filled == 0 and expected_notional > 10:
                    log_func(f"  ⚠️ 订单未完全成交 {sym}: filled={actual_filled}, 预期名义=${expected_notional}")
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
                    # ── 🔭 四视角扫描: 权哥价值+西蒙斯量化+Serenity长期价值 ──
                    try:
                        from perspective_scanner import MultiPerspectiveScanner
                        p_scanner = MultiPerspectiveScanner()
                        p_context = {
                            "fg_value": fg.current_value if fg else 50,
                            "fg_label": fg.classification if fg else "Neutral",
                            "btc_trend": "uptrend",  # will be refined by MarketRegimeClassifier
                            "vol_level": "medium",
                        }
                        # 试着用 regime 分类器的 BTC 数据
                        try:
                            from market_regime import MarketRegimeClassifier
                            rc = MarketRegimeClassifier()
                            btc_data = rc._fetch_btc_data()
                            p_context["btc_trend"] = btc_data.get("trend", "uptrend")
                            p_context["vol_level"] = btc_data.get("vol_level", "medium")
                        except Exception:
                            pass
                        perspective_signals = p_scanner.scan_all(p_context)
                        if perspective_signals:
                            n_val = sum(1 for s in perspective_signals if s.get("strategy_name") == "权哥价值投资")
                            n_quant = sum(1 for s in perspective_signals if s.get("strategy_name") == "西蒙斯量化统计")
                            n_serenity = sum(1 for s in perspective_signals if s.get("strategy_name") == "Serenity长期价值")
                            log(f"🔭 四视角扫描: +{len(perspective_signals)}信号 "
                                f"(权哥{n_val} + 西蒙斯{n_quant} + Serenity{n_serenity})")
                            strategy_signals.extend(perspective_signals)
                    except Exception as e:
                        log(f"⚠️ 多视角扫描跳过: {e}")

                    # ── 🏛️ 市场Regime安检门 (交易前第一道防线) ──
                    from market_regime import check_regime, MarketRegimeClassifier
                    alpha_summary = MarketRegimeClassifier.summarize_alpha_signals(strategy_signals)
                    regime = check_regime(
                        fg_value=fg.current_value if fg else 50,
                        fg_label=fg.classification if fg else "Neutral",
                        alpha_signals=strategy_signals,
                    )
                    state['last_regime'] = {
                        "regime": regime.regime,
                        "action_bias": regime.action_bias,
                        "max_risk": regime.max_risk_percent,
                        "recommended_lev": regime.recommended_leverage,
                        "pass": regime.pass_,
                        "reason": regime.reason,
                    }

                    icon = "🟢" if regime.pass_ else "🔴"
                    log(f"🏛️ 安检门: {icon} {regime.regime} | bias={regime.action_bias} "
                        f"| risk={regime.max_risk_percent:.0%} | lev≤{regime.recommended_leverage}x")
                    log(f"   📋 {regime.reason}")

                    if not regime.pass_:
                        log(f"   🚫 市场环境禁止交易，跳过本轮全部信号")
                        strategy_signals = []  # 清空所有信号
                    else:
                        # 🛡️ 方向过滤: 根据regime的action_bias过滤信号
                        before_filter = len(strategy_signals)
                        if regime.action_bias == "long_only":
                            strategy_signals = [s for s in strategy_signals
                                              if s.get("action") == "BUY"]
                        elif regime.action_bias == "short_only":
                            strategy_signals = [s for s in strategy_signals
                                              if s.get("action") == "SELL"]
                        elif regime.action_bias == "neutral":
                            # 中性模式: 仅保留置信度 > 70% 的信号 (高度确认才做)
                            strategy_signals = [s for s in strategy_signals
                                              if s.get("confidence", 0) >= 0.70]
                        filtered = before_filter - len(strategy_signals)
                        if filtered:
                            log(f"   🔍 方向过滤: {before_filter}→{len(strategy_signals)} "
                                f"({regime.action_bias}模式，过滤{filtered}条)")

                        # 🛡️ 风险限制: 覆盖杠杆上限
                        max_lev = regime.recommended_leverage
                        for s in strategy_signals:
                            s["_max_leverage"] = max_lev
                            s["_max_risk_pct"] = regime.max_risk_percent

                        # ── ⚖️ L2+L3 策略共识 + 执行官审批 ──
                        if strategy_signals:
                            try:
                                from trade_gates import run_trade_gates
                                # 获取当前持仓列表
                                try:
                                    current_pos_list = list(leverage_engine.fetch_open_positions() or [])
                                except Exception:
                                    current_pos_list = []
                                equity = leverage_engine.fetch_equity() if leverage_engine else 100
                                trade_decisions = run_trade_gates(
                                    signals=strategy_signals,
                                    regime_result=regime,
                                    current_positions=current_pos_list,
                                    total_equity=equity,
                                    fg_value=fg.current_value if fg else 50,
                                )
                                # 统计并过滤
                                approved = {sym: d for sym, d in trade_decisions.items() if d.execute}
                                rejected = {sym: d for sym, d in trade_decisions.items() if not d.execute}

                                if rejected:
                                    for sym, d in rejected.items():
                                        reasons_short = d.reject_reason[:80] if d.reject_reason else "未知"
                                        log(f"  🚫 [{sym}] 审批拒绝: {reasons_short}")

                                if approved:
                                    log(f"  ⚖️ 三层审批通过: {len(approved)}个信号 "
                                        f"({', '.join(approved.keys())})")
                                    # 更新信号: 应用执行官批准的仓位
                                    for sym, d in approved.items():
                                        for s in strategy_signals:
                                            if s.get("symbol") == sym:
                                                s["suggested_size"] = d.suggested_position_size
                                                s["_approved_confidence"] = d.final_confidence
                                    # 只保留批准的信号
                                    strategy_signals = [s for s in strategy_signals
                                                      if s.get("symbol") in approved]
                                else:
                                    log(f"  ⚖️ 三层审批: 无信号通过")
                                    strategy_signals = []
                            except Exception as e:
                                log(f"  ⚠️ 共识/执行官审批异常，跳过L2+L3: {e}")
                                # 异常时保守处理: 只保留高置信度信号
                                strategy_signals = [s for s in strategy_signals
                                                  if s.get("confidence", 0) >= 0.70]

                    # 🆕 冷却期过滤: 最近被止损的币种跳过
                    stopped_out = set(state.get('stopped_out_cooldown', {}).keys()) if state.get('stopped_out_cooldown') else set()
                    if stopped_out:
                        before = len(strategy_signals)
                        strategy_signals = [s for s in strategy_signals if s.get('symbol') not in stopped_out]
                        skipped = before - len(strategy_signals)
                        if skipped:
                            log(f"  🧊 冷却期过滤: 跳过 {skipped} 条信号 ({', '.join(sorted(stopped_out))})")

                    if strategy_signals:
                        strat_results = execute_strategy_signals(
                            strategy_signals, pf, leverage_engine,
                            live_trader=getattr(trader, 'live_trader', None)
                        )
                        if strat_results:
                            results.extend(strat_results)
                            log(f"🧠 多策略引擎: {len(strat_results)} 笔执行")

                        # ── ⚡ 杠杆决策 (实盘 + 合约标的) ──
                        if leverage_engine and trading_mode.is_real:
                            swap_results = _run_leverage_decisions(
                                strategy_signals, sentiment_engine,
                                leverage_engine, pf, log,
                                stopped_out_symbols=stopped_out
                            )
                            if swap_results:
                                results.extend(swap_results)
                    else:
                        log("💤 安检后无有效信号")
                else:
                    log("💤 多策略引擎: 无信号")
        except Exception as e:
            log(f"⚠️ 多策略引擎异常: {e}")

        # ── 🏭 股票合约扫描 (半导体/AI软件/太空等, ML策略不覆盖) ──
        # 🆕 股票合约与加密货币共享仓位上限，不再有独立执行路径
        if leverage_engine and trading_mode.is_real:
            try:
                from stock_swap_scanner import scan_stock_swaps

                # 🛡️ 重新获取实盘持仓 (含刚开的crypto仓位)
                existing_swap_positions = set()
                try:
                    okx_pos = leverage_engine.fetch_open_positions()
                    existing_swap_positions = okx_pos if isinstance(okx_pos, set) else set(okx_pos.keys() if hasattr(okx_pos, 'keys') else okx_pos)
                except Exception:
                    pass

                # 🛡️ 统一仓位上限: 股票+crypto共享同一个限制
                total_equity = leverage_engine.fetch_equity()
                if total_equity <= 100:
                    MAX_TOTAL_POSITIONS = 2
                elif total_equity <= 200:
                    MAX_TOTAL_POSITIONS = 3
                elif total_equity <= 500:
                    MAX_TOTAL_POSITIONS = 4
                else:
                    MAX_TOTAL_POSITIONS = 5

                remaining_slots = MAX_TOTAL_POSITIONS - len(existing_swap_positions)
                if remaining_slots <= 0:
                    log(f"  🛑 总仓位已达上限 {MAX_TOTAL_POSITIONS}个 (含crypto+股票)。"
                        f"当前: {', '.join(sorted(existing_swap_positions)[:8])}")
                else:
                    # 每3个周期全量扫描一次扩展列表
                    force_full = (state.get('cycles', 0) % 3 == 0)
                    stock_signals = scan_stock_swaps(
                        sentiment_engine=sentiment_engine,
                        existing_positions=existing_swap_positions,
                        force_full=force_full,
                    )
                    if stock_signals:
                        log(f"🏭 股票合约扫描: {len(stock_signals)} 条信号 (剩余{remaining_slots}槽)")
                        for ss in stock_signals:
                            log(f"  📈 [{ss['strategy_name']}] {ss['name']} | "
                                f"评分{ss['score']:.0f} | 置信{ss['confidence']:.1%} | "
                                f"{', '.join(ss['reasons'][:2])}")
                        # 🆕 传入剩余槽数 + 冷却名单
                        stopped_out = set(state.get('stopped_out_cooldown', {}).keys()) if state.get('stopped_out_cooldown') else set()
                        stock_results = _run_leverage_decisions(
                            stock_signals, sentiment_engine,
                            leverage_engine, pf, log,
                            remaining_slots=remaining_slots,
                            stopped_out_symbols=stopped_out
                        )
                        if stock_results:
                            results.extend(stock_results)
                            log(f"🏭 股票合约: {len(stock_results)} 笔执行")
                    else:
                        log(f"🏭 股票合约扫描: 0 条信号 (剩余{remaining_slots}槽, force_full={force_full})")
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

        # ── 🚨 硬性最低保证金检查: <$20的直接平仓，不商量 ──
        if leverage_engine:
            try:
                enforced = _enforce_min_margin(leverage_engine, log)
                if enforced:
                    results.extend(enforced)
                    log(f"🚨 最低保证金执法: {len(enforced)} 笔强制平仓")
            except Exception as e:
                log(f"⚠️ 最低保证金执法异常: {e}")

        # ── 🧹 低确信仓位清理: 盈利>手续费就收 (不勉强) ──
        if leverage_engine:
            try:
                cleaned = _clean_low_conviction_positions(leverage_engine, log)
                if cleaned:
                    results.extend(cleaned)
                    log(f"🧹 低确信清理: {len(cleaned)} 笔回收")
            except Exception as e:
                log(f"⚠️ 低确信清理异常: {e}")

        # ── 🛡️ 硬止损强制执行 (所有持仓) ──
        if pf and pf.open_positions:
            sl_results = _enforce_hard_stop_loss(pf, log)
            if sl_results:
                results.extend(sl_results)
                log(f"🛡️ 硬止损: {len(sl_results)} 笔强制平仓")

        # ── 🕯️ 裸K持仓管理 (结构止盈止损) ──
        if pf and pf.open_positions:
            kline_positions = [p for p in pf.open_positions
                             if p.market == "crypto" and
                             hasattr(p, 'kline_signal_score') and
                             p.kline_signal_score > 0]
            if kline_positions:
                kline_results = _manage_kline_positions(kline_positions, pf, log)
                if kline_results:
                    results.extend(kline_results)

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

    # ── 🆕 启动时同步OKX现有持仓 (防止重启后stale数据重复开仓) ──
    if trading_mode.is_real:
        try:
            from leverage_engine import LeverageEngine
            le = LeverageEngine()
            okx_startup_positions = le.fetch_open_positions()
            pos_list = list(okx_startup_positions) if isinstance(okx_startup_positions, set) else list(okx_startup_positions.keys()) if hasattr(okx_startup_positions, 'keys') else []
            if pos_list:
                log(f"🔗 OKX现有持仓({len(pos_list)}): {', '.join(sorted(pos_list))}")
                state['okx_positions_at_start'] = sorted(pos_list)
            else:
                log("🔗 OKX现有持仓: 空仓")
                state['okx_positions_at_start'] = []
        except Exception as e:
            log(f"⚠️ OKX持仓同步失败: {e}")
            state['okx_positions_at_start'] = []

    # ── 🛡️ 启动时检查所有持仓SL/TP保护 ──
    if trading_mode.is_real:
        try:
            from leverage_engine import LeverageEngine
            sltp_le = LeverageEngine()
            sltp_status = sltp_le.check_all_positions_sl_tp()
            if sltp_status.get("naked_positions"):
                log(f"🚨 发现裸奔仓位: {', '.join(sltp_status['naked_positions'])}")
                log("  ⚠️ 这些仓位没有SL/TP保护!")
            if sltp_status.get("protected_positions"):
                for sym, prot in sltp_status["protected_positions"].items():
                    log(f"  🛡️ {sym}: SL={prot.get('sl','?')} TP={prot.get('tp','?')}")
            if sltp_status.get("all_protected", False) and sltp_status.get("protected_positions"):
                log(f"✅ 全部{len(sltp_status['protected_positions'])}个持仓均有SL/TP保护")
        except Exception as e:
            log(f"⚠️ SL/TP检查异常: {e}")

    # 启动通知
    push_wechat(
        f"🐾 Yina自主交易已启动 [{mode_label}]",
        f"> 🚀 交易引擎已就绪\n"
        f"> 📡 [查看仪表板](https://runs-student-skill-seeds.trycloudflare.com)\n"
        f"> ⏱️ 扫描间隔: {SCAN_INTERVAL_MINUTES}分钟 | 推送间隔: {PUSH_INTERVAL_MINUTES}分钟\n"
        f"> 🧠 引擎: v5 Qlib融合 + 图增强 + Alpha + SMART拆单\n"
        f"> 🎮 模式: {mode_label}\n"
        f"> ⏰ {beijing_now().strftime('%Y-%m-%d %H:%M')}"
    )

    def run_cycle():
        nonlocal state
        cycle_start = time.time()
        state["cycles"] += 1
        state["today_cycles"] = state.get("today_cycles", 0) + 1

        # 🆕 冷却期管理: 每周期递减计数器，过期移除
        cooldown = state.get('stopped_out_cooldown', {})
        if cooldown:
            expired = []
            for sym in list(cooldown.keys()):
                cooldown[sym] -= 1
                if cooldown[sym] <= 0:
                    expired.append(sym)
            for sym in expired:
                del cooldown[sym]
                log(f"  🧊 冷却期结束: {sym} 重新纳入候选")
            state['stopped_out_cooldown'] = cooldown

        # 🛡️ 每轮扫描前: 检查所有持仓SL/TP保护 (每5轮≈50分钟检查一次)
        if trading_mode.is_real and state["cycles"] % 5 == 1:
            try:
                from leverage_engine import LeverageEngine
                sltp_le = LeverageEngine()
                sltp_status = sltp_le.check_all_positions_sl_tp()
                naked = sltp_status.get("naked_positions", [])
                if naked:
                    log(f"🚨 裸奔仓位检测: {', '.join(naked)} — 不补设，等待下一轮新开仓自带SL/TP")
                else:
                    protected = sltp_status.get("protected_positions", {})
                    if protected:
                        log(f"🛡️ SL/TP全部在线({len(protected)}个仓位)")
            except Exception as e:
                log(f"⚠️ 周期SL/TP检查异常: {e}")

        log(f"\n{'='*50}")
        log(f"🔄 第 {state['cycles']} 次扫描 [{beijing_now().strftime('%H:%M')}]")
        log(f"{'='*50}")

        result = run_trade_cycle(state, trading_mode)

        # 🆕 止损冷却追踪: 将本周期被止损/强平的币种加入冷却名单
        global _stopped_out_this_cycle
        if _stopped_out_this_cycle:
            cooldown = state.get('stopped_out_cooldown', {})
            for sym in _stopped_out_this_cycle:
                cooldown[sym] = STOPPED_OUT_COOLDOWN_CYCLES
                log(f"  🧊 加入冷却: {sym} (跳过{STOPPED_OUT_COOLDOWN_CYCLES}个周期)")
            state['stopped_out_cooldown'] = cooldown
        _stopped_out_this_cycle = set()  # 清空本周期记录

        if result["ok"]:
            pf = result["pf"]
            results = result["results"]

            n_trades = len(results)
            if n_trades > 0:
                state["total_trades"] += n_trades
                state["today_trades"] = state.get("today_trades", 0) + n_trades

            # 🆕 每30分钟推送一次 (有交易才推, 避免刷屏)
            if n_trades > 0:
                now_ts = time.time()
                last_push = state.get("last_push_ts", 0)
                if now_ts - last_push >= PUSH_INTERVAL_MINUTES * 60:
                    log(f"📋 {n_trades} 笔操作, 推送企微...")
                    summary = build_trade_summary(results, pf, beijing_now().strftime("%Y-%m-%d %H:%M"))
                    push_wechat("📊 Yina交易通知", summary)
                    state["last_push_ts"] = now_ts
                else:
                    remaining = PUSH_INTERVAL_MINUTES * 60 - (now_ts - last_push)
                    log(f"📋 {n_trades} 笔操作 (下次推送 {remaining/60:.0f}分钟后)")
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
