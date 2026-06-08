"""
Yina 自主模拟盘交易守护进程 🐾
==============================
持续运行: 扫单 → 决策 → 执行 → 企微通知
每30分钟扫描一次, 有交易时推送企微简报, 每日22:00发送日报

使用:
  python3 auto_trade_daemon.py              # 前台运行
  python3 auto_trade_daemon.py --daemon     # 后台运行
  python3 auto_trade_daemon.py --once       # 单次运行+推送
"""
from __future__ import annotations
import sys
import os
import json
import time
import signal
import traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

# ── 配置 ──
SCAN_INTERVAL_MINUTES = 30  # 扫描间隔
DATA_DIR = Path(__file__).parent / "data"
LOG_DIR = DATA_DIR / "daemon_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = DATA_DIR / "daemon_state.json"

# 夜间休眠: 北京时间 01:00-07:00 暂停扫描 (加密市场仍开盘但波动低)
SLEEP_START_HOUR = 1
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


def run_trade_cycle(state: dict) -> dict:
    """执行一次交易周期"""
    from portfolio import PortfolioManager
    from execution import ExecutionConfig

    # 构建执行配置: SMART 自适应策略
    exec_cfg = ExecutionConfig(
        strategy="smart",
        horizon_minutes=60,
        n_slices=0,  # 自动最优
        urgency=0.5,
    )

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
            )
            log(f"🚀 引擎: {trader.engine_label}")
            results, pf = trader.run()
        else:
            from auto_trade import MLAutoTrader, ML_SIGNAL_AVAILABLE
            if ML_SIGNAL_AVAILABLE:
                trader = MLAutoTrader(use_lgbm=True)
                from auto_trade import auto_scan_and_trade
                results, pf = auto_scan_and_trade(["crypto"], use_ml=True, use_rolling=False)
            else:
                from auto_trade import auto_scan_and_trade
                results, pf = auto_scan_and_trade(["crypto"], use_ml=False, use_rolling=False)

        # ── A股 + 美股传统信号扫描 (ML引擎不覆盖) ──
        try:
            from auto_trade import auto_scan_and_trade
            trad_results, pf = auto_scan_and_trade(
                ["a_stock", "us_stock"], use_ml=False, use_rolling=False
            )
            if trad_results:
                results.extend(trad_results)
                log(f"📊 A股/美股: {len(trad_results)} 笔信号")
        except Exception as e:
            log(f"⚠️ A股/美股扫描异常: {e}")

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
    args = parser.parse_args()

    log("🐾 Yina 自主模拟盘交易守护进程 启动!")
    log(f"📡 API: http://localhost:8766")
    log(f"⏱️  扫描间隔: {SCAN_INTERVAL_MINUTES}分钟")
    log(f"🌙 夜间休眠: {SLEEP_START_HOUR:02d}:00-{SLEEP_END_HOUR:02d}:00 (北京时间)")

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
        "🐾 Yina自主交易已启动",
        f"> 🚀 交易引擎已就绪\n"
        f"> 📡 [查看仪表板](https://runs-student-skill-seeds.trycloudflare.com)\n"
        f"> ⏱️ 扫描间隔: {SCAN_INTERVAL_MINUTES}分钟\n"
        f"> 🧠 引擎: v5 Qlib融合 + 图增强 + Alpha + SMART拆单\n"
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

        result = run_trade_cycle(state)

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
