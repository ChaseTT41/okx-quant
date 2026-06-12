#!/usr/bin/env python3
"""量化仪表板 → 企业微信定时推送
每2小时抓取 portfolio + 最近交易，按 Chase哥 偏好的简洁格式推送
"""
import json, urllib.request, urllib.error, sys
from datetime import datetime, timezone, timedelta

API_BASE = "http://localhost:8766"
WEBHOOK_URL = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=2c602b48-5da2-4989-9193-30c0e226c769"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
TZ = timezone(timedelta(hours=8))  # 北京时间

MARKET_NAMES = {
    "crypto": "₿ 加密货币",
    "a_stock": "🇨🇳 A股",
    "us_stock": "🇺🇸 美股",
    "hk_stock": "🇭🇰 港股",
}


def fetch(endpoint):
    url = f"{API_BASE}{endpoint}"
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def fmt_money(v):
    if abs(v) < 1:
        return f"¥{v:.2f}"
    return f"¥{v:,.0f}"


def fmt_pnl(v):
    sign = "+" if v >= 0 else ""
    return f"{sign}¥{v:,.0f}"


def build_markdown(portfolio, trades):
    now = datetime.now(TZ).strftime("%m/%d %H:%M")
    lines = [f"## 📊 量化战报 <font color=\"comment\">{now}</font>", ""]

    # 总览
    p = portfolio
    pnl = p.get("total_pnl", 0)
    pnl_pct = p.get("total_pnl_pct", 0)
    emoji = "🟢" if pnl >= 0 else "🔴"
    lines.append(f"### {emoji} 总资产 {fmt_money(p.get('total_value', 0))} ｜ 盈亏 {fmt_pnl(pnl)}（{pnl_pct:+.1f}%）")
    lines.append("")

    # 各市场账户
    markets = p.get("markets", [])
    for m in markets:
        name = MARKET_NAMES.get(m["market"], m["market"])
        cash = m["cash"]
        pos_val = m["positions_value"]
        total = m["total"]
        pnl_m = m["pnl"]
        pnl_pct_m = m["pnl_pct"]

        lines.append(f"**{name}**")
        lines.append(f"> 💵 现金 {fmt_money(cash)} ｜ 📦 持仓 {fmt_money(pos_val)}")
        lines.append(f"> 合计 {fmt_money(total)} ｜ 盈亏 {fmt_pnl(pnl_m)}（{pnl_pct_m:+.1f}%）")
        lines.append("")

    # 最近交易
    trades_list = trades.get("trades", [])[:5]
    if trades_list:
        lines.append("### 📋 最近交易")
        lines.append("")
        for t in trades_list:
            side = "🟢买入" if t["side"] == "buy" else "🔴卖出"
            t_time = datetime.fromisoformat(t["time"].replace("Z", "+00:00")).astimezone(TZ).strftime("%m/%d %H:%M")
            lines.append(f"> {side} **{t['name']}**（{t['symbol']}）")
            lines.append(f"> {t_time} ｜ {fmt_money(t['amount'])} ｜ {t.get('reason', '')[:60]}")
            if t.get("pnl"):
                pnl_line = fmt_pnl(t["pnl"])
                lines.append(f"> 盈亏: {pnl_line}")
            lines.append("")

    # 风控提示
    risk = p.get("risk", {}) if isinstance(p, dict) else {}
    warnings = risk.get("warnings", [])
    if warnings:
        lines.append("### ⚠️ 风控提醒")
        for w in warnings:
            lines.append(f"> {w}")
        lines.append("")

    # 月目标进度
    progress = p.get("monthly_progress_pct", 0)
    lines.append(f"🎯 月目标 30% ｜ 当前 {progress:.0f}%")
    lines.append("")
    lines.append("[打开仪表板](https://long-equity-houston-checkout.trycloudflare.com)")

    return "\n".join(lines)


def send_to_wecom(md_content):
    payload = json.dumps({
        "msgtype": "markdown",
        "markdown": {"content": md_content}
    }).encode("utf-8")

    req = urllib.request.Request(
        WEBHOOK_URL,
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            return result.get("errcode") == 0
    except Exception as e:
        print(f"推送失败: {e}", file=sys.stderr)
        return False


def main():
    portfolio = fetch("/api/portfolio")
    trades = fetch("/api/trades?limit=5")

    if "error" in portfolio:
        print(f"获取 portfolio 失败: {portfolio['error']}", file=sys.stderr)
        sys.exit(1)

    md = build_markdown(portfolio, trades)
    ok = send_to_wecom(md)

    if ok:
        print("✅ 量化战报已推送到企业微信")
    else:
        print("❌ 推送失败", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
