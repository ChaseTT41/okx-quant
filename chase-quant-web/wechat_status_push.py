"""
Yina OKX 实盘状态定时推送 🐾
每10分钟推送到企业微信「金融监控」群
"""
import os, sys, json, hmac, base64, requests, urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).parent


def load_env():
    env_path = PROJECT_DIR / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())


def okx_get(path):
    api_key = os.environ['OKX_API_KEY']
    secret = os.environ['OKX_SECRET_KEY']
    passphrase = os.environ['OKX_PASSPHRASE']
    base_url = os.environ['OKX_API_URL']

    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    sign_str = ts + 'GET' + path + ''
    sign = base64.b64encode(hmac.new(
        secret.encode(), sign_str.encode(), 'sha256'
    ).digest()).decode()

    r = requests.get(f'{base_url}{path}', headers={
        'OK-ACCESS-KEY': api_key, 'OK-ACCESS-SIGN': sign,
        'OK-ACCESS-TIMESTAMP': ts, 'OK-ACCESS-PASSPHRASE': passphrase,
    }, timeout=10)
    return r.json()


def push_wechat(markdown_content: str) -> bool:
    webhook_key = os.environ.get('WECHAT_WEBHOOK_KEY', '')
    if not webhook_key:
        return False

    payload = json.dumps({
        "msgtype": "markdown",
        "markdown": {"content": markdown_content}
    }).encode()

    req = urllib.request.Request(
        f"https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={webhook_key}",
        data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        result = json.loads(resp.read())
        return result.get('errcode') == 0


def get_fear_greed():
    try:
        r = requests.get('https://api.alternative.me/fng/?limit=1', timeout=10)
        d = r.json()['data'][0]
        return d['value'], d['value_classification']
    except Exception:
        return '?', '?'


def get_ticker_price(symbols):
    """Batch fetch prices via OKX public ticker"""
    try:
        base_url = os.environ['OKX_API_URL']
        r = requests.get(f'{base_url}/api/v5/market/tickers?instType=SWAP', timeout=10)
        data = r.json()
        prices = {}
        if data.get('code') == '0':
            for t in data['data']:
                inst = t['instId'].replace('-USDT-SWAP', '')
                prices[inst] = float(t['last'])
        return prices
    except Exception:
        return {}


def format_pnl(upl: float) -> str:
    if upl > 0:
        return f"<font color=\"info\">${upl:+.4f}</font>"
    elif upl < 0:
        return f"<font color=\"warning\">${upl:+.4f}</font>"
    return f"${upl:+.4f}"


def build_status():
    beijing_now = datetime.now(timezone.utc) + timedelta(hours=8)
    time_str = beijing_now.strftime('%m-%d %H:%M')

    # Positions
    pos_data = okx_get('/api/v5/account/positions?instType=SWAP')
    positions = []
    if pos_data.get('code') == '0':
        positions = [p for p in pos_data.get('data', []) if float(p['pos']) > 0]

    # Balance
    bal = okx_get('/api/v5/account/balance')
    equity = '?'
    avail = '?'
    if bal.get('code') == '0':
        d = bal['data'][0]
        equity = f"${float(d.get('totalEq', 0)):.2f}"
        usdt = next((x for x in d['details'] if x['ccy'] == 'USDT'), {})
        avail = f"${float(usdt.get('availBal', 0)):.2f}"

    # Fear & Greed
    fg_val, fg_class = get_fear_greed()

    # Daemon check
    daemon_running = False
    try:
        out = os.popen('ps aux | grep "auto_trade_daemon.*--live" | grep -v grep').read()
        daemon_running = bool(out.strip())
    except Exception:
        pass

    # Last scan from log
    last_scan = 'N/A'
    log_file = Path('/tmp/yina_daemon_live.log')
    if log_file.exists():
        with open(log_file) as f:
            lines = f.readlines()
            for l in reversed(lines):
                if '耗时:' in l:
                    last_scan = l.strip()[-40:]
                    break

    # PnL
    pnl_today = 0.0
    try:
        pnl_path = PROJECT_DIR / 'data' / 'live_risk_state.json'
        if pnl_path.exists():
            with open(pnl_path) as f:
                rs = json.load(f)
                pnl_today = float(rs.get('daily_pnl', 0))
    except Exception:
        pass

    # Build message
    lines = [
        f"## 🐾 Yina OKX 实盘 `{time_str}`",
        "",
        f"守护进程: {'🟢 运行中' if daemon_running else '🔴 已停止'} | {last_scan}",
        "",
        f"💰 权益: **{equity}** | 可用: {avail}",
        f"😱 F&G: **{fg_val}** ({fg_class}) | 今日Pnl: {format_pnl(pnl_today)}",
        "",
    ]

    if positions:
        lines.append("### 📊 持仓")
        total_margin = 0
        total_upl = 0
        for p in positions:
            margin = float(p['margin'])
            upl = float(p.get('upl', 0))
            total_margin += margin
            total_upl += upl
            name = p['instId'].replace('-USDT-SWAP', '')
            lines.append(
                f"> **{name}** {p['posSide']} {p['lever']}x | "
                f"{p['pos']}张 @ ${p['avgPx']} | "
                f"保证金${margin:.2f} | UPL {format_pnl(upl)}"
            )
        lines.append(f"> ")
        lines.append(f"> 📌 总保证金: **${total_margin:.2f}** | 总UPL: {format_pnl(total_upl)}")
    else:
        lines.append("> 📭 暂无持仓，等待信号中...")

    lines += [
        "",
        "---",
        f"> 🛡️ 信号驱动 | 最大杠杆10x | 回撤熔断10%",
        f"> ⚡ 胜率62% | ⏰ 下次推送 ~{(beijing_now + timedelta(minutes=10)).strftime('%H:%M')}",
    ]

    return "\n".join(lines)


if __name__ == '__main__':
    load_env()
    content = build_status()
    ok = push_wechat(content)
    if ok:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 企微推送成功")
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 推送失败")
