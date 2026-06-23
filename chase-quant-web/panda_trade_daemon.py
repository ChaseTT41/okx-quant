#!/usr/bin/env python3
"""
🐼 熊猫教练策略实盘交易守护进程

Chase哥 2026-06-21 要求:
  - 10u保证金/笔，杠杆按信心浮动
  - 按企微推送频率运行（每30分钟扫描）
  - 三指标共振 + 多时间框架对齐 → 入场
  - "趋势一旦破坏，无条件空仓"

用法:
  python3 panda_trade_daemon.py               # 前台运行
  python3 panda_trade_daemon.py --daemon      # 后台守护进程
  python3 panda_trade_daemon.py --once        # 单次扫描
  python3 panda_trade_daemon.py --once --push # 单次扫描+推送
"""

import sys
import os
import json
import time
import signal
import hashlib
import hmac
import base64
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

import requests

# 项目路径
PROJ_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJ_DIR))

# 加载 .env
try:
    from dotenv import load_dotenv
    load_dotenv(PROJ_DIR / ".env")
except ImportError:
    pass

from panda_trade_strategy import (
    PandaSignal, analyze_coin, scan_all, push_to_wecom,
    WATCH_COINS, MARGIN_PER_TRADE, MIN_CONFIDENCE, MAX_POSITIONS,
    LEVERAGE_TABLE, STOP_LOSS_ATR_MULT,
)

# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════

SCAN_INTERVAL_MINUTES = 5  # 扫描间隔（分钟）✨ 从10改为5，更快捕捉信号
LOG_FILE = PROJ_DIR / "data" / "daemon_logs" / "panda_daemon.log"
STATE_FILE = PROJ_DIR / "data" / "panda_state.json"
OKX_HOST = "www.okx.cab"

# WeCom
WECOM_WEBHOOK = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=2c602b48-5da2-4989-9193-30c0e226c769"

# 北京时间
def beijing_now():
    return datetime.now(timezone.utc) + timedelta(hours=8)


# ═══════════════════════════════════════════════════════════
# OKX 交易接口
# ═══════════════════════════════════════════════════════════

class PandaOKXTrader:
    """熊猫策略专用的 OKX 交易执行器"""

    def __init__(self):
        self._api_key = os.environ.get("OKX_API_KEY", "")
        self._secret = os.environ.get("OKX_SECRET_KEY", "")
        self._passphrase = os.environ.get("OKX_PASSPHRASE", "")
        self._ok = bool(self._api_key and self._secret and self._passphrase)

    def _sign(self, method: str, path: str, body: str = "") -> tuple:
        """OKX V5 签名"""
        ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        sign_str = ts + method + path + body
        sign = base64.b64encode(hmac.new(
            self._secret.encode(), sign_str.encode(), "sha256"
        ).digest()).decode()
        return ts, sign

    def _headers(self, method: str, path: str, body: str = ""):
        ts, sign = self._sign(method, path, body)
        return {
            "OK-ACCESS-KEY": self._api_key,
            "OK-ACCESS-SIGN": sign,
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": self._passphrase,
            "Content-Type": "application/json",
        }

    def _inst_id(self, symbol: str) -> str:
        """BTC → BTC-USDT-SWAP"""
        return f"{symbol}-USDT-SWAP"

    def fetch_equity(self) -> float:
        """获取账户权益"""
        if not self._ok:
            return 0.0
        try:
            path = "/api/v5/account/balance"
            r = requests.get(f"https://{OKX_HOST}{path}",
                           headers=self._headers("GET", path), timeout=10)
            data = r.json()
            if data.get("code") == "0" and data.get("data"):
                return float(data["data"][0].get("totalEq", 0))
            return 0.0
        except Exception as e:
            print(f"⚠️ 获取余额失败: {e}")
            return 0.0

    def fetch_positions(self) -> dict:
        """获取当前持仓 {symbol: {pos, posSide, avgPx, lever, upl}}"""
        if not self._ok:
            return {}
        try:
            path = "/api/v5/account/positions?instType=SWAP"
            r = requests.get(f"https://{OKX_HOST}{path}",
                           headers=self._headers("GET", path), timeout=10)
            data = r.json()
            positions = {}
            if data.get("code") == "0":
                for p in data.get("data", []):
                    qty = float(p.get("pos", 0))
                    if qty > 0:
                        inst_id = p.get("instId", "")
                        sym = inst_id.replace("-USDT-SWAP", "")
                        positions[sym] = {
                            "pos": qty,
                            "posSide": p.get("posSide", "long"),
                            "avgPx": float(p.get("avgPx", 0)),
                            "lever": int(float(p.get("lever", 1))),
                            "upl": float(p.get("upl", 0)),
                            "margin": float(p.get("margin", 0)),
                        }
            return positions
        except Exception as e:
            print(f"⚠️ 获取持仓失败: {e}")
            return {}

    def set_leverage(self, symbol: str, leverage: int, pos_side: str = "long") -> bool:
        """设置杠杆倍数"""
        if not self._ok:
            return False
        try:
            body = json.dumps({
                "instId": self._inst_id(symbol),
                "lever": str(leverage),
                "mgnMode": "isolated",
                "posSide": pos_side,
            })
            path = "/api/v5/account/set-leverage"
            r = requests.post(f"https://{OKX_HOST}{path}",
                            headers=self._headers("POST", path, body),
                            data=body, timeout=10)
            data = r.json()
            if data.get("code") == "0":
                print(f"  ⚡ {symbol} 杠杆设为 {leverage}x {pos_side}")
                return True
            print(f"  ⚠️ 设置杠杆失败 {symbol}: {data.get('msg', '')}")
            return False
        except Exception as e:
            print(f"  ❌ 设置杠杆异常 {symbol}: {e}")
            return False

    def place_market_order(self, symbol: str, side: str, quantity_ct: float,
                           leverage: int, sl_price: float = None) -> Optional[str]:
        """
        下市价单

        Args:
          symbol: 'BTC' (不用带/USDT)
          side: 'buy' 做多, 'sell' 做空
          quantity_ct: 合约张数
          leverage: 杠杆倍数
          sl_price: 止损价（可选）

        Returns: order_id or None
        """
        if not self._ok:
            print(f"  ❌ OKX未配置，跳过 {symbol} {side}")
            return None

        try:
            # 1. 设杠杆
            pos_side = "long" if side == "buy" else "short"
            self.set_leverage(symbol, leverage, pos_side)

            # 2. 下市价单
            ts_suffix = str(int(time.time() * 1000))[-12:]
            body = json.dumps({
                "instId": self._inst_id(symbol),
                "tdMode": "isolated",
                "side": side,
                "posSide": pos_side,
                "ordType": "market",
                "sz": str(quantity_ct),
                "lever": str(leverage),
                "clOrdId": f"panda{ts_suffix}",
            })
            path = "/api/v5/trade/order"
            r = requests.post(f"https://{OKX_HOST}{path}",
                            headers=self._headers("POST", path, body),
                            data=body, timeout=10)
            data = r.json()
            if data.get("code") == "0":
                ord_id = data["data"][0].get("ordId", "")
                print(f"  ✅ {symbol} {side} {quantity_ct}张 @{leverage}x → {ord_id}")

                # 3. 设止损（如果有）
                if sl_price:
                    self._place_stop_loss(symbol, pos_side, quantity_ct, sl_price)

                return ord_id
            else:
                print(f"  ❌ 下单失败 {symbol}: {data.get('msg', '')}")
                return None
        except Exception as e:
            print(f"  ❌ 下单异常 {symbol}: {e}")
            return None

    def _place_stop_loss(self, symbol: str, pos_side: str, sz: float, sl_price: float):
        """设止损单 (algo order)"""
        try:
            body = json.dumps({
                "instId": self._inst_id(symbol),
                "tdMode": "isolated",
                "side": "sell" if pos_side == "long" else "buy",
                "posSide": pos_side,
                "ordType": "conditional",
                "sz": str(sz),
                "slTriggerPx": str(round(sl_price, 2)),
                "slOrdPx": str(round(sl_price * 0.995, 2)),
            })
            path = "/api/v5/trade/order-algo"
            r = requests.post(f"https://{OKX_HOST}{path}",
                            headers=self._headers("POST", path, body),
                            data=body, timeout=10)
            data = r.json()
            if data.get("code") == "0":
                print(f"  🛡️ 止损设于 {sl_price:.2f}")
            else:
                print(f"  ⚠️ 止损设置失败: {data.get('msg', '')}")
        except Exception as e:
            print(f"  ⚠️ 止损异常: {e}")

    def close_position(self, symbol: str, pos_side: str, sz: float) -> bool:
        """平仓"""
        try:
            body = json.dumps({
                "instId": self._inst_id(symbol),
                "tdMode": "isolated",
                "side": "sell" if pos_side == "long" else "buy",
                "posSide": pos_side,
                "ordType": "market",
                "sz": str(sz),
            })
            path = "/api/v5/trade/order"
            r = requests.post(f"https://{OKX_HOST}{path}",
                            headers=self._headers("POST", path, body),
                            data=body, timeout=10)
            data = r.json()
            if data.get("code") == "0":
                print(f"  🔒 平仓 {symbol} {pos_side} {sz}张")
                return True
            print(f"  ⚠️ 平仓失败: {data.get('msg', '')}")
            return False
        except Exception as e:
            print(f"  ❌ 平仓异常: {e}")
            return False


# ═══════════════════════════════════════════════════════════
# 仓位管理
# ═══════════════════════════════════════════════════════════

def load_state() -> dict:
    """加载持久化状态"""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except:
            pass
    return {
        "trades_today": 0,
        "total_trades": 0,
        "trade_history": [],
        "last_scan": "",
    }


def save_state(state: dict):
    """保存状态"""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def decide_trade(signal: PandaSignal, positions: dict, state: dict,
                 trader: PandaOKXTrader) -> Optional[dict]:
    """
    根据熊猫策略信号决定是否交易

    Returns: trade record dict or None
    """
    if not signal.entry_ready:
        return None

    symbol = signal.symbol

    # ── 持仓检查 ──
    if symbol in positions:
        print(f"  ⏭️ {symbol} 已有持仓，跳过")
        return None

    if len(positions) >= MAX_POSITIONS:
        print(f"  🛑 已达最大持仓数 {MAX_POSITIONS}")
        return None

    # ── 保证金计算 ──
    margin = MARGIN_PER_TRADE
    leverage = signal.suggested_leverage
    notional = margin * leverage  # 名义价值

    # 检查余额
    equity = trader.fetch_equity()
    if equity < margin * 1.5:
        print(f"  💸 余额不足: ${equity:.2f} < ${margin * 1.5:.0f}")
        return None

    # ── 获取当前价格计算合约张数 ──
    try:
        clean = symbol.split("/")[0]  # "BTC/USDT" → "BTC"
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={clean}USDT"
        price = float(requests.get(url, timeout=5).json()["price"])
    except:
        print(f"  ⚠️ 无法获取 {symbol} 价格")
        return None

    # OKX合约: 1张 = 0.01 BTC/ETH, 1 SOL (根据币种不同)
    ct_size_map = {"BTC": 0.01, "ETH": 0.1, "SOL": 1.0}
    ct_size = ct_size_map.get(symbol, 0.01)
    quantity_ct = notional / price  # 名义价值/价格 = 币数量
    quantity_ct = int(quantity_ct / ct_size) * ct_size  # 向下取到合约单位
    if quantity_ct < ct_size:
        quantity_ct = ct_size  # 至少1张

    # ── 止损价 ──
    sl_pct = signal.stop_loss_pct / 100
    sl_price = price * (1 - sl_pct)  # 做多止损在下

    # ── 执行交易 ──
    side = "buy"  # 做多
    if signal.direction == "short":
        side = "sell"
        sl_price = price * (1 + sl_pct)  # 做空止损在上

    print(f"\n🚀 熊猫策略入场!")
    print(f"  {symbol} {side} | 保证金${margin} | {leverage}x | 名义${notional:.0f}")
    print(f"  入场价: ${price:.2f} | 止损: ${sl_price:.2f} | 合约: {quantity_ct}张")

    order_id = trader.place_market_order(
        symbol, side, quantity_ct, leverage, sl_price
    )

    if order_id:
        trade = {
            "time": beijing_now().isoformat(),
            "symbol": symbol,
            "side": side,
            "margin": margin,
            "leverage": leverage,
            "notional": notional,
            "price": price,
            "quantity_ct": quantity_ct,
            "sl_price": sl_price,
            "confidence": signal.confidence,
            "order_id": order_id,
        }
        state["trades_today"] += 1
        state["total_trades"] += 1
        state["trade_history"].append(trade)

        # 推送到企微
        _push_trade_alert(trade, signal)

        return trade

    return None


def _push_trade_alert(trade: dict, signal: PandaSignal):
    """推送交易提醒到企微"""
    lines = [
        "🐼 **熊猫策略开仓!**",
        f"> 币种: **{trade['symbol']}** | {'🟢做多' if trade['side'] == 'buy' else '🔴做空'}",
        f"> 保证金: ${trade['margin']} | 杠杆: {trade['leverage']}x",
        f"> 入场价: ${trade['price']:.2f} | 止损: ${trade['sl_price']:.2f}",
        f"> 信心分: {signal.confidence:.0f}/100",
        f"> 共振: {signal.tf_alignment_detail}",
        f"> 时间: {trade['time'][:19]}",
    ]
    if signal.has_divergence:
        lines.append(f"> ⚡ 背离确认: {signal.divergence_type}")

    try:
        requests.post(WECOM_WEBHOOK, json={
            "msgtype": "markdown",
            "markdown": {"content": "\n".join(lines)}
        }, timeout=10)
    except:
        pass


# ═══════════════════════════════════════════════════════════
# 持仓监控
# ═══════════════════════════════════════════════════════════

def monitor_positions(trader: PandaOKXTrader, state: dict):
    """监控现有持仓，检查止盈止损条件"""
    positions = trader.fetch_positions()

    for sym, pos in positions.items():
        upl_pct = pos['upl'] / pos['margin'] * 100 if pos['margin'] > 0 else 0
        print(f"  📦 {sym} {pos['posSide']} | UPL: {upl_pct:+.1f}% | 杠杆: {pos['lever']}x")

        # ── 止盈: UPL > 30% ──
        if upl_pct > 30:
            print(f"  🎯 {sym} 盈利{upl_pct:.0f}%，建议止盈!")

        # ── 止损警告: UPL < -15% ──
        if upl_pct < -15:
            print(f"  ⚠️ {sym} 亏损{upl_pct:.0f}%，接近止损!")


# ═══════════════════════════════════════════════════════════
# 主循环
# ═══════════════════════════════════════════════════════════

def run_scan_cycle(state: dict, trader: PandaOKXTrader, push: bool = True):
    """执行一次完整的扫描周期"""
    now = beijing_now()
    scan_time = now.strftime("%m-%d %H:%M")
    print(f"\n{'='*60}")
    print(f"🐼 熊猫策略扫描 [{scan_time}]")
    print(f"{'='*60}")

    # 1. 检查账户
    equity = trader.fetch_equity()
    positions = trader.fetch_positions()
    print(f"💰 权益: ${equity:.2f} | 持仓: {len(positions)}个")

    if positions:
        monitor_positions(trader, state)

    # 2. 扫描信号
    print(f"\n🔍 扫描 {', '.join(WATCH_COINS)}...")
    signals = scan_all()

    # 3. 检查入场
    trades_made = []
    for sig in signals:
        if sig.entry_ready:
            trade = decide_trade(sig, positions, state, trader)
            if trade:
                trades_made.append(trade)
                # 更新持仓列表（避免同币种重复开仓）
                positions[sig.symbol] = {"pos": 1}

    # 4. 推送报告
    if push and not trades_made:
        # 只在有值得关注的信号时推送
        interesting = [s for s in signals if s.confidence >= 40]
        if interesting:
            push_to_wecom(signals)

    # 5. 保存状态
    state["last_scan"] = scan_time
    save_state(state)

    print(f"\n⏱️ 本次扫描完成 | {'🟢 入场!' if trades_made else '💤 无入场信号'}")

    return len(trades_made)


def run_daemon():
    """后台守护进程模式"""
    print("🐼 熊猫策略实盘守护进程启动...")
    print(f"   保证金/笔: ${MARGIN_PER_TRADE}")
    print(f"   最低信心: {MIN_CONFIDENCE}分")
    print(f"   最多持仓: {MAX_POSITIONS}个")
    print(f"   扫描间隔: {SCAN_INTERVAL_MINUTES}分钟")
    print(f"   关注币种: {', '.join(WATCH_COINS)}")

    # 确保日志目录存在
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    trader = PandaOKXTrader()
    if not trader._ok:
        print("⚠️ OKX API未配置! 仅做信号分析，不执行交易")

    state = load_state()
    cycle = 0

    # 信号处理
    def handle_signal(signum, frame):
        print(f"\n🐼 收到停止信号，保存状态...")
        save_state(state)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    while True:
        cycle += 1
        try:
            trades = run_scan_cycle(state, trader, push=True)

            # 写日志
            log_line = f"[{beijing_now().strftime('%m-%d %H:%M:%S')}] 第{cycle}次 | 权益${trader.fetch_equity():.2f} | {'入场!' if trades else '观望'}\n"
            with open(LOG_FILE, "a") as f:
                f.write(log_line)

        except Exception as e:
            print(f"❌ 扫描异常: {e}")
            import traceback
            traceback.print_exc()

        # 等下次扫描
        print(f"\n⏳ 等待 {SCAN_INTERVAL_MINUTES} 分钟后下次扫描...")
        time.sleep(SCAN_INTERVAL_MINUTES * 60)


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="🐼 熊猫策略实盘交易守护进程")
    p.add_argument("--daemon", action="store_true", help="后台守护进程模式")
    p.add_argument("--once", action="store_true", help="单次扫描")
    p.add_argument("--push", action="store_true", help="推送到企业微信")
    p.add_argument("--interval", type=int, default=SCAN_INTERVAL_MINUTES,
                   help=f"扫描间隔分钟 (默认{SCAN_INTERVAL_MINUTES})")
    args = p.parse_args()

    if args.interval != SCAN_INTERVAL_MINUTES:
        SCAN_INTERVAL_MINUTES = args.interval

    if args.daemon:
        run_daemon()
    elif args.once:
        trader = PandaOKXTrader()
        state = load_state()
        run_scan_cycle(state, trader, push=args.push)
    else:
        # 默认前台运行一次
        trader = PandaOKXTrader()
        state = load_state()
        run_scan_cycle(state, trader, push=True)
