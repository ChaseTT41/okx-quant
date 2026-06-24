"""
🐺 激进交易策略 — 目标月化30%
多时间框架 RSI + MACD + 布林带
1h/4h 信号确认, 5分钟扫描
大仓位, 快进快出
"""
import ccxt, pandas as pd, numpy as np, time, json, os, traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

# ── 配置 ──
DATA_DIR = Path(__file__).parent / "data"
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
TRADES_FILE = DATA_DIR / "trades.json"

# 从统一配置中心加载 Tier1 ML深度扫描标的 (30+主流币)
try:
    from symbol_config import get_all_crypto_symbols
    SYMBOLS = get_all_crypto_symbols(tiers=[1])  # Tier1 ML深度扫描
except ImportError:
    SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
SCAN_INTERVAL = 300  # 5分钟

# 激进参数
RSI_OVERSOLD = 40      # RSI < 40 → 超卖加分
RSI_OVERBOUGHT = 65    # RSI > 65 → 超买减分
STOP_LOSS_PCT = -0.05  # -5% 硬止损
TAKE_PROFIT_PCT = 0.08 # +8% 止盈
POSITION_PCT = 0.30    # 每次用30%现金建仓
MIN_TRADE = 200        # 最低¥200

BJ_TZ = timezone(timedelta(hours=8))


def bj_now():
    return datetime.now(BJ_TZ)


# ── 数据获取 ──
_exchange = None


def get_exchange():
    global _exchange
    if _exchange is None:
        _exchange = ccxt.binance({"enableRateLimit": True, "timeout": 15000})
    return _exchange


def fetch_ohlcv(symbol: str, timeframe: str = "1h", limit: int = 200) -> Optional[pd.DataFrame]:
    try:
        ex = get_exchange()
        ohlcv = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        return df
    except Exception as e:
        print(f"  ⚠️ {symbol} {timeframe} 数据获取失败: {e}")
        return None


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算 RSI, MACD, 布林带, ATR"""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    # RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # 布林带 (20)
    df["bb_mid"] = pd.Series(close).rolling(20).mean()
    bb_std = pd.Series(close).rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    # ATR (14)
    tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    df["atr"] = pd.Series(tr).rolling(14).mean()

    # 短期动量
    df["ret_4h"] = close / np.roll(close, 4) - 1
    df["ret_12h"] = close / np.roll(close, 12) - 1

    return df


@dataclass
class TradeSignal:
    symbol: str
    name: str
    action: str  # BUY / SELL
    price: float
    score: float  # 0-100
    reason: str
    rsi_1h: float
    rsi_4h: float
    stop_loss: float
    take_profit: float


def scan_symbol(symbol: str) -> Optional[TradeSignal]:
    """多时间框架扫描单个币种"""
    name = symbol.replace("/USDT", "")

    # 1h 数据
    df_1h = fetch_ohlcv(symbol, "1h", 200)
    if df_1h is None or len(df_1h) < 50:
        return None
    df_1h = compute_indicators(df_1h)

    # 4h 数据
    df_4h = fetch_ohlcv(symbol, "4h", 100)
    if df_4h is None or len(df_4h) < 30:
        return None
    df_4h = compute_indicators(df_4h)

    price = float(df_1h["close"].iloc[-1])
    rsi_1h = float(df_1h["rsi"].iloc[-1])
    rsi_4h = float(df_4h["rsi"].iloc[-1])
    macd_hist_1h = float(df_1h["macd_hist"].iloc[-1])
    macd_hist_4h = float(df_4h["macd_hist"].iloc[-1])
    bb_lower = float(df_1h["bb_lower"].iloc[-1])
    bb_upper = float(df_1h["bb_upper"].iloc[-1])
    ret_4h = float(df_1h["ret_4h"].iloc[-1])
    ret_12h = float(df_1h["ret_12h"].iloc[-1])
    atr = float(df_1h["atr"].iloc[-1])

    score = 50
    reasons = []

    # 🆕 市场环境检测: 从feature_engine获取F&G+BTC趋势
    bearish_market = False
    try:
        from feature_engine import _SENTIMENT_CTX
        fg = _SENTIMENT_CTX.get("fg_value", 50)
        btc_below_ma = _SENTIMENT_CTX.get("btc_below_ma20", False)
        # F&G < 40 或 BTC在MA20下方 = 偏空市场
        bearish_market = (fg < 40) or btc_below_ma
    except Exception:
        pass

    # === 信号评分 ===
    buy_triggers = 0
    sell_triggers = 0

    # 🆕 偏空市场: RSI超卖不再盲目加分
    rsi_buy_mult = 0.5 if bearish_market else 1.0
    rsi_sell_bonus = 1.5 if bearish_market else 1.0

    # 1h RSI
    if rsi_1h < 38:
        bonus = int(15 * rsi_buy_mult)
        score += bonus; buy_triggers += 1 if bonus >= 8 else 0
        reasons.append(f"1hRSI超卖({rsi_1h:.0f})")
    elif rsi_1h < 45:
        bonus = int(8 * rsi_buy_mult)
        score += bonus
        if bonus >= 5:
            reasons.append(f"1hRSI偏低({rsi_1h:.0f})")
    elif rsi_1h > 70:
        penalty = int(10 * rsi_sell_bonus)
        score -= penalty; sell_triggers += 1
        reasons.append(f"1hRSI超买({rsi_1h:.0f})")
    elif rsi_1h > 60 and bearish_market:
        # 🆕 偏空市场中 RSI>60 = 反弹即空
        score -= 6; sell_triggers += 1
        reasons.append(f"1hRSI偏强反弹({rsi_1h:.0f})=做空机会")

    # 4h RSI
    if rsi_4h < 42:
        bonus = int(12 * rsi_buy_mult)
        score += bonus; buy_triggers += 1 if bonus >= 7 else 0
        reasons.append(f"4hRSI超卖({rsi_4h:.0f})")
    elif rsi_4h > 72:
        penalty = int(8 * rsi_sell_bonus)
        score -= penalty; sell_triggers += 1
        reasons.append(f"4hRSI超买({rsi_4h:.0f})")

    # MACD
    if macd_hist_1h > 0 and df_1h["macd_hist"].iloc[-2] <= 0:
        score += 10; buy_triggers += 1
        reasons.append("1hMACD金叉")
    elif macd_hist_1h > 0 and macd_hist_1h > df_1h["macd_hist"].iloc[-2]:
        score += 5; buy_triggers += 1
        reasons.append("1hMACD增强")
    elif macd_hist_1h < 0 and df_1h["macd_hist"].iloc[-2] >= 0:
        score -= 10; sell_triggers += 1
        reasons.append("1hMACD死叉")

    # 布林带位置
    bb_position = (price - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
    if bb_position < 0.15:
        if bearish_market:
            # 🆕 偏空市场触下轨 = 下跌趋势延续，不是抄底信号
            score -= 5; sell_triggers += 1
            reasons.append("偏空触布林下轨=趋势延续")
        else:
            score += 12
            buy_triggers += 1
            reasons.append("触布林下轨")
    elif bb_position < 0.35:
        score += 5
        reasons.append("低于布林中轨")
    elif bb_position > 0.85:
        penalty = 15 if bearish_market else 12
        score -= penalty
        sell_triggers += 1
        reasons.append("触布林上轨")

    # 短期超跌反弹 → 🆕 偏空市场中 = 下跌加速信号
    if ret_4h < -0.03 and ret_12h < -0.05:
        if bearish_market:
            score -= 8; sell_triggers += 1
            reasons.append(f"下跌加速({ret_4h:.1%})")
        else:
            score += 10
            buy_triggers += 1
            reasons.append(f"超跌反弹({ret_4h:.1%})")

    score = max(0, min(100, score))

    # 决策 — 🆕 偏空市场调整阈值
    if bearish_market:
        buy_threshold = 62   # 更难触发买入
        sell_threshold = 45  # 更容易触发卖出
    else:
        buy_threshold = 52
        sell_threshold = 48

    if score >= buy_threshold:
        action = "BUY"
    elif score <= sell_threshold:
        action = "SELL"
    else:
        action = "HOLD"

    # 计算止损止盈 (基于ATR)
    stop_loss = price * (1 + STOP_LOSS_PCT)
    take_profit = price * (1 + TAKE_PROFIT_PCT)

    return TradeSignal(
        symbol=symbol, name=name, action=action, price=price,
        score=score, reason=" | ".join(reasons) if reasons else "无明显信号",
        rsi_1h=rsi_1h, rsi_4h=rsi_4h,
        stop_loss=stop_loss, take_profit=take_profit,
    )


# ── 仓位管理 (使用 portfolio.py) ──
def load_portfolio():
    if PORTFOLIO_FILE.exists():
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    return {
        "cash": {"crypto": 5000.0, "a_stock": 2500.0, "us_stock": 2500.0},
        "positions": {},
        "initial_capital": 10000.0,
    }


def save_portfolio(pf):
    """保存持仓 — 使用绝对路径确保写入正确"""
    import traceback
    try:
        # 添加时间戳
        pf["updated_at"] = bj_now().isoformat()
        path = PORTFOLIO_FILE.resolve()
        with open(path, "w") as f:
            json.dump(pf, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"  ❌ 保存portfolio失败: {e}")
        traceback.print_exc()


def load_trades():
    if TRADES_FILE.exists():
        with open(TRADES_FILE) as f:
            return json.load(f)
    return []


def save_trades(trades):
    """保存交易记录"""
    import traceback
    try:
        path = TRADES_FILE.resolve()
        with open(path, "w") as f:
            json.dump(trades, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"  ❌ 保存trades失败: {e}")
        traceback.print_exc()


def get_position(pf, symbol):
    """获取指定币种的持仓"""
    for pid, pos in pf.get("positions", {}).items():
        if pos.get("symbol") == symbol and pos.get("status") == "open":
            return pid, pos
    return None, None


def execute_trade(signal: TradeSignal, pf: dict, trades: list) -> bool:
    """执行交易"""
    now = bj_now()
    cash = pf["cash"].get("crypto", 0)

    if signal.action == "BUY":
        # 检查是否已持有
        pid, existing = get_position(pf, signal.symbol)
        if existing:
            return False  # 已持有, 不重复买

        # 仓位大小: 30% 现金 或 最低¥200
        amount = max(MIN_TRADE, cash * POSITION_PCT)
        if cash < amount:
            print(f"    💰 现金不足: ¥{cash:.0f} < ¥{amount:.0f}")
            return False

        quantity = amount / signal.price
        pos_id = f"crypto_{signal.symbol}_{now.strftime('%Y%m%d%H%M%S')}"

        pf["cash"]["crypto"] = cash - amount
        pf["positions"][pos_id] = {
            "id": pos_id,
            "market": "crypto",
            "symbol": signal.symbol,
            "name": signal.name,
            "entry_price": signal.price,
            "current_price": signal.price,
            "quantity": quantity,
            "entry_time": now.isoformat(),
            "entry_reason": f"[🐺] {signal.reason} | 评分{signal.score:.0f}",
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "status": "open",
            "avg_entry_price": signal.price,
            "entry_prices": [signal.price],
            "execution_strategy": "aggressive",
        }

        trades.append({
            "id": pos_id,
            "market": "crypto",
            "symbol": signal.symbol,
            "name": signal.name,
            "side": "buy",
            "price": signal.price,
            "quantity": quantity,
            "amount": amount,
            "fee": amount * 0.001,
            "time": now.isoformat(),
            "reason": f"[🐺] {signal.reason} | 1hRSI={signal.rsi_1h:.0f} 4hRSI={signal.rsi_4h:.0f}",
            "pnl": 0.0,
            "pnl_pct": 0.0,
        })

        print(f"  🟢 买入 {signal.name} ¥{amount:.0f} @ {signal.price:.2f} | "
              f"评分{signal.score:.0f} | 1hRSI={signal.rsi_1h:.0f}")
        return True

    elif signal.action == "SELL":
        pid, existing = get_position(pf, signal.symbol)
        if not existing:
            return False

        entry_price = existing["entry_price"]
        quantity = existing["quantity"]
        pnl = (signal.price - entry_price) * quantity
        pnl_pct = (signal.price / entry_price - 1) * 100

        pf["cash"]["crypto"] += signal.price * quantity * 0.999  # 扣0.1%手续费
        pf["positions"][pid]["status"] = "closed"
        pf["positions"][pid]["current_price"] = signal.price

        trades.append({
            "id": pid,
            "market": "crypto",
            "symbol": signal.symbol,
            "name": signal.name,
            "side": "sell",
            "price": signal.price,
            "quantity": quantity,
            "amount": signal.price * quantity,
            "fee": signal.price * quantity * 0.001,
            "time": now.isoformat(),
            "reason": f"[🐺] {signal.reason} | PnL={pnl_pct:+.1f}%",
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
        })

        print(f"  🔴 卖出 {signal.name} ¥{signal.price*quantity:.0f} @ {signal.price:.2f} | "
              f"PnL={pnl_pct:+.1f}% | {signal.reason[:40]}")
        return True

    return False


def check_stop_loss(pf: dict, trades: list):
    """检查持仓止损止盈"""
    for pid, pos in list(pf.get("positions", {}).items()):
        if pos.get("status") != "open":
            continue

        symbol = pos["symbol"]
        entry = pos["entry_price"]

        # 获取最新价格
        df = fetch_ohlcv(symbol, "15m", 5)
        if df is None or len(df) < 1:
            continue
        price = float(df["close"].iloc[-1])

        pnl_pct = (price / entry - 1) * 100
        pos["current_price"] = price

        # 硬止损
        if pnl_pct <= STOP_LOSS_PCT * 100:
            quantity = pos["quantity"]
            pf["cash"]["crypto"] += price * quantity * 0.999
            pos["status"] = "closed"
            trades.append({
                "id": pid, "market": "crypto", "symbol": symbol,
                "name": pos["name"], "side": "sell", "price": price,
                "quantity": quantity, "amount": price * quantity,
                "fee": price * quantity * 0.001,
                "time": bj_now().isoformat(),
                "reason": f"🛑 硬止损: {pnl_pct:.1f}%",
                "pnl": round((price - entry) * quantity, 2),
                "pnl_pct": round(pnl_pct, 2),
            })
            print(f"  🛑 止损 {pos['name']}: {pnl_pct:.1f}%")

        # 止盈
        elif pnl_pct >= TAKE_PROFIT_PCT * 100:
            quantity = pos["quantity"]
            pf["cash"]["crypto"] += price * quantity * 0.999
            pos["status"] = "closed"
            trades.append({
                "id": pid, "market": "crypto", "symbol": symbol,
                "name": pos["name"], "side": "sell", "price": price,
                "quantity": quantity, "amount": price * quantity,
                "fee": price * quantity * 0.001,
                "time": bj_now().isoformat(),
                "reason": f"🎯 止盈: +{pnl_pct:.1f}%",
                "pnl": round((price - entry) * quantity, 2),
                "pnl_pct": round(pnl_pct, 2),
            })
            print(f"  🎯 止盈 {pos['name']}: +{pnl_pct:.1f}%")

    # 清理已平仓的
    closed = [pid for pid, p in pf["positions"].items() if p.get("status") == "closed"]
    for pid in closed:
        del pf["positions"][pid]


def run_cycle():
    """执行一次完整交易周期"""
    print(f"\n{'='*60}")
    print(f"🐺 [{bj_now().strftime('%H:%M:%S')}] 激进扫描...")
    print(f"{'='*60}")

    pf = load_portfolio()
    trades = load_trades()

    # 1. 止损止盈检查
    check_stop_loss(pf, trades)

    # 2. 扫描所有币种
    signals = []
    for sym in SYMBOLS:
        try:
            sig = scan_symbol(sym)
            if sig:
                signals.append(sig)
                icon = "🟢" if sig.action == "BUY" else "🔴" if sig.action == "SELL" else "⚪"
                print(f"  {icon} {sig.name:5s} ¥{sig.price:>10,.2f} | "
                      f"评分{sig.score:3.0f} | 1hRSI={sig.rsi_1h:4.0f} 4hRSI={sig.rsi_4h:4.0f} | "
                      f"{sig.action:4s} | {sig.reason[:60]}")
        except Exception as e:
            print(f"  ❌ {sym}: {e}")

    # 3. 执行信号
    executed = 0
    # 先执行卖出
    for sig in sorted(signals, key=lambda s: s.score):
        if sig.action == "SELL":
            if execute_trade(sig, pf, trades):
                executed += 1

    # 再执行买入 (最多3个)
    buy_count = 0
    open_positions = sum(1 for p in pf["positions"].values() if p.get("status") == "open")
    max_new = max(0, 5 - open_positions)  # 最多5个同时持仓

    for sig in sorted(signals, key=lambda s: s.score, reverse=True):
        if sig.action == "BUY" and buy_count < max_new:
            if execute_trade(sig, pf, trades):
                executed += 1
                buy_count += 1

    # 4. 最小仓位规则: 如果完全没有加密持仓, 买评分最高的
    open_positions = [p for p in pf["positions"].values() if p.get("status") == "open"]
    if not open_positions and signals:
        best = max(signals, key=lambda s: s.score)
        if best.score >= 40:
            print(f"  🐾 最小仓位: 无持仓, 强制买入评分最高的 {best.name}")
            # 绕过 action 检查
            best.action = "BUY"
            execute_trade(best, pf, trades)

    # 5. 更新持仓市价 (用最新信号价格)
    for sig in signals:
        for pid, pos in pf["positions"].items():
            if pos.get("symbol") == sig.symbol and pos.get("status") == "open":
                pos["current_price"] = sig.price
                # 同步更新 PortfolioManager 格式兼容字段
                pos["entry_price"] = pos.get("entry_price", pos.get("avg_entry_price", sig.price))
                if "avg_entry_price" not in pos or pos.get("avg_entry_price", 0) == 0:
                    pos["avg_entry_price"] = pos.get("entry_price", sig.price)

    # 6. 始终保存 (价格更新也需要持久化!)
    save_portfolio(pf)
    save_trades(trades)

    # 7. 汇总
    open_pos = [p for p in pf["positions"].values() if p.get("status") == "open"]
    crypto_cash = pf["cash"]["crypto"]
    total_value = crypto_cash + sum(p["current_price"] * p["quantity"] for p in open_pos)
    total_pnl = total_value - 5000  # crypto初始资金

    print(f"\n  📊 持仓{len(open_pos)}个 | 加密现金¥{crypto_cash:,.0f} | "
          f"加密总值¥{total_value:,.0f} | 加密PnL{total_pnl:+.0f}")
    print(f"  💾 已保存 → data/portfolio.json")

    return {
        "executed": executed,
        "signals": len(signals),
        "positions": len(open_pos),
        "crypto_value": total_value,
        "crypto_pnl": total_pnl,
    }


def main():
    print(f"🐺 激进交易策略启动")
    print(f"   扫描间隔: {SCAN_INTERVAL // 60}分钟")
    print(f"   标的: {', '.join(s.replace('/USDT', '') for s in SYMBOLS)}")
    print(f"   参数: RSI<{RSI_OVERSOLD}买 RSI>{RSI_OVERBOUGHT}卖")
    print(f"   仓位: {POSITION_PCT:.0%} 止损{STOP_LOSS_PCT:.0%} 止盈{TAKE_PROFIT_PCT:.0%}")
    print(f"   启动时间: {bj_now().strftime('%Y-%m-%d %H:%M:%S')}")

    while True:
        try:
            result = run_cycle()
            print(f"  ⏱️  下次扫描: {SCAN_INTERVAL // 60}分钟后")
        except Exception as e:
            print(f"  ❌ 周期异常: {e}")
            traceback.print_exc()

        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
