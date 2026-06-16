#!/usr/bin/env python3
"""
🐾 Yina 金融日报引擎 v2.2
= 五维评分卡 + 白毛股神 Serenity 瓶颈逻辑 + 大白话产业链叙事
每整点过后5分钟运行，推送企业微信「金融监控」群

v2.2 升级:
  📰 日报格式重构 — 30秒速读，Bloomberg 5 Things 风格
  🦊 白毛视角 — 大白话讲清瓶颈叙事，不是堆数据而是讲道理
  🎯 三只值得盯 — Top3 标的 + 一句话理由 + 止损位
  📋 紧凑速查 — 一屏看完20币评分

v2.1 基础:
  ① 供应链 DAG 映射 / ② 证据分级 / ③ 证伪条件 / ④ 跨主题加权 / ⑤ 前向验证
"""
from __future__ import annotations
import json, sys, os, urllib.request, urllib.error
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════
WEBHOOK_URL = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=2c602b48-5da2-4989-9193-30c0e226c769"
TZ = timezone(timedelta(hours=7))  # 曼谷时间 UTC+7 (Chase哥在泰国)
DATA_DIR = Path(__file__).parent / "data" / "hourly"
FORWARD_PICKS_FILE = DATA_DIR / "forward_picks.json"
MARKET_SNAPSHOTS_DIR = DATA_DIR / "snapshots"
UA = "Yina-Hourly/2.1 (Chase-Quant; macOS)"

# 确保目录存在
DATA_DIR.mkdir(parents=True, exist_ok=True)
MARKET_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 币种元数据 ──
COIN_META = {
    "BTC": {"name": "Bitcoin", "theme": ["store_of_value", "macro_sensitive", "etf_flow"], "supply_chain": None},
    "ETH": {"name": "Ethereum", "theme": ["smart_contract_platform", "defi", "layer0"], "supply_chain": "l1_infrastructure"},
    "BNB": {"name": "BNB Chain", "theme": ["exchange_token", "defi", "layer1"], "supply_chain": None},
    "SOL": {"name": "Solana", "theme": ["smart_contract_platform", "high_performance", "defi"], "supply_chain": "l1_infrastructure"},
    "XRP": {"name": "XRP", "theme": ["payment", "institutional", "legal_sensitive"], "supply_chain": None},
    "ADA": {"name": "Cardano", "theme": ["smart_contract_platform", "research_driven"], "supply_chain": None},
    "DOGE": {"name": "Dogecoin", "theme": ["meme", "social_sentiment", "whale_driven"], "supply_chain": None},
    "AVAX": {"name": "Avalanche", "theme": ["smart_contract_platform", "gaming", "subnet"], "supply_chain": "l1_infrastructure"},
    "DOT": {"name": "Polkadot", "theme": ["interoperability", "parachain"], "supply_chain": "cross_chain_infrastructure"},
    "LINK": {"name": "Chainlink", "theme": ["oracle", "infrastructure", "rwa"], "supply_chain": "data_infrastructure"},
    "MATIC": {"name": "Polygon", "theme": ["scaling", "enterprise", "zk"], "supply_chain": "l2_scaling"},
    "ATOM": {"name": "Cosmos", "theme": ["interoperability", "ibc"], "supply_chain": "cross_chain_infrastructure"},
    "LTC": {"name": "Litecoin", "theme": ["payment", "btc_beta"], "supply_chain": None},
    "UNI": {"name": "Uniswap", "theme": ["defi", "dex"], "supply_chain": "defi_infrastructure"},
    "APT": {"name": "Aptos", "theme": ["smart_contract_platform", "move_ecosystem"], "supply_chain": "l1_infrastructure"},
    "NEAR": {"name": "NEAR Protocol", "theme": ["smart_contract_platform", "ai_integration"], "supply_chain": "l1_infrastructure"},
    "OP": {"name": "Optimism", "theme": ["scaling", "superchain"], "supply_chain": "l2_scaling"},
    "ARB": {"name": "Arbitrum", "theme": ["scaling", "defi"], "supply_chain": "l2_scaling"},
    "SUI": {"name": "Sui", "theme": ["smart_contract_platform", "move_ecosystem", "gaming"], "supply_chain": "l1_infrastructure"},
    "TON": {"name": "Toncoin", "theme": ["social_platform", "messaging"], "supply_chain": None},
    "RENDER": {"name": "Render", "theme": ["ai_infrastructure", "gpu_compute"], "supply_chain": "ai_compute_infrastructure"},
    "FIL": {"name": "Filecoin", "theme": ["storage", "ai_data"], "supply_chain": "storage_infrastructure"},
}

# ── AI/半导体 供应链 DAG (Serenity风格) ──
# 定义 Crypto 世界中的 "卡脖子" 位置
SUPPLY_CHAIN_DAG = {
    "ai_compute_layer": {
        "name": "🧠 AI 算力层",
        "bottleneck_score": 9.2,  # 极高瓶颈
        "crypto_plays": ["RENDER", "FIL", "NEAR"],
        "thesis": "AI模型训练需要分布式GPU算力 → Render/FIL等去中心化算力网络是瓶颈节点",
        "key_question": "中心化云 (AWS/Azure) 会不会碾压去中心化方案？",
        "evidence_level": "inferred",  # 已证实/声称/推断/推测
    },
    "defi_settlement_layer": {
        "name": "💱 DeFi 结算层",
        "bottleneck_score": 7.5,
        "crypto_plays": ["ETH", "UNI", "LINK"],
        "thesis": "链上金融的清算/报价/预言机 → ETH是最终结算层, LINK是数据入口, UNI是流动性枢纽",
        "key_question": "L2是否会完全吸走L1的DeFi活动？",
        "evidence_level": "claimed",
    },
    "cross_chain_infrastructure": {
        "name": "🌉 跨链互操作层",
        "bottleneck_score": 6.8,
        "crypto_plays": ["DOT", "ATOM"],
        "thesis": "多链未来需要跨链通信协议 → DOT/ATOM 卡位互操作标准",
        "key_question": "L2原生互操作（如Superchain）会取代独立跨链协议吗？",
        "evidence_level": "inferred",
    },
    "l2_scaling": {
        "name": "⚡ L2 扩容层",
        "bottleneck_score": 7.0,
        "crypto_plays": ["MATIC", "OP", "ARB"],
        "thesis": "ETH扩容需求确定 → L2是必经之路, 但竞争激烈",
        "key_question": "ZK Rollup 会不会让 Optimistic Rollup 过时？",
        "evidence_level": "confirmed",
    },
    "data_oracle": {
        "name": "🔮 数据预言机",
        "bottleneck_score": 8.0,
        "crypto_plays": ["LINK"],
        "thesis": "所有DeFi/RWA都需要外部数据 → LINK几乎是唯一选择",
        "key_question": "竞争对手 (Pyth/Band) 市场份额增长有多快？",
        "evidence_level": "confirmed",
    },
}

# ═══════════════════════════════════════════
# 数据获取
# ═══════════════════════════════════════════

def fetch_binance_klines(symbol: str, interval: str = "1h", limit: int = 100) -> dict:
    """从 Binance 公开 API 获取K线"""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol.replace('/', '')}&interval={interval}&limit={limit}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            if not data:
                return {"error": f"无数据: {symbol}"}
            closes = [float(c[4]) for c in data]
            highs = [float(c[2]) for c in data]
            lows = [float(c[3]) for c in data]
            volumes = [float(c[5]) for c in data]
            opens = [float(c[1]) for c in data]
            return {
                "symbol": symbol,
                "close": np_array(closes),
                "high": np_array(highs),
                "low": np_array(lows),
                "volume": np_array(volumes),
                "open": np_array(opens),
                "price": closes[-1],
                "ts": data[-1][0],
            }
    except Exception as e:
        return {"error": str(e)}

def fetch_24hr_ticker(symbols: list) -> dict:
    """批量获取24小时行情"""
    syms = ",".join([f'"{s.replace("/", "")}"' for s in symbols])
    url = f"https://api.binance.com/api/v3/ticker/24hr?symbols=[{syms}]"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            result = {}
            for t in data:
                sym = t["symbol"]
                result[sym] = {
                    "price": float(t["lastPrice"]),
                    "change_pct": float(t["priceChangePercent"]),
                    "volume_usdt": float(t["quoteVolume"]),
                    "high_24h": float(t["highPrice"]),
                    "low_24h": float(t["lowPrice"]),
                }
            return result
    except Exception as e:
        return {"error": str(e)}

def np_array(lst):
    try:
        import numpy as np
        return np.array(lst)
    except ImportError:
        return lst

# ═══════════════════════════════════════════
# 五维评分卡 (v2.1 — 增强版)
# ═══════════════════════════════════════════

@dataclass
class ScoreCard:
    symbol: str
    price: float
    change_24h: float
    volume_usdt: float
    scores: dict  # {trend, ob_os, sr, fundamental, risk}
    composite: float
    action: str
    stars: int
    confidence: float
    reasons: list
    # ── Serenity v2.1 新增 ──
    supply_chain_position: Optional[dict] = None  # 供应链位置
    evidence_grade: str = "inferred"  # 证据分级
    falsification_conditions: list = field(default_factory=list)  # 证伪条件
    cross_theme_signals: list = field(default_factory=list)  # 跨主题命中
    cross_theme_bonus: float = 0.0  # 跨主题加成
    forward_pick_id: Optional[str] = None  # 前向验证ID


def calc_five_dimension(klines: dict, ticker: dict, meta: dict) -> ScoreCard:
    """对单个币种运行五维评分 + Serenity增强"""
    symbol = klines["symbol"]
    base = symbol.split("/")[0]
    close = klines["close"]
    high = klines["high"]
    low = klines["low"]
    volume = klines["volume"]
    price = ticker.get("price", klines["price"])
    change_24h = ticker.get("change_pct", 0)
    vol_usdt = ticker.get("volume_usdt", 0)
    reasons = []

    # ── 维度1: 趋势强度 (0-100) ──
    trend = 50
    try:
        import numpy as np
        # MACD
        ema12 = pd_ewma(close, 12)
        ema26 = pd_ewma(close, 26)
        macd_line = ema12[-1] - ema26[-1]
        signal_line = pd_ewma(np_array([ema12[i] - ema26[i] for i in range(len(close))]), 9)[-1]
        hist = macd_line - signal_line
        if hist > 0: trend += 10
        # 均线排列
        sma5 = np.mean(close[-5:])
        sma20 = np.mean(close[-20:])
        sma60 = np.mean(close[-60:]) if len(close) >= 60 else sma20
        if sma5 > sma20: trend += 10
        if sma20 > sma60: trend += 10
        # ADX 简化
        if len(close) >= 14:
            plus_dm = np.maximum(np.diff(high), 0)
            minus_dm = np.maximum(-np.diff(low), 0)
            tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
            tr = np.maximum(tr, np.abs(low[1:] - close[:-1]))
            atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            di_plus = np.mean(plus_dm[-14:]) / (atr + 1e-9) * 100 if len(plus_dm) >= 14 else 25
            di_minus = np.mean(minus_dm[-14:]) / (atr + 1e-9) * 100 if len(minus_dm) >= 14 else 25
            adx = abs(di_plus - di_minus) / (di_plus + di_minus + 1e-9) * 100
            if adx > 25: trend += 10
        # 价格相对位置
        if len(close) >= 20:
            high_20d = np.max(high[-20:])
            low_20d = np.min(low[-20:])
            pos = (price - low_20d) / (high_20d - low_20d + 1e-9)
            if 0.4 < pos < 0.8: trend += 10  # 健康上涨区
        if change_24h > 0: trend += 5
        if change_24h > 5: trend += 5
        trend = min(100, trend)
        if trend >= 70: reasons.append(f"📈 趋势偏强 ({trend}分): MACD+均线共振")
        elif trend >= 50: reasons.append(f"📊 趋势中性 ({trend}分)")
        else: reasons.append(f"📉 趋势偏弱 ({trend}分)")
    except Exception:
        pass

    # ── 维度2: 超买超卖 (0-100) ──
    ob_os = 50
    try:
        rsi = calc_rsi_manual(close, 14)
        if 40 <= rsi <= 60: ob_os = 75
        elif 30 <= rsi < 40: ob_os = 55
        elif 60 < rsi <= 70: ob_os = 45
        elif rsi < 30: ob_os = 25
        elif rsi > 70: ob_os = 20
        if ob_os >= 70: reasons.append(f"⚖️ RSI={rsi:.0f} 健康区间")
        elif rsi > 70: reasons.append(f"⚠️ RSI={rsi:.0f} 超买警示")
        elif rsi < 30: reasons.append(f"💡 RSI={rsi:.0f} 超卖机会")
    except Exception:
        pass

    # ── 维度3: 支撑阻力 (0-100) ──
    sr = 50
    try:
        import numpy as np
        high_24h = ticker.get("high_24h", price * 1.05)
        low_24h = ticker.get("low_24h", price * 0.95)
        dist_to_res = (high_24h - price) / price * 100
        dist_to_sup = (price - low_24h) / price * 100
        if dist_to_sup < 3: sr += 20  # 接近支撑
        elif dist_to_sup < 5: sr += 10
        if dist_to_res > 8: sr += 15  # 上方空间大
        elif dist_to_res > 4: sr += 8
        if dist_to_res < 2: sr -= 15  # 接近阻力
        sr = max(10, min(100, sr))
        if sr >= 70: reasons.append(f"🏗️ 支撑坚实 ({sr}分): 距支撑仅{dist_to_sup:.1f}%")
        elif sr <= 30: reasons.append(f"⚠️ 接近阻力 ({sr}分): 距阻力仅{dist_to_res:.1f}%")
    except Exception:
        pass

    # ── 维度4: 基本面 (0-100) v2.1增强: 证据分级 ──
    fundamental = 50
    evidence_grade = "inferred"
    try:
        # 交易量活跃度 (高成交量 = 市场认可)
        if vol_usdt > 1_000_000_000: fundamental += 15  # >$1B 24h vol
        elif vol_usdt > 200_000_000: fundamental += 10
        elif vol_usdt > 50_000_000: fundamental += 5

        # 24h价格变化 (强势/弱势信号)
        if change_24h > 3: fundamental += 10
        elif change_24h < -5: fundamental -= 8

        # 证据分级判断
        if vol_usdt > 500_000_000 and abs(change_24h) > 2:
            evidence_grade = "confirmed"  # 大成交量+明确方向
        elif vol_usdt > 100_000_000:
            evidence_grade = "claimed"    # 活跃但无明确方向
        elif vol_usdt > 10_000_000:
            evidence_grade = "inferred"   # 一般活跃
        else:
            evidence_grade = "speculative"  # 低流动性

        fundamental = max(0, min(100, fundamental))
    except Exception:
        pass

    # ── 维度5: 风险度 (0-100, 低风险=高分) ──
    risk = 60
    try:
        import numpy as np
        # 历史波动率
        if len(close) >= 24:
            returns = np.diff(np.log(close[-24:]))
            hv_24h = np.std(returns) * np.sqrt(365) * 100
            if hv_24h > 100: risk -= 25
            elif hv_24h > 60: risk -= 15
            elif hv_24h < 30: risk += 15

        # 24h波动幅度
        high_low_range = (ticker.get("high_24h", price * 1.1) - ticker.get("low_24h", price * 0.9)) / price * 100
        if high_low_range > 10: risk -= 15
        elif high_low_range > 5: risk -= 5
        elif high_low_range < 3: risk += 10

        risk = max(5, min(100, risk))
    except Exception:
        pass

    # ── 跨主题信号加权 (Serenity v2.1) ──
    themes = meta.get("theme", [])
    cross_theme_bonus = 0
    cross_theme_signals = []
    # 如果标的属于多个高价值主题，给加成
    high_value_themes = ["ai_compute_layer", "defi_settlement_layer", "data_oracle", "cross_chain_infrastructure"]
    supply_pos = meta.get("supply_chain")
    if supply_pos:
        # 查找该标的是否在供应链瓶颈节点上
        for layer_id, layer_info in SUPPLY_CHAIN_DAG.items():
            if base in layer_info["crypto_plays"]:
                bt_score = layer_info["bottleneck_score"]
                cross_theme_bonus += bt_score * 0.5  # 瓶颈分越高，加成越大
                cross_theme_signals.append(f"{layer_info['name']} 瓶颈分={bt_score:.1f}")
                if bt_score >= 8.0:
                    reasons.append(f"🔬 高瓶颈节点: {layer_info['name']} ({bt_score}/10)")
        if len(cross_theme_signals) >= 2:
            fundamental += 8  # 多主题命中 → 基本面加分
            reasons.append(f"🎯 跨主题信号: {', '.join(cross_theme_signals)}")

    # ── 证据分级加成 ──
    evidence_weight = {"confirmed": 1.0, "claimed": 0.85, "inferred": 0.7, "speculative": 0.5}
    evidence_mult = evidence_weight.get(evidence_grade, 0.7)

    # ── 加权综合 ──
    weights = {"trend": 0.25, "ob_os": 0.15, "sr": 0.20, "fundamental": 0.25, "risk": 0.15}
    composite = (
        trend * weights["trend"] +
        ob_os * weights["ob_os"] +
        sr * weights["sr"] +
        fundamental * weights["fundamental"] +
        risk * weights["risk"]
    )
    # 证据分级调整置信度
    confidence = 0.5 + (composite / 100) * 0.4 + (evidence_mult - 0.5) * 0.2
    confidence = max(0.15, min(0.95, confidence))

    # ── 行动映射 ──
    if composite >= 70: action = "BUY"
    elif composite >= 50: action = "WATCH"
    elif composite >= 35: action = "REDUCE"
    else: action = "SELL"
    stars = min(5, max(1, round(composite / 20)))

    # ── 证伪条件 (Serenity v2.1) ──
    falsification = []
    if action == "BUY":
        if trend < 50: falsification.append("若趋势反转(trend<50)持续2小时→止损")
        if change_24h > 0:
            sl_price = price * 0.92
            falsification.append(f"若跌破{symbol} ${sl_price:.0f} (-8%)→硬止损")
        falsification.append("若BTC大盘24h跌幅>5%→减仓50%")
    elif action == "SELL":
        falsification.append("若RSI<30且放量反弹→空头离场")
        falsification.append("若出现重大利好新闻→重新评估")

    return ScoreCard(
        symbol=symbol,
        price=price,
        change_24h=change_24h,
        volume_usdt=vol_usdt,
        scores={"trend": trend, "ob_os": ob_os, "sr": sr, "fundamental": fundamental, "risk": risk},
        composite=composite,
        action=action,
        stars=stars,
        confidence=confidence,
        reasons=reasons,
        evidence_grade=evidence_grade,
        falsification_conditions=falsification,
        cross_theme_signals=cross_theme_signals,
        cross_theme_bonus=cross_theme_bonus,
    )


def pd_ewma(data, span):
    """简易指数加权移动平均"""
    result = []
    alpha = 2 / (span + 1)
    ema = data[0] if isinstance(data[0], (int, float)) else float(data[0])
    for val in data:
        if isinstance(val, (list, np_array)):
            val = float(val[0]) if hasattr(val, '__iter__') else float(val)
        ema = alpha * float(val) + (1 - alpha) * ema
        result.append(ema)
    return result


def calc_rsi_manual(close, period=14):
    """手算RSI，避免依赖ta库"""
    deltas = []
    for i in range(1, len(close)):
        deltas.append(float(close[i]) - float(close[i-1]))
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss < 1e-9:
        return 100.0
    return float(100 - 100 / (1 + avg_gain / avg_loss))


# ═══════════════════════════════════════════
# 供应链 DAG 分析 (Serenity 核心逻辑)
# ═══════════════════════════════════════════

def analyze_supply_chain_bottlenecks(scores: List[ScoreCard]) -> str:
    """对当前市场运行供应链瓶颈分析，识别非线性定价权标的"""
    lines = []
    lines.append("### 🔬 供应链瓶颈分析 (Serenity 逻辑)")
    lines.append("")

    # 按瓶颈层聚合
    for layer_id, layer in SUPPLY_CHAIN_DAG.items():
        layer_coins = []
        for sc in scores:
            base = sc.symbol.split("/")[0]
            if base in layer["crypto_plays"]:
                layer_coins.append(sc)

        if not layer_coins:
            continue

        avg_score = sum(c.composite for c in layer_coins) / len(layer_coins)
        avg_change = sum(c.change_24h for c in layer_coins) / len(layer_coins)
        evidence = layer["evidence_level"]
        evidence_emoji = {"confirmed": "✅", "claimed": "📋", "inferred": "🔍", "speculative": "❓"}

        lines.append(f"**{layer['name']}** | 瓶颈分: {layer['bottleneck_score']:.1f}/10 | {evidence_emoji.get(evidence, '?')} {evidence}")
        coins_str = ", ".join([f"{c.symbol}({c.composite:.0f}分 ⭐{c.stars})" for c in layer_coins])
        lines.append(f"> 标的: {coins_str}")
        lines.append(f"> 综合: {avg_score:.0f}分 | 24h变化: {avg_change:+.1f}%")
        lines.append(f"> 核心命题: {layer['thesis']}")
        lines.append(f"> ⚡ 反向问题: {layer['key_question']}")
        lines.append("")

    # 瓶颈交叉信号
    high_bottleneck_coins = []
    for sc in scores:
        base = sc.symbol.split("/")[0]
        for layer_id, layer in SUPPLY_CHAIN_DAG.items():
            if base in layer["crypto_plays"] and layer["bottleneck_score"] >= 7.5:
                high_bottleneck_coins.append((sc, layer["bottleneck_score"], layer["name"]))

    if high_bottleneck_coins:
        lines.append("#### 🎯 高瓶颈节点 (≥7.5分)")
        for sc, bt, lname in sorted(high_bottleneck_coins, key=lambda x: x[1], reverse=True):
            action_color = "🟢" if sc.action == "BUY" else "🟡" if sc.action == "WATCH" else "🔴"
            lines.append(f"> {action_color} **{sc.symbol}** ({lname}, 瓶颈{bt:.1f}) — {sc.action} ⭐{sc.stars} | 综合{sc.composite:.0f}分")

    return "\n".join(lines)


# ═══════════════════════════════════════════
# 前向验证 (Serenity forward_picks)
# ═══════════════════════════════════════════

def record_forward_picks(scores: List[ScoreCard]):
    """记录本次分析判断，供30天后回验"""
    now = datetime.now(TZ).isoformat()
    picks = []

    existing = []
    if FORWARD_PICKS_FILE.exists():
        try:
            with open(FORWARD_PICKS_FILE) as f:
                existing = json.load(f)
        except Exception:
            existing = []

    for sc in scores:
        if sc.action in ("BUY", "SELL"):
            pick = {
                "id": f"{sc.symbol.replace('/', '-')}_{datetime.now(TZ).strftime('%Y%m%d_%H%M')}",
                "timestamp": now,
                "symbol": sc.symbol,
                "price": sc.price,
                "action": sc.action,
                "composite": round(sc.composite, 1),
                "stars": sc.stars,
                "confidence": round(sc.confidence, 3),
                "reasons": sc.reasons[:3],
                "falsification": sc.falsification_conditions[:2],
                "evidence_grade": sc.evidence_grade,
                "verified_30d": False,
                "verified_price": None,
                "verified_correct": None,
            }
            picks.append(pick)
            sc.forward_pick_id = pick["id"]

    existing.extend(picks)
    with open(FORWARD_PICKS_FILE, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    return picks


# ═══════════════════════════════════════════
# 市场快照存档
# ═══════════════════════════════════════════

def save_market_snapshot(scores: List[ScoreCard], supply_chain_text: str):
    """每小时存档完整市场快照"""
    now = datetime.now(TZ)
    filename = now.strftime("%Y%m%d_%H00.json")
    snapshot = {
        "timestamp": now.isoformat(),
        "hour": now.strftime("%Y-%m-%d %H:00"),
        "assets": [],
        "supply_chain": supply_chain_text,
        "summary": {},
    }
    for sc in scores:
        snapshot["assets"].append({
            "symbol": sc.symbol,
            "price": sc.price,
            "change_24h": sc.change_24h,
            "composite": round(sc.composite, 1),
            "action": sc.action,
            "stars": sc.stars,
            "confidence": round(sc.confidence, 2),
            "scores": sc.scores,
            "evidence_grade": sc.evidence_grade,
            "cross_theme_signals": sc.cross_theme_signals,
        })

    # 汇总
    scores_list = [s.composite for s in scores]
    snapshot["summary"] = {
        "n_assets": len(scores),
        "avg_score": round(sum(scores_list) / len(scores_list), 1) if scores_list else 0,
        "n_buy": sum(1 for s in scores if s.action == "BUY"),
        "n_watch": sum(1 for s in scores if s.action == "WATCH"),
        "n_sell": sum(1 for s in scores if s.action == "SELL"),
        "market_sentiment": "bullish" if sum(1 for s in scores if s.change_24h > 0) > len(scores) * 0.6 else (
            "bearish" if sum(1 for s in scores if s.change_24h < 0) > len(scores) * 0.6 else "neutral"
        ),
    }

    filepath = MARKET_SNAPSHOTS_DIR / filename
    with open(filepath, "w") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════
# 完整报告生成
# ═══════════════════════════════════════════

WATCHLIST = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
    "MATIC/USDT", "UNI/USDT", "ATOM/USDT", "APT/USDT", "OP/USDT",
    "ARB/USDT", "SUI/USDT", "NEAR/USDT", "RENDER/USDT", "FIL/USDT",
]


# ═══════════════════════════════════════════
# 白毛股神 叙事引擎 — 大白话讲清市场逻辑
# ═══════════════════════════════════════════

def _bottleneck_story(layer_id: str, avg_score: float, avg_change: float, coins: list) -> str:
    """为每个供应链层生成大白话叙事"""
    stories = {
        "ai_compute_layer": (
            f"AI算力这块，{', '.join(coins)} 扛着去中心化GPU的大旗。"
            f"这层是整个Crypto最'卡脖子'的地方——AI公司抢GPU抢疯了，中心化云又贵又排他，"
            f"去中心化算力网络成了唯一溢出通道。"
        ),
        "l2_scaling": (
            f"L2扩容层（{', '.join(coins)}）——ETH想做大必须过L2这道坎。"
            f"但ZK和OP两派互卷，赢家通吃风险高，所以虽然需求确定，估值却容易被新技术迭代打趴。"
        ),
        "defi_settlement_layer": (
            f"DeFi结算层（{', '.join(coins)}）——链上金融的清算所+数据口。"
            f"LINK几乎垄断预言机这个入口，ETH是最终结算账本，UNI管流动性枢纽。"
            f"这三个是DeFi最硬的'基础设施税'收取者。"
        ),
        "cross_chain_infrastructure": (
            f"跨链互操作（{', '.join(coins)}）——多链是事实，但跨链协议能不能收到'过桥费'还两说。"
            f"L2原生互操作方案（Superchain之类）有可能让独立跨链协议边缘化。"
        ),
        "data_oracle": (
            f"预言机层（{', '.join(coins)}）——所有DeFi和RWA都依赖外部数据喂养。"
            f"LINK几乎是唯一选择，这就像所有大楼共用一根水管——换水管成本高到没人愿意换。"
        ),
    }
    base = stories.get(layer_id, f"{layer_id} 层：{', '.join(coins)}")
    direction = "涨" if avg_change > 1 else "跌" if avg_change < -1 else "横盘"
    return f"{base} 本小时{direction}势，综合{avg_score:.0f}分。"


def generate_market_pulse(scores: List[ScoreCard]) -> str:
    """一句话市场脉搏 — Chase哥扫一眼就知道现在什么情况"""
    now = datetime.now(TZ)
    bull = sum(1 for s in scores if s.change_24h > 0)
    bear = sum(1 for s in scores if s.change_24h < 0)
    avg_change = sum(s.change_24h for s in scores) / len(scores) if scores else 0
    avg_score = sum(s.composite for s in scores) / len(scores) if scores else 0

    # 找最亮的和最暗的
    sorted_scores = sorted(scores, key=lambda s: s.composite, reverse=True)
    top3 = sorted_scores[:3]
    bottom3 = sorted_scores[-3:]

    if avg_change > 2:
        pulse = "🔥 市场偏热，注意追高风险"
    elif avg_change > 0:
        pulse = "🌤️ 温和偏暖，以持仓观察为主"
    elif avg_change > -2:
        pulse = "🌥️ 小幅回调，汰弱留强窗口"
    else:
        pulse = "🌧️ 全线承压，严格止损观望"

    # 找结构热点
    top_supply_chain = None
    top_bt_score = 0
    for sc in scores:
        base = sc.symbol.split("/")[0]
        for lid, l in SUPPLY_CHAIN_DAG.items():
            if base in l["crypto_plays"] and l["bottleneck_score"] > top_bt_score:
                top_bt_score = l["bottleneck_score"]
                top_supply_chain = l

    sc_hint = ""
    if top_supply_chain:
        sc_hint = f" | 最强结构: {top_supply_chain['name']}"

    return (
        f"**{now.strftime('%m/%d %H:%M')} 曼谷时间**\n"
        f"> {pulse}\n"
        f"> 20币均分 **{avg_score:.0f}**/100 | 涨{bull}跌{bear} | 24h均值 {avg_change:+.1f}%{sc_hint}\n"
        f"> 🏆 {top3[0].symbol.split('/')[0]} {top3[0].composite:.0f}分 · "
        f"{top3[1].symbol.split('/')[0]} {top3[1].composite:.0f}分 · "
        f"{top3[2].symbol.split('/')[0]} {top3[2].composite:.0f}分"
    )


def generate_serenity_bottleneck_narrative(scores: List[ScoreCard]) -> str:
    """白毛股神视角 — 用大白话讲清楚：什么在涨？为什么涨？谁卡住了谁的脖子？"""
    # 按瓶颈层聚合
    layer_summaries = []
    for layer_id, layer in SUPPLY_CHAIN_DAG.items():
        layer_coins = []
        for sc in scores:
            base = sc.symbol.split("/")[0]
            if base in layer["crypto_plays"]:
                layer_coins.append(sc)

        if not layer_coins:
            continue

        avg_score = sum(c.composite for c in layer_coins) / len(layer_coins)
        avg_change = sum(c.change_24h for c in layer_coins) / len(layer_coins)
        coin_names = [f"**{c.symbol.split('/')[0]}**({c.change_24h:+.0f}%,{c.composite:.0f}分)" for c in layer_coins]

        layer_summaries.append({
            "layer": layer,
            "layer_id": layer_id,
            "avg_score": avg_score,
            "avg_change": avg_change,
            "coins": layer_coins,
            "coin_names": coin_names,
            "bt_score": layer["bottleneck_score"],
        })

    # 按瓶颈分排序
    layer_summaries.sort(key=lambda x: x["bt_score"], reverse=True)

    # 生成叙事
    narrative_lines = []
    narrative_lines.append(f"### 🦊 白毛视角 · 瓶颈叙事")
    narrative_lines.append("")

    # 找出最有故事性的层（瓶颈高 + 变化大的）
    top_layer = layer_summaries[0] if layer_summaries else None
    hot_layer = max(layer_summaries, key=lambda x: abs(x["avg_change"])) if layer_summaries else None

    if top_layer:
        l = top_layer
        coin_str = "、".join([c.symbol.split("/")[0] for c in l["coins"]])
        bt = l["bt_score"]
        direction = "往上冲" if l["avg_change"] > 0 else "在回调" if l["avg_change"] < 0 else "横着走"

        narrative_lines.append(f"**{l['layer']['name']}** (瓶颈{bt:.1f}/10) 今天{direction}：")
        narrative_lines.append(f"> {_bottleneck_story(l['layer_id'], l['avg_score'], l['avg_change'], [c.symbol.split('/')[0] for c in l['coins']])}")

        # 找该层里最强的标的
        best_in_layer = max(l["coins"], key=lambda c: c.composite)
        if best_in_layer.action == "BUY":
            narrative_lines.append(f"> 👉 这层最值得盯的是 **{best_in_layer.symbol}**，综合{best_in_layer.composite:.0f}分")
            if best_in_layer.falsification_conditions:
                narrative_lines.append(f"> ⚠️ 但要记住：{best_in_layer.falsification_conditions[0]}")

        narrative_lines.append("")

    # 第二大层（如果有趣的话）
    if len(layer_summaries) >= 2:
        l2 = layer_summaries[1]
        # 只有当它的瓶颈分也高或者变化剧烈时才提
        if l2["bt_score"] >= 7.0 or abs(l2["avg_change"]) > 3:
            direction2 = "涨得不错" if l2["avg_change"] > 1 else "在跌" if l2["avg_change"] < -1 else "没大动静"
            narrative_lines.append(f"**{l2['layer']['name']}** (瓶颈{l2['bt_score']:.1f}/10) {direction2}：")
            narrative_lines.append(f"> {_bottleneck_story(l2['layer_id'], l2['avg_score'], l2['avg_change'], [c.symbol.split('/')[0] for c in l2['coins']])}")
            narrative_lines.append("")

    # 识别跨层交叉信号
    cross_layer_hits = {}
    for sc in scores:
        base = sc.symbol.split("/")[0]
        layers_hit = []
        for lid, l in SUPPLY_CHAIN_DAG.items():
            if base in l["crypto_plays"]:
                layers_hit.append(l["name"])
        if len(layers_hit) >= 2:
            cross_layer_hits[base] = layers_hit

    if cross_layer_hits:
        narrative_lines.append(f"**🔗 跨层交叉信号**:")
        for coin, layers in sorted(cross_layer_hits.items(), key=lambda x: len(x[1]), reverse=True):
            narrative_lines.append(f"> {coin} 横跨 {', '.join(layers)} — 多层命中的标的结构上更抗跌")
        narrative_lines.append("")

    # 白毛股神的"反向灵魂拷问"
    if hot_layer:
        narrative_lines.append(f"**❓ 反向拷问**（白毛式自我质疑）：")
        narrative_lines.append(f"> {hot_layer['layer']['key_question']}")
        narrative_lines.append("")

    return "\n".join(narrative_lines)


def generate_top_picks(scores: List[ScoreCard]) -> str:
    """选出最值得交易的3个标的，各配一句话理由 + 止损位"""
    sorted_scores = sorted(scores, key=lambda s: s.composite, reverse=True)

    picks = []
    for sc in sorted_scores[:5]:  # 从Top5里挑
        if len(picks) >= 3:
            break
        # 优先挑有供应链位置 + 交易信号明确的
        if sc.supply_chain_position or sc.cross_theme_signals:
            picks.append(sc)
        elif sc.action in ("BUY", "SELL"):
            picks.append(sc)

    # 如果还不够3个，补充分最高的
    if len(picks) < 3:
        for sc in sorted_scores:
            if sc not in picks and len(picks) < 3:
                picks.append(sc)

    lines = []
    lines.append(f"### 🎯 三只值得盯的")
    lines.append("")

    for i, sc in enumerate(picks[:3], 1):
        base = sc.symbol.split("/")[0]
        emoji = "🟢" if sc.action == "BUY" else "🟡" if sc.action == "WATCH" else "🔴"
        name = COIN_META.get(base, {}).get("name", base)

        # 一句话理由（取最强理由）
        best_reason = sc.reasons[0] if sc.reasons else "五维综合评分"
        # 去掉emoji前缀，只要文字
        clean_reason = best_reason.split(" ", 1)[-1] if " " in best_reason else best_reason

        lines.append(f"**{i}. {emoji} {base}** ({name}) ${sc.price:,.2f} | {sc.change_24h:+.1f}%")
        lines.append(f"> 综合 **{sc.composite:.0f}** 分 ⭐{sc.stars} | {clean_reason}")

        # 供应链定位
        if sc.cross_theme_signals:
            sc_info = sc.cross_theme_signals[0]
            lines.append(f"> 🔬 {sc_info}")

        # 止损
        if sc.falsification_conditions:
            lines.append(f"> ⚠️ {sc.falsification_conditions[0]}")
        elif sc.action == "BUY":
            sl = sc.price * 0.92
            lines.append(f"> ⚠️ 硬止损 ${sl:.0f} (-8%)")

        lines.append("")

    return "\n".join(lines)


def build_compact_scoreboard(scores: List[ScoreCard]) -> str:
    """紧凑型全币种速查表 — 两列布局，一屏看完"""
    sorted_scores = sorted(scores, key=lambda s: s.composite, reverse=True)

    lines = []
    lines.append(f"### 📋 速查 · 20币评分卡")
    lines.append("")

    # 紧凑表头
    header = ("| 标的 | 价格 | 24h | 评分 | 行动 |\n"
              "|------|------|-----|------|------|")
    lines.append(header)

    for sc in sorted_scores:
        emoji = "🟢" if sc.action == "BUY" else "🟡" if sc.action == "WATCH" else "🔴"
        lines.append(
            f"| {emoji} {sc.symbol.split('/')[0]} | ${sc.price:,.2f} | {sc.change_24h:+.1f}% | "
            f"**{sc.composite:.0f}** | {sc.action} |"
        )

    lines.append("")
    return "\n".join(lines)


def build_serenity_report(scores: List[ScoreCard]) -> str:
    """🐾 Yina 金融日报 — 30秒速读版
    格式灵感：Bloomberg 5 Things + 白毛股神 大白话产业链叙事
    """
    now = datetime.now(TZ)
    n_buy = sum(1 for s in scores if s.action == "BUY")
    n_sell = sum(1 for s in scores if s.action == "SELL")

    lines = []
    # ── 头部 ──
    lines.append(f"# 📰 Yina 金融日报")
    lines.append(f"## {now.strftime('%m/%d')} {now.strftime('%H:%M')} 曼谷 · 五维评分v2.1 · 白毛逻辑")
    lines.append("")

    # ── 1. 市场脉搏 (2行) ──
    lines.append(generate_market_pulse(scores))
    lines.append("")

    # ── 2. 白毛股神 瓶颈叙事 ──
    lines.append(generate_serenity_bottleneck_narrative(scores))

    # ── 3. 三只值得盯 ──
    lines.append(generate_top_picks(scores))

    # ── 4. 紧凑速查表 ──
    lines.append(build_compact_scoreboard(scores))

    # ── 5. 风控一行 ──
    btc = next((s for s in scores if s.symbol == "BTC/USDT"), None)
    btc_hint = ""
    if btc:
        if btc.change_24h < -3:
            btc_hint = " ⚠️BTC大盘承压，所有买入谨慎！"
        elif btc.change_24h > 5:
            btc_hint = " BTC强势，多头氛围偏暖。"

    lines.append(f"### 🛡️ 风控")
    lines.append(f"> 硬止损-8% · 日熔断-5% · 持仓≤5只{btc_hint}")
    lines.append(f"> 🟢{n_buy}只可买 🟡{20-n_buy-n_sell}只观望 🔴{n_sell}只回避")
    lines.append("")

    # ── 尾部 ──
    lines.append("---")
    lines.append(f"🐾 Yina 自动日报 · 白毛股神 Serenity 瓶颈逻辑驱动 · {now.strftime('%H:%M')}")

    # ── 后台：记录前向验证 + 存档快照 (不进入报告正文) ──
    record_forward_picks(scores)
    save_market_snapshot(scores, "")

    return "\n".join(lines)


# ═══════════════════════════════════════════
# 企业微信推送
# ═══════════════════════════════════════════

def send_to_wecom(md_content: str) -> bool:
    """推送 Markdown 到企业微信群"""
    # 企业微信 markdown 最大 4096 字节，超长需要分段
    max_len = 4000
    chunks = []
    current = ""
    for line in md_content.split("\n"):
        if len((current + line + "\n").encode("utf-8")) > max_len:
            chunks.append(current)
            current = line + "\n"
        else:
            current += line + "\n"
    if current:
        chunks.append(current)

    success = True
    for i, chunk in enumerate(chunks):
        prefix = f"({i+1}/{len(chunks)})\n" if len(chunks) > 1 else ""
        payload = json.dumps({
            "msgtype": "markdown",
            "markdown": {"content": prefix + chunk}
        }).encode("utf-8")

        req = urllib.request.Request(
            WEBHOOK_URL, data=payload,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                if result.get("errcode") != 0:
                    print(f"   ⚠️ 分段{i+1}推送失败: {result}", file=sys.stderr)
                    success = False
        except Exception as e:
            print(f"   ❌ 分段{i+1}推送异常: {e}", file=sys.stderr)
            success = False

    return success


# ═══════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════

def main():
    now = datetime.now(TZ)
    print(f"🐾 Yina 金融日报 — {now.strftime('%Y-%m-%d %H:%M')} (曼谷时间)")
    print(f"   v2.2: 日报格式 + 白毛股神叙事 + 五维评分 + 瓶颈逻辑")
    print()

    # Step 1: 获取24h行情
    print("📡 获取 24h 行情...")
    tickers = fetch_24hr_ticker(WATCHLIST)
    if "error" in tickers:
        print(f"   ❌ 获取行情失败: {tickers['error']}")
        sys.exit(1)
    print(f"   ✅ 已获取 {len(tickers)} 个交易对行情")

    # Step 2: 逐个获取K线 + 五维评分
    print("🧮 运行五维评分...")
    scores = []
    for i, sym in enumerate(WATCHLIST):
        base = sym.split("/")[0]
        klines = fetch_binance_klines(sym, interval="1h", limit=100)
        if "error" in klines:
            print(f"   ⚠️ {sym}: {klines['error']} — 跳过")
            continue

        sym_key = sym.replace("/", "")
        ticker = tickers.get(sym_key, {"price": klines["price"], "change_pct": 0, "volume_usdt": 0})
        meta = COIN_META.get(base, {"name": base, "theme": [], "supply_chain": None})
        sc = calc_five_dimension(klines, ticker, meta)
        scores.append(sc)

        if (i + 1) % 5 == 0:
            print(f"   ... {i+1}/{len(WATCHLIST)} 完成")

    print(f"   ✅ 评分完成: {len(scores)} 个标的")

    # Step 3: 生成报告 (金融日报格式 + 白毛股神叙事)
    print("📝 生成金融日报...")
    report = build_serenity_report(scores)
    print(f"   ✅ 日报生成完毕 ({len(report)} 字符, ~{len(report.encode('utf-8'))} bytes)")

    # Step 4: 推送到企业微信
    print("📤 推送企业微信「金融监控」...")
    ok = send_to_wecom(report)
    if ok:
        print("✅ 金融日报已推送到企业微信")
    else:
        print("❌ 推送失败，请检查网络和企业微信 Webhook", file=sys.stderr)
        sys.exit(1)

    # Step 5: 语音通知 (可选)
    print("🔊 语音通知...")
    os.system(f'python3 ~/.claude/tools/yina-notify.py "每小时市场分析已推送" &')
    print("✅ 金融日报发送完毕! 🐾")


if __name__ == "__main__":
    main()
