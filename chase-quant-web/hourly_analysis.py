#!/usr/bin/env python3
"""
🐾 Yina 金融日报引擎 v3.0 — 五市场全覆盖
= 加密货币 + A股 + 美股 + 港股 + bStocks
= 白毛股神 Serenity 瓶颈逻辑 + 大白话产业链叙事
每整点过后5分钟运行，推送企业微信「金融监控」群
"""
import json, sys, urllib.request, urllib.error
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
from ai_news_fetcher import get_ai_news_section

warnings.filterwarnings("ignore")

WEBHOOK_URL = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=2c602b48-5da2-4989-9193-30c0e226c769"
API_BASE = "http://localhost:8766"
TZ = timezone(timedelta(hours=7))
DATA_DIR = Path(__file__).parent / "data" / "hourly"
UA = "Yina-Hourly/3.0"
API_UA = "Mozilla/5.0"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════
# 元数据: 五市场 主题+产业链DAG
# ═══════════════════════════════════════════

COIN_META = {
    "BTC":{"name":"Bitcoin","theme":["store_of_value","macro_sensitive","etf_flow"],"supply_chain":None},
    "ETH":{"name":"Ethereum","theme":["smart_contract_platform","defi","layer0"],"supply_chain":"l1_infrastructure"},
    "BNB":{"name":"BNB Chain","theme":["exchange_token","defi","layer1"],"supply_chain":None},
    "SOL":{"name":"Solana","theme":["smart_contract_platform","high_performance","defi"],"supply_chain":"l1_infrastructure"},
    "XRP":{"name":"XRP","theme":["payment","institutional","legal_sensitive"],"supply_chain":None},
    "ADA":{"name":"Cardano","theme":["smart_contract_platform","research_driven"],"supply_chain":None},
    "DOGE":{"name":"Dogecoin","theme":["meme","social_sentiment","whale_driven"],"supply_chain":None},
    "AVAX":{"name":"Avalanche","theme":["smart_contract_platform","gaming","subnet"],"supply_chain":"l1_infrastructure"},
    "DOT":{"name":"Polkadot","theme":["interoperability","parachain"],"supply_chain":"cross_chain_infrastructure"},
    "LINK":{"name":"Chainlink","theme":["oracle","infrastructure","rwa"],"supply_chain":"data_infrastructure"},
    "MATIC":{"name":"Polygon","theme":["scaling","enterprise","zk"],"supply_chain":"l2_scaling"},
    "ATOM":{"name":"Cosmos","theme":["interoperability","ibc"],"supply_chain":"cross_chain_infrastructure"},
    "LTC":{"name":"Litecoin","theme":["payment","btc_beta"],"supply_chain":None},
    "UNI":{"name":"Uniswap","theme":["defi","dex"],"supply_chain":"defi_infrastructure"},
    "APT":{"name":"Aptos","theme":["smart_contract_platform","move_ecosystem"],"supply_chain":"l1_infrastructure"},
    "NEAR":{"name":"NEAR Protocol","theme":["smart_contract_platform","ai_integration"],"supply_chain":"l1_infrastructure"},
    "OP":{"name":"Optimism","theme":["scaling","superchain"],"supply_chain":"l2_scaling"},
    "ARB":{"name":"Arbitrum","theme":["scaling","defi"],"supply_chain":"l2_scaling"},
    "SUI":{"name":"Sui","theme":["smart_contract_platform","move_ecosystem","gaming"],"supply_chain":"l1_infrastructure"},
    "TON":{"name":"Toncoin","theme":["social_platform","messaging"],"supply_chain":None},
    "RENDER":{"name":"Render","theme":["ai_infrastructure","gpu_compute"],"supply_chain":"ai_compute_infrastructure"},
    "FIL":{"name":"Filecoin","theme":["storage","ai_data"],"supply_chain":"storage_infrastructure"},
}

ASTOCK_THEMES = {
    "600900":"电力/国企","000001":"银行/金融","601888":"免税/消费","000002":"地产/国企",
    "000858":"白酒/消费","002415":"安防/AI","002594":"新能源/汽车","300750":"电池/新能源",
    "600519":"白酒/消费","601318":"保险/金融","600036":"银行/金融","600276":"医药/创新药",
    "601012":"光伏/新能源","000568":"白酒/消费","002475":"消费电子/苹果链",
    "300059":"券商/互金","603259":"医药/CXO","600809":"白酒/消费","002714":"养殖/农业","300124":"工控/机器人",
}

USTOCK_THEMES = {
    "IWM":"小盘股ETF","SMH":"半导体ETF","TLT":"长期国债ETF","AAPL":"消费电子/AI","MSFT":"云计算/AI",
    "NVDA":"AI芯片","GOOGL":"搜索/AI","AMZN":"电商/云","TSLA":"电动车/AI","META":"社交/广告",
    "AMD":"芯片","NFLX":"流媒体","QQQ":"纳指ETF","SPY":"标普ETF","DIA":"道指ETF",
    "BRK.B":"保险/价值","JPM":"银行","V":"支付","JNJ":"医药/消费","XOM":"能源",
}

HKSTOCK_THEMES = {
    "01109":"地产/央企","02020":"运动消费","00939":"银行/国企","00700":"社交/游戏/AI","09988":"电商/云/AI",
    "01810":"手机/汽车","09618":"电商/物流","03690":"本地生活","02318":"保险/金融","00883":"能源/央企",
    "00388":"交易所","00175":"汽车/新能源","02269":"医药/CXO","01024":"短视频/直播","09888":"搜索/AI",
    "01299":"保险","00005":"银行/国际","02628":"保险/央企","01211":"汽车/新能源","01898":"能源/煤炭",
}

BSTOCK_THEMES = {
    "NVDAB":"AI芯片/币安","TSLAB":"电动车/币安","CRCLB":"稳定币/USDC","MUB":"比特币持仓","SNDKB":"存储/芯片",
}

MARKET_LABELS = {
    "crypto":"₿ 加密货币","a_stock":"🇨🇳 A股","us_stock":"🇺🇸 美股",
    "hk_stock":"🇭🇰 港股","b_stock":"🏦 bStocks",
}

SUPPLY_CHAIN_DAG = {
    "ai_compute_layer":{"name":"🧠 AI 算力层","bottleneck_score":9.2,"crypto_plays":["RENDER","FIL","NEAR"],"thesis":"AI训练需要分布式GPU算力 → Render/FIL等去中心化算力网络是瓶颈节点","key_question":"中心化云(AWS/Azure)会不会碾压去中心化方案？"},
    "defi_settlement_layer":{"name":"💱 DeFi 结算层","bottleneck_score":7.5,"crypto_plays":["ETH","UNI","LINK"],"thesis":"链上金融的清算/报价/预言机 → ETH是最终结算层, LINK是数据入口, UNI是流动性枢纽","key_question":"L2是否会完全吸走L1的DeFi活动？"},
    "cross_chain_infrastructure":{"name":"🌉 跨链互操作层","bottleneck_score":6.8,"crypto_plays":["DOT","ATOM"],"thesis":"多链未来需要跨链通信协议 → DOT/ATOM卡位互操作标准","key_question":"L2原生互操作(Superchain)会取代独立跨链协议吗？"},
    "l2_scaling":{"name":"⚡ L2 扩容层","bottleneck_score":7.0,"crypto_plays":["MATIC","OP","ARB"],"thesis":"ETH扩容需求确定 → L2是必经之路, 但竞争激烈","key_question":"ZK Rollup会不会让Optimistic Rollup过时？"},
    "data_oracle":{"name":"🔮 数据预言机","bottleneck_score":8.0,"crypto_plays":["LINK"],"thesis":"所有DeFi/RWA都需要外部数据 → LINK几乎是唯一选择","key_question":"竞争对手(Pyth/Band)市场份额增长有多快？"},
}

ASTOCK_DAG = {
    "baijiu_consumer":{"name":"🍶 白酒消费","bottleneck_score":7.5,"plays":["600519","000858","000568","600809"],"thesis":"白酒是A股最硬的消费护城河——品牌溢价+渠道垄断+成瘾性复购","key_question":"年轻人不喝白酒的趋势会不会加速？"},
    "new_energy":{"name":"⚡ 新能源产业链","bottleneck_score":8.0,"plays":["300750","002594","601012"],"thesis":"电池+光伏+整车——中国唯一全球领先的先进制造产业链","key_question":"产能过剩会不会导致利润率长期低迷？"},
    "finance_bank":{"name":"🏦 金融银行","bottleneck_score":5.0,"plays":["000001","600036","601318","300059"],"thesis":"高股息+低估值+国家队托盘——防御性配置的压舱石","key_question":"净息差持续收窄，银行利润从哪来？"},
    "ai_tech":{"name":"🤖 AI/科技","bottleneck_score":8.5,"plays":["002415","300124"],"thesis":"安防+机器人——AI落地最实在的两个赛道，政策加持","key_question":"海康受制裁影响多大？汇川的机器人量产进度？"},
    "pharma":{"name":"💊 医药健康","bottleneck_score":7.0,"plays":["600276","603259"],"thesis":"创新药+CXO——中国医药从仿制到创新的转型期","key_question":"中美关系会不会切断CXO的海外订单？"},
}

USTOCK_DAG = {
    "mag7_ai":{"name":"🧠 Mag7 AI核心","bottleneck_score":9.5,"plays":["NVDA","MSFT","GOOGL","AMZN","META","AAPL","TSLA"],"thesis":"AI军备竞赛的最大受益者——算力(NVDA)+云(MSFT/AMZN/GOOGL)+应用(META/AAPL/TSLA)","key_question":"AI投入的ROI什么时候能兑现？2万亿Capex何时回本？"},
    "semiconductor":{"name":"🔬 半导体","bottleneck_score":9.0,"plays":["SMH","NVDA","AMD"],"thesis":"芯片是现代经济的石油——SMH是整个半导体产业链的ETF代理","key_question":"台海地缘风险溢价是否已经定价？"},
    "rate_sensitive":{"name":"📊 利率敏感","bottleneck_score":7.0,"plays":["TLT","IWM","JPM","SPY"],"thesis":"降息预期→TLT涨+小盘股(IWM)涨+银行(JPM)息差改善","key_question":"美联储到底什么时候降息？市场是不是太乐观了？"},
}

HKSTOCK_DAG = {
    "china_internet":{"name":"🌐 中国互联网","bottleneck_score":8.5,"plays":["00700","09988","09618","03690","01024","09888"],"thesis":"腾讯+阿里+美团+京东+快手+百度——中国互联网六根柱子，政策底已过","key_question":"监管会不会再来一轮？AI对互联网的增量有多大？"},
    "auto_ev":{"name":"🚗 新能源汽车","bottleneck_score":8.0,"plays":["01810","00175","01211"],"thesis":"小米+吉利+比亚迪——中国EV全球扩张，但价格战猛烈","key_question":"欧洲关税壁垒会不会封杀中国EV出海？"},
    "finance_hk":{"name":"🏦 金融央企","bottleneck_score":5.5,"plays":["00939","02318","00388","01299","02628","00005"],"thesis":"高股息+低估值+南下资金持续流入——港股的价值锚","key_question":"港股流动性什么时候能真正改善？"},
    "energy_soe":{"name":"⛽ 能源央企","bottleneck_score":6.0,"plays":["00883","01898","01109"],"thesis":"中海油+中煤+华润置地——垄断性资源+稳定分红","key_question":"油价/煤价下行周期会不会压缩分红？"},
}

BSTOCK_DAG = {
    "binance_tokenized":{"name":"🔗 币安代币化美股","bottleneck_score":7.0,"plays":["NVDAB","TSLAB","CRCLB","MUB","SNDKB"],"thesis":"24/7交易+USDT计价+免手续费——传统美股的Crypto入口","key_question":"监管风险：币安代币化股票的法律地位？"},
}


# ═══════════════════════════════════════════
# 数据获取 (从本地API)
# ═══════════════════════════════════════════

def api_get(path: str) -> dict:
    url = f"{API_BASE}{path}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": API_UA})
        with urllib.request.urlopen(req, timeout=90) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def fetch_all_signals() -> Dict[str, list]:
    all_signals = {}
    for market in ["crypto", "a_stock", "us_stock", "hk_stock", "b_stock"]:
        data = api_get(f"/api/signals?market={market}")
        if "error" in data:
            print(f"   ⚠️ {market} 信号获取失败: {data['error']}")
            all_signals[market] = []
        else:
            sigs = data.get("signals", {}).get(market, [])
            all_signals[market] = sigs
            print(f"   ✅ {MARKET_LABELS[market]}: {len(sigs)} 个信号")
    return all_signals


def fetch_crypto_24hr(symbols: list) -> dict:
    if not symbols:
        return {}
    syms = ",".join([f'"{s.replace("/","")}"' for s in symbols])
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


# ═══════════════════════════════════════════
# 统一资产模型
# ═══════════════════════════════════════════

@dataclass
class MarketAsset:
    symbol: str
    market: str
    name: str
    price: float
    change_24h: float
    score: int
    action: str
    stars: int
    confidence: float
    reasons: list
    risk_level: str
    suggested_size: float
    stop_loss: float
    take_profit: float
    theme: str = ""
    supply_chain: str = ""
    bottleneck_score: float = 0.0


def score_from_signal(sig: dict, market: str, ticker_24h: dict = None) -> MarketAsset:
    symbol = sig["symbol"]
    price = sig.get("price", 0)
    score = int(sig.get("adjusted_score", sig.get("score", 50)))
    action = sig.get("action", "WATCH")
    reasons = sig.get("reasons", [])
    change_24h = 0.0
    if ticker_24h and market in ("crypto", "b_stock"):
        sym_key = symbol.replace("/", "")
        t = ticker_24h.get(sym_key, {})
        change_24h = t.get("change_pct", 0)
        if t.get("price"):
            price = t["price"]
    theme = ""
    supply_chain = ""
    bt_score = 0.0
    if market == "crypto":
        base = symbol.split("/")[0]
        meta = COIN_META.get(base, {})
        theme = ", ".join(meta.get("theme", [])[:2])
        for lid, l in SUPPLY_CHAIN_DAG.items():
            if base in l["crypto_plays"]:
                supply_chain = l["name"]
                bt_score = l["bottleneck_score"]
                break
    elif market == "a_stock":
        theme = ASTOCK_THEMES.get(symbol, "")
        for lid, l in ASTOCK_DAG.items():
            if symbol in l["plays"]:
                supply_chain = l["name"]
                bt_score = l["bottleneck_score"]
                break
    elif market == "us_stock":
        theme = USTOCK_THEMES.get(symbol, "")
        for lid, l in USTOCK_DAG.items():
            if symbol in l["plays"]:
                supply_chain = l["name"]
                bt_score = l["bottleneck_score"]
                break
    elif market == "hk_stock":
        theme = HKSTOCK_THEMES.get(symbol, "")
        for lid, l in HKSTOCK_DAG.items():
            if symbol in l["plays"]:
                supply_chain = l["name"]
                bt_score = l["bottleneck_score"]
                break
    elif market == "b_stock":
        base = symbol.split("/")[0] if "/" in symbol else symbol
        theme = BSTOCK_THEMES.get(base, "")
        for lid, l in BSTOCK_DAG.items():
            if base in l["plays"]:
                supply_chain = l["name"]
                bt_score = l["bottleneck_score"]
                break
    stars = min(5, max(1, round(score / 20)))
    return MarketAsset(
        symbol=symbol, market=market, name=sig.get("name", symbol),
        price=price, change_24h=change_24h,
        score=score, action=action, stars=stars,
        confidence=sig.get("confidence", 0.5),
        reasons=reasons, risk_level=sig.get("risk_level", "medium"),
        suggested_size=sig.get("suggested_size", 0),
        stop_loss=sig.get("stop_loss", 0),
        take_profit=sig.get("take_profit", 0),
        theme=theme, supply_chain=supply_chain, bottleneck_score=bt_score,
    )


# ═══════════════════════════════════════════
# 报告生成
# ═══════════════════════════════════════════

def generate_market_pulse(all_assets: List) -> str:
    now = datetime.now(TZ)
    lines = [f"**{now.strftime('%m/%d %H:%M')} 曼谷时间 · 五市场全景**", ""]
    for market in ["crypto", "a_stock", "us_stock", "hk_stock", "b_stock"]:
        assets = [a for a in all_assets if a.market == market]
        if not assets:
            continue
        label = MARKET_LABELS[market]
        n_buy = sum(1 for a in assets if a.action == "BUY")
        n_sell = sum(1 for a in assets if a.action == "SELL")
        n_watch = len(assets) - n_buy - n_sell
        avg_score = sum(a.score for a in assets) / len(assets)
        buy_ratio = n_buy / len(assets) if assets else 0
        heat = "🔥" if buy_ratio >= 0.5 else "🌤️" if buy_ratio >= 0.3 else "🌧️" if n_sell > n_buy else "🌥️"
        top3 = sorted(assets, key=lambda a: a.score, reverse=True)[:3]
        top_str = " · ".join([f"{a.symbol}({a.score})" for a in top3])
        lines.append(f"{heat} **{label}**: 均分{avg_score:.0f} | 🟢{n_buy} 🟡{n_watch} 🔴{n_sell}")
        lines.append(f"> {top_str}")
    return "\n".join(lines)


def generate_crypto_narrative(assets: List) -> str:
    crypto_assets = [a for a in assets if a.market == "crypto"]
    if not crypto_assets:
        return ""
    lines = ["### 🦊 白毛视角 · Crypto 瓶颈叙事", ""]
    layer_summaries = []
    for layer_id, layer in SUPPLY_CHAIN_DAG.items():
        layer_assets = [a for a in crypto_assets if a.symbol.split("/")[0] in layer["crypto_plays"]]
        if not layer_assets:
            continue
        avg_score = sum(a.score for a in layer_assets) / len(layer_assets)
        layer_summaries.append({"layer": layer, "avg_score": avg_score, "assets": layer_assets, "bt_score": layer["bottleneck_score"]})
    layer_summaries.sort(key=lambda x: x["bt_score"], reverse=True)
    for ls in layer_summaries[:3]:
        l = ls["layer"]
        coins_str = "、".join([a.symbol.split("/")[0] for a in ls["assets"]])
        lines.append(f"**{l['name']}** (瓶颈{l['bottleneck_score']:.1f}/10): {coins_str}")
        lines.append(f"> {l['thesis']}")
        lines.append(f"> ⚡ {l['key_question']}")
        best = max(ls["assets"], key=lambda a: a.score)
        if best.action == "BUY":
            lines.append(f"> 👉 盯 **{best.symbol}** ({best.score}分)")
        lines.append("")
    return "\n".join(lines)


def generate_stock_narrative(assets: List, market: str, dag: dict) -> str:
    market_assets = [a for a in assets if a.market == market]
    if not market_assets:
        return ""
    label = MARKET_LABELS[market]
    lines = [f"### 🦊 白毛视角 · {label} 产业链叙事", ""]
    chain_summaries = []
    for chain_id, chain in dag.items():
        chain_assets = [a for a in market_assets if a.symbol in chain["plays"]]
        if not chain_assets:
            continue
        avg_score = sum(a.score for a in chain_assets) / len(chain_assets)
        chain_summaries.append({"chain": chain, "avg_score": avg_score, "assets": chain_assets, "bt_score": chain["bottleneck_score"]})
    chain_summaries.sort(key=lambda x: x["bt_score"], reverse=True)
    for cs in chain_summaries[:2]:
        c = cs["chain"]
        names = "、".join([a.symbol for a in cs["assets"]])
        lines.append(f"**{c['name']}** (瓶颈{c['bottleneck_score']:.1f}/10): {names}")
        lines.append(f"> {c['thesis']}")
        lines.append(f"> ⚡ {c['key_question']}")
        best = max(cs["assets"], key=lambda a: a.score)
        if best.action == "BUY":
            lines.append(f"> 👉 盯 **{best.symbol}** {best.name} ({best.score}分)")
        lines.append("")
    top3 = sorted(market_assets, key=lambda a: a.score, reverse=True)[:3]
    lines.append(f"**🏆 {label} Top3**:")
    for a in top3:
        emoji = "🟢" if a.action == "BUY" else "🟡" if a.action == "WATCH" else "🔴"
        theme_str = f" | {a.theme}" if a.theme else ""
        lines.append(f"> {emoji} **{a.symbol}** {a.name} {a.score}分{theme_str}")
    lines.append("")
    return "\n".join(lines)


def generate_top5_cross_market(all_assets: List) -> str:
    sorted_assets = sorted(all_assets, key=lambda a: a.score, reverse=True)
    picked = []
    seen_markets = set()
    for a in sorted_assets:
        if a.market not in seen_markets or len(picked) < 5:
            if a not in picked:
                picked.append(a)
                seen_markets.add(a.market)
        if len(picked) >= 5:
            break
    if len(picked) < 5:
        for a in sorted_assets:
            if a not in picked:
                picked.append(a)
            if len(picked) >= 5:
                break
    lines = ["### 🎯 五市场 · 最值得盯的", ""]
    for i, a in enumerate(picked[:5], 1):
        emoji = "🟢" if a.action == "BUY" else "🟡" if a.action == "WATCH" else "🔴"
        mkt_emoji = MARKET_LABELS.get(a.market, "").split(" ")[0]
        theme_str = f" | {a.theme}" if a.theme else ""
        reason_str = a.reasons[0] if a.reasons else ""
        lines.append(f"**{i}. {emoji} {mkt_emoji} {a.symbol}** {a.name} | {a.action} ⭐{a.stars} | {a.score}分")
        if a.market == "a_stock":
            lines.append(f"> 💰 ¥{a.price:.2f}")
        elif a.price > 100:
            lines.append(f"> 💰 ${a.price:,.0f}")
        elif a.price > 1:
            lines.append(f"> 💰 ${a.price:.2f}")
        else:
            lines.append(f"> 💰 ${a.price:.4f}")
        if a.supply_chain:
            lines.append(f"> 🔬 {a.supply_chain}{theme_str}")
        if reason_str:
            lines.append(f"> 📊 {reason_str}")
        if a.stop_loss and a.stop_loss > 0:
            lines.append(f"> ⚠️ 止损: {a.stop_loss:,.2f}")
        lines.append("")
    return "\n".join(lines)


def build_market_scoreboard(all_assets: List) -> str:
    lines = ["### 📋 速查 · 五市场评分卡", ""]
    for market in ["crypto", "a_stock", "us_stock", "hk_stock", "b_stock"]:
        assets = sorted([a for a in all_assets if a.market == market], key=lambda a: a.score, reverse=True)
        if not assets:
            continue
        label = MARKET_LABELS[market]
        lines.append(f"**{label}**")
        lines.append("| 标的 | 价格 | 评分 | 行动 | 主题 |")
        lines.append("|------|------|------|------|------|")
        for a in assets[:8]:
            emoji = "🟢" if a.action == "BUY" else "🟡" if a.action == "WATCH" else "🔴"
            if a.market == "a_stock":
                price_str = f"¥{a.price:.2f}"
            elif a.price > 100:
                price_str = f"${a.price:,.0f}"
            else:
                price_str = f"${a.price:.2f}"
            theme_str = a.theme[:12] if a.theme else "-"
            lines.append(f"| {emoji} {a.symbol} | {price_str} | **{a.score}** | {a.action} | {theme_str} |")
        if len(assets) > 8:
            lines.append(f"| ... | +{len(assets)-8}只 | | | |")
        lines.append("")
    return "\n".join(lines)


def build_five_market_report(all_signals: Dict[str, list]) -> str:
    now = datetime.now(TZ)
    crypto_symbols = [s["symbol"] for s in all_signals.get("crypto", [])]
    bstock_symbols = [s["symbol"] for s in all_signals.get("b_stock", [])]
    ticker_24h = fetch_crypto_24hr(crypto_symbols + bstock_symbols)

    all_assets = []
    for market in ["crypto", "a_stock", "us_stock", "hk_stock", "b_stock"]:
        for sig in all_signals.get(market, []):
            all_assets.append(score_from_signal(sig, market, ticker_24h))

    total_n = len(all_assets)
    n_buy = sum(1 for a in all_assets if a.action == "BUY")
    n_sell = sum(1 for a in all_assets if a.action == "SELL")
    n_watch = total_n - n_buy - n_sell
    btc_asset = next((a for a in all_assets if a.symbol == "BTC/USDT"), None)

    lines = []
    lines.append(f"# 📰 Yina 金融日报 v3.0")
    lines.append(f"## {now.strftime('%m/%d %H:%M')} 曼谷 · 五市场全覆盖 · 白毛逻辑")
    lines.append("")
    lines.append(f"> 🟢{n_buy}只可买 🟡{n_watch}只观望 🔴{n_sell}只回避 | 共{total_n}标的 五市场")
    lines.append("")
    lines.append(generate_market_pulse(all_assets))
    lines.append("")
    lines.append(generate_crypto_narrative(all_assets))
    lines.append(generate_stock_narrative(all_assets, "us_stock", USTOCK_DAG))
    lines.append(generate_stock_narrative(all_assets, "a_stock", ASTOCK_DAG))
    lines.append(generate_stock_narrative(all_assets, "hk_stock", HKSTOCK_DAG))
    bstock_assets = [a for a in all_assets if a.market == "b_stock"]
    if bstock_assets:
        lines.append(f"### 🏦 bStocks · 币安代币化美股")
        lines.append("")
        for a in sorted(bstock_assets, key=lambda x: x.score, reverse=True):
            emoji = "🟢" if a.action == "BUY" else "🟡" if a.action == "WATCH" else "🔴"
            theme_str = f" | {a.theme}" if a.theme else ""
            lines.append(f"> {emoji} **{a.symbol}** {a.name} | {a.score}分 | ${a.price:.2f}{theme_str}")
        lines.append("")
    lines.append(generate_top5_cross_market(all_assets))
    lines.append(build_market_scoreboard(all_assets))

    # 🤖 AI信源动态
    try:
        ai_section = get_ai_news_section()
        lines.append(ai_section)
        lines.append("")
    except Exception as e:
        print(f"   ⚠️ AI信源采集失败: {e}")

    btc_hint = ""
    if btc_asset and btc_asset.change_24h < -3:
        btc_hint = " ⚠️ BTC大盘承压，所有买入谨慎！"
    elif btc_asset and btc_asset.change_24h > 5:
        btc_hint = " BTC强势，多头氛围偏暖。"
    lines.append(f"### 🛡️ 风控")
    lines.append(f"> 硬止损-8% · 日熔断-5% · 持仓≤5只{btc_hint}")
    lines.append(f"> 配置: Crypto 35% | A股 30% | 美股 20% | bStocks 15% | 港股 10%")
    lines.append("")
    lines.append("---")
    lines.append(f"🐾 Yina 自动日报 v3.0 · 五市场全覆盖 · 白毛股神瓶颈逻辑 · {now.strftime('%H:%M')}")
    return "\n".join(lines)


# ═══════════════════════════════════════════
# 企业微信推送
# ═══════════════════════════════════════════

def send_to_wecom(md_content: str) -> bool:
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
        payload = json.dumps({"msgtype": "markdown", "markdown": {"content": prefix + chunk}}).encode("utf-8")
        req = urllib.request.Request(WEBHOOK_URL, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                if result.get("errcode") != 0:
                    print(f"   ⚠️ 分段{i+1}推送失败: {result}")
                    success = False
                else:
                    print(f"   ✅ 分段{i+1}/{len(chunks)} 推送成功")
        except Exception as e:
            print(f"   ❌ 分段{i+1}推送异常: {e}")
            success = False
    return success


# ═══════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════

def main():
    now = datetime.now(TZ)
    print(f"🐾 Yina 金融日报 v3.0 — {now.strftime('%Y-%m-%d %H:%M')} (曼谷时间)")
    print(f"   五市场全覆盖: Crypto + A股 + 美股 + 港股 + bStocks")
    print()

    print("📡 获取五市场信号...")
    all_signals = fetch_all_signals()
    total_signals = sum(len(v) for v in all_signals.values())
    print(f"   ✅ 共获取 {total_signals} 个信号 (应为~80个)")
    print()

    print("📝 生成五市场全景日报...")
    report = build_five_market_report(all_signals)
    report_bytes = len(report.encode('utf-8'))
    print(f"   ✅ 日报生成完毕 ({len(report)} 字符, ~{report_bytes} bytes)")
    print()

    print("📤 推送企业微信「金融监控」...")
    ok = send_to_wecom(report)
    if ok:
        print("✅ 金融日报已推送到企业微信！")
    else:
        print("❌ 推送失败，请检查")
        sys.exit(1)

    print("✅ 五市场金融日报发送完毕! 🐾")


if __name__ == "__main__":
    main()
