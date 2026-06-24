"""
Yina 统一标的配置中心 🐾
========================
分层扫描: ML深度扫描 → 技术面轻扫 → 动量筛选
扩展至 OKX 全量加密货币 + 股票永续合约 + ETF + 大宗商品

配置来源:
  - OKX 现货 USDT 交易对 (~297个) → 加密货币
  - OKX 永续合约 (swap, linear) → 139只美股 + 12 ETF + 5 商品 + 3 韩股
  - 永续合约说明: type=swap, linear=True (USDT保证金), 可做多/空, 有杠杆

Chase哥 实盘交易: OKX 上的股票都是永续合约, 不是现货!
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# ═══════════════════════════════════════════════════════════════
# Tier 1 — ML 深度扫描 (Top ~36 高流动性加密货币)
# 每个币跑完整的 Qlib + LightGBM + 图增强 + 五维评分
# ═══════════════════════════════════════════════════════════════

TIER1_ML_HEAVY = [
    # ── 大蓝筹 ──
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    # ── 主流币 ──
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
    "POL/USDT", "UNI/USDT", "ATOM/USDT", "APT/USDT", "OP/USDT",
    # ── H开头 (Chase哥 钦点) ──
    "HBAR/USDT",
    # HYPE/USDT — OKX only, Binance无此交易对, ML扫描需用OKX数据源
    # ⚠️ 暂放Tier1外, 等OKX数据源接入后再加入
    # ── 强势赛道 ──
    "ARB/USDT", "NEAR/USDT", "INJ/USDT", "LTC/USDT", "ETC/USDT",
    "PEPE/USDT", "SHIB/USDT", "WIF/USDT", "BONK/USDT",  # Meme
    "RENDER/USDT", "FET/USDT", "WLD/USDT",  # AI赛道
    "TIA/USDT", "SEI/USDT", "SUI/USDT", "STRK/USDT", "JUP/USDT",  # 新公链+DeFi
    "ONDO/USDT", "ORDI/USDT",  # RWA + BRC20
]

# ═══════════════════════════════════════════════════════════════
# Tier 2 — 技术面快扫 (~50个次主流加密货币)
# 跑传统指标 (RSI/MACD/BB/量价) 不用ML模型
# ═══════════════════════════════════════════════════════════════

TIER2_TECHNICAL_LIGHT = [
    "FIL/USDT", "TRX/USDT", "ICP/USDT", "GRT/USDT", "ALGO/USDT",
    "SAND/USDT", "MANA/USDT", "AXS/USDT", "GALA/USDT", "APE/USDT",
    "FLOW/USDT", "CHZ/USDT", "ENJ/USDT", "LDO/USDT", "CRV/USDT",
    "COMP/USDT", "AAVE/USDT", "MKR/USDT", "SNX/USDT", "YFI/USDT",
    "1INCH/USDT", "DYDX/USDT", "GMX/USDT", "PENDLE/USDT", "LRC/USDT",
    "IMX/USDT", "ENS/USDT", "BAT/USDT", "ZRX/USDT", "KNC/USDT",
    "BAND/USDT", "OCEAN/USDT", "ROSE/USDT", "IOTA/USDT", "NEO/USDT",
    "ONT/USDT", "VET/USDT", "QTUM/USDT", "ZIL/USDT", "RVN/USDT",
    "CELO/USDT", "ONE/USDT", "THETA/USDT", "XTZ/USDT", "KSM/USDT",
    "EGLD/USDT", "DASH/USDT", "ZEC/USDT", "WAVES/USDT", "IOST/USDT",
]

# ═══════════════════════════════════════════════════════════════
# Tier 3 — 动量过滤器 (低流动性加密货币)
# 只在 24h 成交量飙升 or 价格突破关键位 时触发扫描
# ═══════════════════════════════════════════════════════════════

TIER3_MOMENTUM_FILTER = [
    # 运行时从 OKX 动态拉取，只保留Tier1/2之外的
    # 触发器: 24h vol > $5M 且 price change > 5% 时进入快扫
]

# ═══════════════════════════════════════════════════════════════
# 📊 OKX 永续合约 — 股票 / ETF / 商品
# 这些都是 USDT 保证金的线性永续合约 (swap, linear)
# 可以直接用 USDT 做多/做空, 支持杠杆
# Chase哥 实盘在用!!
# ═══════════════════════════════════════════════════════════════

# ── 半导体 & 芯片 (Chase哥 重点关注) ──
OKX_SEMICONDUCTOR_SWAPS = [
    "NVDA/USDT",     # 🥇 英伟达 NVIDIA — AI算力之王
    "AMD/USDT",      # 🥈 AMD — 数据中心GPU
    "INTC/USDT",     # 英特尔 Intel — 老牌芯片
    "MU/USDT",       # 🔥 美光科技 Micron — Chase哥持仓!
    "MRVL/USDT",     # Marvell — 数据中心芯片
    "AVGO/USDT",     # 博通 Broadcom — 网络芯片
    "QCOM/USDT",     # 高通 Qualcomm — 移动芯片
    "TSM/USDT",      # 台积电 TSMC — 芯片代工之王
    "ARM/USDT",      # ARM — 芯片架构
    "ASML/USDT",     # 🥇 阿斯麦 ASML — 光刻机垄断
    "AMAT/USDT",     # 应用材料 Applied Materials — 半导体设备
    "COHR/USDT",     # Coherent — 光模块
    "CIEN/USDT",     # Ciena — 光网络
    "CRDO/USDT",     # Credo — 高速互联
    "CGNX/USDT",     # Cognex — 机器视觉
    "AXTI/USDT",     # AXT — 砷化镓衬底
    "POET/USDT",     # POET Technologies — 光引擎
    "WDC/USDT",      # 西部数据 Western Digital — 存储
    "SNDK/USDT",     # 闪迪 Sandisk — NAND闪存
    "FLNC/USDT",     # Fluence Energy — 储能 (与芯片周期联动)
]

# ── AI & 软件 (非芯片但AI驱动) ──
OKX_AI_SOFTWARE_SWAPS = [
    "MSFT/USDT",     # 微软 Microsoft — OpenAI 最大股东
    "GOOGL/USDT",    # 谷歌 Alphabet — Gemini
    "META/USDT",     # Meta — Llama开源模型
    "AMZN/USDT",     # 亚马逊 Amazon — AWS AI
    "AAPL/USDT",     # 苹果 Apple — Apple Intelligence
    "ORCL/USDT",     # 甲骨文 Oracle — AI云
    "ADBE/USDT",     # Adobe — AI创意
    "CRM/USDT",      # Salesforce — (check if listed)
    "NOW/USDT",      # ServiceNow — AI自动化
    "PANW/USDT",     # Palo Alto Networks — AI安全
    "CRWD/USDT",     # CrowdStrike — AI安全
    "PLTR/USDT",     # Palantir — AI数据分析
    "TWLO/USDT",     # Twilio — 通信API
    "BILL/USDT",     # Bill.com — 财务AI
    "NBIS/USDT",     # Nebius — AI云基础设施
]

# ── 太空经济 🚀 (Chase哥 感兴趣) ──
# ⚠️ SpaceX 未上市 (私有公司)
# ⚠️ SPCX/USDT 是太空ETF的永续合约, 间接持有SpaceX敞口!
OKX_SPACE_SWAPS = [
    "RKLB/USDT",     # 🚀 Rocket Lab — 小卫星发射, SpaceX竞争对手
    "ASTS/USDT",     # 📡 AST SpaceMobile — 卫星直连手机
    "LUNR/USDT",     # 🌙 Intuitive Machines — 月球着陆器
    "RDW/USDT",      # 🛰️ Redwire — 太空制造
    "SPCX/USDT",     # 🌌 太空ETF合约 — 间接持有SpaceX!
]

# ── 消费 & 金融科技 ──
OKX_CONSUMER_FINTECH_SWAPS = [
    "TSLA/USDT",     # 🔥 特斯拉 Tesla — Chase哥持仓!
    "COIN/USDT",     # Coinbase — 加密交易所
    "MSTR/USDT",     # MicroStrategy — 比特币大户
    "HOOD/USDT",     # Robinhood — 散户交易平台
    "COST/USDT",     # Costco — 零售
    "HIMS/USDT",     # Hims & Hers — 在线医疗
    "GME/USDT",      # GameStop — 模因股鼻祖
    "DELL/USDT",     # 戴尔 Dell — AI服务器
    "IBM/USDT",      # IBM — 量子计算
    "NFLX/USDT",     # 奈飞 Netflix
    "LLY/USDT",      # 礼来 Eli Lilly — 减肥药
    "ISRG/USDT",     # Intuitive Surgical — 手术机器人
    "HPE/USDT",      # 惠普企业 HPE
    "GEV/USDT",      # GE Vernova — 能源转型
    "GLW/USDT",      # 康宁 Corning — 玻璃/光纤
    "NOK/USDT",      # 诺基亚 Nokia
    "CSCO/USDT",     # 思科 Cisco
    "CRWV/USDT",     # CoreWeave — AI云IPO新贵
]

# ── 🇰🇷 韩国股票 (Chase哥 特别关注) ──
OKX_KOREAN_STOCKS = [
    "SKHYNIX/USDT",  # 🔥 SK海力士 — HBM高带宽内存龙头!
    "SAMSUNG/USDT",  # 三星电子 — 全球最大存储芯片
    "HYUNDAI/USDT",  # 现代汽车 — 韩国制造业
]

# ── 📈 ETF 永续合约 ──
OKX_ETF_SWAPS = [
    "SPY/USDT",      # 标普500 ETF
    "QQQ/USDT",      # 纳斯达克100 ETF
    "IWM/USDT",      # 罗素2000 小盘股
    "XLE/USDT",      # 能源板块 ETF
    "SOXL/USDT",     # 半导体3倍做多 ETF 🔥
    "EWJ/USDT",      # 日本 iShares MSCI
    "EWT/USDT",      # 台湾 iShares MSCI
    "EWY/USDT",      # 韩国 iShares MSCI
    "USO/USDT",      # 美国原油 ETF
]

# ── 🏆 大宗商品 ──
OKX_COMMODITY_SWAPS = [
    "XAU/USDT",      # 🥇 黄金 Gold
    "XAG/USDT",      # 🥈 白银 Silver
    "XCU/USDT",      # 🥉 铜 Copper
    "XPD/USDT",      # 钯金 Palladium
    "XPT/USDT",      # 铂金 Platinum
]

# ── 🤖 AI 概念币 (Crypto AI, 非股票) ──
OKX_AI_CONCEPT_SWAPS = [
    "OPENAI/USDT",       # OpenAI 概念 (预测市场)
    "ANTHROPIC/USDT",    # Anthropic 概念 (预测市场)
    "AIXBT/USDT",        # AI 交易信号
    "COAI/USDT",         # AI 概念
]

# ── 汇总: OKX 上所有股票/ETF/商品永续合约 ──
OKX_ALL_STOCK_SWAPS = (
    OKX_SEMICONDUCTOR_SWAPS
    + OKX_AI_SOFTWARE_SWAPS
    + OKX_SPACE_SWAPS
    + OKX_CONSUMER_FINTECH_SWAPS
    + OKX_KOREAN_STOCKS
)

OKX_ALL_NON_CRYPTO_SWAPS = (
    OKX_ALL_STOCK_SWAPS
    + OKX_ETF_SWAPS
    + OKX_COMMODITY_SWAPS
    + OKX_AI_CONCEPT_SWAPS
)

# ═══════════════════════════════════════════════════════════════
# 美股 (通过 Yahoo Finance 等外部API)
# 用于补充基本面数据 (PE/市值/财报), OKX合约不提供这些
# ═══════════════════════════════════════════════════════════════

US_STOCKS_WATCH = [
    {"symbol": "MU", "name": "美光科技 Micron", "market": "NASDAQ"},
    {"symbol": "TSLA", "name": "特斯拉 Tesla", "market": "NASDAQ"},
    {"symbol": "NVDA", "name": "英伟达 NVIDIA", "market": "NASDAQ"},
    {"symbol": "AMD", "name": "AMD", "market": "NASDAQ"},
    {"symbol": "SMCI", "name": "超微电脑 Super Micro", "market": "NASDAQ"},
    {"symbol": "COIN", "name": "Coinbase", "market": "NASDAQ"},
    {"symbol": "MSTR", "name": "MicroStrategy", "market": "NASDAQ"},
    {"symbol": "ASML", "name": "阿斯麦 ASML", "market": "NASDAQ"},
    {"symbol": "AVGO", "name": "博通 Broadcom", "market": "NASDAQ"},
    {"symbol": "QCOM", "name": "高通 Qualcomm", "market": "NASDAQ"},
    {"symbol": "PLTR", "name": "Palantir", "market": "NYSE"},
    {"symbol": "TSM", "name": "台积电 TSMC", "market": "NYSE"},
    {"symbol": "ARM", "name": "ARM Holdings", "market": "NASDAQ"},
    {"symbol": "CRWD", "name": "CrowdStrike", "market": "NASDAQ"},
    {"symbol": "MRVL", "name": "Marvell Technology", "market": "NASDAQ"},
]

KR_STOCKS_WATCH = [
    {"symbol": "000660.KS", "name": "SK海力士 SK Hynix", "market": "KOSPI"},
    {"symbol": "005930.KS", "name": "三星电子 Samsung", "market": "KOSPI"},
]

# ⚠️ SpaceX — 未上市, 是私有公司!
# 间接参与方式:
#   - SPCX/USDT (OKX永续合约) — 太空ETF, 持有SpaceX股份
#   - DXYZ — Destiny Tech100 封闭基金
#   - RKLB, ASTS, LUNR, RDW — 已上市的太空公司


# ═══════════════════════════════════════════════════════════════
# 动态拉取: OKX REST API
# ═══════════════════════════════════════════════════════════════

def _okx_rest_instruments(inst_type: str) -> List[dict]:
    """
    通过 OKX REST API 获取交易对列表 (绕过 ccxt load_markets bug)。

    Args:
        inst_type: "SPOT" / "SWAP"

    Returns:
        [{baseCcy, quoteCcy, instId, state, ...}, ...]
    """
    import requests
    try:
        url = "https://www.okx.cab/api/v5/public/instruments"
        params = {"instType": inst_type}
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        if data.get("code") == "0":
            instruments = data.get("data", [])
            # 只保留 live 状态
            return [i for i in instruments if i.get("state") == "live"]
        else:
            print(f"⚠️ OKX REST instruments({inst_type}) 异常: {data.get('msg', '?')}")
            return []
    except Exception as e:
        print(f"⚠️ 无法拉取 OKX {inst_type} 交易对: {e}")
        return []


def fetch_okx_spot_symbols() -> List[str]:
    """从 OKX REST API 实时拉取所有 USDT 现货交易对 (纯加密货币)"""
    try:
        instruments = _okx_rest_instruments("SPOT")
        symbols = sorted([
            f"{i['baseCcy']}/{i['quoteCcy']}"
            for i in instruments
            if i.get("quoteCcy") == "USDT"
        ])
        return symbols
    except Exception as e:
        print(f"⚠️ 无法拉取 OKX 现货交易对: {e}")
        return []


def fetch_okx_swap_symbols() -> Dict[str, List[str]]:
    """
    从 OKX REST API 拉取所有 USDT 永续合约, 分类返回。

    OKX SWAP instruments API: instId="BTC-USDT-SWAP", settleCcy="USDT", ctType="linear"
    需要从 instId 解析 base/quote。

    Returns:
        {
            "stocks": [...],       # 股票
            "etfs": [...],         # ETF
            "commodities": [...],  # 大宗商品
            "crypto_swaps": [...], # 加密货币合约
            "all": [...],          # 全部
        }
    """
    try:
        instruments = _okx_rest_instruments("SWAP")

        # 已知分类
        etf_bases = {"SPY", "QQQ", "IWM", "XLE", "SOXL", "EWJ", "EWT", "EWY",
                     "ROBO", "URNM", "USO", "SPX", "SPCX", "SPACE"}
        commodity_bases = {"XAU", "XAG", "XCU", "XPD", "XPT"}
        # 所有加密货币的 base (从 SPOT 获取)
        spot_instruments = _okx_rest_instruments("SPOT")
        crypto_bases = {
            i["baseCcy"] for i in spot_instruments
            if i.get("quoteCcy") == "USDT"
        }
        # 手动补充 (可能没spot但肯定是crypto)
        for base in ["BTC", "ETH", "USDC", "DAI", "WBTC", "STETH"]:
            crypto_bases.add(base)

        categorized = {"stocks": [], "etfs": [], "commodities": [],
                      "crypto_swaps": [], "all": []}

        for inst in instruments:
            # 只取 USDT-margined linear swaps
            if inst.get("settleCcy") != "USDT" or inst.get("ctType") != "linear":
                continue

            inst_id = inst["instId"]  # "BTC-USDT-SWAP"
            # 解析: "BTC-USDT-SWAP" → ("BTC", "USDT")
            parts = inst_id.split("-")
            if len(parts) < 2:
                continue
            base = parts[0]
            quote = "USDT"
            std_sym = f"{base}/{quote}"

            if base in crypto_bases:
                categorized["crypto_swaps"].append(std_sym)
            elif base in commodity_bases:
                categorized["commodities"].append(std_sym)
            elif base in etf_bases:
                categorized["etfs"].append(std_sym)
            else:
                categorized["stocks"].append(std_sym)

            categorized["all"].append(std_sym)

        return categorized
    except Exception as e:
        print(f"⚠️ 无法拉取 OKX 永续合约: {e}")
        return {"stocks": [], "etfs": [], "commodities": [],
                "crypto_swaps": [], "all": []}


def build_tiered_symbols() -> dict:
    """
    构建分层标的池 (加密货币现货 + 股票永续合约)

    Returns:
        {
            "crypto_spot": {...},      # 加密货币现货
            "stocks": [...],           # OKX股票永续合约
            "etfs": [...],             # OKX ETF合约
            "commodities": [...],      # OKX 大宗商品合约
            "semiconductor": [...],    # 半导体专属
            "korean": [...],           # 韩国股票
            "space": [...],            # 太空经济
        }
    """
    all_okx_spot = fetch_okx_spot_symbols()
    swap_data = fetch_okx_swap_symbols()

    # Crypto spot tiers
    tier1 = [s for s in TIER1_ML_HEAVY if s in all_okx_spot]
    tier2 = [s for s in TIER2_TECHNICAL_LIGHT if s in all_okx_spot and s not in tier1]
    tier1_tier2 = set(tier1 + tier2)
    tier3 = [s for s in all_okx_spot if s not in tier1_tier2]

    # 股票: 优先用API动态拉取的, API失败则fallback到预定义列表
    api_stocks = swap_data.get("stocks", [])
    if not api_stocks:
        api_stocks = OKX_ALL_STOCK_SWAPS

    api_etfs = swap_data.get("etfs", [])
    if not api_etfs:
        api_etfs = OKX_ETF_SWAPS

    api_commodities = swap_data.get("commodities", [])
    if not api_commodities:
        api_commodities = OKX_COMMODITY_SWAPS

    # 半导体股票 (从预定义列表中筛选OKX实际支持的)
    all_okx_swaps_set = set(swap_data.get("all", []))
    semiconductor = [s for s in OKX_SEMICONDUCTOR_SWAPS if s in all_okx_swaps_set]
    if not semiconductor:
        semiconductor = OKX_SEMICONDUCTOR_SWAPS  # fallback

    korean = [s for s in OKX_KOREAN_STOCKS if s in all_okx_swaps_set]
    if not korean:
        korean = OKX_KOREAN_STOCKS

    space_ = [s for s in OKX_SPACE_SWAPS if s in all_okx_swaps_set]
    if not space_:
        space_ = OKX_SPACE_SWAPS

    return {
        "crypto_spot": {
            "tier1_ml": tier1,
            "tier2_technical": tier2,
            "tier3_momentum": tier3,
            "total": len(all_okx_spot),
        },
        "stocks": api_stocks,
        "etfs": api_etfs,
        "commodities": api_commodities,
        "semiconductor": semiconductor,
        "korean": korean,
        "space": space_,
        "total_stock_swaps": len(api_stocks),
        "total_etf_swaps": len(api_etfs),
        "total_commodity_swaps": len(api_commodities),
        "note": "股票/ETF/商品为永续合约(swap, linear, USDT保证金), 非现货! 支持做多/做空/杠杆",
        "spacex_note": "SpaceX 未上市! SPCX/USDT 是太空ETF永续合约, 间接持有SpaceX敞口",
    }


def get_all_crypto_symbols(tiers: List[int] = None) -> List[str]:
    """
    获取所有需要扫描的加密货币 (现货)

    Args:
        tiers: 要包含的层级 (默认 [1,2] = ML + 技术面)
    """
    if tiers is None:
        tiers = [1, 2]

    all_okx = fetch_okx_spot_symbols()
    result = []

    if 1 in tiers:
        result.extend([s for s in TIER1_ML_HEAVY if s in all_okx])
    if 2 in tiers:
        t1 = set([s for s in TIER1_ML_HEAVY if s in all_okx])
        result.extend([s for s in TIER2_TECHNICAL_LIGHT if s in all_okx and s not in t1])
    if 3 in tiers:
        t1t2 = set(result)
        result.extend([s for s in all_okx if s not in t1t2])

    return result


def get_all_scan_symbols() -> dict:
    """
    获取完整扫描标的 — 加密货币 + 股票/ETF/商品

    Returns:
        {
            "crypto_spot": [...],       # 加密货币现货 (Tier1+2)
            "crypto_swaps": [...],       # 加密货币永续合约
            "stocks": [...],             # 股票永续合约
            "etfs": [...],               # ETF合约
            "commodities": [...],        # 商品合约
            "semiconductor": [...],      # 🔥 半导体 (Chase哥最爱)
            "all_scannable": [...],      # 全部可扫描标的
        }
    """
    config = build_tiered_symbols()
    swap_data = fetch_okx_swap_symbols()

    return {
        "crypto_spot": (
            config["crypto_spot"]["tier1_ml"] + config["crypto_spot"]["tier2_technical"]
        ),
        "crypto_swaps": swap_data.get("crypto_swaps", []),
        "stocks": config["stocks"],
        "etfs": config["etfs"],
        "commodities": config["commodities"],
        "semiconductor": config["semiconductor"],
        "korean": config["korean"],
        "space": config["space"],
        "all_scannable": (
            config["crypto_spot"]["tier1_ml"]
            + config["crypto_spot"]["tier2_technical"]
            + config["stocks"]
            + config["etfs"]
            + config["commodities"]
        ),
    }


# ═══════════════════════════════════════════════════════════════
# 向后兼容 — 覆盖原有的小列表
# ═══════════════════════════════════════════════════════════════

COMPAT_SYMBOLS = get_all_crypto_symbols(tiers=[1, 2])  # 默认: Tier1+2 加密货币


if __name__ == "__main__":
    config = build_tiered_symbols()
    print("📊 OKX 标的池全景:")
    print()
    print("🪙 加密货币现货:")
    print(f"  Tier 1 (ML深度): {len(config['crypto_spot']['tier1_ml'])} 个")
    for s in config["crypto_spot"]["tier1_ml"]:
        print(f"    {s}")
    print(f"  Tier 2 (技术面): {len(config['crypto_spot']['tier2_technical'])} 个")
    print(f"  Tier 3 (动量过滤): {len(config['crypto_spot']['tier3_momentum'])} 个")
    print(f"  现货总计: {config['crypto_spot']['total']} 个")
    print()
    print("📈 OKX 永续合约 (USDT保证金):")
    print(f"  📀 半导体: {len(config['semiconductor'])} 个")
    for s in config["semiconductor"]:
        print(f"    {s}")
    print(f"  🇰🇷 韩股: {len(config['korean'])} 个")
    for s in config["korean"]:
        print(f"    {s}")
    print(f"  🚀 太空: {len(config['space'])} 个")
    for s in config["space"]:
        print(f"    {s}")
    print(f"  🏢 美股总计: {config['total_stock_swaps']} 个")
    print(f"  📊 ETF: {config['total_etf_swaps']} 个")
    for s in config["etfs"]:
        print(f"    {s}")
    print(f"  🏆 商品: {config['total_commodity_swaps']} 个")
    for s in config["commodities"]:
        print(f"    {s}")
    print()
    print(f"⚠️  {config['note']}")
    print(f"🚀 {config['spacex_note']}")
    print()
    total_scan = (
        len(config["crypto_spot"]["tier1_ml"])
        + len(config["crypto_spot"]["tier2_technical"])
        + len(config["stocks"])
        + len(config["etfs"])
        + len(config["commodities"])
    )
    print(f"🎯 可扫描标的合计: {total_scan} 个 (加密现货{len(config['crypto_spot']['tier1_ml'])+len(config['crypto_spot']['tier2_technical'])}+ 股票{config['total_stock_swaps']}+ ETF{config['total_etf_swaps']}+ 商品{config['total_commodity_swaps']})")
