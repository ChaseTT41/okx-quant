"""
Yina 市场情绪引擎 🎭
===================
每轮扫描前拉取 OKX 链上情绪数据 + Fear & Greed，
计算板块资金流、AI叙事相关性、板块轮动预测。

数据源:
  - OKX 公开API: 资金费率 / 持仓量 / 多空比 / Taker买卖量
  - alternative.me: Fear & Greed Index
  - 已有 Gemini NLP 情绪 (sentiment_analyzer.py, 独立使用)

缓存策略: data/sentiment_cache/ 下JSON文件, 按TTL自动刷新
OKX限频: 公开端点 20req/2s, 用批量端点 + 速率控制

用法:
  engine = MarketSentimentEngine()
  snapshot = engine.refresh_all()
  overlay = engine.get_sentiment_overlay("NVDA/USDT")
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import time
import logging

import numpy as np
import requests

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
SENTIMENT_CACHE_DIR = DATA_DIR / "sentiment_cache"

# OKX 公开API base (用 .cab 域名绕过GFW)
OKX_PUBLIC_BASE = "https://www.okx.cab"

# ── 缓存TTL常量 ──
FUNDING_RATE_TTL_HOURS = 1.0
OI_TTL_HOURS = 0.5
LONG_SHORT_TTL_HOURS = 2.0
TAKER_VOLUME_TTL_HOURS = 1.0
FEAR_GREED_TTL_HOURS = 4.0
CORRELATION_TTL_HOURS = 6.0
SECTOR_FLOW_TTL_HOURS = 0.5
ROTATION_TTL_HOURS = 1.0

# OKX 速率限制: 公开端点 20 req / 2s
OKX_RATE_LIMIT_INTERVAL = 0.12  # 稍保守, 留给其他系统余量


# ── 辅助函数 ──

def _df_to_ccxt_ohlcv(df) -> list:
    """Convert okx_rest_data DataFrame → ccxt-compatible OHLCV list format.

    DataFrame columns: [date, open, high, low, close, volume]
    Returns: [[timestamp_ms, open, high, low, close, volume], ...]
    """
    if df is None or df.empty:
        return []
    result = []
    for _, row in df.iterrows():
        ts = int(row['date'].timestamp() * 1000)
        result.append([ts, row['open'], row['high'], row['low'], row['close'], row['volume']])
    return result


# ═══════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════

@dataclass
class FundingRateSnapshot:
    """单个标的的当前资金费率"""
    symbol: str
    funding_rate: float             # 当期资金费率 (每8小时)
    annualized_rate: float          # 年化 ≈ funding_rate * 3 * 365
    next_funding_time: str = ""     # 下次结算时间 ISO
    timestamp: str = ""

@dataclass
class OpenInterestSnapshot:
    """单个标的的持仓量数据"""
    symbol: str
    open_interest: float            # OI (USDT)
    oi_change_24h_pct: float        # 24h变化率
    timestamp: str = ""

@dataclass
class LongShortRatioSnapshot:
    """单个币种的多空比 (OKX Rubik 按币种聚合)"""
    currency: str                   # e.g. "BTC"
    long_short_ratio: float         # >1 = 多>空
    long_pct: float
    short_pct: float
    timestamp: str = ""

@dataclass
class TakerBuySellSnapshot:
    """单个币种的主动买卖量比"""
    currency: str                   # e.g. "BTC"
    taker_buy_volume: float         # 24h 主动买入量
    taker_sell_volume: float        # 24h 主动卖出量
    taker_buy_sell_ratio: float     # >1 = 买方主导
    timestamp: str = ""

@dataclass
class FearGreedState:
    """恐惧贪婪指数 + 历史上下文"""
    current_value: int              # 0-100
    classification: str             # "Extreme Fear" / "Fear" / "Neutral" / "Greed" / "Extreme Greed"
    value_1d_ago: int = 0
    value_7d_ago: int = 0
    change_1d: int = 0
    change_7d: int = 0
    is_extreme_fear: bool = False   # <= 25
    is_extreme_greed: bool = False  # >= 75
    timestamp: str = ""

@dataclass
class SectorFlow:
    """板块资金流向"""
    sector_name: str
    symbols: List[str] = field(default_factory=list)
    avg_price_change_24h_pct: float = 0.0
    avg_oi_change_24h_pct: float = 0.0
    avg_funding_rate: float = 0.0
    avg_taker_buy_sell_ratio: float = 0.0
    avg_volume_change_24h_pct: float = 0.0
    flow_score: float = 0.0             # [-1, +1] 综合资金流向
    dominant_direction: str = "neutral" # "inflow" / "outflow" / "neutral"
    n_symbols_with_data: int = 0

@dataclass
class AIStockCorrelation:
    """AI股票 vs 加密AI代币相关性"""
    stock_symbol: str               # "NVDA/USDT"
    crypto_symbol: str              # "FET/USDT"
    correlation_30d: float
    correlation_7d: float
    divergence_zscore: float        # 价格比的z-score
    signal: str = "neutral"         # "convergence_buy" / "divergence_sell" / "neutral"
    note: str = ""

@dataclass
class SectorRotationPrediction:
    """板块轮动预测"""
    current_leader: str
    next_leader: str
    rotation_confidence: float      # 0-1
    reasoning: List[str] = field(default_factory=list)
    sector_scores: Dict[str, float] = field(default_factory=dict)

@dataclass
class MarketSentimentSnapshot:
    """一次性返回全部情绪数据"""
    fear_greed: Optional[FearGreedState] = None
    funding_rates: Dict[str, FundingRateSnapshot] = field(default_factory=dict)
    open_interests: Dict[str, OpenInterestSnapshot] = field(default_factory=dict)
    long_short_ratios: Dict[str, LongShortRatioSnapshot] = field(default_factory=dict)
    taker_volumes: Dict[str, TakerBuySellSnapshot] = field(default_factory=dict)
    sector_flows: Dict[str, SectorFlow] = field(default_factory=dict)
    ai_correlations: List[AIStockCorrelation] = field(default_factory=list)
    rotation_prediction: Optional[SectorRotationPrediction] = None
    updated_at: str = ""


# ═══════════════════════════════════════════════════════════════
# MarketSentimentEngine
# ═══════════════════════════════════════════════════════════════

class MarketSentimentEngine:
    """
    市场情绪数据引擎 — 每轮扫描调用 refresh_all()

    缓存策略: 各组数据独立TTL, 避免重复请求OKX
    速率控制: 公开端点控制在 ~8 req/s
    """

    # ── 板块定义 (与 symbol_config.py 对齐) ──
    SECTORS = {
        "semiconductor": [
            "NVDA/USDT", "AMD/USDT", "INTC/USDT", "MU/USDT", "MRVL/USDT",
            "AVGO/USDT", "QCOM/USDT", "TSM/USDT", "ARM/USDT", "ASML/USDT",
            "AMAT/USDT", "COHR/USDT", "CIEN/USDT", "CRDO/USDT", "CGNX/USDT",
            "AXTI/USDT", "POET/USDT", "WDC/USDT", "SNDK/USDT", "FLNC/USDT",
        ],
        "ai_software": [
            "MSFT/USDT", "GOOGL/USDT", "META/USDT", "AMZN/USDT", "AAPL/USDT",
            "ORCL/USDT", "ADBE/USDT", "NOW/USDT", "PANW/USDT", "CRWD/USDT",
            "PLTR/USDT", "TWLO/USDT", "BILL/USDT", "NBIS/USDT",
        ],
        "crypto_ai": [
            "FET/USDT", "RENDER/USDT", "WLD/USDT", "AIXBT/USDT",
            "COAI/USDT", "OPENAI/USDT", "ANTHROPIC/USDT",
        ],
        "meme_coins": [
            "PEPE/USDT", "SHIB/USDT", "WIF/USDT", "BONK/USDT", "DOGE/USDT",
            "FARTCOIN/USDT", "MUBARAK/USDT",
        ],
        "defi": [
            "UNI/USDT", "AAVE/USDT", "LDO/USDT", "CRV/USDT", "COMP/USDT",
            "MKR/USDT", "SNX/USDT", "PENDLE/USDT", "GMX/USDT", "DYDX/USDT",
        ],
        "commodities": [
            "XAU/USDT", "XAG/USDT", "XCU/USDT", "XPD/USDT", "XPT/USDT",
        ],
        "space": [
            "RKLB/USDT", "ASTS/USDT", "LUNR/USDT", "RDW/USDT", "SPCX/USDT",
        ],
        "consumer_fintech": [
            "TSLA/USDT", "COIN/USDT", "MSTR/USDT", "HOOD/USDT", "COST/USDT",
            "HIMS/USDT", "GME/USDT", "DELL/USDT", "IBM/USDT", "NFLX/USDT",
            "LLY/USDT", "ISRG/USDT", "HPE/USDT", "GEV/USDT", "GLW/USDT",
            "NOK/USDT", "CSCO/USDT", "CRWV/USDT",
        ],
        "korean": [
            "SKHYNIX/USDT", "SAMSUNG/USDT", "HYUNDAI/USDT",
        ],
    }

    # AI叙事配对: (股票, 加密AI代币)
    AI_NARRATIVE_PAIRS = [
        ("NVDA/USDT", "FET/USDT"),
        ("NVDA/USDT", "RENDER/USDT"),
        ("NVDA/USDT", "WLD/USDT"),
        ("TSLA/USDT", "FET/USDT"),
        ("AMD/USDT", "RENDER/USDT"),
    ]

    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or SENTIMENT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session: Optional[requests.Session] = None
        self._last_request_time = 0.0

    @property
    def session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
            })
        return self._session

    # ── 速率控制 ──

    def _rate_limit(self):
        """确保两次OKX请求间隔 >= OKX_RATE_LIMIT_INTERVAL"""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < OKX_RATE_LIMIT_INTERVAL:
            time.sleep(OKX_RATE_LIMIT_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()

    # ── 缓存工具 ──

    def _cache_path(self, name: str) -> Path:
        return self.cache_dir / f"{name}.json"

    def _cache_valid(self, name: str, ttl_hours: float) -> bool:
        p = self._cache_path(name)
        if not p.exists():
            return False
        age = (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)).total_seconds()
        return age < ttl_hours * 3600

    def _read_cache(self, name: str) -> Optional[dict]:
        p = self._cache_path(name)
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def _write_cache(self, name: str, data: dict):
        p = self._cache_path(name)
        with open(p, "w") as f:
            json.dump(data, f, ensure_ascii=False, default=str)

    def _okx_get(self, path: str, params: dict = None) -> Optional[dict]:
        """调用 OKX 公开API"""
        self._rate_limit()
        try:
            url = f"{OKX_PUBLIC_BASE}{path}"
            resp = self.session.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            else:
                log.warning(f"OKX API {path} -> {resp.status_code}")
                return None
        except Exception as e:
            log.warning(f"OKX API {path} error: {e}")
            return None

    # ── 1. 资金费率 ──

    def _to_okx_inst_id(self, symbol: str) -> str:
        """将 'BTC/USDT' 转为 OKX instId 'BTC-USDT-SWAP'"""
        base = symbol.replace("/USDT", "")
        return f"{base}-USDT-SWAP"

    def _from_okx_inst_id(self, inst_id: str) -> str:
        """将 'BTC-USDT-SWAP' 转回 'BTC/USDT'"""
        return inst_id.replace("-USDT-SWAP", "/USDT")

    def fetch_funding_rates(self, symbols: List[str],
                            use_cache: bool = True) -> Dict[str, FundingRateSnapshot]:
        """
        批量拉取所有 swap 标的的资金费率。

        OKX API: GET /api/v5/public/funding-rate?instId=BTC-USDT-SWAP
        """
        cache_name = "funding_rates"
        if use_cache and self._cache_valid(cache_name, FUNDING_RATE_TTL_HOURS):
            cached = self._read_cache(cache_name)
            if cached:
                return {k: FundingRateSnapshot(**v) for k, v in cached.items()}

        results = {}
        now_ts = datetime.now(timezone.utc).isoformat()
        fetched = 0

        for symbol in symbols:
            inst_id = self._to_okx_inst_id(symbol)
            data = self._okx_get("/api/v5/public/funding-rate", {"instId": inst_id})
            if not data or data.get("code") != "0":
                continue

            rate_info = data.get("data", [])
            if not rate_info:
                continue

            r = rate_info[0]
            try:
                fr = float(r.get("fundingRate", 0))
            except (ValueError, TypeError):
                fr = 0.0

            annualized = abs(fr) * 3 * 365  # 每8小时 * 3次/天 * 365天

            results[symbol] = FundingRateSnapshot(
                symbol=symbol,
                funding_rate=fr,
                annualized_rate=annualized,
                next_funding_time=r.get("nextFundingTime", ""),
                timestamp=now_ts,
            )
            fetched += 1

        log.info(f"💰 资金费率: 拉取 {fetched}/{len(symbols)} 个标的")
        self._write_cache(cache_name, {k: v.__dict__ for k, v in results.items()})
        return results

    # ── 2. 持仓量 (Open Interest) ──

    def fetch_open_interests(self, symbols: List[str],
                              use_cache: bool = True) -> Dict[str, OpenInterestSnapshot]:
        """
        批量拉取持仓量。

        OKX API: GET /api/v5/public/open-interest?instId=BTC-USDT-SWAP
        """
        cache_name = "open_interests"
        if use_cache and self._cache_valid(cache_name, OI_TTL_HOURS):
            cached = self._read_cache(cache_name)
            if cached:
                return {k: OpenInterestSnapshot(**v) for k, v in cached.items()}

        results = {}
        now_ts = datetime.now(timezone.utc).isoformat()
        fetched = 0

        # 只对主要标的拉取OI (全部456个太慢), 优先Tier1+2加密 + 半导体 + 韩股
        priority_symbols = []
        from symbol_config import (
            TIER1_ML_HEAVY, TIER2_TECHNICAL_LIGHT,
            OKX_SEMICONDUCTOR_SWAPS, OKX_KOREAN_STOCKS, OKX_SPACE_SWAPS,
        )
        priority_set = set()
        for s_list in [TIER1_ML_HEAVY, TIER2_TECHNICAL_LIGHT,
                       OKX_SEMICONDUCTOR_SWAPS, OKX_KOREAN_STOCKS, OKX_SPACE_SWAPS]:
            for s in s_list:
                if s in symbols or f"{s}:USDT" in str(symbols):
                    priority_set.add(s)

        priority_set = {s for s in symbols if any(
            s.startswith(p.split('/')[0]) for p in priority_set
        )}

        fetch_symbols = list(priority_set)[:200]  # 最多200个

        for symbol in fetch_symbols:
            inst_id = self._to_okx_inst_id(symbol)
            data = self._okx_get("/api/v5/public/open-interest", {"instId": inst_id})
            if not data or data.get("code") != "0":
                continue

            oi_data = data.get("data", [])
            if not oi_data:
                continue

            d = oi_data[0]
            try:
                oi = float(d.get("oi", 0))
            except (ValueError, TypeError):
                oi = 0.0

            # OKX OI API 可能不直接返回 24h 变化, 用粗略估计
            try:
                oi_24h = float(d.get("oiCcy", 0))  # 或 oiUsd
            except (ValueError, TypeError):
                oi_24h = oi

            results[symbol] = OpenInterestSnapshot(
                symbol=symbol,
                open_interest=oi,
                oi_change_24h_pct=0.0,  # 后续从历史对比计算
                timestamp=now_ts,
            )
            fetched += 1

        # 计算 OI 变化率 (对比上次缓存)
        prev = self._read_cache(cache_name)
        if prev:
            for sym, snap in results.items():
                if sym in prev:
                    prev_oi = prev[sym].get("open_interest", 0)
                    if prev_oi and prev_oi > 0:
                        snap.oi_change_24h_pct = (
                            (snap.open_interest - prev_oi) / prev_oi * 100
                        )

        log.info(f"📊 持仓量: 拉取 {fetched}/{len(fetch_symbols)} 个标的")
        self._write_cache(cache_name, {k: v.__dict__ for k, v in results.items()})
        return results

    # ── 3. 多空比 ──

    def fetch_long_short_ratios(self,
                                 use_cache: bool = True) -> Dict[str, LongShortRatioSnapshot]:
        """
        拉取各币种的多空账户比。
        OKX Rubik: GET /api/v5/rubik/stat/contracts/long-short-account-ratio?ccy=BTC&period=5M

        注意: OKX 按币种 (ccy) 聚合，不是按合约
        """
        cache_name = "long_short_ratios"
        if use_cache and self._cache_valid(cache_name, LONG_SHORT_TTL_HOURS):
            cached = self._read_cache(cache_name)
            if cached:
                return {k: LongShortRatioSnapshot(**v) for k, v in cached.items()}

        results = {}
        now_ts = datetime.now(timezone.utc).isoformat()

        # 主要币种
        currencies = [
            "BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "AVAX",
            "DOT", "LINK", "MATIC", "UNI", "ATOM", "APT", "OP",
            "HBAR", "ARB", "NEAR", "LTC", "PEPE", "SHIB", "SUI",
        ]

        for ccy in currencies:
            data = self._okx_get(
                "/api/v5/rubik/stat/contracts/long-short-account-ratio",
                {"ccy": ccy, "period": "5m"}  # 小写m, 5分钟粒度
            )
            if not data or data.get("code") != "0":
                continue

            ls_data = data.get("data", [])
            if not ls_data:
                continue

            # Rubik 返回格式: [["timestamp_ms", "ratio_value"], ...]
            # ratio_value = 多/空 比值, >1 表示多头多于空头
            latest = ls_data[0]
            try:
                if isinstance(latest, list) and len(latest) >= 2:
                    ratio = float(latest[1])
                elif isinstance(latest, dict):
                    ratio = float(latest.get("longAccount", 50)) / float(latest.get("shortAccount", 50))
                else:
                    ratio = 1.0
                long_pct = ratio / (1 + ratio) * 100
                short_pct = 100 - long_pct
            except (ValueError, TypeError, ZeroDivisionError):
                long_pct, short_pct, ratio = 50.0, 50.0, 1.0

            results[ccy] = LongShortRatioSnapshot(
                currency=ccy,
                long_short_ratio=round(ratio, 3),
                long_pct=long_pct,
                short_pct=short_pct,
                timestamp=now_ts,
            )
            self._rate_limit()

        log.info(f"⚖️  多空比: 拉取 {len(results)} 个币种")
        self._write_cache(cache_name, {k: v.__dict__ for k, v in results.items()})
        return results

    # ── 4. Taker买卖量比 (基于 Ticker 推断) ──

    def fetch_taker_volumes(self,
                             use_cache: bool = True) -> Dict[str, TakerBuySellSnapshot]:
        """
        推断买卖压力。
        OKX Rubik taker-volume 端点已下线，改用 ticker 批量端点推断。

        从 ticker 数据计算:
          - buy_sell_ratio ≈ last / open24h (价格方向代理买卖压力)
          - vol24h 作为总成交量
        """
        cache_name = "taker_volumes"
        if use_cache and self._cache_valid(cache_name, TAKER_VOLUME_TTL_HOURS):
            cached = self._read_cache(cache_name)
            if cached:
                return {k: TakerBuySellSnapshot(**v) for k, v in cached.items()}

        results = {}
        now_ts = datetime.now(timezone.utc).isoformat()

        # 从 ticker 批量端点拉取 (全部SWAP一起拿, 只需要1次请求!)
        data = self._okx_get("/api/v5/market/tickers", {"instType": "SWAP"})
        if not data or data.get("code") != "0":
            log.warning("Ticker 批量拉取失败, 无法推断买卖压力")
            return results

        for ticker in data.get("data", []):
            inst_id = ticker.get("instId", "")
            if not inst_id.endswith("-USDT-SWAP"):
                continue

            base = inst_id.replace("-USDT-SWAP", "")
            try:
                open24 = float(ticker.get("open24h", 0))
                last = float(ticker.get("last", 0))
                vol24 = float(ticker.get("volCcy24h", 0))  # USDT计价成交量
            except (ValueError, TypeError):
                continue

            # 价格方向作为买卖压力代理
            # last > open24h → 买方主导, ratio > 1
            if open24 > 0 and last > 0:
                # 用价格变化率映射到 [0.3, 3.0] 范围
                price_change = (last - open24) / open24
                ratio = 1.0 + price_change * 5  # 1% price up → 1.05 ratio
                ratio = max(0.3, min(3.0, ratio))
            else:
                ratio = 1.0

            sell_vol = vol24 / (1 + ratio) if ratio > 0 else vol24 / 2
            buy_vol = vol24 - sell_vol

            results[base] = TakerBuySellSnapshot(
                currency=base,
                taker_buy_volume=round(buy_vol, 2),
                taker_sell_volume=round(sell_vol, 2),
                taker_buy_sell_ratio=round(ratio, 3),
                timestamp=now_ts,
            )

        log.info(f"🔄 买卖压力: 推断 {len(results)} 个币种 (from tickers)")
        self._write_cache(cache_name, {k: v.__dict__ for k, v in results.items()})
        return results

    # ── 5. Fear & Greed ──

    def fetch_fear_greed(self, use_cache: bool = True) -> FearGreedState:
        """
        从 alternative.me 拉取恐惧贪婪指数。

        API: GET https://api.alternative.me/fng/?limit=30
        """
        cache_name = "fear_greed"
        if use_cache and self._cache_valid(cache_name, FEAR_GREED_TTL_HOURS):
            cached = self._read_cache(cache_name)
            if cached:
                return FearGreedState(**cached)

        try:
            resp = self.session.get(
                "https://api.alternative.me/fng/?limit=30", timeout=15
            )
            data = resp.json().get("data", [])
            if not data:
                raise ValueError("Empty response")
        except Exception as e:
            log.warning(f"Fear & Greed API 失败: {e}")
            return FearGreedState(
                current_value=50, classification="Neutral", timestamp=""
            )

        current = int(data[0]["value"])
        v1d = int(data[1]["value"]) if len(data) > 1 else current
        v7d = int(data[7]["value"]) if len(data) > 7 else current

        # 分类
        if current <= 25:
            classification = "Extreme Fear"
        elif current <= 45:
            classification = "Fear"
        elif current <= 55:
            classification = "Neutral"
        elif current <= 75:
            classification = "Greed"
        else:
            classification = "Extreme Greed"

        result = FearGreedState(
            current_value=current,
            classification=classification,
            value_1d_ago=v1d,
            value_7d_ago=v7d,
            change_1d=current - v1d,
            change_7d=current - v7d,
            is_extreme_fear=(current <= 25),
            is_extreme_greed=(current >= 75),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._write_cache(cache_name, result.__dict__)
        return result

    # ── 6. 价格变动 (用于板块聚合) ──

    def _get_price_changes(self, symbols: List[str]) -> Dict[str, float]:
        """拉取24h价格变动 % (通过 OKX ticker 批量端点)"""
        # OKX ticker 批量: GET /api/v5/market/tickers?instType=SWAP
        # 先读缓存
        cache_name = "price_changes"
        if self._cache_valid(cache_name, 0.17):  # 10 min
            cached = self._read_cache(cache_name)
            if cached:
                return cached

        changes = {}
        try:
            data = self._okx_get("/api/v5/market/tickers", {"instType": "SWAP"})
            if data and data.get("code") == "0":
                for ticker in data.get("data", []):
                    inst_id = ticker.get("instId", "")
                    if not inst_id.endswith("-USDT-SWAP"):
                        continue
                    symbol = self._from_okx_inst_id(inst_id)
                    try:
                        # OKX ticker: open24h + last → change %
                        open24 = float(ticker.get("open24h", 0))
                        last = float(ticker.get("last", 0))
                        if open24 > 0:
                            changes[symbol] = (last - open24) / open24 * 100
                    except (ValueError, TypeError, ZeroDivisionError):
                        pass
        except Exception as e:
            log.warning(f"价格变动拉取失败: {e}")

        log.info(f"📈 价格变动: 获取 {len(changes)} 个标的")
        self._write_cache(cache_name, changes)
        return changes

    # ── 7. 板块资金流 ──

    def compute_sector_flows(self,
                              funding_data: Dict[str, FundingRateSnapshot],
                              oi_data: Dict[str, OpenInterestSnapshot],
                              taker_data: Dict[str, TakerBuySellSnapshot],
                              price_changes: Dict[str, float],
                              ) -> Dict[str, SectorFlow]:
        """
        按板块聚合资金流向指标。

        flow_score (每个板块, 跨板块归一化):
          + 0.25 * norm(avg_price_change)       价格动量
          + 0.25 * norm(avg_oi_change)          持仓变化
          - 0.15 * norm(|avg_funding_rate|)     高资金费率=拥挤, 扣分
          + 0.20 * norm(avg_taker_ratio)        主动买入压力
          + 0.15 * norm(avg_volume_change)      成交量变化
        """
        cache_name = "sector_flows"
        if self._cache_valid(cache_name, SECTOR_FLOW_TTL_HOURS):
            cached = self._read_cache(cache_name)
            if cached:
                return {k: SectorFlow(**v) for k, v in cached.items()}

        sector_flows = {}

        for sector_name, sector_symbols in self.SECTORS.items():
            prices = []
            oi_changes = []
            funding_rates = []
            taker_ratios = []

            for sym in sector_symbols:
                if sym in price_changes:
                    prices.append(price_changes[sym])
                if sym in oi_data:
                    oi_changes.append(oi_data[sym].oi_change_24h_pct)
                if sym in funding_data:
                    funding_rates.append(funding_data[sym].funding_rate)
                # Taker 数据按币种聚合 — 映射 symbol → currency
                base = sym.split("/")[0]
                if base in taker_data:
                    taker_ratios.append(taker_data[base].taker_buy_sell_ratio)

            n = max(len(prices), len(oi_changes), len(funding_rates), len(taker_ratios))
            if n == 0:
                continue

            avg_price = np.mean(prices) if prices else 0.0
            avg_oi = np.mean(oi_changes) if oi_changes else 0.0
            avg_fr = np.mean(funding_rates) if funding_rates else 0.0
            avg_taker = np.mean(taker_ratios) if taker_ratios else 1.0
            avg_vol = avg_price  # 用价格变动做 volume proxy (OKX ticker有vol24h可扩展)

            sector_flows[sector_name] = SectorFlow(
                sector_name=sector_name,
                symbols=sector_symbols,
                avg_price_change_24h_pct=round(avg_price, 2),
                avg_oi_change_24h_pct=round(avg_oi, 2),
                avg_funding_rate=round(avg_fr, 4),
                avg_taker_buy_sell_ratio=round(avg_taker, 3),
                avg_volume_change_24h_pct=round(avg_vol, 2),
                flow_score=0.0,  # computed below
                n_symbols_with_data=n,
            )

        # ── 跨板块归一化, 计算 flow_score ──
        if sector_flows:
            # 提取各维度值
            def _norm(values: List[float]) -> List[float]:
                """Min-max normalize to [0,1], handle all-same case"""
                if not values:
                    return []
                mn, mx = min(values), max(values)
                if mx == mn:
                    return [0.5] * len(values)
                return [(v - mn) / (mx - mn) for v in values]

            sectors = list(sector_flows.keys())
            prices_raw = [sector_flows[s].avg_price_change_24h_pct for s in sectors]
            oi_raw = [sector_flows[s].avg_oi_change_24h_pct for s in sectors]
            fr_raw = [abs(sector_flows[s].avg_funding_rate) for s in sectors]
            taker_raw = [sector_flows[s].avg_taker_buy_sell_ratio for s in sectors]

            prices_n = _norm(prices_raw)
            oi_n = _norm(oi_raw)
            fr_n = _norm(fr_raw)
            taker_n = _norm(taker_raw)
            vol_n = prices_n  # proxy

            for i, sector in enumerate(sectors):
                # flow_score: 高 = 钱在流入
                score = (
                    0.25 * prices_n[i]
                    + 0.25 * oi_n[i]
                    - 0.15 * fr_n[i]    # 高费率 = 拥挤, 扣分
                    + 0.20 * taker_n[i]
                    + 0.15 * vol_n[i]
                )
                # re-center to [-1, +1]
                sector_flows[sector].flow_score = round(score * 2 - 1, 3)

                if score > 0.6:
                    sector_flows[sector].dominant_direction = "inflow"
                elif score < 0.4:
                    sector_flows[sector].dominant_direction = "outflow"
                else:
                    sector_flows[sector].dominant_direction = "neutral"

        self._write_cache(cache_name, {k: v.__dict__ for k, v in sector_flows.items()})
        return sector_flows

    # ── 8. AI叙事相关性 ──

    def compute_ai_narrative_correlations(self,
                                           use_cache: bool = True) -> List[AIStockCorrelation]:
        """
        计算 AI 股票 vs 加密 AI 代币的相关性和背离。

        用最近30天日回报的Pearson相关性。
        背离z-score: (当前价格比 - 均值) / std
        """
        cache_name = "ai_correlations"
        if use_cache and self._cache_valid(cache_name, CORRELATION_TTL_HOURS):
            cached = self._read_cache(cache_name)
            if cached:
                return [AIStockCorrelation(**v) for v in cached]

        results = []
        try:
            from okx_rest_data import fetch_okx_ohlcv

            for stock_sym, crypto_sym in self.AI_NARRATIVE_PAIRS:
                try:
                    # 拉30天日线 (OKX REST, 零 ccxt)
                    stock_df = fetch_okx_ohlcv(
                        stock_sym.replace("/USDT", "-USDT"), "1D", limit=35
                    )
                    crypto_df = fetch_okx_ohlcv(
                        crypto_sym.replace("/USDT", "-USDT"), "1D", limit=35
                    )
                    # 转回 ccxt 兼容格式 [[ts, o, h, l, c, vol], ...]
                    stock_ohlcv = _df_to_ccxt_ohlcv(stock_df) if stock_df is not None else []
                    crypto_ohlcv = _df_to_ccxt_ohlcv(crypto_df) if crypto_df is not None else []

                    if len(stock_ohlcv) < 10 or len(crypto_ohlcv) < 10:
                        results.append(AIStockCorrelation(
                            stock_symbol=stock_sym, crypto_symbol=crypto_sym,
                            correlation_30d=0.0, correlation_7d=0.0,
                            divergence_zscore=0.0, signal="neutral",
                            note="数据不足 (需要≥10天)"
                        ))
                        continue

                    # 对齐长度
                    n = min(len(stock_ohlcv), len(crypto_ohlcv))
                    stock_close = np.array([c[4] for c in stock_ohlcv[-n:]])
                    crypto_close = np.array([c[4] for c in crypto_ohlcv[-n:]])

                    # 日回报
                    stock_ret = np.diff(stock_close) / stock_close[:-1]
                    crypto_ret = np.diff(crypto_close) / crypto_close[:-1]

                    # 相关性
                    if len(stock_ret) >= 7:
                        corr_30d = np.corrcoef(stock_ret, crypto_ret)[0, 1]
                        if np.isnan(corr_30d):
                            corr_30d = 0.0
                        corr_7d = np.corrcoef(stock_ret[-7:], crypto_ret[-7:])[0, 1]
                        if np.isnan(corr_7d):
                            corr_7d = 0.0
                    else:
                        corr_30d, corr_7d = 0.0, 0.0

                    # 背离 z-score: price_ratio = stock / crypto
                    price_ratio = stock_close / (crypto_close + 1e-10)
                    ratio_mean = np.mean(price_ratio)
                    ratio_std = np.std(price_ratio)
                    if ratio_std > 0:
                        divergence_z = (price_ratio[-1] - ratio_mean) / ratio_std
                    else:
                        divergence_z = 0.0

                    # 信号
                    if divergence_z < -2:
                        signal = "convergence_buy"
                        note = f"加密AI相对股票折价 (z={divergence_z:.1f}), 可能补涨"
                    elif divergence_z > 2:
                        signal = "divergence_sell"
                        note = f"加密AI相对股票溢价 (z={divergence_z:.1f}), 可能回调"
                    else:
                        signal = "neutral"
                        note = f"无明显背离 (z={divergence_z:.1f})"

                    results.append(AIStockCorrelation(
                        stock_symbol=stock_sym,
                        crypto_symbol=crypto_sym,
                        correlation_30d=round(corr_30d, 3),
                        correlation_7d=round(corr_7d, 3),
                        divergence_zscore=round(divergence_z, 2),
                        signal=signal,
                        note=note,
                    ))
                except Exception as e:
                    log.debug(f"AI相关性 {stock_sym} vs {crypto_sym}: {e}")
                    results.append(AIStockCorrelation(
                        stock_symbol=stock_sym, crypto_symbol=crypto_sym,
                        correlation_30d=0.0, correlation_7d=0.0,
                        divergence_zscore=0.0, signal="neutral", note=f"Error: {e}"
                    ))
        except Exception as e:
            log.warning(f"AI叙事相关性计算失败: {e}")

        self._write_cache(cache_name, [r.__dict__ for r in results])
        return results

    # ── 9. 板块轮动预测 ──

    def predict_sector_rotation(self,
                                 sector_flows: Dict[str, SectorFlow],
                                 fear_greed: FearGreedState) -> SectorRotationPrediction:
        """
        根据资金流向动量 + 恐惧贪婪极值 → 预测下个领涨板块。

        逻辑:
          1. 基础分 = flow_score (资金流向)
          2. 极度恐惧 → 防御板块加分 (商品、大市值加密)
          3. 极度贪婪 → 风险板块加分 (meme、太空、AI概念)
          4. 中性 → 跟随动量
        """
        cache_name = "rotation_prediction"
        if self._cache_valid(cache_name, ROTATION_TTL_HOURS):
            cached = self._read_cache(cache_name)
            if cached:
                return SectorRotationPrediction(**cached)

        if not sector_flows:
            return SectorRotationPrediction(
                current_leader="unknown", next_leader="unknown",
                rotation_confidence=0.0,
                reasoning=["板块数据不足"]
            )

        # 基础分
        sector_scores = {
            name: flow.flow_score for name, flow in sector_flows.items()
        }

        # Fear & Greed 修正
        fg = fear_greed.current_value
        fg_modifier = (fg - 50) / 50  # [-1, +1]: -1=extreme fear, +1=extreme greed

        for sector in sector_scores:
            if sector in ("commodities", "korean"):
                # 防御板块: 恐惧时加分
                sector_scores[sector] += -fg_modifier * 0.15
            elif sector in ("meme_coins", "space", "crypto_ai"):
                # 风险板块: 贪婪时加分
                sector_scores[sector] += fg_modifier * 0.15
            elif sector == "semiconductor":
                # 半导体: 中性偏进攻, 跟随动量
                sector_scores[sector] += fg_modifier * 0.05

        # 排序
        ranked = sorted(sector_scores.items(), key=lambda x: -x[1])
        current_leader = ranked[0][0] if ranked else "unknown"
        next_leader = ranked[1][0] if len(ranked) > 1 else current_leader

        # 置信度: 基于 leader 与第二名的差距
        gap = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 0
        confidence = min(0.9, max(0.1, 0.5 + gap))

        reasoning = [
            f"当前领涨: {current_leader} (score={sector_scores[current_leader]:.2f})",
            f"F&G={fg} ({fear_greed.classification}), modifier={fg_modifier:+.2f}",
        ]
        # 添加强势/弱势板块备注
        inflows = [s for s, f in sector_flows.items() if f.dominant_direction == "inflow"]
        outflows = [s for s, f in sector_flows.items() if f.dominant_direction == "outflow"]
        if inflows:
            reasoning.append(f"资金流入: {', '.join(inflows)}")
        if outflows:
            reasoning.append(f"资金流出: {', '.join(outflows)}")

        result = SectorRotationPrediction(
            current_leader=current_leader,
            next_leader=next_leader,
            rotation_confidence=round(confidence, 2),
            reasoning=reasoning,
            sector_scores={k: round(v, 3) for k, v in sorted(
                sector_scores.items(), key=lambda x: -x[1]
            )},
        )
        self._write_cache(cache_name, result.__dict__)
        return result

    # ── 10. 情绪叠加 (单标的) ──

    def get_sentiment_overlay(self, symbol: str) -> dict:
        """
        返回单个标的的情绪叠加因子。

        Returns:
          {
            "symbol": str,
            "fear_greed_modifier": float,      # [-1, +1]
            "sector_flow_modifier": float,      # [-1, +1]
            "funding_rate_pressure": float,     # [-1, +1] 负费率=看涨
            "oi_momentum_signal": float,        # [-1, +1]
            "taker_pressure": float,            # [-1, +1]
            "composite_sentiment": float,       # [-1, +1] 加权综合
            "sector": str,
            "ai_divergence_signal": str,        # 对加密AI代币
          }
        """
        # 找标的所属板块
        sector = "unknown"
        for sec_name, sec_symbols in self.SECTORS.items():
            if symbol in sec_symbols:
                sector = sec_name
                break

        # 读取各缓存
        fr_data = self._read_cache("funding_rates") or {}
        oi_data = self._read_cache("open_interests") or {}
        taker_data = self._read_cache("taker_volumes") or {}
        fg_data = self._read_cache("fear_greed") or {}
        sector_data = self._read_cache("sector_flows") or {}
        ai_corr = self._read_cache("ai_correlations") or []

        # ── Fear & Greed modifier ──
        fg_val = fg_data.get("current_value", 50) if isinstance(fg_data, dict) else 50
        fg_modifier = (fg_val - 50) / 50  # [-1, +1]

        # ── 板块资金流 modifier ──
        sec_info = sector_data.get(sector, {})
        if isinstance(sec_info, dict):
            sector_modifier = sec_info.get("flow_score", 0.0)
        else:
            sector_modifier = 0.0

        # ── 资金费率压力 (负费率 = 空头付钱给多头 = 看涨) ──
        fr = fr_data.get(symbol, {})
        if isinstance(fr, dict):
            funding_rate = fr.get("funding_rate", 0.0)
        else:
            funding_rate = 0.0
        # normalize: 0.1% = neutral, >1% = extreme
        funding_pressure = -np.clip(funding_rate / 0.005, -1, 1)

        # ── OI 动量 ──
        oi = oi_data.get(symbol, {})
        if isinstance(oi, dict):
            oi_change = oi.get("oi_change_24h_pct", 0.0)
        else:
            oi_change = 0.0
        oi_signal = np.clip(oi_change / 10, -1, 1)  # 10% OI change = max signal

        # ── Taker 压力 ──
        base = symbol.split("/")[0]
        taker = taker_data.get(base, {})
        if isinstance(taker, dict):
            taker_ratio = taker.get("taker_buy_sell_ratio", 1.0)
        else:
            taker_ratio = 1.0
        taker_pressure = np.clip((taker_ratio - 1) * 2, -1, 1)

        # ── AI 背离信号 ──
        ai_signal = "neutral"
        if symbol in ["FET/USDT", "RENDER/USDT", "WLD/USDT"]:
            for corr in ai_corr:
                if isinstance(corr, dict) and corr.get("crypto_symbol") == symbol:
                    ai_signal = corr.get("signal", "neutral")
                    break

        # ── 综合情绪 ──
        composite = (
            0.10 * fg_modifier
            + 0.30 * sector_modifier
            + 0.25 * funding_pressure
            + 0.15 * oi_signal
            + 0.20 * taker_pressure
        )

        return {
            "symbol": symbol,
            "sector": sector,
            "fear_greed_modifier": round(fg_modifier, 3),
            "sector_flow_modifier": round(sector_modifier, 3),
            "funding_rate_pressure": round(funding_pressure, 3),
            "oi_momentum_signal": round(oi_signal, 3),
            "taker_pressure": round(taker_pressure, 3),
            "composite_sentiment": round(np.clip(composite, -1, 1), 3),
            "ai_divergence_signal": ai_signal,
        }

    # ── 11a. OKX OHLCV (Fix #6: OKX-only 币种需要) ──

    def fetch_okx_ohlcv(self, symbol: str, limit: int = 400, bar: str = "1D") -> Optional[list]:
        """
        从 OKX 拉取 K线数据, 返回 ccxt 兼容格式: [[ts, open, high, low, close, vol], ...]
        用于 OKX-only 币种 (如 HYPE) 无 Binance 数据时的 fallback。
        """
        inst_id = self._to_okx_inst_id(symbol)
        data = self._okx_get("/api/v5/market/candles", {
            "instId": inst_id,
            "bar": bar,
            "limit": str(limit),
        })
        if not data or data.get("code") != "0":
            log.warning(f"OKX OHLCV {symbol} 拉取失败")
            return None
        candles = data.get("data", [])
        if not candles:
            return None
        # OKX 返回格式: [ts, open, high, low, close, vol, volCcy] (反转时间序)
        ohlcv = []
        for c in reversed(candles):  # OKX 最新在前, 反转为升序
            try:
                ts = int(c[0])
                o = float(c[1])
                h = float(c[2])
                l = float(c[3])
                cl = float(c[4])
                vol = float(c[5])
                ohlcv.append([ts, o, h, l, cl, vol])
            except (ValueError, IndexError, TypeError):
                continue
        return ohlcv if ohlcv else None

    # ── 11b. 实时点差 (Fix #5) ──

    def fetch_okx_spread(self, symbol: str) -> float:
        """
        从 OKX ticker 获取实时 bid/ask spread。
        Returns: spread as fraction (e.g. 0.0005 = 5bps)
        """
        inst_id = self._to_okx_inst_id(symbol)
        data = self._okx_get("/api/v5/market/ticker", {"instId": inst_id})
        if not data or data.get("code") != "0":
            return None
        tickers = data.get("data", [])
        if not tickers:
            return None
        t = tickers[0]
        try:
            bid = float(t.get("bidPx", 0))
            ask = float(t.get("askPx", 0))
            if bid > 0 and ask > 0:
                return (ask - bid) / bid  # relative spread
        except (ValueError, TypeError):
            pass
        return None

    # ── 11. 主入口: refresh_all() ──

    def refresh_all(self,
                     symbols: List[str] = None,
                     include_ai_correlation: bool = False,
                     force: bool = False) -> MarketSentimentSnapshot:
        """
        主入口 — 刷新所有情绪数据。

        Args:
          symbols: 要拉取的 swap 标的列表 (None = 自动从 symbol_config 获取)
          include_ai_correlation: 是否计算 AI 相关性 (较慢, 默认关)
          force: 忽略缓存强制刷新

        Returns MarketSentimentSnapshot
        """
        if symbols is None:
            from symbol_config import TIER1_ML_HEAVY, TIER2_TECHNICAL_LIGHT
            from symbol_config import OKX_ALL_NON_CRYPTO_SWAPS
            crypto_swaps = [
                s.replace('/USDT', '/USDT') for s in TIER1_ML_HEAVY + TIER2_TECHNICAL_LIGHT
            ]
            symbols = list(set(crypto_swaps + OKX_ALL_NON_CRYPTO_SWAPS))

        use_cache = not force
        now_ts = datetime.now(timezone.utc).isoformat()

        log.info(f"🎭 刷新市场情绪数据 (force={force}, symbols={len(symbols)})")

        # 并行拉取各项数据 (按依赖顺序)
        funding_rates = self.fetch_funding_rates(symbols, use_cache=use_cache)
        open_interests = self.fetch_open_interests(symbols, use_cache=use_cache)
        long_short = self.fetch_long_short_ratios(use_cache=use_cache)
        taker_volumes = self.fetch_taker_volumes(use_cache=use_cache)
        fear_greed = self.fetch_fear_greed(use_cache=use_cache)
        price_changes = self._get_price_changes(symbols)

        # 聚合分析
        sector_flows = self.compute_sector_flows(
            funding_rates, open_interests, taker_volumes, price_changes
        )
        rotation = self.predict_sector_rotation(sector_flows, fear_greed)

        ai_correlations = []
        if include_ai_correlation:
            ai_correlations = self.compute_ai_narrative_correlations(use_cache=use_cache)

        snapshot = MarketSentimentSnapshot(
            fear_greed=fear_greed,
            funding_rates=funding_rates,
            open_interests=open_interests,
            long_short_ratios=long_short,
            taker_volumes=taker_volumes,
            sector_flows=sector_flows,
            ai_correlations=ai_correlations,
            rotation_prediction=rotation,
            updated_at=now_ts,
        )

        log.info(f"✅ 市场情绪刷新完成: F&G={fear_greed.current_value} "
                 f"({fear_greed.classification}), "
                 f"板块={len(sector_flows)}, "
                 f"资金费率={len(funding_rates)}, "
                 f"轮动→{rotation.next_leader} "
                 f"(conf={rotation.rotation_confidence:.0%})")

        return snapshot

    # ── 12. 状态报告 ──

    def get_status_report(self) -> str:
        """生成 Markdown 格式的市场情绪摘要"""
        fg = self.fetch_fear_greed(use_cache=True)
        sector_data = self._read_cache("sector_flows") or {}
        rotation = self._read_cache("rotation_prediction") or {}
        ls_data = self._read_cache("long_short_ratios") or {}

        lines = [
            f"## 🎭 市场情绪",
            f"",
            f"**恐惧贪婪**: {fg.current_value} ({fg.classification})",
            f"  1d变化: {fg.change_1d:+d}, 7d变化: {fg.change_7d:+d}",
            f"",
            f"### 📊 板块资金流",
        ]

        if sector_data:
            # 按 flow_score 排序
            sorted_sectors = sorted(
                sector_data.items(),
                key=lambda x: x[1].get("flow_score", 0) if isinstance(x[1], dict) else 0,
                reverse=True,
            )
            lines.append(f"| 板块 | 流向 | 价格24h | OI变化 | 费率 | Taker比 |")
            lines.append(f"|------|------|---------|--------|------|---------|")
            for name, info in sorted_sectors:
                if isinstance(info, dict):
                    direction = info.get("dominant_direction", "neutral")
                    emoji = {"inflow": "🟢", "outflow": "🔴", "neutral": "⚪"}.get(direction, "⚪")
                    lines.append(
                        f"| {emoji} {name} | {direction} | "
                        f"{info.get('avg_price_change_24h_pct', 0):+.1f}% | "
                        f"{info.get('avg_oi_change_24h_pct', 0):+.1f}% | "
                        f"{info.get('avg_funding_rate', 0):.4f} | "
                        f"{info.get('avg_taker_buy_sell_ratio', 1):.2f} |"
                    )

        if rotation:
            lines.append(f"")
            lines.append(f"### 🔮 轮动预测")
            if isinstance(rotation, dict):
                lines.append(f"- 当前领涨: **{rotation.get('current_leader', '?')}**")
                lines.append(f"- 预测下个: **{rotation.get('next_leader', '?')}** "
                            f"(置信={rotation.get('rotation_confidence', 0):.0%})")
                for r in rotation.get("reasoning", []):
                    lines.append(f"  - {r}")

        if ls_data:
            lines.append(f"")
            lines.append(f"### ⚖️ 多空比 (Top 偏高/偏低)")
            # 极端多空比
            extremes = []
            for ccy, info in ls_data.items():
                if isinstance(info, dict):
                    ratio = info.get("long_short_ratio", 1.0)
                    if ratio > 2 or ratio < 0.5:
                        extremes.append((ccy, ratio))
            if extremes:
                for ccy, ratio in sorted(extremes, key=lambda x: -x[1]):
                    lines.append(f"- {ccy}: {ratio:.2f} {'⚠️ 过度看多' if ratio > 2 else '💡 过度看空'}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# CLI 测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("🎭 Yina 市场情绪引擎 测试")
    print("=" * 50)
    engine = MarketSentimentEngine()
    snapshot = engine.refresh_all(force=True)

    print(f"\n😱 Fear & Greed: {snapshot.fear_greed.current_value} "
          f"({snapshot.fear_greed.classification})")
    print(f"   Extreme Fear: {snapshot.fear_greed.is_extreme_fear}, "
          f"Extreme Greed: {snapshot.fear_greed.is_extreme_greed}")

    print(f"\n📊 板块资金流 (Top 5):")
    sorted_flows = sorted(snapshot.sector_flows.items(),
                          key=lambda x: -x[1].flow_score)
    for name, flow in sorted_flows[:5]:
        emoji = {"inflow": "🟢", "outflow": "🔴", "neutral": "⚪"}.get(
            flow.dominant_direction, "⚪"
        )
        print(f"  {emoji} {name:20s} | flow={flow.flow_score:+.3f} | "
              f"price={flow.avg_price_change_24h_pct:+.1f}% | "
              f"OI={flow.avg_oi_change_24h_pct:+.1f}% | "
              f"FR={flow.avg_funding_rate:.4f}")

    if snapshot.rotation_prediction:
        r = snapshot.rotation_prediction
        print(f"\n🔮 轮动预测: {r.current_leader} → {r.next_leader} "
              f"(置信={r.rotation_confidence:.0%})")
        for reason in r.reasoning:
            print(f"   {reason}")

    print(f"\n💰 资金费率 (前5极端):")
    sorted_fr = sorted(snapshot.funding_rates.items(),
                       key=lambda x: -abs(x[1].funding_rate))
    for sym, fr in sorted_fr[:5]:
        print(f"  {sym:20s} | FR={fr.funding_rate:+.4f}% "
              f"| 年化={fr.annualized_rate:+.1f}%")

    print(f"\n" + "=" * 50)
    print(engine.get_status_report())
