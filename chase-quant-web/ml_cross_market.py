#!/usr/bin/env python3
"""
Phase 6: 跨市场 + 链上数据接入
===============================
CrossMarketFetcher — 拉取跨市场实际数据, 接入 FeatureFactoryV4 的 T 类特征

数据源:
  - ccxt Binance: ETH/USDT, BTC funding rate, ETH funding rate
  - yfinance: SPY, QQQ, GLD (黄金), DXY (美元指数), VIX (恐慌指数)
  - alternative.me: Fear & Greed Index (加密恐慌贪婪指数)

使用:
  from ml_cross_market import CrossMarketFetcher
  fetcher = CrossMarketFetcher()
  df_enriched = fetcher.enrich_dataframe(df_btc)  # 返回含 cross-market 列的 DF
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import warnings
from typing import Optional, Dict, Tuple

warnings.filterwarnings("ignore")


# ── Cache paths ──
DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = DATA_DIR / "cross_market_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class CrossMarketFetcher:
    """跨市场数据获取器 — 一站式拉取 ETH/SPY/DXY/VIX/F&G/Funding Rate"""

    # 外部资产定义
    EXTERNAL_ASSETS = {
        "eth":    {"ticker": "ETH/USDT",    "source": "ccxt",   "col": "eth_close"},
        "spy":    {"ticker": "SPY",         "source": "yfinance", "col": "spy_close"},
        "qqq":    {"ticker": "QQQ",         "source": "yfinance", "col": "qqq_close"},
        "gld":    {"ticker": "GLD",         "source": "yfinance", "col": "gld_close"},
        "dxy":    {"ticker": "DX-Y.NYB",    "source": "yfinance", "col": "dxy_close"},
        "vix":    {"ticker": "^VIX",        "source": "yfinance", "col": "vix_close"},
    }

    def __init__(self, cache_ttl_hours: int = 4):
        """
        Args:
            cache_ttl_hours: 缓存有效期, 默认4小时
        """
        self.cache_ttl = cache_ttl_hours
        self._data_available: Dict[str, bool] = {}
        self._status: Dict[str, str] = {}

    # ── 主入口 ──────────────────────────────────

    def enrich_dataframe(self, df_btc: pd.DataFrame,
                         include_funding: bool = True,
                         include_fear_greed: bool = True,
                         use_cache: bool = True) -> pd.DataFrame:
        """
        用跨市场数据增强 BTC OHLCV DataFrame.

        Args:
            df_btc: BTC OHLCV DataFrame (必须含 'date' 或 datetime index)
            include_funding: 是否加入资金费率
            include_fear_greed: 是否加入恐慌贪婪指数

        Returns:
            合并后的 DataFrame (side-effect: 添加 eth_close/spy_close/... 列)
        """
        df = df_btc.copy()

        # 确保有 datetime index
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], unit="ms")
            df.set_index("date", inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            if isinstance(df.index[0], (int, float)):
                df.index = pd.to_datetime(df.index, unit="ms")

        # 1. ETH 从 ccxt
        self._fetch_eth_ccxt(df, use_cache)

        # 2. SPY/QQQ/GLD/DXY/VIX 从 yfinance
        self._fetch_external_yfinance(df, use_cache)

        # 3. Funding rate
        if include_funding:
            self._fetch_funding_rates(df, use_cache)

        # 4. Fear & Greed
        if include_fear_greed:
            self._fetch_fear_greed(df, use_cache)

        return df

    # ── ETH (ccxt) ──────────────────────────────

    def _fetch_eth_ccxt(self, df: pd.DataFrame, use_cache: bool = True):
        """拉取 ETH/USDT 日线, 合并到 df"""
        try:
            cache_file = CACHE_DIR / "eth_usdt_daily.json"

            if use_cache and self._cache_valid(cache_file):
                data = json.loads(cache_file.read_text())
                self._status["eth"] = "cache"
            else:
                import ccxt
                exchange = ccxt.binance({"enableRateLimit": True, "timeout": 15000})
                ohlcv = exchange.fetch_ohlcv("ETH/USDT", "1d", limit=500)
                data = [
                    {"t": o[0], "c": o[4]}  # timestamp, close
                    for o in ohlcv
                ]
                cache_file.write_text(json.dumps(data))
                self._status["eth"] = "live"

            # 对齐到 BTC index
            eth_map = {}
            for d in data:
                ts = pd.Timestamp(d["t"], unit="ms")
                eth_map[ts.date()] = d["c"]

            df["eth_close"] = [eth_map.get(ts.date(), np.nan) for ts in df.index]
            df["eth_close"] = df["eth_close"].ffill()
            self._data_available["eth"] = True
            self._status.setdefault("eth", "ok")

        except Exception as e:
            self._status["eth"] = f"error: {str(e)[:40]}"
            self._data_available["eth"] = False
            df["eth_close"] = np.nan

    # ── 外部资产 (yfinance) ─────────────────────

    def _fetch_external_yfinance(self, df: pd.DataFrame, use_cache: bool = True):
        """拉取 SPY/QQQ/GLD/DXY/VIX 日线"""
        try:
            import yfinance as yf

            start_date = df.index[0].strftime("%Y-%m-%d") if hasattr(df.index[0], 'strftime') else str(df.index[0])[:10]
            end_date = df.index[-1].strftime("%Y-%m-%d") if hasattr(df.index[-1], 'strftime') else str(df.index[-1])[:10]

            # 扩展日期范围, yfinance 有时需要 buffer
            start_dt = pd.Timestamp(start_date) - timedelta(days=10)
            end_dt = pd.Timestamp(end_date) + timedelta(days=5)

            tickers = ["SPY", "QQQ", "GLD", "DX-Y.NYB", "^VIX"]
            col_map = {
                "SPY": "spy_close", "QQQ": "qqq_close", "GLD": "gld_close",
                "DX-Y.NYB": "dxy_close", "^VIX": "vix_close",
            }

            cache_file = CACHE_DIR / "external_yfinance.json"
            all_data = {}

            if use_cache and self._cache_valid(cache_file):
                all_data = json.loads(cache_file.read_text())
                self._status["yfinance"] = "cache"
            else:
                for ticker in tickers:
                    try:
                        ticker_obj = yf.Ticker(ticker)
                        hist = ticker_obj.history(start=start_dt.strftime("%Y-%m-%d"),
                                                   end=end_dt.strftime("%Y-%m-%d"))
                        if not hist.empty:
                            all_data[ticker] = [
                                {"t": str(idx.date()), "c": float(row["Close"])}
                                for idx, row in hist.iterrows()
                            ]
                    except Exception as exc:
                        self._status[f"yf_{ticker}"] = f"skip: {str(exc)[:30]}"
                        continue

                if all_data:
                    cache_file.write_text(json.dumps(all_data))
                self._status["yfinance"] = "live"

            # 合并各资产
            for ticker, col in col_map.items():
                data = all_data.get(ticker, [])
                if data:
                    price_map = {}
                    for d in data:
                        try:
                            price_map[pd.Timestamp(d["t"]).date()] = d["c"]
                        except Exception:
                            continue
                    df[col] = [price_map.get(ts.date(), np.nan) for ts in df.index]
                    df[col] = df[col].ffill()
                    self._data_available[col.replace("_close", "")] = True
                else:
                    df[col] = np.nan
                    self._data_available[col.replace("_close", "")] = False

        except Exception as e:
            self._status["yfinance"] = f"error: {str(e)[:40]}"
            for col in ["spy_close", "qqq_close", "gld_close", "dxy_close", "vix_close"]:
                df[col] = np.nan
                self._data_available[col.replace("_close", "")] = False

    # ── Funding Rate (ccxt) ─────────────────────

    def _fetch_funding_rates(self, df: pd.DataFrame, use_cache: bool = True):
        """
        拉取 BTC 和 ETH 永续合约资金费率.

        币安资金费率每8小时结算, 我们取日均值.
        正值=多头付空头(市场偏多), 负值=空头付多头(市场偏空).
        """
        try:
            cache_file = CACHE_DIR / "funding_rates.json"

            if use_cache and self._cache_valid(cache_file):
                fr_data = json.loads(cache_file.read_text())
                self._status["funding"] = "cache"
            else:
                import ccxt
                exchange = ccxt.binance({"enableRateLimit": True, "timeout": 15000})

                fr_data = {}
                for symbol in ["BTC/USDT:USDT", "ETH/USDT:USDT"]:
                    try:
                        # 获取最近500条 funding rate (每8小时, 约166天)
                        raw = exchange.fetch_funding_rate_history(symbol, limit=500, params={})
                        prefix = "btc" if "BTC" in symbol else "eth"
                        fr_data[prefix] = []
                        for r in raw:
                            fr_data[prefix].append({
                                "t": r.get("timestamp", 0),
                                "r": r.get("fundingRate", 0),
                            })
                    except Exception:
                        # older ccxt might not support fetch_funding_rate_history
                        fr_data["btc"] = []
                        fr_data["eth"] = []

                if fr_data:
                    cache_file.write_text(json.dumps(fr_data))
                self._status["funding"] = "live"

            # 日均 funding rate
            for prefix in ["btc", "eth"]:
                col_name = f"{prefix}_funding_rate"
                entries = fr_data.get(prefix, [])
                if entries:
                    # 按天聚合
                    daily_fr: Dict[str, list] = {}
                    for e in entries:
                        day = pd.Timestamp(e["t"], unit="ms").date()
                        if day not in daily_fr:
                            daily_fr[day] = []
                        daily_fr[day].append(e["r"])

                    fr_map = {day: float(np.mean(rates)) for day, rates in daily_fr.items()}
                    df[col_name] = [fr_map.get(ts.date(), 0.0) for ts in df.index]
                    df[col_name] = df[col_name].fillna(0.0)
                else:
                    df[col_name] = 0.0

            self._data_available["funding"] = True

        except Exception as e:
            self._status["funding"] = f"error: {str(e)[:40]}"
            df["btc_funding_rate"] = 0.0
            df["eth_funding_rate"] = 0.0
            self._data_available["funding"] = False

    # ── Fear & Greed (alternative.me) ────────────

    def _fetch_fear_greed(self, df: pd.DataFrame, use_cache: bool = True):
        """
        拉取 crypto Fear & Greed Index.

        API: https://api.alternative.me/fng/?limit=0
        返回最近约6年的每日数据 (免费, 无API key).

        值: 0-100, 0=极度恐惧, 100=极度贪婪
        作为情绪指标: 极度恐惧(≤25)通常是买入机会, 极度贪婪(≥75)是卖出信号.
        """
        try:
            import requests

            cache_file = CACHE_DIR / "fear_greed.json"

            if use_cache and self._cache_valid(cache_file):
                fg_data = json.loads(cache_file.read_text())
                self._status["fear_greed"] = "cache"
            else:
                resp = requests.get("https://api.alternative.me/fng/?limit=0", timeout=15)
                if resp.status_code == 200:
                    fg_data = resp.json()
                    cache_file.write_text(json.dumps(fg_data))
                    self._status["fear_greed"] = "live"
                else:
                    self._status["fear_greed"] = f"HTTP {resp.status_code}"
                    df["fear_greed"] = 50  # neutral fallback
                    self._data_available["fear_greed"] = False
                    return

            # 解析数据
            data_list = fg_data.get("data", [])
            fg_map: Dict[str, int] = {}
            for entry in data_list:
                try:
                    ts = datetime.fromtimestamp(int(entry["timestamp"]))
                    fg_map[ts.date()] = int(entry["value"])
                except Exception:
                    continue

            df["fear_greed"] = [fg_map.get(ts.date(), 50) for ts in df.index]
            df["fear_greed"] = df["fear_greed"].ffill().fillna(50).astype(int)
            self._data_available["fear_greed"] = True

        except Exception as e:
            self._status["fear_greed"] = f"error: {str(e)[:40]}"
            df["fear_greed"] = 50
            self._data_available["fear_greed"] = False

    # ── 状态查询 ────────────────────────────────

    def status_report(self) -> str:
        """返回数据获取状态 Markdown 表格"""
        lines = [
            "| 数据源 | 状态 | 列名 |",
            "|--------|------|------|",
        ]
        all_sources = [
            ("ETH/USDT (ccxt)", "eth", "eth_close"),
            ("SPY (yfinance)", "spy", "spy_close"),
            ("QQQ (yfinance)", "qqq", "qqq_close"),
            ("GLD 黄金 (yfinance)", "gld", "gld_close"),
            ("DXY 美元指数 (yfinance)", "dxy", "dxy_close"),
            ("VIX 恐慌指数 (yfinance)", "vix", "vix_close"),
            ("BTC Funding Rate (ccxt)", "funding", "btc_funding_rate"),
            ("ETH Funding Rate (ccxt)", "funding", "eth_funding_rate"),
            ("F&G 恐慌贪婪 (alternative.me)", "fear_greed", "fear_greed"),
        ]
        for name, key, col in all_sources:
            avail = self._data_available.get(key, None)
            if avail is True:
                status = "✅"
            elif avail is False:
                status = "❌"
            else:
                status = "—"
            detail = self._status.get(key, "")
            if detail and detail not in ("ok",):
                status += f" ({detail})"
            lines.append(f"| {name} | {status} | `{col}` |")

        return "\n".join(lines)

    def available_count(self) -> int:
        """返回成功获取的数据源数量"""
        return sum(1 for v in self._data_available.values() if v)

    @property
    def is_healthy(self) -> bool:
        """至少核心数据 (ETH + F&G) 可用"""
        return self._data_available.get("eth", False) and \
               self._data_available.get("fear_greed", False)

    # ── 内部 ────────────────────────────────────

    def _cache_valid(self, cache_file: Path) -> bool:
        """检查缓存是否有效"""
        if not cache_file.exists():
            return False
        age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return age.total_seconds() < self.cache_ttl * 3600


# ── CLI 测试 ────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🌐 Phase 6: 跨市场 + 链上数据接入 — 数据获取测试")
    print("=" * 60)

    # 构造 BTC 日期范围
    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True, "timeout": 15000})
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1d", limit=400)
        df = pd.DataFrame(ohlcv, columns=["date", "open", "high", "low", "close", "volume"])
        print(f"\n📊 BTC数据: {len(df)} 日")

        fetcher = CrossMarketFetcher(cache_ttl_hours=4)
        df_enriched = fetcher.enrich_dataframe(df, use_cache=False)

        print(f"\n{fetcher.status_report()}")
        print(f"\n数据源可用: {fetcher.available_count()}/9")

        # 显示新增列
        new_cols = ["eth_close", "spy_close", "qqq_close", "gld_close",
                    "dxy_close", "vix_close", "btc_funding_rate",
                    "eth_funding_rate", "fear_greed"]
        present = [c for c in new_cols if c in df_enriched.columns]
        print(f"\n新增列: {present}")
        if present:
            last_vals = df_enriched[present].iloc[-1]
            print(f"\n最新值:")
            for col in present:
                val = last_vals[col]
                if pd.notna(val):
                    print(f"  {col}: {val:.4f}" if abs(val) < 100 else f"  {col}: {val:.0f}")
                else:
                    print(f"  {col}: NaN")

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
