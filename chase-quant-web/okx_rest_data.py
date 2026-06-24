#!/usr/bin/env python3
"""
🔗 OKX REST Data Provider — 纯 REST API 市场数据获取 (绕过 ccxt)

ccxt 的 load_markets() 在解析 OKX 某些字段时会抛 NoneType 比较错误。
此模块完全不依赖 ccxt，直接用 OKX REST API 获取所有市场数据。

用法:
    from okx_rest_data import OKXDataProvider

    okx = OKXDataProvider()

    # 单币种 OHLCV
    df = okx.fetch_ohlcv("BTC-USDT", "1D", limit=200)

    # 批量 OHLCV
    data = okx.fetch_all_ohlcv(["BTC-USDT", "ETH-USDT", "SOL-USDT"], limit=200)

    # Ticker
    ticker = okx.fetch_ticker("BTC-USDT")

    # 批量 Tickers
    tickers = okx.fetch_tickers(["BTC-USDT", "ETH-USDT"])
"""

from __future__ import annotations
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timezone

import requests
import pandas as pd

log = logging.getLogger(__name__)

# OKX API 域名池 (自动切换)
OKX_HOSTS = [
    "https://www.okx.cab",
    "https://www.okx.com",
]


class OKXDataProvider:
    """
    OKX 纯 REST 数据提供器 — 零 ccxt 依赖。

    功能:
      - 日线/4h/1h/30m OHLCV
      - 实时 Ticker
      - 批量获取
      - 自动域名切换
    """

    # OKX bar 格式映射
    BAR_MAP = {
        "1D": "1D", "1d": "1D",
        "4H": "4H", "4h": "4H",
        "1H": "1H", "1h": "1H",
        "30m": "30m", "30M": "30m",
        "15m": "15m", "15M": "15m",
        "5m": "5m", "5M": "5m",
    }

    def __init__(self, host: str = None, timeout: int = 15):
        """
        Args:
            host: OKX API 域名 (None=自动探测)
            timeout: 请求超时秒数
        """
        self._host = host
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Yina-Quant/3.0",
            "Accept": "application/json",
        })
        # 找到可用的 host
        if self._host is None:
            self._host = self._find_host()

    def _find_host(self) -> str:
        """探测可用的 OKX API 域名"""
        for host in OKX_HOSTS:
            try:
                url = f"{host}/api/v5/public/time"
                resp = self._session.get(url, timeout=5)
                if resp.status_code == 200 and resp.json().get("code") == "0":
                    log.info(f"✅ OKX REST 数据源: {host}")
                    return host
            except Exception:
                continue
        log.warning(f"⚠️ 所有 OKX 域名不可达，使用默认 {OKX_HOSTS[0]}")
        return OKX_HOSTS[0]

    @property
    def host(self) -> str:
        return self._host

    # ═══════════════════════════════════════════════════════
    # Ticker
    # ═══════════════════════════════════════════════════════

    def fetch_ticker(self, symbol: str) -> Optional[dict]:
        """
        获取单个币种的 Ticker。

        Args:
            symbol: "BTC-USDT" 或 "BTC/USDT" (自动转换)

        Returns:
            {"last": 63836.1, "high24h": 65629.5, "low24h": 63806.3, ...}
        """
        inst_id = self._to_inst_id(symbol)
        try:
            url = f"{self._host}/api/v5/market/ticker?instId={inst_id}"
            resp = self._session.get(url, timeout=self._timeout)
            data = resp.json()
            if data.get("code") == "0" and data.get("data"):
                t = data["data"][0]
                return {
                    "symbol": symbol,
                    "instId": inst_id,
                    "last": float(t["last"]),
                    "high24h": float(t["high24h"]),
                    "low24h": float(t["low24h"]),
                    "bid": float(t.get("bidPx", 0)),
                    "ask": float(t.get("askPx", 0)),
                    "volume24h": float(t.get("vol24h", 0)),
                    "volumeCcy24h": float(t.get("volCcy24h", 0)),
                    "change24h": float(t.get("sodUtc0", 0)),
                    "timestamp": t.get("ts", ""),
                }
            else:
                log.warning(f"⚠️ Ticker {inst_id} 返回异常: {data.get('msg', '?')}")
                return None
        except Exception as e:
            log.error(f"❌ Ticker {inst_id} 获取失败: {e}")
            return None

    def fetch_tickers(self, symbols: List[str]) -> Dict[str, dict]:
        """
        批量获取 Tickers。

        Args:
            symbols: ["BTC-USDT", "ETH-USDT", ...]

        Returns:
            {symbol: ticker_dict, ...}
        """
        if not symbols:
            return {}

        results = {}
        for sym in symbols:
            ticker = self.fetch_ticker(sym)
            if ticker:
                results[sym] = ticker
            time.sleep(0.05)  # 30 req/s limit (每对0.05s = 20 req/s, 安全)
        return results

    # ═══════════════════════════════════════════════════════
    # OHLCV (K线)
    # ═══════════════════════════════════════════════════════

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1D",
                    limit: int = 200) -> Optional[pd.DataFrame]:
        """
        获取单个币种的 OHLCV 数据。

        Args:
            symbol: "BTC-USDT" 或 "BTC/USDT"
            timeframe: "1D" / "4H" / "1H" / "30m" / "15m" / "5m"
            limit: 最多返回的K线数量 (max 300)

        Returns:
            DataFrame with columns [date, open, high, low, close, volume]
        """
        inst_id = self._to_inst_id(symbol)
        bar = self.BAR_MAP.get(timeframe, "1D")

        try:
            url = f"{self._host}/api/v5/market/candles"
            params = {"instId": inst_id, "bar": bar, "limit": str(min(limit, 300))}
            resp = self._session.get(url, params=params, timeout=self._timeout)
            data = resp.json()

            if data.get("code") != "0":
                log.warning(f"⚠️ OHLCV {inst_id} 返回异常: {data.get('msg', '?')}")
                return None

            candles = data["data"]
            if not candles:
                log.warning(f"⚠️ OHLCV {inst_id} 无数据")
                return None

            # OKX 返回: [ts, o, h, l, c, vol, volCcy] — newest first
            rows = []
            for c in candles:
                rows.append({
                    "date": pd.to_datetime(int(c[0]), unit="ms"),
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": float(c[5]),
                })

            df = pd.DataFrame(rows)
            df = df.sort_values("date").reset_index(drop=True)
            return df

        except Exception as e:
            log.error(f"❌ OHLCV {inst_id} 获取失败: {e}")
            return None

    def fetch_all_ohlcv(self, symbols: List[str], timeframe: str = "1D",
                        limit: int = 200) -> Dict[str, pd.DataFrame]:
        """
        批量获取 OHLCV 数据。

        Args:
            symbols: ["BTC-USDT", "ETH-USDT", ...]
            timeframe: K线周期
            limit: 每币种最多K线数

        Returns:
            {symbol: DataFrame, ...}  只返回数据充足的币种 (≥50行)
        """
        if not symbols:
            return {}

        market_data = {}
        for i, sym in enumerate(symbols):
            try:
                df = self.fetch_ohlcv(sym, timeframe, limit)
                if df is not None and len(df) >= min(50, limit):
                    market_data[sym] = df
                elif df is not None:
                    log.warning(f"  ⚠️ {sym} 数据不足 ({len(df)}行 < 50)")
            except Exception as e:
                log.warning(f"  ⚠️ {sym} 拉取失败: {e}")

            # 速率限制: OKX 允许 20 req/2s
            if (i + 1) % 15 == 0:
                time.sleep(0.3)
            else:
                time.sleep(0.05)

        return market_data

    # ═══════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _to_inst_id(symbol: str) -> str:
        """BTC/USDT → BTC-USDT"""
        return symbol.replace("/", "-")

    @staticmethod
    def from_inst_id(inst_id: str) -> str:
        """BTC-USDT → BTC/USDT"""
        return inst_id.replace("-", "/")

    @staticmethod
    def inst_id_to_symbol(inst_id: str) -> str:
        """BTC-USDT-SWAP → BTC/USDT"""
        return inst_id.replace("-USDT-", "/USDT").replace("-SWAP", "")


# ═══════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════

_provider: Optional[OKXDataProvider] = None


def get_okx_provider() -> OKXDataProvider:
    """获取全局单例 OKXDataProvider"""
    global _provider
    if _provider is None:
        _provider = OKXDataProvider()
    return _provider


# ═══════════════════════════════════════════════════════
# Convenience Functions (compatible with existing code)
# ═══════════════════════════════════════════════════════

def fetch_okx_ohlcv(symbol: str, timeframe: str = "1D",
                    limit: int = 200) -> Optional[pd.DataFrame]:
    """便捷函数: 获取单币种 OHLCV"""
    return get_okx_provider().fetch_ohlcv(symbol, timeframe, limit)


def fetch_okx_all_ohlcv(symbols: List[str], timeframe: str = "1D",
                        limit: int = 200) -> Dict[str, pd.DataFrame]:
    """便捷函数: 批量获取 OHLCV"""
    return get_okx_provider().fetch_all_ohlcv(symbols, timeframe, limit)


def fetch_okx_ticker(symbol: str) -> Optional[dict]:
    """便捷函数: 获取单币种 Ticker"""
    return get_okx_provider().fetch_ticker(symbol)


def fetch_okx_tickers(symbols: List[str]) -> Dict[str, dict]:
    """便捷函数: 批量获取 Tickers"""
    return get_okx_provider().fetch_tickers(symbols)


# ═══════════════════════════════════════════════════════
# CLI Test
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("🔗 OKX REST Data Provider — 测试\n")

    okx = OKXDataProvider()
    print(f"📍 使用 Host: {okx.host}\n")

    # 1. 测试 Ticker
    print("=" * 50)
    print("📊 Ticker 测试")
    print("=" * 50)
    btc_ticker = okx.fetch_ticker("BTC-USDT")
    if btc_ticker:
        print(f"  BTC: last=${btc_ticker['last']:.1f}, "
              f"high24h=${btc_ticker['high24h']:.1f}, "
              f"low24h=${btc_ticker['low24h']:.1f}, "
              f"vol=${btc_ticker['volumeCcy24h']:.0f}")

    # 2. 测试 OHLCV
    print(f"\n{'='*50}")
    print("📈 OHLCV 测试")
    print("=" * 50)
    df = okx.fetch_ohlcv("BTC-USDT", "1D", limit=5)
    if df is not None:
        print(f"  BTC 日线 (最近5天):")
        for _, row in df.iterrows():
            chg = (row['close'] / row['open'] - 1) * 100
            print(f"    {row['date'].strftime('%m-%d'):8s} "
                  f"O={row['open']:.0f} H={row['high']:.0f} "
                  f"L={row['low']:.0f} C={row['close']:.0f} "
                  f"({chg:+.1f}%)")

    # 3. 测试批量
    print(f"\n{'='*50}")
    print("📊 批量测试 (3 币种)")
    print("=" * 50)
    data = okx.fetch_all_ohlcv(["BTC-USDT", "ETH-USDT", "SOL-USDT"], limit=200)
    for sym, _df in data.items():
        print(f"  {sym}: {len(_df)} 行, "
              f"最新价=${_df['close'].iloc[-1]:.1f}")

    print(f"\n✅ OKX REST 数据源测试完成")
