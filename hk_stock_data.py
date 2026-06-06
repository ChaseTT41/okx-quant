"""
HK Stock Data Loader — 港股数据加载器
数据源: hk.zip → ~/yina-app/data/hk_stocks/
覆盖: 3136只港股, 2007-2026 日线 OHLCV
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path
from functools import lru_cache
from typing import Optional

DATA_DIR = Path(__file__).parent / "data" / "hk_stocks"
ALL_DAILY = DATA_DIR / "all_daily.parquet"
STOCK_LIST = DATA_DIR / "stock_list.parquet"
DAILY_DIR = DATA_DIR / "daily"


class HKStockData:
    """港股数据加载 + 缓存"""

    def __init__(self):
        self._all_daily: pd.DataFrame | None = None
        self._stock_list: pd.DataFrame | None = None
        self._daily_cache: dict[str, pd.DataFrame] = {}

    # ── 股票列表 ──
    @property
    def stock_list(self) -> pd.DataFrame:
        """返回: DataFrame[代码, 名称] — 3136只港股"""
        if self._stock_list is None:
            self._stock_list = pd.read_parquet(STOCK_LIST)
            self._stock_list["代码"] = self._stock_list["代码"].astype(str).str.zfill(5)
        return self._stock_list

    def search(self, keyword: str) -> pd.DataFrame:
        """按名称/代码搜索股票"""
        df = self.stock_list
        mask = df["名称"].str.contains(keyword, case=False, na=False) | df["代码"].str.startswith(keyword)
        return df[mask]

    def lookup(self, code_or_name: str) -> Optional[dict]:
        """代码→名称 或 名称→代码"""
        df = self.stock_list
        code_or_name = str(code_or_name).strip()
        # try exact code match
        row = df[df["代码"] == code_or_name.zfill(5)]
        if not row.empty:
            return {"代码": row.iloc[0]["代码"], "名称": row.iloc[0]["名称"]}
        # try name match
        row = df[df["名称"] == code_or_name]
        if not row.empty:
            return {"代码": row.iloc[0]["代码"], "名称": row.iloc[0]["名称"]}
        # try fuzzy
        row = df[df["名称"].str.contains(code_or_name, case=False, na=False)]
        if not row.empty:
            return {"代码": row.iloc[0]["代码"], "名称": row.iloc[0]["名称"]}
        return None

    # ── 全部日线 ──
    @property
    def all_daily(self) -> pd.DataFrame:
        """返回: 合并日线 DataFrame[日期, 开盘, 收盘, 最高, 最低, 成交量, 代码] — 145万行"""
        if self._all_daily is None:
            df = pd.read_parquet(ALL_DAILY)
            df["日期"] = pd.to_datetime(df["日期"])
            df["代码"] = df["代码"].astype(str).str.zfill(5)
            df = df.sort_values(["代码", "日期"]).reset_index(drop=True)
            self._all_daily = df
        return self._all_daily

    # ── 单只股票日线 ──
    def get_daily(self, code: str) -> pd.DataFrame:
        """获取单只股票的完整日线"""
        code = str(code).zfill(5)
        if code in self._daily_cache:
            return self._daily_cache[code]

        fpath = DAILY_DIR / f"{code}.parquet"
        if not fpath.exists():
            # fallback: slice from all_daily
            df = self.all_daily[self.all_daily["代码"] == code].copy()
        else:
            df = pd.read_parquet(fpath)
            df["日期"] = pd.to_datetime(df["日期"])
            df["代码"] = df["代码"].astype(str).str.zfill(5)

        df = df.sort_values("日期").reset_index(drop=True)
        self._daily_cache[code] = df
        return df

    # ── 多股票批量获取 ──
    def get_multi_daily(self, codes: list[str], start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        """批量获取多只股票日线, 返回合并DataFrame"""
        df = self.all_daily
        codes = [str(c).zfill(5) for c in codes]
        mask = df["代码"].isin(codes)
        if start_date:
            mask &= df["日期"] >= pd.Timestamp(start_date)
        if end_date:
            mask &= df["日期"] <= pd.Timestamp(end_date)
        return df[mask].copy()

    # ── 市场概览 ──
    def market_summary(self) -> dict:
        """数据概览统计"""
        df = self.all_daily
        return {
            "股票数量": df["代码"].nunique(),
            "日期范围": f"{df['日期'].min().date()} ~ {df['日期'].max().date()}",
            "总记录数": len(df),
            "数据年数": round((df["日期"].max() - df["日期"].min()).days / 365.25, 1),
        }

    # ── 基准指数 ──
    def get_benchmark(self, benchmark: str = "HSI") -> pd.Series | None:
        """
        获取基准指数日线收益率
        HSI=恒生指数(^HSI未直接覆盖, 用代表性股票等权近似)
        或直接返回 all_daily 的平均收盘价变化作为市场 proxy
        """
        # 用包含"恒生"关键词的ETF或指标
        if benchmark == "HSI":
            # 02800 盈富基金 = 追踪恒指ETF, 流动性最好
            try:
                df = self.get_daily("02800")
                s = df.set_index("日期")["收盘"]
                return s
            except Exception:
                pass
        return None


# ── 全局单例 ──
_data: HKStockData | None = None


def get_data() -> HKStockData:
    global _data
    if _data is None:
        _data = HKStockData()
    return _data


# ── CLI ──
if __name__ == "__main__":
    d = get_data()
    print("📊 HK Stock Data Loader")
    print(d.market_summary())
    print("\n🔍 搜索 '腾讯':")
    print(d.search("腾讯"))
    print("\n📈 00005 汇丰控股 最近5天:")
    print(d.get_daily("00005").tail())
