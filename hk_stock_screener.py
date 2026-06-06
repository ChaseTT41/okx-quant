"""
HK Stock Screener — 港股全市场批量评分筛选器
对 3136 只港股批量运行五维评分, 输出 Top 榜单 + 行业分布
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from hk_stock_data import get_data
from hk_five_dim_scorer import FiveDimScorer, FiveDimResult


def _score_one(code: str) -> Optional[dict]:
    """单只评分 (用于并行)"""
    try:
        scorer = FiveDimScorer()
        r = scorer.score(code)
        if r.stars >= 1:
            return r.to_dict()
    except Exception:
        pass
    return None


class HKScreener:
    """港股全市场筛选器"""

    def __init__(self, min_volume: int = 500_000, min_price: float = 0.5, min_days: int = 60):
        self.data = get_data()
        self.min_volume = min_volume       # 最小日均成交量
        self.min_price = min_price          # 最低股价 (过滤仙股)
        self.min_days = min_days            # 最少数据天数

    def _prefilter(self) -> List[str]:
        """预筛选: 过滤仙股、无流动性、数据不足的股票"""
        df = self.data.all_daily
        stock_list = self.data.stock_list
        codes = stock_list["代码"].tolist()

        qualified = []
        for code in codes:
            try:
                sub = df[df["代码"] == code]
                if len(sub) < self.min_days:
                    continue
                recent = sub.tail(20)
                avg_vol = recent["成交量"].mean()
                avg_price = recent["收盘"].mean()
                if avg_vol >= self.min_volume and avg_price >= self.min_price:
                    qualified.append(code)
            except Exception:
                continue

        print(f"预筛选: {len(codes)} → {len(qualified)} 只 (过滤{len(codes)-len(qualified)}只)")
        return qualified

    def run(self, top_n: int = 50, parallel: bool = True, max_workers: int = 4) -> pd.DataFrame:
        """
        全市场扫描, 返回 Top N 综合评分排名
        """
        codes = self._prefilter()

        results = []
        if parallel and len(codes) > 100:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_score_one, c): c for c in codes}
                for i, future in enumerate(as_completed(futures)):
                    if (i + 1) % 200 == 0:
                        print(f"  进度: {i+1}/{len(codes)}")
                    try:
                        r = future.result(timeout=30)
                        if r:
                            results.append(r)
                    except Exception:
                        pass
        else:
            scorer = FiveDimScorer()
            for i, code in enumerate(codes):
                if (i + 1) % 200 == 0:
                    print(f"  进度: {i+1}/{len(codes)}")
                try:
                    r = scorer.score(code)
                    if r.stars >= 1:
                        results.append(r.to_dict())
                except Exception:
                    pass

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values("综合分", ascending=False).reset_index(drop=True)
            df = df.head(top_n)
        return df

    def run_quick(self, codes: List[str]) -> pd.DataFrame:
        """快速评分指定股票列表"""
        scorer = FiveDimScorer()
        results = []
        for code in codes:
            try:
                r = scorer.score(code)
                if r.stars >= 1:
                    results.append(r.to_dict())
            except Exception:
                pass
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values("综合分", ascending=False).reset_index(drop=True)
        return df

    def report(self, df: pd.DataFrame) -> str:
        """格式化输出筛选报告"""
        if df.empty:
            return "📭 无符合条件的股票"

        lines = [
            f"## 🎯 港股五维评分 Top {len(df)} 榜单",
            f"*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
            "",
            "| # | 代码 | 名称 | 收盘价 | 综合分 | 操作 | 星级 | 置信度 | 趋势 | RS | SR | 基本 | 风险 |",
            "|---|------|------|:---:|:---:|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]

        for i, row in df.iterrows():
            stars_str = "⭐" * int(row["星级"]) + "☆" * (5 - int(row["星级"]))
            lines.append(
                f"| {i+1} | {row['代码']} | {row['名称']} | "
                f"¥{row['收盘价']:.2f} | {row['综合分']:.0f} | "
                f"{row['操作']} | {stars_str} | {row['置信度']} | "
                f"{int(row['趋势'])} | {int(row['超买超卖'])} | {int(row['支撑阻力'])} | "
                f"{int(row['基本面'])} | {int(row['风险'])} |"
            )

        # 统计分布
        actions = df["操作"].value_counts()
        lines.append("")
        lines.append("### 📊 操作分布")
        for action, count in actions.items():
            lines.append(f"- {action}: {count} 只")

        stars_dist = df["星级"].value_counts().sort_index(ascending=False)
        lines.append("")
        lines.append("### ⭐ 星级分布")
        for stars, count in stars_dist.items():
            lines.append(f"- {'⭐'*int(stars)}{'☆'*(5-int(stars))}: {count} 只")

        # Top 5 详情
        lines.append("")
        lines.append("### 🏆 Top 5 详情")
        scorer = FiveDimScorer()
        for code in df.head(5)["代码"].values:
            r = scorer.score(str(code))
            lines.append(f"**{r.code} {r.name}** — {r.action} {r.stars}⭐ | ¥{r.close:.2f} | 综合{r.composite:.1f}")
            lines.append(f"> 趋势:{r.trend.score:.0f} | 超买超卖:{r.ob_os.score:.0f} | 支撑阻力:{r.sr.score:.0f} | 基本面:{r.fundamental.score:.0f} | 风险:{r.risk.score:.0f}")
            lines.append(f"> 止损:¥{r.stop_loss:.2f} | 信号: {', '.join(r.trend.signals + r.fundamental.signals)[:120]}")

        return "\n".join(lines)


# ── CLI ──
if __name__ == "__main__":
    import sys

    screener = HKScreener(min_volume=300_000, min_price=1.0)

    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        # 全市场扫描 (耗时较长)
        print("🔍 全市场扫描中...")
        df = screener.run(top_n=50, parallel=False)
    else:
        # 快速模式: 知名港股 + 热门标的
        print("⚡ 快速模式: 扫描知名港股...")
        watchlist = [
            # 科技
            "00700", "09988", "09618", "03690", "09888", "01810", "09999", "02382",
            # 金融
            "00005", "00388", "01299", "02388", "00939", "01398", "03968",
            # 地产/综合
            "00001", "00016", "01109", "00688",
            # 消费
            "02020", "09633", "09961", "09901",
            # 医药
            "02269", "01801", "09926", "06160",
            # 能源/工业
            "00883", "00386", "01088",
            # ETF
            "02800", "02828", "03033",
            # 其他热门
            "00175", "02331", "01833", "02269", "09626", "09992",
        ]
        df = screener.run_quick(watchlist)

    if not df.empty:
        report = screener.report(df)
        print(report)

        # 保存
        out = Path(__file__).parent / "reports" / f"hk_screener_{datetime.now().strftime('%Y%m%d')}.md"
        out.parent.mkdir(exist_ok=True)
        out.write_text(report)
        print(f"\n📄 报告已保存: {out}")
