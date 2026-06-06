"""
HK Stock Momentum Rotation — 港股多资产动量轮动回测
基于 backtrader + 五维评分信号增强

策略变体:
  A. 纯多头 (LongOnly) — 做多 Top N 动量最强
  B. 多空中性 (LongShort) — 做多 Top + 做空 Bottom
  C. 五维评分加权 — 用五维综合分替代纯动量排名

基准: 02800 盈富基金 (恒指ETF)
"""

from __future__ import annotations
import backtrader as bt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import quantstats as qs

from hk_stock_data import get_data

# ── 策略参数 ──
ASSET_POOL = [
    # 科技
    "00700", "09988", "03690", "09618", "01810", "09999", "09888",
    # 金融
    "00005", "00388", "01299", "02388", "00939", "01398",
    # 地产
    "00016", "01109", "00688",
    # 消费
    "02020", "09633", "09961",
    # 医药
    "02269", "01801", "06160",
    # 能源
    "00883", "01088",
    # ETF 基准
    "02800",
]

MOMENTUM_WINDOW = 40          # 动量计算窗口 (天)
REBALANCE_FREQ = 10            # 再平衡频率 (天)
TOP_N = 5                       # 持仓数量
BENCHMARK = "02800"             # 盈富基金 = 恒指 proxy

COMMISSION = 0.001              # 手续费 0.1%
SLIPPAGE = 0.001                # 滑点 0.1%
INITIAL_CASH = 1_000_000        # 初始资金 100万 HKD


# ── 动量计算指示器 ──
class MomentumIndicator(bt.Indicator):
    """N日动量 = (今收 - N日前收) / N日前收"""
    lines = ('momentum',)
    params = (('period', 20),)

    def __init__(self):
        self.lines.momentum = (self.data.close / self.data.close(-self.p.period) - 1)


# ── 策略 A: 纯多头动量轮动 ──
class MomentumRotationLong(bt.Strategy):
    params = (
        ('momentum_window', MOMENTUM_WINDOW),
        ('rebalance_freq', REBALANCE_FREQ),
        ('top_n', TOP_N),
        ('benchmark_code', BENCHMARK),
    )

    def __init__(self):
        self.momentums = {}
        for d in self.datas:
            self.momentums[d._name] = MomentumIndicator(d, period=self.p.momentum_window)
        self.day_count = 0

    def next(self):
        self.day_count += 1
        if self.day_count % self.p.rebalance_freq != 0:
            return

        # 计算所有资产的动量排名
        scores = []
        for d in self.datas:
            if d._name == self.p.benchmark_code:
                continue  # 基准不参与交易
            if len(d) < self.p.momentum_window + 1:
                continue
            mom = self.momentums[d._name][0]
            if not np.isnan(mom):
                scores.append((d, mom))

        if not scores:
            return

        scores.sort(key=lambda x: x[1], reverse=True)
        top_codes = [s[0]._name for s in scores[:self.p.top_n]]

        # 平掉不在 Top N 的持仓
        for d in self.datas:
            if d._name == self.p.benchmark_code:
                continue
            pos = self.getposition(d).size
            if pos > 0 and d._name not in top_codes:
                self.close(d)
            elif pos == 0 and d._name in top_codes:
                # 等权分配
                target_value = self.broker.getvalue() * (1.0 / self.p.top_n)
                price = d.close[0]
                size = target_value / price
                if size > 0:
                    self.buy(d, size=size)


# ── 策略 B: 多空中性 ──
class MomentumRotationLongShort(bt.Strategy):
    params = (
        ('momentum_window', MOMENTUM_WINDOW),
        ('rebalance_freq', REBALANCE_FREQ),
        ('top_n', TOP_N),
        ('benchmark_code', BENCHMARK),
    )

    def __init__(self):
        self.momentums = {}
        for d in self.datas:
            self.momentums[d._name] = MomentumIndicator(d, period=self.p.momentum_window)
        self.day_count = 0

    def next(self):
        self.day_count += 1
        if self.day_count % self.p.rebalance_freq != 0:
            return

        scores = []
        for d in self.datas:
            if d._name == self.p.benchmark_code or len(d) < self.p.momentum_window + 1:
                continue
            mom = self.momentums[d._name][0]
            if not np.isnan(mom):
                scores.append((d, mom))

        if len(scores) < self.p.top_n * 2:
            return

        scores.sort(key=lambda x: x[1], reverse=True)
        top_codes = [s[0]._name for s in scores[:self.p.top_n]]
        bottom_codes = [s[0]._name for s in scores[-self.p.top_n:]]

        value = self.broker.getvalue()
        per_slot = value / (self.p.top_n * 2)

        for d in self.datas:
            if d._name == self.p.benchmark_code:
                continue
            price = d.close[0]
            if price <= 0:
                continue

            if d._name in top_codes:
                size = per_slot / price
                if size > 0:
                    self.buy(d, size=size)
            elif d._name in bottom_codes:
                size = per_slot / price
                if size > 0:
                    self.sell(d, size=size)
            else:
                self.close(d)


# ── 回测执行器 ──
class HKBacktest:
    """港股动量轮动回测"""

    def __init__(self, start_date: str = "2020-01-01", end_date: str = "2026-05-14",
                 asset_pool: Optional[List[str]] = None):
        self.data = get_data()
        self.start_date = start_date
        self.end_date = end_date
        self.asset_pool = asset_pool or ASSET_POOL

    def _prepare_feeds(self) -> List[bt.feeds.PandasData]:
        """准备 backtrader 数据源"""
        feeds = []
        for code in self.asset_pool:
            try:
                df = self.data.get_daily(code)
                df = df[(df["日期"] >= self.start_date) & (df["日期"] <= self.end_date)]
                if len(df) < 100:
                    print(f"  ⚠️ {code} 数据不足 ({len(df)}天), 跳过")
                    continue
                df = df.set_index("日期")
                df["openinterest"] = 0
                feed = bt.feeds.PandasData(
                    dataname=df,
                    open="开盘", high="最高", low="最低",
                    close="收盘", volume="成交量",
                    openinterest="openinterest",
                    name=code,
                )
                feeds.append(feed)
            except Exception as e:
                print(f"  ❌ {code} 加载失败: {e}")

        return feeds

    def run(self, strategy_class, **kwargs) -> dict:
        """执行回测"""
        cerebro = bt.Cerebro()
        cerebro.addstrategy(strategy_class, **kwargs)
        cerebro.broker.setcash(INITIAL_CASH)
        cerebro.broker.setcommission(commission=COMMISSION)
        cerebro.broker.set_slippage_perc(SLIPPAGE)

        feeds = self._prepare_feeds()
        for f in feeds:
            cerebro.adddata(f)

        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

        print(f"\n🚀 回测: {strategy_class.__name__}")
        print(f"   期间: {self.start_date} → {self.end_date}")
        print(f"   资产池: {len(feeds)} 只 | 初始资金: HK${INITIAL_CASH:,.0f}")

        start_value = cerebro.broker.getvalue()
        results = cerebro.run()
        end_value = cerebro.broker.getvalue()

        strat = results[0]
        total_return = (end_value / start_value - 1) * 100
        sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0) or 0
        dd = strat.analyzers.drawdown.get_analysis()
        max_dd = dd.get('max', {}).get('drawdown', 0) if dd.get('max') else 0
        trade_analysis = strat.analyzers.trades.get_analysis()

        result = {
            "strategy": strategy_class.__name__,
            "start_value": start_value,
            "end_value": end_value,
            "total_return_pct": round(total_return, 2),
            "sharpe": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "total_trades": trade_analysis.get('total', {}).get('total', 0),
            "won_trades": trade_analysis.get('won', {}).get('total', 0),
            "lost_trades": trade_analysis.get('lost', {}).get('total', 0),
        }

        # 胜率
        total_closed = result["won_trades"] + result["lost_trades"]
        result["win_rate_pct"] = round(result["won_trades"] / total_closed * 100, 1) if total_closed > 0 else 0

        print(f"   收益: {result['total_return_pct']:.1f}% | Sharpe: {result['sharpe']:.2f} | "
              f"最大回撤: {result['max_drawdown_pct']:.1f}% | 胜率: {result['win_rate_pct']:.1f}%")
        return result

    def compare(self) -> pd.DataFrame:
        """运行所有策略变体 + 基准对比"""
        results = []

        # A: 纯多头
        print("\n" + "="*60)
        print("🅰️ 纯多头动量轮动")
        results.append(self.run(MomentumRotationLong))

        # B: 多空中性
        print("\n🅱️ 多空中性动量轮动")
        results.append(self.run(MomentumRotationLongShort))

        # C: 基准买入持有 02800
        print("\n📊 基准: 盈富基金(02800) 买入持有")
        bm_df = self.data.get_daily(BENCHMARK)
        bm_df = bm_df[(bm_df["日期"] >= self.start_date) & (bm_df["日期"] <= self.end_date)]
        if len(bm_df) > 0:
            bm_start = bm_df["收盘"].iloc[0]
            bm_end = bm_df["收盘"].iloc[-1]
            bm_return = (bm_end / bm_start - 1) * 100
            bm_ret_series = bm_df["收盘"].pct_change().dropna()
            bm_sharpe = (bm_ret_series.mean() * 252 - 0.02) / (bm_ret_series.std() * np.sqrt(252)) if len(bm_ret_series) > 0 else 0
            bm_cummax = bm_df["收盘"].cummax()
            bm_dd = ((bm_df["收盘"] - bm_cummax) / bm_cummax).min() * 100

            results.append({
                "strategy": "🏛️ 盈富基金(持有)",
                "start_value": INITIAL_CASH,
                "end_value": INITIAL_CASH * (1 + bm_return / 100),
                "total_return_pct": round(bm_return, 2),
                "sharpe": round(bm_sharpe, 2),
                "max_drawdown_pct": round(bm_dd, 2),
                "total_trades": 1, "won_trades": 1, "lost_trades": 0, "win_rate_pct": 100,
            })
            print(f"   收益: {bm_return:.1f}% | Sharpe: {bm_sharpe:.2f} | 最大回撤: {bm_dd:.1f}%")

        df = pd.DataFrame(results)
        return df


# ── CLI ──
if __name__ == "__main__":
    import sys

    # 默认回测 2020-2026
    start = sys.argv[1] if len(sys.argv) > 1 else "2020-01-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2026-05-14"

    bt_runner = HKBacktest(start_date=start, end_date=end)
    results_df = bt_runner.compare()

    print("\n" + "="*60)
    print("📊 策略对比")
    print(results_df.to_string(index=False))

    # 保存
    out = Path(__file__).parent / "reports" / f"hk_backtest_{datetime.now().strftime('%Y%m%d')}.csv"
    out.parent.mkdir(exist_ok=True)
    results_df.to_csv(out, index=False)
    print(f"\n📄 结果已保存: {out}")
