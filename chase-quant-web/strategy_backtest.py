"""
Strategy Backtester v1.0 — ML策略回测引擎
==========================================
Chase量化策略 Phase 7: Walk-forward回测验证7主题ML策略

核心理念:
  逐日遍历历史数据, 每根bar只用当前已知信息生成信号, 严格避免前视偏差。

  前200天 warm-up → 后200天逐日:
    1. 提取截至当日的特征
    2. 构建7个子信号 (线性IC加权)
    3. LightGBM预测 (可选)
    4. 交易逻辑: signal>0.5→买, signal<-0.5→卖, 持仓signal<0→平仓
    5. 记录权益/回撤/交易

输出:
  - 回测报告 (Sharpe, 最大回撤, 胜率, 盈亏比, Calmar)
  - 权益曲线 vs 买入持有基准
  - 交易明细
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings("ignore")

from feature_ts import FeatureFactoryV4, _ret
from ml_signal_v4 import MLSignalEngineV4, SubSignalBuilder, EnsembleSignal


# ── 回测参数 ──
INITIAL_CAPITAL = 10_000.0       # 初始资金 (RMB)
FEE_RATE = 0.001                 # 手续费 0.1%
SLIPPAGE = 0.0005                # 滑点 0.05%
MAX_POSITION_SIZE = 0.40         # 单仓位 ≤ 40%
STOP_LOSS_PCT = -0.08            # 硬止损 -8%
TAKE_PROFIT_PCT = 0.15           # 止盈 +15%
WARMUP_DAYS = 200                # Warm-up期
MIN_SCORE_FOR_ENTRY = 0.5        # 最低入场信号值


@dataclass
class BacktestTrade:
    """单笔交易记录"""
    entry_idx: int
    entry_time: str
    entry_price: float
    exit_idx: int
    exit_time: str
    exit_price: float
    size_pct: float              # 仓位比例
    pnl: float
    pnl_pct: float
    holding_days: int
    exit_reason: str             # signal_exit / stop_loss / take_profit / end_of_test
    entry_signal: float
    entry_consensus: float


@dataclass
class BacktestResult:
    """回测结果"""
    symbol: str
    start_date: str
    end_date: str
    n_days: int
    n_trades: int

    # 收益指标
    total_return_pct: float
    annual_return_pct: float
    benchmark_return_pct: float   # 买入持有
    alpha_pct: float              # 超额收益

    # 风险指标
    annual_volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    max_drawdown_days: int
    calmar_ratio: float

    # 交易统计
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    avg_holding_days: float

    # 信号统计
    signal_accuracy: float        # 信号方向正确率

    # 详细数据
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: Optional[pd.Series] = None
    benchmark_curve: Optional[pd.Series] = None
    signal_history: Optional[pd.DataFrame] = None
    monthly_returns: Optional[pd.Series] = None

    def to_dict(self) -> dict:
        """序列化 (不含trades和curves)"""
        return {
            "symbol": self.symbol,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "n_days": self.n_days,
            "n_trades": self.n_trades,
            "total_return_pct": round(self.total_return_pct, 2),
            "annual_return_pct": round(self.annual_return_pct, 2),
            "benchmark_return_pct": round(self.benchmark_return_pct, 2),
            "alpha_pct": round(self.alpha_pct, 2),
            "annual_volatility_pct": round(self.annual_volatility_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "sortino_ratio": round(self.sortino_ratio, 3),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "max_drawdown_days": self.max_drawdown_days,
            "calmar_ratio": round(self.calmar_ratio, 3),
            "win_rate_pct": round(self.win_rate_pct, 1),
            "avg_win_pct": round(self.avg_win_pct, 2),
            "avg_loss_pct": round(self.avg_loss_pct, 2),
            "profit_factor": round(self.profit_factor, 2),
            "avg_holding_days": round(self.avg_holding_days, 1),
            "signal_accuracy": round(self.signal_accuracy, 2),
        }

    def save(self, path: str = None):
        """保存回测结果"""
        if path is None:
            path = Path(__file__).parent / "data" / "backtest_results" / "latest.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        # 保存trades
        data["trades"] = [
            {k: v for k, v in t.__dict__.items()}
            for t in self.trades
        ]
        # 保存curves
        if self.equity_curve is not None:
            data["equity_curve"] = self.equity_curve.to_dict()
        if self.benchmark_curve is not None:
            data["benchmark_curve"] = self.benchmark_curve.to_dict()
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)


class StrategyBacktester:
    """
    ML策略 Walk-Forward 回测器

    使用方式:
        backtester = StrategyBacktester(use_lgbm=True)
        result = backtester.run(df_ohlcv)
        print(backtester.report())
    """

    def __init__(self, use_lgbm: bool = True, use_cross_market: bool = True):
        self.use_lgbm = use_lgbm
        self.use_cross_market = use_cross_market
        self.engine = None
        self.result: Optional[BacktestResult] = None

    def run(self, df: pd.DataFrame, symbol: str = "BTC/USDT") -> BacktestResult:
        """
        运行Walk-forward回测

        Args:
            df: OHLCV DataFrame (date, open, high, low, close, volume)
            symbol: 交易对

        Returns:
            BacktestResult with all metrics
        """
        df = df.copy()
        df = df.reset_index(drop=True)
        n_bars = len(df)

        # 保存原始日期 (在特征计算前, 防止date列被修改)
        if "date" in df.columns:
            _dates = [self._fmt_date(df["date"].values[i]) for i in range(n_bars)]
        else:
            _dates = [str(i) for i in range(n_bars)]

        if n_bars < WARMUP_DAYS + 20:
            raise ValueError(f"数据不足: 需要至少{WARMUP_DAYS + 20}根bar, 实际{n_bars}")

        print(f"📊 初始化ML引擎...")
        self.engine = MLSignalEngineV4()

        # Step 1: 预计算全量特征时序 (滚动窗口保证无前视偏差)
        print(f"🔧 计算特征矩阵 ({n_bars}天 × 286特征)...")
        try:
            if self.use_cross_market:
                df = self.engine._enrich_cross_market(df)
            feature_matrix = self.engine.factory.compute_timeseries(df)
            feature_ids = [f.id for f in self.engine.factory.features]
        except Exception as e:
            print(f"⚠️ 特征计算失败: {e}, 尝试降级模式...")
            feature_matrix = pd.DataFrame()
            feature_ids = []

        if feature_matrix.empty or len(feature_ids) == 0:
            print("❌ 特征计算失败, 无法回测")
            return self._empty_result(symbol, df)

        # 对齐索引
        feature_matrix = feature_matrix.reset_index(drop=True)
        if len(feature_matrix) != n_bars:
            # 截断到较短者
            min_len = min(len(feature_matrix), n_bars)
            feature_matrix = feature_matrix.iloc[:min_len]
            df = df.iloc[:min_len]
            n_bars = min_len

        # Step 2: Walk-forward 回测
        print(f"🏃 Walk-forward 回测 (t={WARMUP_DAYS}..{n_bars-1})...")

        capital = INITIAL_CAPITAL
        position = 0.0            # 持仓数量 (BTC)
        entry_price = 0.0
        entry_idx = 0
        entry_signal = 0.0
        entry_consensus = 0.0

        equity = np.full(n_bars, INITIAL_CAPITAL)
        trades: List[BacktestTrade] = []
        signal_log = []
        drawdowns = []
        peak_equity = INITIAL_CAPITAL

        total_bars = n_bars - WARMUP_DAYS
        report_every = max(1, total_bars // 10)

        for t in range(WARMUP_DAYS, n_bars):
            if (t - WARMUP_DAYS) % report_every == 0:
                pct = (t - WARMUP_DAYS) / total_bars * 100
                print(f"  ... {pct:.0f}% (day {t}/{n_bars})", end="\r")

            price = float(df["close"].values[t])

            # 提取截至t的特征行
            feat_row = {}
            for fid in feature_ids:
                if fid in feature_matrix.columns:
                    val = feature_matrix[fid].values[t]
                    if not np.isnan(val) and not np.isinf(val):
                        feat_row[fid] = float(val)
                    else:
                        feat_row[fid] = 0.0
                else:
                    feat_row[fid] = 0.0

            if not feat_row:
                continue

            # 构建子信号 + 生成信号
            try:
                sub_signals = self.engine.builder.build(feat_row)
            except Exception:
                continue

            active_sigs = [s for s in sub_signals if s.confidence > 0.1]

            # IC加权信号
            ic_weights = {}
            for s in sub_signals:
                feat_icirs = []
                for fid in s.contributing_features:
                    r = self.engine.builder.results.get(fid)
                    feat_icirs.append(abs(r.icir) if r else 0.5)
                ic_weights[s.id] = np.mean(feat_icirs) if feat_icirs else 0.5

            if active_sigs:
                ic_w_sum = sum(ic_weights.get(s.id, 0.5) * s.value for s in active_sigs)
                ic_w_total = sum(ic_weights.get(s.id, 0.5) for s in active_sigs)
                signal_val = ic_w_sum / (ic_w_total + 1e-9)
            else:
                signal_val = 0.0

            # LightGBM预测 (如果有)
            lgbm_val = 0.0
            if self.use_lgbm and self.engine._lgbm_loaded:
                lgbm_preds = self.engine._lgbm_predictor.predict(feat_row)
                if lgbm_preds:
                    lgbm_vals = []
                    lgbm_ws = []
                    for s in sub_signals:
                        pred = lgbm_preds.get(s.id, 0.0)
                        lgbm_vals.append(float(np.tanh(pred * 20)))
                        lgbm_ws.append(ic_weights.get(s.id, 0.5))
                    if lgbm_ws and sum(lgbm_ws) > 0:
                        lgbm_val = sum(v * w for v, w in zip(lgbm_vals, lgbm_ws)) / sum(lgbm_ws)

            final_signal = lgbm_val if (self.use_lgbm and self.engine._lgbm_loaded and lgbm_val != 0) else signal_val

            # 活跃子信号数
            long_count = sum(1 for s in sub_signals if s.direction == "LONG")
            short_count = sum(1 for s in sub_signals if s.direction == "SHORT")
            active_count = long_count + short_count

            # 记录信号
            signal_log.append({
                "idx": t, "date": _dates[t],
                "signal": final_signal, "signal_ic": signal_val, "signal_lgbm": lgbm_val,
                "active_count": active_count, "price": price,
            })

            # ── 交易逻辑 ──
            in_position = position > 1e-9

            if not in_position:
                # 空仓 → 寻找入场
                if final_signal > MIN_SCORE_FOR_ENTRY and active_count >= 2:
                    size_pct = min(MAX_POSITION_SIZE, 0.05 + abs(final_signal) * 0.05)
                    size_amount = capital * size_pct * (1 - FEE_RATE - SLIPPAGE)
                    position = size_amount / price
                    entry_price = price
                    entry_idx = t
                    entry_signal = final_signal
                    entry_consensus = max(long_count, short_count) / max(1, len(sub_signals))
                    capital -= size_amount  # 扣除投入资金

            else:
                # 持仓 → 检查出场条件
                pnl_pct = (price - entry_price) / entry_price
                exit_now = False
                exit_reason = "signal_exit"

                # 硬止损
                if pnl_pct <= STOP_LOSS_PCT:
                    exit_now = True
                    exit_reason = "stop_loss"
                # 止盈
                elif pnl_pct >= TAKE_PROFIT_PCT:
                    exit_now = True
                    exit_reason = "take_profit"
                # 信号反转
                elif final_signal < 0:
                    exit_now = True
                    exit_reason = "signal_exit"

                if exit_now:
                    exit_value = position * price * (1 - FEE_RATE - SLIPPAGE)
                    capital += exit_value
                    trade_pnl = exit_value - (position * entry_price)
                    trade_pnl_pct = trade_pnl / (position * entry_price) * 100

                    trades.append(BacktestTrade(
                        entry_idx=entry_idx,
                        entry_time=_dates[entry_idx],
                        entry_price=entry_price,
                        exit_idx=t,
                        exit_time=_dates[t],
                        exit_price=price,
                        size_pct=position * price / equity[entry_idx] if equity[entry_idx] > 0 else 0,
                        pnl=trade_pnl,
                        pnl_pct=trade_pnl_pct,
                        holding_days=t - entry_idx,
                        exit_reason=exit_reason,
                        entry_signal=entry_signal,
                        entry_consensus=entry_consensus,
                    ))
                    position = 0.0
                    entry_price = 0.0

            # 权益 = 现金 + 持仓市值
            equity[t] = capital + position * price * (1 - FEE_RATE - SLIPPAGE)

            # 追踪回撤
            if equity[t] > peak_equity:
                peak_equity = equity[t]
            dd = (peak_equity - equity[t]) / peak_equity * 100
            drawdowns.append(dd)

        # 最后一天强制平仓
        if position > 1e-9:
            final_price = float(df["close"].values[-1])
            exit_value = position * final_price * (1 - FEE_RATE - SLIPPAGE)
            capital += exit_value
            trade_pnl = exit_value - (position * entry_price)
            trade_pnl_pct = trade_pnl / (position * entry_price) * 100
            trades.append(BacktestTrade(
                entry_idx=entry_idx,
                entry_time=_dates[entry_idx],
                entry_price=entry_price,
                exit_idx=n_bars - 1,
                exit_time=_dates[-1],
                exit_price=final_price,
                size_pct=position * final_price / equity[entry_idx] if equity[entry_idx] > 0 else 0,
                pnl=trade_pnl,
                pnl_pct=trade_pnl_pct,
                holding_days=n_bars - 1 - entry_idx,
                exit_reason="end_of_test",
                entry_signal=entry_signal,
                entry_consensus=entry_consensus,
            ))
            equity[-1] = capital
            position = 0.0

        print(f"\n✅ 回测完成: {n_bars - WARMUP_DAYS}天, {len(trades)}笔交易")

        # ── 计算指标 ──
        result = self._compute_metrics(
            df, equity, trades, signal_log, symbol, drawdowns, _dates
        )
        self.result = result
        return result

    @staticmethod
    def _fmt_date(date_val) -> str:
        """安全格式化日期"""
        try:
            ts = pd.Timestamp(date_val)
            return ts.strftime("%Y-%m-%d")
        except Exception:
            return str(date_val)[:10]

    def _compute_metrics(self, df: pd.DataFrame, equity: np.ndarray,
                         trades: List[BacktestTrade], signal_log: list,
                         symbol: str, drawdowns: list, _dates: list) -> BacktestResult:
        """计算所有回测指标"""
        equity_series = pd.Series(equity, index=range(len(equity)))
        n_bars = len(df)

        # 买入持有基准
        bench_start_price = float(df["close"].values[WARMUP_DAYS])
        bench_end_price = float(df["close"].values[-1])
        benchmark_return = (bench_end_price - bench_start_price) / bench_start_price * 100
        benchmark_curve = pd.Series(
            [float(df["close"].values[i]) / bench_start_price * INITIAL_CAPITAL
             for i in range(n_bars)]
        )

        # 总收益
        total_return = (equity[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        annual_factor = 365 / max(1, n_bars - WARMUP_DAYS)
        annual_return = ((equity[-1] / INITIAL_CAPITAL) ** annual_factor - 1) * 100

        # 日收益
        daily_returns = pd.Series([
            (equity[i] - equity[i-1]) / equity[i-1] if equity[i-1] > 0 else 0
            for i in range(WARMUP_DAYS + 1, n_bars)
        ])

        # 波动率
        annual_vol = float(np.std(daily_returns) * np.sqrt(365) * 100) if len(daily_returns) > 0 else 0

        # Sharpe
        avg_daily_ret = float(np.mean(daily_returns)) if len(daily_returns) > 0 else 0
        sharpe = (avg_daily_ret / max(1e-9, float(np.std(daily_returns)))) * np.sqrt(365)

        # Sortino (只 penalize 下行波动)
        neg_returns = daily_returns[daily_returns < 0]
        downside_std = float(np.std(neg_returns)) if len(neg_returns) > 0 else 1e-9
        sortino = (avg_daily_ret / downside_std) * np.sqrt(365)

        # 最大回撤
        max_dd = max(drawdowns) if drawdowns else 0
        # 回撤持续期
        dd_start = 0
        max_dd_days = 0
        current_dd_days = 0
        peak = equity[WARMUP_DAYS]
        for i in range(WARMUP_DAYS, n_bars):
            if equity[i] >= peak:
                peak = equity[i]
                max_dd_days = max(max_dd_days, current_dd_days)
                current_dd_days = 0
            else:
                current_dd_days += 1
        max_dd_days = max(max_dd_days, current_dd_days)

        # Calmar
        calmar = annual_return / max(max_dd, 1e-9)

        # 交易统计
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        win_rate = len(winning_trades) / max(1, len(trades)) * 100
        avg_win = np.mean([t.pnl_pct for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl_pct for t in losing_trades]) if losing_trades else 0
        total_profits = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_profits / max(1e-9, total_losses) if total_losses > 0 else 999.0 if total_profits > 0 else 0.0
        avg_hold = np.mean([t.holding_days for t in trades]) if trades else 0

        # 信号方向准确率
        if signal_log:
            correct_signals = 0
            for i, sl in enumerate(signal_log):
                if i < len(signal_log) - 1:
                    future_ret = (signal_log[i+1]["price"] - sl["price"]) / sl["price"]
                    if (sl["signal"] > 0 and future_ret > 0) or (sl["signal"] < 0 and future_ret < 0):
                        correct_signals += 1
            signal_acc = correct_signals / max(1, len(signal_log) - 1)
        else:
            signal_acc = 0

        # 月收益
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"].values)
            monthly = pd.Series(daily_returns.values,
                              index=dates[WARMUP_DAYS + 1:WARMUP_DAYS + 1 + len(daily_returns)])
            monthly_returns = monthly.resample("ME").apply(
                lambda x: np.prod(1 + x) - 1
            ) * 100
        else:
            monthly_returns = None

        return BacktestResult(
            symbol=symbol,
            start_date=_dates[WARMUP_DAYS],
            end_date=_dates[-1],
            n_days=n_bars - WARMUP_DAYS,
            n_trades=len(trades),
            total_return_pct=total_return,
            annual_return_pct=annual_return,
            benchmark_return_pct=benchmark_return,
            alpha_pct=total_return - benchmark_return,
            annual_volatility_pct=annual_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_dd,
            max_drawdown_days=max_dd_days,
            calmar_ratio=calmar,
            win_rate_pct=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            profit_factor=profit_factor,
            avg_holding_days=avg_hold,
            signal_accuracy=signal_acc,
            trades=trades,
            equity_curve=equity_series,
            benchmark_curve=benchmark_curve,
            signal_history=pd.DataFrame(signal_log) if signal_log else None,
            monthly_returns=monthly_returns,
        )

    def _empty_result(self, symbol: str, df: pd.DataFrame) -> BacktestResult:
        """返回空结果"""
        return BacktestResult(
            symbol=symbol,
            start_date="", end_date="",
            n_days=0, n_trades=0,
            total_return_pct=0, annual_return_pct=0,
            benchmark_return_pct=0, alpha_pct=0,
            annual_volatility_pct=0, sharpe_ratio=0,
            sortino_ratio=0, max_drawdown_pct=0,
            max_drawdown_days=0, calmar_ratio=0,
            win_rate_pct=0, avg_win_pct=0, avg_loss_pct=0,
            profit_factor=0, avg_holding_days=0,
            signal_accuracy=0,
        )

    def report(self) -> str:
        """生成Markdown回测报告"""
        if self.result is None:
            return "⚠️ 尚未运行回测"

        r = self.result
        lines = [
            "=" * 60,
            f"📈 ML策略回测报告 — {r.symbol}",
            "=" * 60,
            f"",
            f"📅 回测期: {r.start_date} → {r.end_date} ({r.n_days}天)",
            f"📊 交易笔数: {r.n_trades}",
            f"",
            f"## 💰 收益指标",
            f"",
            f"| 指标 | 策略 | 买入持有 | 超额 |",
            f"|------|:---:|:---:|:---:|",
            f"| 总收益 | {r.total_return_pct:+.2f}% | {r.benchmark_return_pct:+.2f}% | {r.alpha_pct:+.2f}% |",
            f"| 年化收益 | {r.annual_return_pct:+.2f}% | — | — |",
            f"| 年化波动 | {r.annual_volatility_pct:.1f}% | — | — |",
            f"",
            f"## 🛡️ 风险指标",
            f"",
            f"| 指标 | 数值 | 评价 |",
            f"|------|:---:|------|",
        ]

        # Sharpe评价
        if r.sharpe_ratio > 2:
            sharpe_grade = "🏆 优秀"
        elif r.sharpe_ratio > 1:
            sharpe_grade = "✅ 良好"
        elif r.sharpe_ratio > 0:
            sharpe_grade = "⚠️ 一般"
        else:
            sharpe_grade = "❌ 负值"

        lines.append(f"| Sharpe比率 | {r.sharpe_ratio:.3f} | {sharpe_grade} |")
        lines.append(f"| Sortino比率 | {r.sortino_ratio:.3f} | — |")
        lines.append(f"| 最大回撤 | -{r.max_drawdown_pct:.2f}% | 持续{r.max_drawdown_days}天 |")
        lines.append(f"| Calmar比率 | {r.calmar_ratio:.3f} | — |")

        lines += [
            f"",
            f"## 📋 交易统计",
            f"",
            f"| 指标 | 数值 |",
            f"|------|:---:|",
            f"| 胜率 | {r.win_rate_pct:.1f}% |",
            f"| 平均盈利 | +{r.avg_win_pct:.2f}% |",
            f"| 平均亏损 | {r.avg_loss_pct:.2f}% |",
            f"| 盈亏比 | {r.profit_factor:.2f} |",
            f"| 平均持有时长 | {r.avg_holding_days:.1f}天 |",
            f"| 信号方向准确率 | {r.signal_accuracy:.1%} |",
            f"",
        ]

        # 最近交易
        if r.trades:
            lines.append(f"## 📜 最近10笔交易")
            lines.append(f"")
            lines.append(f"| # | 入场日 | 出场日 | 持有时长 | 盈亏 | 出场原因 |")
            lines.append(f"|---|--------|--------|:---:|:---:|------|")
            for i, t in enumerate(r.trades[-10:]):
                emoji = "🟢" if t.pnl > 0 else "🔴"
                lines.append(
                    f"| {len(r.trades) - 10 + i + 1} | {t.entry_time[:10]} | {t.exit_time[:10]} | "
                    f"{t.holding_days}天 | {emoji} {t.pnl_pct:+.2f}% | {t.exit_reason} |"
                )

        # 月收益
        if r.monthly_returns is not None and len(r.monthly_returns) > 0:
            lines.append(f"")
            lines.append(f"## 📅 月收益分布")
            lines.append(f"")
            for m, ret in r.monthly_returns.items():
                emoji = "🟢" if ret > 0 else "🔴" if ret < 0 else "⚪"
                bar = "█" * min(20, int(abs(ret) / 2))
                lines.append(f"  {str(m)[:7]} {emoji} {ret:+.1f}% {bar}")

        return "\n".join(lines)

    def plot_equity_curve(self):
        """生成权益曲线Plotly图表"""
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("⚠️ plotly 未安装, 跳过图表")
            return None

        if self.result is None or self.result.equity_curve is None:
            return None

        r = self.result
        n = len(r.equity_curve)

        # X轴标签
        if "date" in self.result.signal_history.columns if self.result.signal_history is not None else False:
            dates = self.result.signal_history["date"].values
        else:
            dates = list(range(n))

        fig = go.Figure()

        # 权益曲线
        fig.add_trace(go.Scatter(
            x=dates[-len(r.equity_curve):] if hasattr(r.equity_curve, 'values') else list(range(len(r.equity_curve))),
            y=r.equity_curve.values if hasattr(r.equity_curve, 'values') else r.equity_curve,
            mode="lines",
            name="ML策略",
            line=dict(color="#00ff88", width=2),
        ))

        # 基准
        if r.benchmark_curve is not None:
            fig.add_trace(go.Scatter(
                x=dates[-len(r.benchmark_curve):] if hasattr(r.benchmark_curve, 'values') else list(range(len(r.benchmark_curve))),
                y=r.benchmark_curve.values if hasattr(r.benchmark_curve, 'values') else r.benchmark_curve,
                mode="lines",
                name="买入持有",
                line=dict(color="#888888", width=1, dash="dash"),
            ))

        # 初始资金线
        fig.add_hline(y=INITIAL_CAPITAL, line_dash="dot", line_color="#444444",
                      annotation_text=f"初始 ¥{INITIAL_CAPITAL:,.0f}")

        fig.update_layout(
            title=f"📈 ML策略回测 — {r.symbol}",
            xaxis_title="日期",
            yaxis_title="权益 (RMB)",
            template="plotly_dark",
            height=500,
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        return fig


# ── CLI ──
if __name__ == "__main__":
    print("=" * 60)
    print("📈 Phase 7: ML策略 Walk-Forward 回测")
    print("=" * 60)

    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True, "timeout": 15000})
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1d", limit=400)
        df = pd.DataFrame(ohlcv, columns=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"], unit="ms")
        print(f"📊 BTC/USDT: {len(df)}天 ({df['date'].values[0]} → {df['date'].values[-1]})")

        backtester = StrategyBacktester(use_lgbm=True, use_cross_market=True)
        result = backtester.run(df)

        print()
        print(backtester.report())

        # 保存结果
        result.save()
        print("\n💾 回测结果已保存到 data/backtest_results/latest.json")

        # 尝试生成图表
        try:
            fig = backtester.plot_equity_curve()
            if fig:
                fig.write_html(Path(__file__).parent / "data" / "backtest_results" / "equity_curve.html")
                print("📊 权益曲线已保存到 data/backtest_results/equity_curve.html")
        except Exception as e:
            print(f"⚠️ 图表生成失败: {e}")

    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("   pip install ccxt plotly")
    except Exception as e:
        print(f"❌ 回测失败: {e}")
        import traceback
        traceback.print_exc()
