"""
Walk-Forward Validator v1.0 — 滚动验证框架
===========================================
Chase量化策略 Phase 8: 滚动窗口优化+验证, 真实OOS评估

核心理念:
  模拟真实交易场景 — 用过去数据训练/优化, 未来数据验证
  每窗口: Train集上Grid Search最优参数 → Test集上纯OOS回测
  5个滚动窗口 → 稳定性评估 → 过拟合检测

使用方式:
  python3 walk_forward_validator.py               # 完整W-F验证
  python3 walk_forward_validator.py --n-windows 3  # 3窗口
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings("ignore")

from feature_ts import FeatureFactoryV4
from ml_signal_v4 import MLSignalEngineV4
from strategy_backtest import BacktestResult, INITIAL_CAPITAL
from hyperparam_optimizer import (
    HyperparamOptimizer, STRATEGY_GRID, DEFAULT_PARAMS, OptimizeResult,
)

DATA_DIR = Path(__file__).parent / "data"
OPT_DIR = DATA_DIR / "optimization_results"
OPT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class WFWindow:
    """单次Walk-Forward窗口结果"""
    window_id: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str
    best_params: dict
    test_sharpe: float
    test_maxdd: float
    test_winrate: float
    test_return: float
    test_annual_return: float
    benchmark_return: float
    n_trades: int
    train_days: int
    test_days: int

    def to_dict(self) -> dict:
        return {
            "window_id": self.window_id,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "best_params": self.best_params,
            "test_sharpe": round(self.test_sharpe, 4),
            "test_maxdd": round(self.test_maxdd, 2),
            "test_winrate": round(self.test_winrate, 1),
            "test_return": round(self.test_return, 2),
            "test_annual_return": round(self.test_annual_return, 2),
            "benchmark_return": round(self.benchmark_return, 2),
            "n_trades": self.n_trades,
            "train_days": self.train_days,
            "test_days": self.test_days,
        }


class WalkForwardValidator:
    """
    Walk-Forward滚动验证器

    使用方式:
        wf = WalkForwardValidator(df, feature_matrix, feature_ids)
        windows = wf.run(n_windows=5)
        print(wf.report())
    """

    def __init__(self, df: pd.DataFrame, feature_matrix: pd.DataFrame,
                 feature_ids: List[str],
                 train_days: int = 200, val_days: int = 30, test_days: int = 30):
        self.df = df
        self.feature_matrix = feature_matrix
        self.feature_ids = feature_ids
        self.train_days = train_days
        self.val_days = val_days
        self.test_days = test_days
        self.n_total = len(df)
        self.windows: List[WFWindow] = []

        if "date" in df.columns:
            self._dates = [
                pd.Timestamp(df["date"].values[i]).strftime("%Y-%m-%d")
                for i in range(self.n_total)
            ]
        else:
            self._dates = [str(i) for i in range(self.n_total)]

    def run(self, n_windows: int = 5, verbose: bool = True) -> List[WFWindow]:
        """
        运行Walk-Forward滚动验证。

        Args:
            n_windows: 窗口数 (默认5)
        Returns:
            List[WFWindow]
        """
        self.windows = []

        # 计算步长
        total_days = self.n_total
        required = self.train_days + self.val_days + self.test_days
        max_windows = (total_days - self.train_days) // (self.val_days + self.test_days)
        n_windows = min(n_windows, max_windows)

        if n_windows < 1:
            print(f"❌ 数据不足: 需要{required}天, 实际{total_days}天")
            return []

        step = self.test_days  # 每窗口前进步长

        if verbose:
            print(f"📊 Walk-Forward 滚动验证")
            print(f"   总数据: {total_days}天 | 训练: {self.train_days}d | "
                  f"验证: {self.val_days}d | 测试: {self.test_days}d")
            print(f"   窗口数: {n_windows} | 步长: {step}d")
            print()

        for w in range(n_windows):
            train_start_idx = w * step
            train_end_idx = train_start_idx + self.train_days
            val_start_idx = train_end_idx
            val_end_idx = val_start_idx + self.val_days
            test_start_idx = val_end_idx
            test_end_idx = min(test_start_idx + self.test_days, total_days)

            if test_end_idx > total_days:
                break
            if train_end_idx + self.val_days + 10 > total_days:
                break

            if verbose:
                print(f"🪟 窗口 {w+1}/{n_windows}: "
                      f"Train[{self._dates[train_start_idx]}→{self._dates[train_end_idx-1]}] "
                      f"Test[{self._dates[test_start_idx]}→{self._dates[test_end_idx-1]}]...",
                      end=" ")

            # 在该窗口上进行Grid Search
            train_df = self.df.iloc[train_start_idx:train_end_idx].copy()
            train_fm = self.feature_matrix.iloc[train_start_idx:train_end_idx].copy()

            # 确保数据充足
            if len(train_df) < self.train_days - 10:
                if verbose:
                    print("训练数据不足, 跳过")
                continue

            # 创建该窗口的优化器
            opt = HyperparamOptimizer(train_df, train_fm, self.feature_ids)

            # Grid Search (静默模式)
            try:
                result = opt.optimize_strategy(grid=STRATEGY_GRID, verbose=False)
                best_params = result.best_params
            except Exception as e:
                if verbose:
                    print(f"优化失败: {e}")
                best_params = DEFAULT_PARAMS.copy()

            # 在Test集上用最优参数回测 (纯OOS)
            test_df = self.df.iloc[test_start_idx:test_end_idx].copy()
            test_fm = self.feature_matrix.iloc[test_start_idx:test_end_idx].copy()

            # 需要足够的warmup天数 — 用 best_params 中的 warmup_days
            warmup = best_params.get("warmup_days", 150)
            # Test set需要warmup, 从Train末尾延伸
            extended_start = max(0, test_start_idx - warmup)
            extended_df = self.df.iloc[extended_start:test_end_idx].copy()
            extended_fm = self.feature_matrix.iloc[extended_start:test_end_idx].copy()

            test_opt = HyperparamOptimizer(extended_df, extended_fm, self.feature_ids)
            bt_result = test_opt._run_backtest_with_params(**best_params, verbose=False)

            # 记录
            wf_win = WFWindow(
                window_id=w + 1,
                train_start=self._dates[train_start_idx],
                train_end=self._dates[train_end_idx - 1],
                val_start=self._dates[val_start_idx],
                val_end=self._dates[min(val_end_idx, total_days) - 1],
                test_start=self._dates[test_start_idx],
                test_end=self._dates[test_end_idx - 1],
                best_params=best_params,
                test_sharpe=bt_result.sharpe_ratio,
                test_maxdd=bt_result.max_drawdown_pct,
                test_winrate=bt_result.win_rate_pct,
                test_return=bt_result.total_return_pct,
                test_annual_return=bt_result.annual_return_pct,
                benchmark_return=bt_result.benchmark_return_pct,
                n_trades=bt_result.n_trades,
                train_days=len(train_df),
                test_days=len(test_df),
            )
            self.windows.append(wf_win)

            if verbose:
                emoji = "🟢" if wf_win.test_sharpe > 1 else "🟡" if wf_win.test_sharpe > 0 else "🔴"
                print(f"{emoji} Sharpe={wf_win.test_sharpe:.3f} | "
                      f"MaxDD=-{wf_win.test_maxdd:.1f}% | "
                      f"Return={wf_win.test_return:+.1f}% vs BnH={wf_win.benchmark_return:+.1f}% | "
                      f"{wf_win.n_trades}笔")

        if verbose and self.windows:
            print(f"\n✅ W-F验证完成: {len(self.windows)}个窗口")

        # 保存结果
        self._save_results()
        return self.windows

    def stability_score(self) -> float:
        """
        参数稳定性评分: 各窗口最优参数的一致性。

        低方差 → 参数稳定 → 策略可推广
        高方差 → 过拟合 → 参数在历史数据上过拟合了
        """
        if len(self.windows) < 2:
            return 0.0

        # 收集连续参数
        param_keys = ["entry_threshold", "stop_loss", "take_profit", "max_position"]
        var_scores = []
        for key in param_keys:
            vals = [w.best_params.get(key, DEFAULT_PARAMS[key]) for w in self.windows]
            if len(vals) >= 2:
                mean_v = np.mean(vals)
                if abs(mean_v) > 1e-9:
                    cv = np.std(vals) / abs(mean_v)
                    var_scores.append(cv)

        if not var_scores:
            return 0.0

        # 平均变异系数 → 稳定性得分 (越低越好, 映射到0-100)
        avg_cv = np.mean(var_scores)
        stability = max(0, 100 - avg_cv * 100)
        return float(stability)

    def report(self) -> str:
        """生成Markdown Walk-Forward报告"""
        if not self.windows:
            return "⚠️ 尚未运行W-F验证"

        lines = [
            "=" * 60,
            "📊 Walk-Forward 滚动验证报告",
            "=" * 60,
            "",
            f"🪟 窗口数: {len(self.windows)} | "
            f"训练: {self.train_days}d | 验证: {self.val_days}d | 测试: {self.test_days}d",
            f"",
        ]

        # 指标汇总
        sharpes = [w.test_sharpe for w in self.windows]
        maxdds = [w.test_maxdd for w in self.windows]
        returns = [w.test_return for w in self.windows]
        benchmarks = [w.benchmark_return for w in self.windows]

        lines += [
            "## 📊 OOS指标汇总",
            "",
            "| 指标 | 均值 | 中位 | 最小 | 最大 | 标准差 | 评价 |",
            "|------|:---:|:---:|:---:|:---:|:---:|------|",
        ]

        # Sharpe
        avg_s = np.mean(sharpes)
        grade_s = "🏆 优秀" if avg_s > 2 else "✅ 良好" if avg_s > 1 else "⚠️ 一般" if avg_s > 0 else "❌ 负值"
        lines.append(
            f"| Sharpe | {avg_s:.3f} | {np.median(sharpes):.3f} | "
            f"{min(sharpes):.3f} | {max(sharpes):.3f} | {np.std(sharpes):.3f} | {grade_s} |"
        )

        avg_dd = np.mean(maxdds)
        lines.append(
            f"| 最大回撤 | -{avg_dd:.1f}% | -{np.median(maxdds):.1f}% | "
            f"-{max(maxdds):.1f}% | -{min(maxdds):.1f}% | {np.std(maxdds):.1f}% | "
            f"{'🛡️ 可控' if avg_dd < 15 else '⚠️ 偏高'} |"
        )

        avg_ret = np.mean(returns)
        avg_bm = np.mean(benchmarks)
        lines.append(
            f"| 策略收益 | {avg_ret:+.1f}% | {np.median(returns):+.1f}% | "
            f"{min(returns):+.1f}% | {max(returns):+.1f}% | {np.std(returns):.1f}% | "
            f"{'🟢 跑赢' if avg_ret > avg_bm else '🔴 跑输'} |"
        )
        lines.append(f"| 基准收益 | {avg_bm:+.1f}% | — | — | — | — | BTC买入持有 |")

        # 稳定性
        stability = self.stability_score()
        stability_grade = "🏆 高" if stability > 80 else "✅ 中" if stability > 50 else "⚠️ 低"
        lines += [
            "",
            f"## 🔒 参数稳定性: {stability:.0f}/100 ({stability_grade})",
            "",
            "> 低方差 → 参数跨周期稳定 → 策略可推广  \n"
            "> 高方差 → 过拟合特定时段 → 实盘风险高",
            "",
        ]

        # 各窗口详情
        lines += [
            "## 🪟 各窗口详情",
            "",
            "| # | Train | Test | Sharpe | MaxDD | 策略收益 | 基准收益 | #Trades |",
            "|---|--------|------|:---:|:---:|:---:|:---:|:---:|",
        ]
        for w in self.windows:
            emoji = "🟢" if w.test_sharpe > 1 else "🟡" if w.test_sharpe > 0 else "🔴"
            lines.append(
                f"| {w.window_id} | {w.train_start}→{w.train_end} | "
                f"{w.test_start}→{w.test_end} | "
                f"{emoji} {w.test_sharpe:.3f} | -{w.test_maxdd:.1f}% | "
                f"{w.test_return:+.1f}% | {w.benchmark_return:+.1f}% | {w.n_trades} |"
            )

        # 参数稳定性详情
        lines += [
            "",
            "## 📐 各窗口最优参数",
            "",
            "| # | Entry | StopLoss | TakeProfit | MaxPos | Warmup |",
            "|---|:---:|:---:|:---:|:---:|:---:|",
        ]
        for w in self.windows:
            p = w.best_params
            lines.append(
                f"| {w.window_id} | {p.get('entry_threshold', '-'):.2f} | "
                f"{p.get('stop_loss', '-'):.2f} | {p.get('take_profit', '-'):.2f} | "
                f"{p.get('max_position', '-'):.2f} | {int(p.get('warmup_days', 0))} |"
            )

        # 结论
        all_positive = all(s > 0 for s in sharpes)
        lines += [
            "",
            "## 🎯 结论",
            "",
        ]
        if all_positive and stability > 60:
            lines.append("✅ **策略通过W-F验证**: 所有OOS窗口Sharpe>0 + 参数稳定 → 实盘可行")
        elif all_positive:
            lines.append("⚠️ **策略基本通过**: OOS Sharpe均>0但参数不够稳定 → 建议简化参数或增加训练数据")
        else:
            lines.append("❌ **策略未通过**: 部分窗口OOS Sharpe≤0 → 需重新审视信号逻辑")

        lines.append("")
        lines.append(f"📐 稳定性: {stability:.0f}/100 | "
                     f"OOS Sharpe均值: {avg_s:.3f} | "
                     f"OOS全正窗口: {sum(1 for s in sharpes if s > 0)}/{len(sharpes)}")

        return "\n".join(lines)

    def plot_wf_heatmap(self):
        """绘制 Walk-Forward 窗口×指标 热力图"""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        if not self.windows:
            return None

        metrics = ["Sharpe", "MaxDD", "WinRate", "Strategy%", "Bench%", "#Trades"]
        data = []
        for w in self.windows:
            data.append([
                w.test_sharpe,
                -w.test_maxdd,
                w.test_winrate,
                w.test_return,
                w.benchmark_return,
                w.n_trades,
            ])

        data = np.array(data)
        # 归一化 (每列独立, 用于颜色映射)
        data_norm = np.zeros_like(data)
        for j in range(data.shape[1]):
            col = data[:, j]
            min_v, max_v = col.min(), col.max()
            if max_v > min_v:
                data_norm[:, j] = (col - min_v) / (max_v - min_v)
            else:
                data_norm[:, j] = 0.5

        fig = go.Figure(data=go.Heatmap(
            z=data_norm,
            x=metrics,
            y=[f"W{w.window_id}: {w.test_start}→{w.test_end}" for w in self.windows],
            text=[[f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}" for v in row] for row in data],
            texttemplate="%{text}",
            textfont={"size": 12},
            colorscale="RdYlGn",
        ))

        fig.update_layout(
            title="📊 Walk-Forward OOS表现热力图 (绿=好, 红=差)",
            xaxis_title="指标",
            yaxis_title="窗口",
            template="plotly_dark",
            height=350,
        )

        return fig

    def _save_results(self):
        """保存W-F结果到JSON"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "train_days": self.train_days,
                "val_days": self.val_days,
                "test_days": self.test_days,
                "n_windows": len(self.windows),
            },
            "stability_score": self.stability_score(),
            "summary": {
                "avg_sharpe": round(float(np.mean([w.test_sharpe for w in self.windows])), 4),
                "avg_maxdd": round(float(np.mean([w.test_maxdd for w in self.windows])), 2),
                "avg_return": round(float(np.mean([w.test_return for w in self.windows])), 2),
                "all_positive": all(w.test_sharpe > 0 for w in self.windows),
            },
            "windows": [w.to_dict() for w in self.windows],
        }
        path = OPT_DIR / "walk_forward_results.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"💾 W-F结果已保存: {path}")


# ═══════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chase量化策略 · Walk-Forward验证")
    parser.add_argument("--n-windows", type=int, default=5,
                       help="窗口数 (默认5)")
    parser.add_argument("--train-days", type=int, default=200,
                       help="训练窗口天数 (默认200)")
    parser.add_argument("--val-days", type=int, default=30,
                       help="验证窗口天数 (默认30)")
    parser.add_argument("--test-days", type=int, default=30,
                       help="测试窗口天数 (默认30)")
    args = parser.parse_args()

    print("=" * 60)
    print("📊 Phase 8: Walk-Forward 滚动验证")
    print("=" * 60)

    # 加载数据
    print("\n📊 Step 1: 加载数据...")
    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True, "timeout": 15000})
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1d", limit=400)
        df = pd.DataFrame(ohlcv, columns=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"], unit="ms")
        print(f"   BTC/USDT: {len(df)}天 ({df['date'].values[0]} → {df['date'].values[-1]})")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import sys
        sys.exit(1)

    print("🔧 Step 2: 预计算特征矩阵...")
    engine = MLSignalEngineV4()
    try:
        df = engine._enrich_cross_market(df)
    except Exception:
        pass
    feature_matrix = engine.factory.compute_timeseries(df, verbose=True)
    feature_ids = [f.id for f in engine.factory.features]
    print(f"   特征矩阵: {feature_matrix.shape[0]}行 × {len(feature_ids)}列")

    # 运行W-F
    wf = WalkForwardValidator(
        df, feature_matrix, feature_ids,
        train_days=args.train_days,
        val_days=args.val_days,
        test_days=args.test_days,
    )
    windows = wf.run(n_windows=args.n_windows)

    if windows:
        print()
        print(wf.report())

        # 保存图表
        try:
            fig = wf.plot_wf_heatmap()
            if fig:
                fig.write_html(OPT_DIR / "walk_forward_heatmap.html")
                print("\n💾 W-F热力图已保存: data/optimization_results/walk_forward_heatmap.html")
        except Exception as e:
            print(f"⚠️ 图表失败: {e}")

    print("\n🎉 W-F验证完成!")
