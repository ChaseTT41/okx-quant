"""
Hyperparameter Optimizer v1.0 — 策略参数优化器
==============================================
Chase量化策略 Phase 8: Optuna + Grid Search 双重优化

核心理念:
  先优化ML模型超参 (Optuna Bayesian) → 再优化交易策略参数 (Grid Search)
  特征矩阵预计算一次, 每次回测只改变参数, 1500次回测只需几秒。

优化流程:
  Step 1: 加载BTC 400日OHLCV + 预计算特征矩阵 (一次性)
  Step 2: Optuna优化LightGBM超参 → 目标: 最大化OOS Rank IC
  Step 3: Grid Search优化交易参数 → 目标: 最大化(Sharpe×2 - |MaxDD|/50 + min(n_trades/10, 1))
  Step 4: 敏感性分析 → 每个参数 ±50% 看 Sharpe/MaxDD 变化

使用方式:
  python3 hyperparam_optimizer.py               # 完整优化
  python3 hyperparam_optimizer.py --skip-lgbm   # 只优化交易参数
  python3 hyperparam_optimizer.py --sensitivity-only  # 只做敏感性分析
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings("ignore")

from feature_ts import FeatureFactoryV4
from ml_signal_v4 import MLSignalEngineV4, SubSignalBuilder, EnsembleSignal
from strategy_backtest import (
    StrategyBacktester, BacktestResult, BacktestTrade,
    INITIAL_CAPITAL, FEE_RATE, SLIPPAGE, MAX_POSITION_SIZE,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT, WARMUP_DAYS, MIN_SCORE_FOR_ENTRY,
)

DATA_DIR = Path(__file__).parent / "data"
OPT_DIR = DATA_DIR / "optimization_results"
OPT_DIR.mkdir(parents=True, exist_ok=True)

# ── 默认参数 (baseline) ──
DEFAULT_PARAMS = {
    "entry_threshold": 0.5,
    "stop_loss": -0.08,
    "take_profit": 0.15,
    "max_position": 0.40,
    "warmup_days": 200,
    "fee_rate": 0.001,
    "slippage": 0.0005,
}

# ── 搜索空间 ──
STRATEGY_GRID = {
    "entry_threshold": [0.3, 0.5, 0.7, 1.0, 1.5],
    "stop_loss": [-0.05, -0.08, -0.10, -0.12, -0.15],
    "take_profit": [0.10, 0.15, 0.20, 0.25, 0.30],
    "max_position": [0.20, 0.30, 0.40, 0.50],
    "warmup_days": [150, 200, 250],
}

LGBM_SPACE = {
    "n_estimators": (50, 300),
    "max_depth": (3, 8),
    "num_leaves": (8, 63),
    "learning_rate": (0.01, 0.2),
    "min_child_samples": (20, 200),
}


@dataclass
class OptimizeResult:
    """优化结果"""
    best_params: dict
    best_score: float
    baseline_score: float
    param_importance: dict          # 参数 → Sharpe方差贡献
    sensitivity: Optional[pd.DataFrame] = None  # 敏感度矩阵
    trials_df: Optional[pd.DataFrame] = None    # 所有试验
    best_lgbm_params: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "best_params": self.best_params,
            "best_score": round(self.best_score, 4),
            "baseline_score": round(self.baseline_score, 4),
            "param_importance": {k: round(v, 4) for k, v in self.param_importance.items()},
            "best_lgbm_params": self.best_lgbm_params,
        }

    def save(self, path: str = None):
        if path is None:
            path = OPT_DIR / "latest_optimize.json"
        path = Path(path)
        data = self.to_dict()
        if self.trials_df is not None:
            data["trials"] = self.trials_df.to_dict("records")
        if self.sensitivity is not None:
            data["sensitivity"] = self.sensitivity.to_dict("records")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"💾 优化结果已保存: {path}")


class HyperparamOptimizer:
    """
    策略参数优化器

    使用方式:
        opt = HyperparamOptimizer(df, feature_matrix, feature_ids)
        result = opt.optimize_strategy()  # Grid Search
        # 或者:
        lgbm_params = opt.optimize_lgbm(n_trials=100)  # Optuna
    """

    def __init__(self, df: pd.DataFrame, feature_matrix: pd.DataFrame,
                 feature_ids: List[str]):
        self.df = df
        self.feature_matrix = feature_matrix
        self.feature_ids = feature_ids
        self.n_bars = len(df)
        self.engine = MLSignalEngineV4()
        self._baseline_cache: Optional[float] = None

        # 保存日期
        if "date" in df.columns:
            self._dates = [self._fmt_date(df["date"].values[i]) for i in range(self.n_bars)]
        else:
            self._dates = [str(i) for i in range(self.n_bars)]

    @staticmethod
    def _fmt_date(date_val) -> str:
        try:
            return pd.Timestamp(date_val).strftime("%Y-%m-%d")
        except Exception:
            return str(date_val)[:10]

    # ═══════════════════════════════════════════════════
    # 核心: 参数化回测 (单次)
    # ═══════════════════════════════════════════════════

    def _run_backtest_with_params(self,
                                  entry_threshold: float = 0.5,
                                  stop_loss: float = -0.08,
                                  take_profit: float = 0.15,
                                  max_position: float = 0.40,
                                  warmup_days: int = 200,
                                  fee_rate: float = 0.001,
                                  slippage: float = 0.0005,
                                  use_lgbm: bool = True,
                                  verbose: bool = False) -> BacktestResult:
        # 确保整数参数类型
        warmup_days = int(warmup_days)
        """
        用指定参数运行一次回测。

        核心逻辑与 StrategyBacktester.run() 相同，但参数可覆盖。
        复用已缓存的 engine (含sub_signals builder权重)。
        """
        n_bars = self.n_bars
        feature_matrix = self.feature_matrix
        feature_ids = self.feature_ids
        _dates = self._dates

        if n_bars < warmup_days + 20:
            return self._empty_result("BTC/USDT")

        capital = INITIAL_CAPITAL
        position = 0.0
        entry_price = 0.0
        entry_idx = 0
        entry_signal_val = 0.0
        entry_consensus_val = 0.0

        equity = np.full(n_bars, INITIAL_CAPITAL)
        trades: List[BacktestTrade] = []
        signal_log = []
        drawdowns = []
        peak_equity = INITIAL_CAPITAL

        for t in range(warmup_days, n_bars):
            price = float(self.df["close"].values[t])

            # 提取特征行
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

            # 构建子信号
            try:
                sub_signals = self.engine.builder.build(feat_row)
            except Exception:
                continue

            active_sigs = [s for s in sub_signals if s.confidence > 0.1]

            # IC加权
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

            # LightGBM
            lgbm_val = 0.0
            if use_lgbm and self.engine._lgbm_loaded:
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

            final_signal = lgbm_val if (use_lgbm and self.engine._lgbm_loaded and lgbm_val != 0) else signal_val

            long_count = sum(1 for s in sub_signals if s.direction == "LONG")
            short_count = sum(1 for s in sub_signals if s.direction == "SHORT")
            active_count = long_count + short_count

            signal_log.append({
                "idx": t, "date": _dates[t],
                "signal": final_signal, "signal_ic": signal_val, "signal_lgbm": lgbm_val,
                "active_count": active_count, "price": price,
            })

            # ── 交易逻辑 ──
            in_position = position > 1e-9

            if not in_position:
                if final_signal > entry_threshold and active_count >= 2:
                    size_pct = min(max_position, 0.05 + abs(final_signal) * 0.05)
                    size_amount = capital * size_pct * (1 - fee_rate - slippage)
                    position = size_amount / price
                    entry_price = price
                    entry_idx = t
                    entry_signal_val = final_signal
                    entry_consensus_val = max(long_count, short_count) / max(1, len(sub_signals))
                    capital -= size_amount
            else:
                pnl_pct = (price - entry_price) / entry_price
                exit_now = False
                exit_reason = "signal_exit"

                if pnl_pct <= stop_loss:
                    exit_now = True
                    exit_reason = "stop_loss"
                elif pnl_pct >= take_profit:
                    exit_now = True
                    exit_reason = "take_profit"
                elif final_signal < 0:
                    exit_now = True
                    exit_reason = "signal_exit"

                if exit_now:
                    exit_value = position * price * (1 - fee_rate - slippage)
                    capital += exit_value
                    trade_pnl = exit_value - (position * entry_price)
                    trade_pnl_pct = trade_pnl / (position * entry_price) * 100

                    trades.append(BacktestTrade(
                        entry_idx=entry_idx, entry_time=_dates[entry_idx],
                        entry_price=entry_price, exit_idx=t, exit_time=_dates[t],
                        exit_price=price,
                        size_pct=position * price / equity[entry_idx] if equity[entry_idx] > 0 else 0,
                        pnl=trade_pnl, pnl_pct=trade_pnl_pct,
                        holding_days=t - entry_idx, exit_reason=exit_reason,
                        entry_signal=entry_signal_val, entry_consensus=entry_consensus_val,
                    ))
                    position = 0.0
                    entry_price = 0.0

            equity[t] = capital + position * price * (1 - fee_rate - slippage)
            if equity[t] > peak_equity:
                peak_equity = equity[t]
            dd = (peak_equity - equity[t]) / peak_equity * 100
            drawdowns.append(dd)

        # 最后平仓
        if position > 1e-9:
            final_price = float(self.df["close"].values[-1])
            exit_value = position * final_price * (1 - fee_rate - slippage)
            capital += exit_value
            trade_pnl = exit_value - (position * entry_price)
            trade_pnl_pct = trade_pnl / (position * entry_price) * 100
            trades.append(BacktestTrade(
                entry_idx=entry_idx, entry_time=_dates[entry_idx],
                entry_price=entry_price, exit_idx=n_bars - 1, exit_time=_dates[-1],
                exit_price=final_price,
                size_pct=position * final_price / equity[entry_idx] if equity[entry_idx] > 0 else 0,
                pnl=trade_pnl, pnl_pct=trade_pnl_pct,
                holding_days=n_bars - 1 - entry_idx, exit_reason="end_of_test",
                entry_signal=entry_signal_val, entry_consensus=entry_consensus_val,
            ))
            equity[-1] = capital

        return self._compute_metrics(equity, trades, signal_log, drawdowns, warmup_days)

    def _compute_metrics(self, equity: np.ndarray, trades: List[BacktestTrade],
                         signal_log: list, drawdowns: list,
                         warmup_days: int) -> BacktestResult:
        """计算回测指标 (精简版, 与 strategy_backtest 一致)"""
        n_bars = self.n_bars

        # 基准
        bench_start = float(self.df["close"].values[warmup_days])
        bench_end = float(self.df["close"].values[-1])
        benchmark_return = (bench_end - bench_start) / bench_start * 100

        # 收益
        total_return = (equity[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        annual_factor = 365 / max(1, n_bars - warmup_days)
        annual_return = ((equity[-1] / INITIAL_CAPITAL) ** annual_factor - 1) * 100

        daily_returns = pd.Series([
            (equity[i] - equity[i-1]) / equity[i-1] if equity[i-1] > 0 else 0
            for i in range(warmup_days + 1, n_bars)
        ])

        annual_vol = float(np.std(daily_returns) * np.sqrt(365) * 100) if len(daily_returns) > 0 else 0
        avg_daily_ret = float(np.mean(daily_returns)) if len(daily_returns) > 0 else 0
        sharpe = (avg_daily_ret / max(1e-9, float(np.std(daily_returns)))) * np.sqrt(365)

        neg_returns = daily_returns[daily_returns < 0]
        downside_std = float(np.std(neg_returns)) if len(neg_returns) > 0 else 1e-9
        sortino = (avg_daily_ret / downside_std) * np.sqrt(365)

        max_dd = max(drawdowns) if drawdowns else 0
        max_dd_days = 0
        current_dd_days = 0
        peak = equity[warmup_days]
        for i in range(warmup_days, n_bars):
            if equity[i] >= peak:
                peak = equity[i]
                max_dd_days = max(max_dd_days, current_dd_days)
                current_dd_days = 0
            else:
                current_dd_days += 1
        max_dd_days = max(max_dd_days, current_dd_days)

        calmar = annual_return / max(max_dd, 1e-9)

        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]
        win_rate = len(winning) / max(1, len(trades)) * 100
        avg_win = np.mean([t.pnl_pct for t in winning]) if winning else 0
        avg_loss = np.mean([t.pnl_pct for t in losing]) if losing else 0
        total_profits = sum(t.pnl for t in winning)
        total_losses = abs(sum(t.pnl for t in losing))
        profit_factor = total_profits / max(1e-9, total_losses) if total_losses > 0 else 999.0 if total_profits > 0 else 0.0
        avg_hold = np.mean([t.holding_days for t in trades]) if trades else 0

        signal_acc = 0
        if signal_log:
            correct = 0
            for i, sl in enumerate(signal_log):
                if i < len(signal_log) - 1:
                    future_ret = (signal_log[i+1]["price"] - sl["price"]) / sl["price"]
                    if (sl["signal"] > 0 and future_ret > 0) or (sl["signal"] < 0 and future_ret < 0):
                        correct += 1
            signal_acc = correct / max(1, len(signal_log) - 1)

        equity_series = pd.Series(equity)
        bench_series = pd.Series([
            float(self.df["close"].values[i]) / bench_start * INITIAL_CAPITAL
            for i in range(n_bars)
        ])

        return BacktestResult(
            symbol="BTC/USDT",
            start_date=self._dates[warmup_days],
            end_date=self._dates[-1],
            n_days=n_bars - warmup_days,
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
            benchmark_curve=bench_series,
            signal_history=pd.DataFrame(signal_log) if signal_log else None,
        )

    def _empty_result(self, symbol: str) -> BacktestResult:
        return BacktestResult(
            symbol=symbol, start_date="", end_date="",
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

    # ═══════════════════════════════════════════════════
    # 复合得分
    # ═══════════════════════════════════════════════════

    def _composite_score(self, result: BacktestResult) -> float:
        """
        复合优化目标:
          + Sharpe主导 (×2)
          - 回撤惩罚 (|MaxDD|/50)
          + 交易频率奖励 (cap在10笔)
        """
        return (
            result.sharpe_ratio * 2
            - abs(result.max_drawdown_pct) / 50
            + min(result.n_trades / 10, 1.0)
        )

    def _baseline_score(self) -> float:
        """计算默认参数得分"""
        if self._baseline_cache is not None:
            return self._baseline_cache
        r = self._run_backtest_with_params(**DEFAULT_PARAMS, verbose=True)
        self._baseline_cache = self._composite_score(r)
        return self._baseline_cache

    # ═══════════════════════════════════════════════════
    # Grid Search — 交易策略参数
    # ═══════════════════════════════════════════════════

    def optimize_strategy(self, grid: dict = None, verbose: bool = True) -> OptimizeResult:
        """
        Grid Search 优化交易策略参数。

        Args:
            grid: 参数网格 (默认 STRATEGY_GRID)
        Returns:
            OptimizeResult with best_params, best_score, etc.
        """
        if grid is None:
            grid = STRATEGY_GRID

        baseline = self._baseline_score()
        if verbose:
            print(f"📊 默认参数复合得分: {baseline:.4f}")

        # 展开所有参数组合
        keys = list(grid.keys())
        values = list(grid.values())
        from itertools import product

        total = 1
        for v in values:
            total *= len(v)

        if verbose:
            print(f"🔍 搜索空间: {total} 种组合")
            print(f"   {', '.join(f'{k}={len(v)}' for k, v in zip(keys, values))}")

        trials = []
        best_score = -999
        best_params = None

        for i, combo in enumerate(product(*values)):
            params = dict(zip(keys, combo))
            result = self._run_backtest_with_params(**params)

            score = self._composite_score(result)
            trials.append({
                **params,
                "score": round(score, 4),
                "sharpe": round(result.sharpe_ratio, 4),
                "max_dd": round(result.max_drawdown_pct, 2),
                "n_trades": result.n_trades,
                "win_rate": round(result.win_rate_pct, 1),
                "total_return": round(result.total_return_pct, 2),
                "annual_return": round(result.annual_return_pct, 2),
            })

            if score > best_score:
                best_score = score
                best_params = params

            if verbose and (i + 1) % 300 == 0:
                print(f"  ... {i+1}/{total} ({100*(i+1)/total:.0f}%) | "
                      f"当前最佳: {best_score:.4f}")

        trials_df = pd.DataFrame(trials)
        trials_df = trials_df.sort_values("score", ascending=False)

        # 参数重要性 (每个参数的得分方差)
        param_importance = {}
        for key in keys:
            grouped = trials_df.groupby(key)["score"].mean()
            param_importance[key] = float(grouped.max() - grouped.min())

        # 敏感性
        sensitivity = self.sensitivity_analysis(best_params)

        opt_result = OptimizeResult(
            best_params=best_params,
            best_score=best_score,
            baseline_score=baseline,
            param_importance=param_importance,
            sensitivity=sensitivity,
            trials_df=trials_df,
        )

        if verbose:
            print(f"\n✅ 优化完成!")
            print(f"   最佳得分: {best_score:.4f} (默认: {baseline:.4f}, 提升: {(best_score-baseline):+.4f})")
            print(f"   最优参数: {best_params}")
            print(f"   最佳Sharpe: {trials_df.iloc[0]['sharpe']:.4f}")
            print(f"   最佳回撤: {trials_df.iloc[0]['max_dd']:.2f}%")
            print(f"   最佳交易数: {trials_df.iloc[0]['n_trades']}")

        opt_result.save()
        return opt_result

    # ═══════════════════════════════════════════════════
    # Optuna — LightGBM超参
    # ═══════════════════════════════════════════════════

    def optimize_lgbm(self, n_trials: int = 100, verbose: bool = True) -> dict:
        """
        Optuna Bayesian 优化 LightGBM 超参数。

        目标: 最大化 OOS Rank IC (通过交叉验证评估)

        Returns:
            dict: 最优LGBM超参
        """
        try:
            import optuna
            import lightgbm as lgb
            from scipy import stats
        except ImportError as e:
            print(f"⚠️ 缺少依赖: {e}")
            return {}

        # 准备训练数据
        from ml_lightgbm_trainer import LightGBMTrainer
        trainer = LightGBMTrainer(fwd_window=5, n_splits=3, n_estimators=100, max_depth=4, num_leaves=16, learning_rate=0.05)

        if verbose:
            print(f"🧬 Optuna: 优化LightGBM超参 (n_trials={n_trials})...")

        # 预计算特征
        ts_df = self.feature_matrix
        close = self.df["close"].values
        n = len(close)
        fwd = n - trainer.fwd_window
        if fwd <= 0:
            fwd = max(1, n - trainer.fwd_window)
        fwd_rets = np.zeros(n)
        fwd_rets[:fwd] = close[trainer.fwd_window:n] / close[:fwd] - 1

        # 清理
        valid_mask = ~(np.isnan(ts_df.values).any(axis=1) | np.isnan(fwd_rets))
        X_full = ts_df.values[valid_mask]
        y_full = fwd_rets[valid_mask]

        if len(X_full) < 200:
            print("⚠️ 有效样本不足, 跳过LGBM优化")
            return {}

        # 样本采样 (加速)
        sample_n = min(600, len(X_full))
        indices = np.linspace(0, len(X_full) - 1, sample_n, dtype=int)
        X_opt = X_full[indices]
        y_opt = y_full[indices]

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "num_leaves": trial.suggest_int("num_leaves", 8, 63),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
                "verbose": -1,
                "random_state": 42,
                "force_col_wise": True,
            }

            # 3折交叉验证
            split = len(X_opt) // 3
            ics = []
            for fold in range(3):
                t_start, t_end = fold * split, min((fold + 1) * split, len(X_opt))
                train_end = max(20, t_start - 10)
                if t_end - t_start < 20 or train_end < 50:
                    continue
                X_tr, y_tr = X_opt[:train_end], y_opt[:train_end]
                X_te, y_te = X_opt[t_start:t_end], y_opt[t_start:t_end]

                try:
                    model = lgb.LGBMRegressor(**params)
                    model.fit(X_tr, y_tr)
                    preds = model.predict(X_te)
                    ic, _ = stats.spearmanr(preds, y_te)
                    if not np.isnan(ic):
                        ics.append(ic)
                except Exception:
                    pass

            return np.mean(ics) if ics else -1.0

        # 运行Optuna
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

        best = study.best_params
        best_ic = study.best_value

        if verbose:
            print(f"\n✅ Optuna完成!")
            print(f"   最佳OOS IC: {best_ic:.4f}")
            print(f"   最优LGBM超参: {best}")

        return best

    # ═══════════════════════════════════════════════════
    # 敏感性分析
    # ═══════════════════════════════════════════════════

    def sensitivity_analysis(self, base_params: dict = None,
                             perturbations: tuple = (0.5, 0.75, 1.0, 1.25, 1.5),
                             verbose: bool = True) -> pd.DataFrame:
        """
        参数敏感性分析: 每个参数 ±50% 扰动, 观察 Sharpe/MaxDD 变化。

        Returns:
            DataFrame: param × multiplier → sharpe/maxdd/n_trades
        """
        if base_params is None:
            base_params = DEFAULT_PARAMS.copy()

        if verbose:
            print(f"\n🔬 敏感性分析: {len(base_params)} 参数 × {len(perturbations)} 扰动...")

        rows = []
        for param, base_val in base_params.items():
            for mult in perturbations:
                new_val = base_val * mult
                # 合理性约束
                if param == "entry_threshold" and new_val > 2.0:
                    continue
                if param == "stop_loss" and (new_val > -0.02 or new_val < -0.25):
                    continue
                if param == "take_profit" and (new_val > 0.50 or new_val < 0.03):
                    continue
                if param == "max_position" and (new_val > 0.80 or new_val < 0.05):
                    continue
                if param == "warmup_days" and (new_val < 50 or new_val > 350):
                    continue
                if param in ("fee_rate", "slippage"):
                    continue  # 这些不是优化参数

                test_params = base_params.copy()
                test_params[param] = new_val
                result = self._run_backtest_with_params(**test_params)

                rows.append({
                    "param": param,
                    "multiplier": round(mult, 2),
                    "value": round(new_val, 4),
                    "sharpe": round(result.sharpe_ratio, 4),
                    "max_dd": round(result.max_drawdown_pct, 2),
                    "n_trades": result.n_trades,
                    "win_rate": round(result.win_rate_pct, 1),
                })

        sensitivity = pd.DataFrame(rows)
        if verbose:
            print(f"   ✅ {len(rows)} 个测试点完成")

        return sensitivity

    # ═══════════════════════════════════════════════════
    # 报告 & 可视化
    # ═══════════════════════════════════════════════════

    def report(self, result: OptimizeResult) -> str:
        """生成Markdown优化报告"""
        lines = [
            "=" * 60,
            "🎯 策略参数优化报告",
            "=" * 60,
            "",
            f"📊 默认参数得分: {result.baseline_score:.4f}",
            f"🏆 最优参数得分: {result.best_score:.4f}",
            f"📈 提升: {(result.best_score - result.baseline_score):+.4f} ({(result.best_score/max(0.001, result.baseline_score) - 1)*100:+.1f}%)",
            "",
            "## 🥇 最优参数",
            "",
            "| 参数 | 默认值 | 最优值 | 变化 |",
            "|------|:---:|:---:|:---:|",
        ]
        for k, v in DEFAULT_PARAMS.items():
            best_v = result.best_params.get(k, v)
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.3f} | {best_v:.3f} | {(best_v/v - 1)*100:+.0f}% |")
            else:
                lines.append(f"| {k} | {v} | {best_v} | — |")

        lines += [
            "",
            "## 📊 参数重要性 (得分方差贡献)",
            "",
        ]
        sorted_imp = sorted(result.param_importance.items(), key=lambda x: x[1], reverse=True)
        for param, imp in sorted_imp:
            bar = "█" * min(20, int(imp / max(0.001, sorted_imp[0][1]) * 20))
            lines.append(f"  {param}: {imp:.4f} {bar}")

        if result.trials_df is not None:
            top5 = result.trials_df.head(5)
            lines += [
                "",
                "## 🔝 Top-5 参数组合",
                "",
                "| Rank | Entry | StopLoss | TakeProfit | MaxPos | Warmup | Score | Sharpe | MaxDD | #Trades |",
                "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
            ]
            for i, (_, row) in enumerate(top5.iterrows()):
                lines.append(
                    f"| {i+1} | {row['entry_threshold']:.2f} | {row['stop_loss']:.2f} | "
                    f"{row['take_profit']:.2f} | {row['max_position']:.2f} | {int(row['warmup_days'])} | "
                    f"{row['score']:.4f} | {row['sharpe']:.4f} | {row['max_dd']:.2f}% | {int(row['n_trades'])} |"
                )

        return "\n".join(lines)

    def plot_sensitivity(self, sensitivity: pd.DataFrame):
        """生成参数敏感性热力图"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            print("⚠️ plotly 未安装")
            return None

        # Pivot: param × multiplier → sharpe
        pivot = sensitivity.pivot_table(
            values="sharpe", index="param", columns="multiplier", aggfunc="first"
        )

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[f"×{c}" for c in pivot.columns],
            y=pivot.index,
            colorscale="RdYlGn",
            zmid=sensitivity["sharpe"].median(),
            text=[[f"{v:.3f}" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont={"size": 11},
        ))

        fig.update_layout(
            title="🔬 参数敏感性分析 — Sharpe比率变化",
            xaxis_title="扰动倍数",
            yaxis_title="参数",
            template="plotly_dark",
            height=400,
        )

        return fig

    def plot_trials_distribution(self, trials_df: pd.DataFrame):
        """绘制试验得分分布"""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        baseline = self._baseline_score()

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=trials_df["score"].values,
            nbinsx=50,
            name="所有试验",
            marker_color="#00ff88",
            opacity=0.7,
        ))
        fig.add_vline(x=baseline, line_dash="dash", line_color="#ff4444",
                      annotation_text=f"默认={baseline:.3f}")

        fig.update_layout(
            title="📊 Grid Search 试验分布",
            xaxis_title="复合得分",
            yaxis_title="试验数",
            template="plotly_dark",
            height=350,
        )

        return fig


# ═══════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chase量化策略 · 参数优化器")
    parser.add_argument("--skip-lgbm", action="store_true",
                       help="跳过LightGBM超参优化")
    parser.add_argument("--sensitivity-only", action="store_true",
                       help="仅运行敏感性分析")
    parser.add_argument("--lgbm-trials", type=int, default=100,
                       help="Optuna试验次数 (默认100)")
    args = parser.parse_args()

    print("=" * 60)
    print("🎯 Phase 8: 策略参数优化器")
    print("=" * 60)

    # Step 1: 加载数据 + 预计算特征
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

    opt = HyperparamOptimizer(df, feature_matrix, feature_ids)

    # Step 3: Optuna优化LGBM (可选)
    best_lgbm = {}
    if not args.skip_lgbm:
        print(f"\n🤖 Step 3: Optuna优化LightGBM超参...")
        best_lgbm = opt.optimize_lgbm(n_trials=args.lgbm_trials)
    else:
        print("\n⏭️ 跳过LGBM优化")

    if args.sensitivity_only:
        print("\n🔬 仅运行敏感性分析...")
        sensitivity = opt.sensitivity_analysis()
        print("\n📊 敏感性矩阵 (Sharpe):")
        pivot = sensitivity.pivot_table(values="sharpe", index="param", columns="multiplier")
        print(pivot.to_string())
        # 保存图表
        fig = opt.plot_sensitivity(sensitivity)
        if fig:
            fig.write_html(OPT_DIR / "sensitivity_heatmap.html")
            print("\n💾 热力图已保存: data/optimization_results/sensitivity_heatmap.html")
        import sys
        sys.exit(0)

    # Step 4: Grid Search
    print(f"\n🔍 Step 4: Grid Search 优化交易策略参数...")
    result = opt.optimize_strategy()

    if best_lgbm:
        result.best_lgbm_params = best_lgbm

    print()
    print(opt.report(result))

    # 保存可视化
    print("\n📊 生成可视化...")
    try:
        sensitivity = result.sensitivity
        if sensitivity is not None:
            fig1 = opt.plot_sensitivity(sensitivity)
            if fig1:
                fig1.write_html(OPT_DIR / "sensitivity_heatmap.html")
                print("   ✅ sensitivity_heatmap.html")

        if result.trials_df is not None:
            fig2 = opt.plot_trials_distribution(result.trials_df)
            if fig2:
                fig2.write_html(OPT_DIR / "trials_distribution.html")
                print("   ✅ trials_distribution.html")
    except Exception as e:
        print(f"⚠️ 可视化失败: {e}")

    result.save()
    print("\n🎉 优化完成!")
