"""
Feature Backtester v3.0 — 独立特征验证 + t-stat过滤
=====================================================
西蒙斯风格核心原则:
  "每个特征单独回测。不靠人判断'这个指标有道理' — 让数据说话。
   只保留 t-stat > 1.5 的特征。"

测试方法:
  对每个特征 f_t:
    1. 计算 f_t 的值 (在时间t唯一已知的信息)
    2. 计算未来N日收益 r_{t+N}
    3. 回归: r_{t+N} = α + β * f_t + ε
    4. 计算 β 的 t-statistic
    5. 保留 t-stat > 1.5 的特征

除了t-stat, 还计算:
  - Information Coefficient (IC): rank_corr(f_t, r_{t+N})
  - Hit Rate: sign(f_t) == sign(r_{t+N}) 的比例
  - Sharpe: 基于f_t构建的简单long/short策略的Sharpe

输出:
  - 按 t-stat 排序的特征列表
  - 只保留 t-stat > 1.5 的 → 用于后续子信号构建
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from feature_engine import (
    FeatureFactory, Feature,
    _returns, _rolling_mean, _rolling_std,
)


@dataclass
class FeatureTestResult:
    """单个特征回测结果"""
    feature_id: str
    feature_name: str
    category: str

    # 核心指标
    t_stat: float           # β的t统计量 (H0: β=0)
    p_value: float          # t检验p值
    ic: float               # Information Coefficient (rank correlation)
    ic_std: float           # IC的标准差 → ICIR = ic/ic_std
    hit_rate: float         # 方向正确率
    sharpe: float           # 简单long/short策略Sharpe

    # 稳定性
    icir: float             # Information Coefficient IR
    stability: float        # t-stat在子周期中的标准差 (越小越稳定)

    # 非线性
    nonlinear_r2: float     # 加入f_t^2后的R²增量 (非线性贡献)

    passed: bool            # t-stat > 1.5 ?

    # 样本信息
    n_obs: int = 0
    forward_window: int = 5

    def summary(self) -> str:
        check = "✅" if self.passed else "❌"
        return (
            f"{check} {self.feature_id:35s} | "
            f"t={self.t_stat:+.2f} | IC={self.ic:+.3f} | "
            f"HR={self.hit_rate:.1%} | Sh={self.sharpe:+.2f} | "
            f"IR={self.icir:+.2f} | [{self.category}]"
        )


class FeatureBacktester:
    """独立特征回测器"""

    def __init__(self, forward_window: int = 5,
                 min_obs: int = 100,
                 t_threshold: float = 1.5):
        self.forward_window = forward_window
        self.min_obs = min_obs
        self.t_threshold = t_threshold
        self.factory = FeatureFactory()

    def test_feature(self, feature: Feature,
                     df: pd.DataFrame) -> Optional[FeatureTestResult]:
        """
        对单个特征运行回测:
        1. 计算每个时间点的特征值 (使用当时已知的信息)
        2. 计算未来N日收益
        3. 线性回归 → t-stat
        4. 方向测试 → hit rate
        5. 简单策略 → Sharpe
        """
        close = df["close"].values
        n = len(close)

        if n < self.min_obs + 60:
            return None

        fw = self.forward_window

        # ── 1. 滚动计算特征值 + 前向收益 ──
        feature_vals = []
        forward_rets = []

        for t in range(60, n - fw):
            try:
                # 只用在时间t及之前的数据
                historical_df = df.iloc[:t+1].copy()

                # 计算特征值
                f_val = feature.compute_fn(historical_df)
                if np.isnan(f_val) or np.isinf(f_val):
                    continue

                # 计算未来fw日收益 (前瞻)
                if t + fw < n:
                    fwd_ret = close[t + fw] / close[t] - 1
                    feature_vals.append(f_val)
                    forward_rets.append(fwd_ret)
            except Exception:
                continue

        if len(feature_vals) < self.min_obs:
            return None

        f_arr = np.array(feature_vals)
        r_arr = np.array(forward_rets)

        # ── 2. Winsorize (去极值) ──
        f_arr = self._winsorize(f_arr, 0.01)
        r_arr = self._winsorize(r_arr, 0.01)

        # ── 3. 标准化特征 (Z-score) ──
        f_mean = np.mean(f_arr)
        f_std = np.std(f_arr) + 1e-9
        f_z = (f_arr - f_mean) / f_std

        # ── 4. 线性回归: r = α + β * f + ε ──
        X = np.column_stack([np.ones(len(f_z)), f_z])
        try:
            beta_hat, residuals, rank, singular = np.linalg.lstsq(X, r_arr, rcond=None)
            alpha = beta_hat[0]
            beta = beta_hat[1]

            # t-statistic
            residuals_hat = r_arr - (X @ beta_hat)
            n_obs = len(f_z)
            dof = n_obs - 2
            sigma2 = np.sum(residuals_hat ** 2) / dof
            if sigma2 <= 0:
                return None
            XtX_inv = np.linalg.inv(X.T @ X)
            se_beta = np.sqrt(sigma2 * XtX_inv[1, 1])
            t_stat = beta / se_beta if se_beta > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof))
        except Exception:
            return None

        # ── 5. Information Coefficient ──
        try:
            ic, ic_pval = stats.spearmanr(f_arr, r_arr)
            if np.isnan(ic):
                ic = 0
        except Exception:
            ic = 0

        # ── 6. IC稳定性 (交叉验证) ──
        ic_std = self._calc_ic_stability(f_z, r_arr)

        # ── 7. Hit Rate (方向正确率) ──
        f_sign = np.sign(f_z)
        r_sign = np.sign(r_arr)
        hit_rate = np.mean(f_sign == r_sign)

        # ── 8. 简单Long/Short策略Sharpe ──
        # 做多 f_z > 0.5σ, 做空 f_z < -0.5σ
        long_mask = f_z > 0.5
        short_mask = f_z < -0.5
        strategy_rets = np.zeros(len(r_arr))
        strategy_rets[long_mask] = r_arr[long_mask]
        strategy_rets[short_mask] = -r_arr[short_mask]
        sharpe = np.mean(strategy_rets) / (np.std(strategy_rets) + 1e-9) * np.sqrt(252 / fw)

        # ── 9. 非线性贡献 ──
        nonlinear_r2 = self._calc_nonlinear_r2(f_z, r_arr)

        # ── 10. 稳定性 (子周期t-stat波动) ──
        stability = self._calc_stability(f_z, r_arr)

        return FeatureTestResult(
            feature_id=feature.id,
            feature_name=feature.name,
            category=feature.category,
            t_stat=round(t_stat, 3),
            p_value=round(p_value, 4),
            ic=round(ic, 4),
            ic_std=round(ic_std, 4),
            hit_rate=round(hit_rate, 4),
            sharpe=round(sharpe, 3),
            icir=round(ic / (ic_std + 1e-9), 3),
            stability=round(stability, 4),
            nonlinear_r2=round(nonlinear_r2, 4),
            passed=abs(t_stat) > self.t_threshold,
            n_obs=len(f_arr),
            forward_window=fw,
        )

    def test_all(self, df: pd.DataFrame,
                 categories: Optional[List[str]] = None,
                 verbose: bool = True) -> List[FeatureTestResult]:
        """对所有特征运行回测"""
        results = []

        # 选择特征
        features = self.factory.active_features
        if categories:
            features = [f for f in features if f.category in categories]

        total = len(features)
        for i, feat in enumerate(features):
            if verbose and (i + 1) % 50 == 0:
                print(f"  进度: {i+1}/{total} | 通过: {sum(1 for r in results if r.passed)}")

            result = self.test_feature(feat, df)
            if result:
                results.append(result)

        # 更新特征的t-stat
        for r in results:
            for feat in self.factory.features:
                if feat.id == r.feature_id:
                    feat.t_stat = r.t_stat
                    feat.sharpe = r.sharpe
                    feat.hit_rate = r.hit_rate
                    feat.ic = r.ic
                    feat.retained = r.passed
                    break

        return results

    def filter(self, results: List[FeatureTestResult]) -> List[FeatureTestResult]:
        """过滤: 只保留 t-stat > 1.5 且 IC稳定 的特征"""
        filtered = [r for r in results if r.passed]
        # 按 |t-stat| 排序
        filtered.sort(key=lambda r: abs(r.t_stat), reverse=True)
        return filtered

    def rank_features(self, results: List[FeatureTestResult]) -> pd.DataFrame:
        """返回特征排名 DataFrame"""
        data = []
        for r in sorted(results, key=lambda x: abs(x.t_stat), reverse=True):
            data.append({
                "feature_id": r.feature_id,
                "name": r.feature_name,
                "category": r.category,
                "t_stat": r.t_stat,
                "p_value": r.p_value,
                "ic": r.ic,
                "icir": r.icir,
                "hit_rate": r.hit_rate,
                "sharpe": r.sharpe,
                "stability": r.stability,
                "nonlinear_r2": r.nonlinear_r2,
                "passed": r.passed,
            })
        return pd.DataFrame(data)

    def report(self, results: List[FeatureTestResult]) -> str:
        """生成Markdown报告"""
        passed = self.filter(results)
        all_results = sorted(results, key=lambda r: abs(r.t_stat), reverse=True)

        lines = [
            "## 🔬 Feature Backtest Report",
            f"",
            f"**参数**: forward_window={self.forward_window}d | "
            f"t_threshold={self.t_threshold} | min_obs={self.min_obs}",
            f"**测试特征**: {len(results)} | **通过**: {len(passed)} "
            f"({len(passed)/max(1,len(results))*100:.0f}%)",
            f"",
            f"### 🏆 Top 20 特征 (按|t-stat|排序)",
            f"",
            f"| # | 特征ID | 类别 | t-stat | IC | ICIR | Hit Rate | Sharpe | 通过 |",
            f"|---|--------|------|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]

        for i, r in enumerate(all_results[:20]):
            check = "✅" if r.passed else "❌"
            lines.append(
                f"| {i+1} | {r.feature_id} | {r.category} | "
                f"{r.t_stat:+.2f} | {r.ic:+.3f} | {r.icir:+.2f} | "
                f"{r.hit_rate:.1%} | {r.sharpe:+.2f} | {check} |"
            )

        lines.append("")
        lines.append("### 📊 按类别统计")
        lines.append("")
        lines.append("| 类别 | 测试数 | 通过数 | 通过率 | 平均|t-stat| |")
        lines.append("|------|:---:|:---:|:---:|:---:|")

        cats = {}
        for r in results:
            c = r.category
            if c not in cats:
                cats[c] = {"total": 0, "passed": 0, "t_stats": []}
            cats[c]["total"] += 1
            cats[c]["t_stats"].append(abs(r.t_stat))
            if r.passed:
                cats[c]["passed"] += 1

        for cat, info in sorted(cats.items()):
            lines.append(
                f"| {cat} | {info['total']} | {info['passed']} | "
                f"{info['passed']/info['total']*100:.0f}% | "
                f"{np.mean(info['t_stats']):.2f} |"
            )

        return "\n".join(lines)

    # ── 内部工具 ──
    @staticmethod
    def _winsorize(x: np.ndarray, pct: float) -> np.ndarray:
        lo = np.percentile(x, pct * 100)
        hi = np.percentile(x, (1 - pct) * 100)
        return np.clip(x, lo, hi)

    def _calc_ic_stability(self, f_z: np.ndarray, r_arr: np.ndarray,
                          n_splits: int = 5) -> float:
        """交叉验证IC的标准差"""
        n = len(f_z)
        split_size = n // n_splits
        if split_size < 20:
            return 0.1

        ics = []
        for i in range(n_splits):
            start = i * split_size
            end = min(start + split_size, n)
            if end - start < 20:
                continue
            try:
                ic, _ = stats.spearmanr(f_z[start:end], r_arr[start:end])
                if not np.isnan(ic):
                    ics.append(ic)
            except Exception:
                pass

        return np.std(ics) if ics else 0.1

    def _calc_nonlinear_r2(self, f_z: np.ndarray, r_arr: np.ndarray) -> float:
        """加入二次项后的R²增量"""
        try:
            X_linear = np.column_stack([np.ones(len(f_z)), f_z])
            X_quad = np.column_stack([np.ones(len(f_z)), f_z, f_z**2])

            r2_linear = 1 - np.sum((r_arr - X_linear @ np.linalg.lstsq(X_linear, r_arr, rcond=None)[0])**2) / np.sum((r_arr - np.mean(r_arr))**2)
            r2_quad = 1 - np.sum((r_arr - X_quad @ np.linalg.lstsq(X_quad, r_arr, rcond=None)[0])**2) / np.sum((r_arr - np.mean(r_arr))**2)

            return max(0, r2_quad - r2_linear)
        except Exception:
            return 0

    def _calc_stability(self, f_z: np.ndarray, r_arr: np.ndarray,
                        n_splits: int = 5) -> float:
        """子周期t-stat的标准差 (越小越稳定)"""
        n = len(f_z)
        split_size = n // n_splits
        if split_size < 20:
            return 1.0

        t_stats = []
        for i in range(n_splits):
            start = i * split_size
            end = min(start + split_size, n)
            if end - start < 20:
                continue
            try:
                X = np.column_stack([np.ones(end-start), f_z[start:end]])
                y = r_arr[start:end]
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                resid = y - X @ beta
                sigma2 = np.sum(resid**2) / (end-start-2)
                if sigma2 <= 0:
                    continue
                XtX_inv = np.linalg.inv(X.T @ X)
                se = np.sqrt(sigma2 * XtX_inv[1, 1])
                t_stats.append(beta[1] / (se + 1e-9))
            except Exception:
                pass

        return np.std(t_stats) if len(t_stats) > 1 else 1.0


# ── CLI ──
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("🔬 Feature Backtester v3.0")
    print("=" * 60)

    # 用BTC数据测试 (需要ccxt)
    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True, "timeout": 10000})
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1d", limit=400)
        df = pd.DataFrame(ohlcv, columns=["date", "open", "high", "low", "close", "volume"])

        print(f"📊 数据: BTC/USDT | {len(df)}天")
        print()

        bt = FeatureBacktester(forward_window=5, t_threshold=1.5)
        results = bt.test_all(df, verbose=True)

        print()
        print(bt.report(results))

        # 保存结果
        rank_df = bt.rank_features(results)
        out = Path(__file__).parent / "data" / "feature_backtest_results.csv"
        out.parent.mkdir(exist_ok=True)
        rank_df.to_csv(out, index=False)
        print(f"\n📄 结果已保存: {out}")

        # 统计
        passed = bt.filter(results)
        print(f"\n🎯 最终: {len(passed)}/{len(results)} 特征通过 t > {bt.t_threshold} 检验")
        print(f"   通过率: {len(passed)/len(results)*100:.0f}%")

    except ImportError:
        print("⚠️ ccxt 未安装, 跳过实时测试")
        print("   运行方式: python3 feature_backtest.py")
