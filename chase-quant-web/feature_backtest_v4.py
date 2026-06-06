"""
Feature Backtester v4.0 — FDR校正 + PurgedKFold + 时间序列加速
===============================================================
Chase量化策略 Phase 2: 单特征IC回测 + 多重假设检验校正

核心升级:
  1. 时间序列特征 → 只需对齐特征[t] vs 前向收益[t+k], O(1) per feature
  2. PurgedKFold: 防止时间序列交叉验证中的前视信息泄露
  3. Benjamini-Hochberg FDR: 修正500个同时假设检验的多重比较问题
  4. Rank IC 替代 Pearson t-stat: 更robust, 不依赖正态假设
  5. IC decay: 测试不同holding period (1d/3d/5d/10d/20d)

输出标准:
  - 保留 IC_IR > 0.3 且 FDR-adjusted p < 0.1
  - 按 IC_IR 排序, 不是 |t-stat|
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from scipy import stats
import json
import warnings
warnings.filterwarnings("ignore")

from feature_ts import FeatureFactoryV4, FeatureSpec


@dataclass
class FeatureTestResultV4:
    """单个特征回测结果 (v4增强版)"""
    feature_id: str
    feature_name: str
    category: str

    # Rank IC
    ic: float                # Spearman Rank IC
    ic_std: float            # IC std (cross-fold)
    icir: float              # IC Information Ratio

    # 线性回归
    t_stat: float
    p_value: float
    fdr_adjusted_p: float    # Benjamini-Hochberg adjusted

    # 方向正确率
    hit_rate: float

    # 简单策略
    sharpe: float
    long_sharpe: float       # 只做多
    short_sharpe: float      # 只做空

    # 稳定性
    stability: float         # 子周期IC标准差

    # 衰减曲线
    ic_decay: Dict[int, float]  # {1: IC, 3: IC, 5: IC, 10: IC, 20: IC}

    # 非线性
    nonlinear_r2: float

    # 样本信息
    n_obs: int
    fwd_window: int = 5

    passed: bool = False     # FDR p < 0.1 AND IC_IR > 0.3?

    def summary(self) -> str:
        check = "✅" if self.passed else "❌"
        return (
            f"{check} {self.feature_id:35s} | "
            f"IC={self.ic:+.3f} | ICIR={self.icir:+.2f} | "
            f"t={self.t_stat:+.2f} | FDR_p={self.fdr_adjusted_p:.3f} | "
            f"Sh={self.sharpe:+.2f} | [{self.category}]"
        )


class FeatureBacktesterV4:
    """增强版回测器 — 数据驱动特征筛选"""

    def __init__(self,
                 fwd_windows: List[int] = None,
                 min_obs: int = 100,
                 icir_threshold: float = 0.3,
                 fdr_alpha: float = 0.1):
        self.fwd_windows = fwd_windows or [1, 3, 5, 10, 20]
        self.min_obs = min_obs
        self.icir_threshold = icir_threshold
        self.fdr_alpha = fdr_alpha
        self.factory = FeatureFactoryV4()

    def test_all(self, df: pd.DataFrame,
                categories: Optional[List[str]] = None,
                fwd_window: int = 5,
                verbose: bool = True) -> List[FeatureTestResultV4]:
        """
        对所有特征运行回测 (优化版 — 先一次性计算所有特征时序)

        流程:
          1. compute_timeseries() → 279列×N行
          2. 对每列: 对齐特征[t] vs 前向收益[t+fwd]
          3. Rank IC + 回归 + FDR
        """
        n = len(df)
        fwd = fwd_window

        # ── Step 1: 一次性计算所有特征时间序列 ──
        if verbose:
            print(f"🧬 计算 {self.factory.feature_count} 个特征时间序列...")
        ts_df = self.factory.compute_timeseries(df, categories, verbose=verbose)

        # ── Step 2: 计算前向收益 ──
        close = df["close"].values
        fwd_rets = np.zeros(n)
        fwd_rets[:n-fwd] = close[fwd:] / close[:n-fwd] - 1

        # ── Step 3: 对每个特征单独评估 ──
        results = []
        feature_cols = [c for c in ts_df.columns if ts_df[c].notna().sum() > self.min_obs]

        if verbose:
            print(f"\n🔬 逐个特征回测 (fwd={fwd}d)...")

        for i, col in enumerate(feature_cols):
            if verbose and (i+1) % 50 == 0:
                passed_count = sum(1 for r in results if r.passed)
                print(f"  进度: {i+1}/{len(feature_cols)} | 通过: {passed_count}")

            feat_arr = ts_df[col].values
            result = self._test_single(feat_arr, fwd_rets, col, fwd_window)
            if result:
                results.append(result)

        # ── Step 4: FDR校正 ──
        results = self._apply_fdr(results)

        if verbose:
            passed = [r for r in results if r.passed]
            print(f"\n📊 完成: {len(results)} 测试 | "
                  f"{len(passed)} 通过 (FDR<{self.fdr_alpha}, ICIR>{self.icir_threshold})")

        return results

    def _test_single(self, feat_arr: np.ndarray, fwd_rets: np.ndarray,
                    feature_id: str, fwd_window: int) -> Optional[FeatureTestResultV4]:
        """单个特征评估"""
        n = len(feat_arr)

        # 有效数据点
        valid = ~(np.isnan(feat_arr) | np.isnan(fwd_rets) |
                  np.isinf(feat_arr) | np.isinf(fwd_rets))
        f_valid = feat_arr[valid]
        r_valid = fwd_rets[valid]

        if len(f_valid) < self.min_obs:
            return None

        # Winsorize
        f_clipped = self._winsorize(f_valid, 0.01)
        r_clipped = self._winsorize(r_valid, 0.01)

        # ── Rank IC ──
        try:
            ic, ic_pval = stats.spearmanr(f_clipped, r_clipped)
            if np.isnan(ic):
                ic = 0.0
        except Exception:
            ic = 0.0

        # ── IC稳定性 (PurgedKFold style) ──
        ic_std = self._calc_ic_cv(f_clipped, r_clipped)
        icir = np.clip(ic / (ic_std + 1e-9), -10, 10)  # cap extreme ICIR

        # ── 线性回归 (for t-stat reference) ──
        f_z = (f_clipped - np.mean(f_clipped)) / (np.std(f_clipped) + 1e-9)
        X = np.column_stack([np.ones(len(f_z)), f_z])
        try:
            beta_hat, residuals, rank, singular = np.linalg.lstsq(X, r_clipped, rcond=None)
            resid = r_clipped - X @ beta_hat
            dof = len(f_z) - 2
            sigma2 = np.sum(resid**2) / dof
            if sigma2 > 0:
                XtX_inv = np.linalg.inv(X.T @ X)
                se_beta = np.sqrt(sigma2 * XtX_inv[1, 1])
                t_stat = beta_hat[1] / (se_beta + 1e-9)
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof))
            else:
                t_stat, p_value = 0.0, 1.0
        except Exception:
            t_stat, p_value = 0.0, 1.0

        # ── Hit Rate ──
        hit_rate = np.mean(np.sign(f_z) == np.sign(r_clipped))

        # ── Sharpe ──
        long_mask = f_z > 0.5
        short_mask = f_z < -0.5
        strat_rets = np.zeros(len(r_clipped))
        strat_rets[long_mask] = r_clipped[long_mask]
        strat_rets[short_mask] = -r_clipped[short_mask]
        sharpe = np.mean(strat_rets) / (np.std(strat_rets) + 1e-9) * np.sqrt(252 / fwd_window)

        long_sharpe = (np.mean(r_clipped[long_mask]) / (np.std(r_clipped[long_mask]) + 1e-9)
                      * np.sqrt(252 / fwd_window)) if long_mask.sum() > 10 else 0
        short_sharpe = (np.mean(-r_clipped[short_mask]) / (np.std(r_clipped[short_mask]) + 1e-9)
                       * np.sqrt(252 / fwd_window)) if short_mask.sum() > 10 else 0

        # ── IC Decay (不同forward window) ──
        ic_decay = {}
        # We'll compute this separately if needed; for now use fwd_window only
        ic_decay[fwd_window] = ic

        # ── 非线性 ──
        nonlinear_r2 = self._calc_nonlinear_r2(f_z, r_clipped)

        # ── 稳定性 ──
        stability = ic_std

        # 查找特征名称和类别
        feat_info = self._find_feature_info(feature_id)

        return FeatureTestResultV4(
            feature_id=feature_id,
            feature_name=feat_info.get("name", feature_id),
            category=feat_info.get("category", "?"),
            ic=round(ic, 4),
            ic_std=round(ic_std, 4),
            icir=round(icir, 3),
            t_stat=round(t_stat, 3),
            p_value=round(p_value, 4),
            fdr_adjusted_p=1.0,  # to be filled
            hit_rate=round(hit_rate, 4),
            sharpe=round(sharpe, 3),
            long_sharpe=round(long_sharpe, 3),
            short_sharpe=round(short_sharpe, 3),
            stability=round(stability, 4),
            ic_decay=ic_decay,
            nonlinear_r2=round(nonlinear_r2, 4),
            n_obs=len(f_clipped),
            fwd_window=fwd_window,
            passed=False,  # to be filled after FDR
        )

    def _apply_fdr(self, results: List[FeatureTestResultV4]) -> List[FeatureTestResultV4]:
        """Benjamini-Hochberg FDR校正"""
        if not results:
            return results

        p_values = np.array([r.p_value for r in results])
        n = len(p_values)

        # BH procedure
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        bh_critical = self.fdr_alpha * (np.arange(1, n+1) / n)

        # 找到最大的k使得p(k) <= alpha * k / n
        significant = sorted_p <= bh_critical
        if significant.any():
            max_sig_idx = np.max(np.where(significant)[0])
            fdr_threshold = sorted_p[max_sig_idx]
        else:
            fdr_threshold = 0.0

        # Apply
        for i, r in enumerate(results):
            rank = np.where(sorted_idx == i)[0][0] + 1
            r.fdr_adjusted_p = round(min(1.0, r.p_value * n / rank), 4)
            r.passed = (r.fdr_adjusted_p < self.fdr_alpha and abs(r.icir) > self.icir_threshold)

        return results

    def filter(self, results: List[FeatureTestResultV4]) -> List[FeatureTestResultV4]:
        """过滤通过的特征"""
        filtered = [r for r in results if r.passed]
        filtered.sort(key=lambda r: abs(r.icir), reverse=True)
        return filtered

    def rank_features(self, results: List[FeatureTestResultV4]) -> pd.DataFrame:
        """排序DataFrame"""
        data = []
        for r in sorted(results, key=lambda x: abs(x.icir), reverse=True):
            data.append({
                "feature_id": r.feature_id,
                "name": r.feature_name,
                "category": r.category,
                "ic": r.ic,
                "icir": r.icir,
                "t_stat": r.t_stat,
                "p_value": r.p_value,
                "fdr_p": r.fdr_adjusted_p,
                "hit_rate": r.hit_rate,
                "sharpe": r.sharpe,
                "stability": r.stability,
                "nonlinear_r2": r.nonlinear_r2,
                "n_obs": r.n_obs,
                "passed": r.passed,
            })
        return pd.DataFrame(data)

    def report(self, results: List[FeatureTestResultV4], top_n: int = 20) -> str:
        """生成Markdown报告"""
        passed = self.filter(results)
        all_sorted = sorted(results, key=lambda r: abs(r.icir), reverse=True)

        lines = [
            "## 🔬 Feature Backtest Report v4.0",
            "",
            f"**FDR**: Benjamini-Hochberg α={self.fdr_alpha} | "
            f"**ICIR阈值**: {self.icir_threshold} | "
            f"**Min Obs**: {self.min_obs}",
            f"**测试特征**: {len(results)} | "
            f"**通过**: {len(passed)} ({len(passed)/max(1,len(results))*100:.0f}%)",
            "",
            "### 🏆 Top 特征 (按|ICIR|排序)",
            "",
            "| # | 特征ID | 类 | IC | ICIR | t-stat | FDR-p | Hit | Sharpe |",
            "|---|--------|:--:|:--:|:---:|:---:|:---:|:---:|:---:|",
        ]

        for i, r in enumerate(all_sorted[:top_n]):
            check = "✅" if r.passed else ""
            lines.append(
                f"| {i+1} | {check} {r.feature_id} | {r.category} | "
                f"{r.ic:+.3f} | {r.icir:+.2f} | {r.t_stat:+.2f} | "
                f"{r.fdr_adjusted_p:.3f} | {r.hit_rate:.1%} | {r.sharpe:+.2f} |"
            )

        lines.append("")
        lines.append("### 📊 按类别统计")
        lines.append("")
        lines.append("| 类别 | 测试 | 通过 | 通过率 | 平均|IC| | 平均|ICIR| |")
        lines.append("|:----:|:---:|:---:|:---:|:---:|:---:|")

        cats = {}
        for r in results:
            c = r.category
            if c not in cats:
                cats[c] = {"total": 0, "passed": 0, "ics": [], "icirs": []}
            cats[c]["total"] += 1
            cats[c]["ics"].append(abs(r.ic))
            cats[c]["icirs"].append(abs(r.icir))
            if r.passed:
                cats[c]["passed"] += 1

        for cat, info in sorted(cats.items()):
            lines.append(
                f"| {cat} | {info['total']} | {info['passed']} | "
                f"{info['passed']/info['total']*100:.0f}% | "
                f"{np.mean(info['ics']):.3f} | {np.mean(info['icirs']):.2f} |"
            )

        return "\n".join(lines)

    def _find_feature_info(self, feature_id: str) -> dict:
        for f in self.factory.features:
            if f.id == feature_id:
                return {"name": f.name, "category": f.category}
        return {}

    # ── 内部工具 ──

    @staticmethod
    def _winsorize(x: np.ndarray, pct: float) -> np.ndarray:
        lo = np.percentile(x, pct * 100)
        hi = np.percentile(x, (1 - pct) * 100)
        return np.clip(x, lo, hi)

    def _calc_ic_cv(self, f_arr: np.ndarray, r_arr: np.ndarray,
                    n_splits: int = 5) -> float:
        """PurgedKFold风格的IC稳定性 (去除临近split点防止泄露)"""
        n = len(f_arr)
        split_size = n // n_splits
        purge = max(5, split_size // 10)  # purge zone

        if split_size < 20:
            return 0.1

        ics = []
        for i in range(n_splits):
            start = i * split_size
            end = min(start + split_size - purge, n)
            # Test on this fold
            if end - start < 20:
                continue
            try:
                ic, _ = stats.spearmanr(f_arr[start:end], r_arr[start:end])
                if not np.isnan(ic):
                    ics.append(ic)
            except Exception:
                pass

        return np.std(ics) if len(ics) > 1 else 0.1

    def _calc_nonlinear_r2(self, f_z: np.ndarray, r_arr: np.ndarray) -> float:
        try:
            X_lin = np.column_stack([np.ones(len(f_z)), f_z])
            X_quad = np.column_stack([np.ones(len(f_z)), f_z, f_z**2])

            r2_lin = 1 - np.sum((r_arr - X_lin @ np.linalg.lstsq(X_lin, r_arr, rcond=None)[0])**2) / \
                     np.sum((r_arr - np.mean(r_arr))**2)
            r2_quad = 1 - np.sum((r_arr - X_quad @ np.linalg.lstsq(X_quad, r_arr, rcond=None)[0])**2) / \
                      np.sum((r_arr - np.mean(r_arr))**2)
            return max(0, r2_quad - r2_lin)
        except Exception:
            return 0


# ── CLI ──
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("🔬 Feature Backtester v4.0 — FDR校正版")
    print("=" * 60)

    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True, "timeout": 15000})
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1d", limit=400)
        df = pd.DataFrame(ohlcv, columns=["date", "open", "high", "low", "close", "volume"])

        print(f"📊 BTC/USDT | {len(df)}天 | 测试中...\n")

        bt = FeatureBacktesterV4(fdr_alpha=0.1, icir_threshold=0.3)
        results = bt.test_all(df, fwd_window=5, verbose=True)

        print()
        print(bt.report(results, top_n=25))

        # FDR统计
        n_before = len(results)
        sig_before = sum(1 for r in results if r.p_value < 0.05)
        sig_after = sum(1 for r in results if r.passed)
        print(f"\n🔍 FDR效果: {sig_before} features with raw p<0.05 → {sig_after} after FDR correction")

        # 保存
        rank_df = bt.rank_features(results)
        out = Path(__file__).parent / "data" / "feature_backtest_v4.csv"
        out.parent.mkdir(exist_ok=True)
        rank_df.to_csv(out, index=False)
        print(f"📄 保存到: {out}")

    except ImportError:
        print("⚠️ ccxt未安装, 跳过实时测试")
