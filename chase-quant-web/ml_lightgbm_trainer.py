"""
ML LightGBM Trainer v1.0 — 非线性子信号训练器
==============================================
Chase量化策略 Phase 5: 用LightGBM替代线性IC加权

核心理念 (西蒙斯风格):
  每个子信号 = 独立LightGBM → f(特征组) → 预测5日前向收益
  6个模型独立训练, 独立可debug

训练流程:
  1. 获取BTC/USDT 400日K线 (ccxt)
  2. FeatureFactoryV4.compute_timeseries() → 279列特征矩阵
  3. 计算前向收益 (fwd=5d)
  4. 按SubSignalBuilder主题分组特征 → 6个特征子集
  5. PurgedKFold (5折) → 每折训练LightGBM → 评估OOS
  6. 全量训练最终模型 → 保存 .pkl

输出:
  - data/models/lgbm_<theme>.pkl × 6
  - data/lgbm_train_report.csv
  - data/lgbm_feature_importance.json
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pickle
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

import lightgbm as lgb
from sklearn.model_selection import KFold

from feature_ts import FeatureFactoryV4
from ml_signal_v4 import SubSignalBuilder
from ml_cross_market import CrossMarketFetcher  # Phase 6

# ── 模型注册表 (Phase 8) ──
MODEL_DIR = Path(__file__).parent / "data" / "models"


@dataclass
class ModelVersion:
    """单个模型版本"""
    theme_id: str
    theme_name: str
    trained_at: str             # ISO timestamp
    data_start: str             # 训练数据起始日期
    data_end: str               # 训练数据结束日期
    oos_ic: float
    oos_icir: float
    oos_r2: float
    oos_hit_rate: float
    n_features: int
    n_samples: int
    fwd_window: int
    file_path: str
    feature_cols: List[str] = field(default_factory=list)

    @property
    def age_days(self) -> int:
        """模型年龄 (天)"""
        try:
            trained_date = datetime.fromisoformat(self.trained_at)
            return (datetime.now() - trained_date).days
        except Exception:
            return 99

    @property
    def age_label(self) -> str:
        d = self.age_days
        if d > 60:
            return f"🔴 {d}天 (过期)"
        elif d > 30:
            return f"🟡 {d}天 (偏旧)"
        else:
            return f"🟢 {d}天"

    def to_dict(self) -> dict:
        return {
            "theme_id": self.theme_id,
            "theme_name": self.theme_name,
            "trained_at": self.trained_at,
            "data_start": self.data_start,
            "data_end": self.data_end,
            "oos_ic": round(self.oos_ic, 4),
            "oos_icir": round(self.oos_icir, 3),
            "oos_r2": round(self.oos_r2, 4),
            "oos_hit_rate": round(self.oos_hit_rate, 4),
            "n_features": self.n_features,
            "n_samples": self.n_samples,
            "fwd_window": self.fwd_window,
            "file_path": self.file_path,
            "age_days": self.age_days,
        }


class ModelRegistry:
    """
    模型版本注册表 (Phase 8)

    管理 LightGBM 模型的版本化存储:
    - 保存到 data/models/lgbm_<theme>_<timestamp>.pkl
    - 按 OOS IC 排序获取最佳版本
    - 自动清理旧版本
    """

    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = MODEL_DIR
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Optional[List[ModelVersion]] = None

    def save_version(self, theme_id: str, theme_name: str,
                     model, feature_cols: List[str],
                     oos_ic: float, oos_icir: float, oos_r2: float,
                     oos_hit_rate: float, n_samples: int,
                     fwd_window: int, data_start: str = "",
                     data_end: str = "") -> Path:
        """
        保存新版模型 (带时间戳)

        Returns:
            Path: 保存路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"lgbm_{theme_id}_{timestamp}.pkl"
        path = self.model_dir / filename

        metadata = {
            "model": model,
            "theme_id": theme_id,
            "theme_name": theme_name,
            "feature_cols": feature_cols,
            "fwd_window": fwd_window,
            "trained_at": datetime.now().isoformat(),
            "data_start": data_start,
            "data_end": data_end,
            "oos_ic": oos_ic,
            "oos_icir": oos_icir,
            "oos_r2": oos_r2,
            "oos_hit_rate": oos_hit_rate,
            "n_features": len(feature_cols),
            "n_samples": n_samples,
        }

        with open(path, "wb") as f:
            pickle.dump(metadata, f)

        # 同时保存为默认路径 (向后兼容)
        default_path = self.model_dir / f"lgbm_{theme_id}.pkl"
        with open(default_path, "wb") as f:
            pickle.dump(metadata, f)

        # 清除缓存
        self._cache = None
        return path

    def list_versions(self, theme_id: str = None) -> List[ModelVersion]:
        """列出模型版本 (可选按theme_id过滤)"""
        if self._cache is not None:
            versions = self._cache
        else:
            versions = []
            for path in sorted(self.model_dir.glob("lgbm_*_*.pkl"), reverse=True):
                # 只匹配时间戳版本 (lgbm_<theme>_<YYYYMMDD_HHMM>.pkl)
                stem = path.stem
                if stem.count("_") < 2:
                    continue
                try:
                    with open(path, "rb") as f:
                        data = pickle.load(f)
                    versions.append(ModelVersion(
                        theme_id=data.get("theme_id", ""),
                        theme_name=data.get("theme_name", ""),
                        trained_at=data.get("trained_at", ""),
                        data_start=data.get("data_start", ""),
                        data_end=data.get("data_end", ""),
                        oos_ic=data.get("oos_ic", 0),
                        oos_icir=data.get("oos_icir", 0),
                        oos_r2=data.get("oos_r2", 0),
                        oos_hit_rate=data.get("oos_hit_rate", 0),
                        n_features=data.get("n_features", 0),
                        n_samples=data.get("n_samples", 0),
                        fwd_window=data.get("fwd_window", 5),
                        file_path=str(path),
                        feature_cols=data.get("feature_cols", []),
                    ))
                except Exception:
                    continue
            self._cache = versions

        if theme_id:
            return [v for v in versions if v.theme_id == theme_id]
        return versions

    def get_best(self, theme_id: str) -> Optional[ModelVersion]:
        """获取指定主题OOS IC最高的模型版本"""
        versions = self.list_versions(theme_id)
        if not versions:
            return None
        best = max(versions, key=lambda v: v.oos_icir)
        return best

    def get_latest(self, theme_id: str) -> Optional[ModelVersion]:
        """获取指定主题最新训练的模型版本"""
        versions = self.list_versions(theme_id)
        if not versions:
            return None
        return versions[0]  # 已按时间倒序

    def cleanup_old(self, keep_n: int = 3):
        """每个主题只保留最近N个版本 (按OOS ICIR, 保留最好的)"""
        all_versions = self.list_versions()
        themes = set(v.theme_id for v in all_versions)
        removed = 0
        for theme_id in themes:
            theme_versions = sorted(
                [v for v in all_versions if v.theme_id == theme_id],
                key=lambda v: v.oos_icir, reverse=True
            )
            for v in theme_versions[keep_n:]:
                try:
                    Path(v.file_path).unlink()
                    removed += 1
                except Exception:
                    pass
        self._cache = None
        return removed

    def get_summary(self) -> List[dict]:
        """获取所有主题的模型摘要 (最新版本)"""
        all_versions = self.list_versions()
        themes = set(v.theme_id for v in all_versions)
        summary = []
        for theme_id in sorted(themes):
            best = self.get_best(theme_id)
            latest = self.get_latest(theme_id)
            if latest:
                summary.append({
                    "theme_id": theme_id,
                    "theme_name": latest.theme_name,
                    "latest_trained_at": latest.trained_at[:10],
                    "latest_oos_icir": latest.oos_icir,
                    "best_oos_icir": best.oos_icir if best else latest.oos_icir,
                    "age_days": latest.age_days,
                    "age_label": latest.age_label,
                    "n_versions": len([v for v in all_versions if v.theme_id == theme_id]),
                })
        return summary


def compare_models(model_dir: str = None) -> str:
    """
    对比各主题最佳vs最新模型。

    Returns:
        Markdown格式对比报告
    """
    registry = ModelRegistry(model_dir)
    summary = registry.get_summary()

    if not summary:
        return "⚠️ 未找到模型版本"

    lines = [
        "## 🔄 模型对比报告",
        "",
        f"**检查时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "| 主题 | 最新日期 | 最新ICIR | 最佳ICIR | 差距 | 年龄 | 版本数 |",
        "|------|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for s in summary:
        gap = s["best_oos_icir"] - s["latest_oos_icir"]
        gap_str = f"🔴 {gap:+.3f}" if gap > 0.05 else f"🟢 {gap:+.3f}"
        lines.append(
            f"| {s['theme_name']} | {s['latest_trained_at']} | "
            f"{s['latest_oos_icir']:.3f} | {s['best_oos_icir']:.3f} | "
            f"{gap_str} | {s['age_label']} | {s['n_versions']} |"
        )

    lines.append("")
    lines.append("> 🔴 差距>0.05: 最新模型明显不如历史最佳 → 建议回滚或调查原因")
    return "\n".join(lines)


@dataclass
class ThemeTrainResult:
    """单个主题训练结果"""
    theme_id: str
    theme_name: str
    n_features: int
    n_samples: int

    # 全量OOS评估 (5折平均)
    oos_r2: float
    oos_ic: float          # Spearman
    oos_icir: float         # IC / IC_std
    oos_hit_rate: float

    # 训练集
    train_r2: float

    # Feature importance (top features)
    top_features: List[Tuple[str, float]]  # (feature_name, importance)

    # 模型
    model: Optional[lgb.LGBMRegressor] = None


class LightGBMTrainer:
    """LightGBM训练器 — 每主题一个模型"""

    def __init__(self,
                 fwd_window: int = 5,
                 n_splits: int = 5,
                 purge_pct: float = 0.1,
                 n_estimators: int = 100,
                 max_depth: int = 4,
                 num_leaves: int = 16,
                 learning_rate: float = 0.05,
                 early_stopping_rounds: int = 20,
                 min_obs: int = 100):
        self.fwd_window = fwd_window
        self.n_splits = n_splits
        self.purge_pct = purge_pct
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.min_obs = min_obs

        self.factory = FeatureFactoryV4()
        self.results: Dict[str, ThemeTrainResult] = {}

    # ── 主流程 ──
    def train_all(self, df: pd.DataFrame, verbose: bool = True) -> Dict[str, ThemeTrainResult]:
        """端到端训练7个主题模型 (含跨市场)"""
        # Phase 6: 跨市场数据增强
        if verbose:
            print("🌐 Phase 6: 拉取跨市场数据...")
        try:
            fetcher = CrossMarketFetcher(cache_ttl_hours=4)
            df = fetcher.enrich_dataframe(df, use_cache=True)
            if verbose:
                print(f"   数据源: {fetcher.available_count()}/9 可用")
                if not fetcher.is_healthy:
                    print("   ⚠️ 核心数据不可用, 跨市场特征可能为占位值")
        except Exception as e:
            if verbose:
                print(f"   ⚠️ 跨市场数据获取失败: {e}")

        # Step 1: 计算特征时序
        if verbose:
            n_themes = len(SubSignalBuilder.THEME_CONFIG)
            n_feats = len(self.factory.features)
            print(f"🧬 Step 1: 计算{n_feats}个特征时序...")
        ts_df = self.factory.compute_timeseries(df, verbose=verbose)

        # Step 2: 计算前向收益
        close = df["close"].values
        n = len(close)
        fwd_rets = np.zeros(n)
        fwd_rets[:n-self.fwd_window] = (
            close[self.fwd_window:] / close[:n-self.fwd_window] - 1
        )

        if verbose:
            print(f"\n🎯 Step 2: 前向收益 (fwd={self.fwd_window}d) "
                  f"— 均值={fwd_rets.mean():.4f}, 标准差={fwd_rets.std():.4f}")

        # Step 3: 按主题分组特征
        if verbose:
            print(f"\n📂 Step 3: 按{n_themes}个主题分组特征...")
        theme_features = self._get_theme_feature_cols(ts_df)

        # Step 4: 逐主题训练
        if verbose:
            print(f"\n🤖 Step 4: 逐主题训练LightGBM (PurgedKFold {self.n_splits}折)...\n")

        for theme_id, config in SubSignalBuilder.THEME_CONFIG.items():
            feat_cols = theme_features.get(theme_id, [])
            if len(feat_cols) < 3:
                if verbose:
                    print(f"  ⚠️ {theme_id} ({config['name']}): 特征不足 ({len(feat_cols)}), 跳过")
                continue

            result = self._train_theme(theme_id, config["name"], ts_df, feat_cols, fwd_rets, verbose)
            self.results[theme_id] = result

        # Step 5: 全量训练最终模型
        if verbose:
            print(f"\n📦 Step 5: 全量训练最终模型并保存...")
        self._train_final_models(ts_df, theme_features, fwd_rets, verbose)

        return self.results

    def _get_theme_feature_cols(self, ts_df: pd.DataFrame) -> Dict[str, List[str]]:
        """获取每个主题可用的特征列 (基于SubSignalBuilder规则)"""
        # 先用一个临时 SubSignalBuilder 分配特征
        from feature_backtest_v4 import FeatureTestResultV4

        # 创建模拟回测结果 (只用特征名, 全标记passed)
        mock_results = []
        for col in ts_df.columns:
            mock_results.append(FeatureTestResultV4(
                feature_id=col, feature_name=col, category="?",
                ic=0.1, ic_std=0.1, icir=1.0, t_stat=2.0, p_value=0.01,
                fdr_adjusted_p=0.01, hit_rate=0.55, sharpe=0.5,
                long_sharpe=0.5, short_sharpe=0.5, stability=1.0,
                ic_decay={}, nonlinear_r2=0, n_obs=100, passed=True,
            ))

        builder = SubSignalBuilder(mock_results)
        theme_features = {}

        for theme_id in SubSignalBuilder.THEME_CONFIG:
            feats_with_weights = builder.theme_features.get(theme_id, [])
            # 只保留在ts_df中存在的列
            cols = [fid for fid, w in feats_with_weights if fid in ts_df.columns]
            if len(cols) < 3:
                # 回退: 用特征名关键词匹配
                config = SubSignalBuilder.THEME_CONFIG[theme_id]
                cols = [
                    c for c in ts_df.columns
                    if config["feature_filter"](c, None)
                ]
            theme_features[theme_id] = cols

        return theme_features

    def _train_theme(self, theme_id: str, theme_name: str,
                     ts_df: pd.DataFrame, feat_cols: List[str],
                     fwd_rets: np.ndarray, verbose: bool) -> ThemeTrainResult:
        """PurgedKFold训练+评估一个主题"""
        X_full = ts_df[feat_cols].values
        y_full = fwd_rets.copy()
        n = len(y_full)

        # 清理NaN
        valid_mask = ~(np.isnan(X_full).any(axis=1) | np.isnan(y_full))
        X = X_full[valid_mask]
        y = y_full[valid_mask]

        if len(X) < self.min_obs:
            return ThemeTrainResult(
                theme_id=theme_id, theme_name=theme_name,
                n_features=len(feat_cols), n_samples=len(X),
                oos_r2=0, oos_ic=0, oos_icir=0, oos_hit_rate=0,
                train_r2=0, top_features=[],
            )

        # PurgedKFold
        split_size = len(X) // self.n_splits
        purge = max(1, int(split_size * self.purge_pct))

        oos_preds = np.zeros(len(X))
        train_preds_all = np.zeros(len(X))
        importances = []

        for fold in range(self.n_splits):
            test_start = fold * split_size
            test_end = min(test_start + split_size, len(X))
            train_end = max(0, test_start - purge)

            if test_end - test_start < 20 or train_end < 50:
                continue

            X_train = X[:train_end]
            y_train = y[:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]

            try:
                model = lgb.LGBMRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    num_leaves=self.num_leaves,
                    learning_rate=self.learning_rate,
                    verbose=-1,
                    random_state=42 + fold,
                    force_col_wise=True,
                )
                model.fit(X_train, y_train,
                         eval_set=[(X_test, y_test)],
                         eval_metric="rmse")

                oos_preds[test_start:test_end] = model.predict(X_test)
                if fold == 0:
                    train_preds_all[:train_end] = model.predict(X_train)

                # Feature importance (first fold only)
                if fold == 0:
                    imp = model.feature_importances_
                    top_idx = np.argsort(imp)[-10:][::-1]
                    importances = [(feat_cols[i], float(imp[i])) for i in top_idx]

            except Exception as e:
                if verbose:
                    print(f"    ⚠️ Fold {fold} 失败: {e}")

        # 评估
        valid_oos = oos_preds != 0
        oos_pred_valid = oos_preds[valid_oos]
        y_valid = y[valid_oos]

        if len(oos_pred_valid) < 20:
            return ThemeTrainResult(
                theme_id=theme_id, theme_name=theme_name,
                n_features=len(feat_cols), n_samples=len(X),
                oos_r2=0, oos_ic=0, oos_icir=0, oos_hit_rate=0,
                train_r2=0, top_features=importances,
            )

        # R²
        ss_res = np.sum((y_valid - oos_pred_valid) ** 2)
        ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
        oos_r2 = max(-1, 1 - ss_res / (ss_tot + 1e-9))

        # Rank IC
        try:
            from scipy import stats
            oos_ic, _ = stats.spearmanr(oos_pred_valid, y_valid)
            if np.isnan(oos_ic):
                oos_ic = 0
        except Exception:
            oos_ic = 0

        # ICIR (across folds)
        fold_ics = []
        for fold in range(self.n_splits):
            ts = fold * split_size
            te = min(ts + split_size, len(X))
            if te - ts >= 20:
                fold_pred = oos_preds[ts:te]
                fold_y = y[ts:te]
                mask = fold_pred != 0
                if mask.sum() > 10:
                    try:
                        ic_f, _ = stats.spearmanr(fold_pred[mask], fold_y[mask])
                        if not np.isnan(ic_f):
                            fold_ics.append(ic_f)
                    except Exception:
                        pass
        oos_icir = np.mean(fold_ics) / (np.std(fold_ics) + 1e-9) if len(fold_ics) > 1 else 0
        oos_icir = float(np.clip(oos_icir, -10, 10))

        # Hit rate
        oos_hit_rate = np.mean(np.sign(oos_pred_valid) == np.sign(y_valid))

        # Train R² (from first fold)
        train_mask = train_preds_all != 0
        if train_mask.sum() > 10:
            tr = train_preds_all[train_mask]
            ty = y[train_mask]
            train_r2 = max(-1, 1 - np.sum((ty - tr)**2) / (np.sum((ty - np.mean(ty))**2) + 1e-9))
        else:
            train_r2 = 0

        if verbose:
            status = "✅" if abs(oos_icir) > 0.3 else "⚠️"
            print(f"  {status} {theme_id:20s} | "
                  f"特征:{len(feat_cols):3d} | OOS R²={oos_r2:+.3f} | "
                  f"IC={oos_ic:+.3f} | ICIR={oos_icir:+.2f} | "
                  f"Hit={oos_hit_rate:.1%} | Train R²={train_r2:+.3f}")

        return ThemeTrainResult(
            theme_id=theme_id,
            theme_name=theme_name,
            n_features=len(feat_cols),
            n_samples=len(X),
            oos_r2=round(oos_r2, 4),
            oos_ic=round(oos_ic, 4),
            oos_icir=round(oos_icir, 3),
            oos_hit_rate=round(oos_hit_rate, 4),
            train_r2=round(train_r2, 4),
            top_features=importances,
        )

    def _train_final_models(self, ts_df: pd.DataFrame,
                            theme_features: Dict[str, List[str]],
                            fwd_rets: np.ndarray,
                            verbose: bool,
                            data_start: str = "", data_end: str = ""):
        """全量训练最终模型并保存 (Phase 8: 版本化存储)"""
        model_dir = Path(__file__).parent / "data" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        registry = ModelRegistry(model_dir)
        imp_data = {}

        for theme_id, feat_cols in theme_features.items():
            X = ts_df[feat_cols].values
            y = fwd_rets.copy()

            valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X, y = X[valid], y[valid]

            if len(X) < self.min_obs:
                continue

            model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                num_leaves=self.num_leaves,
                learning_rate=self.learning_rate,
                verbose=-1,
                random_state=42,
                force_col_wise=True,
            )
            model.fit(X, y)

            # 特征重要性 (从最终全量模型)
            imp = model.feature_importances_
            top_idx = np.argsort(imp)[-10:][::-1]
            top_features = [(feat_cols[i], float(imp[i])) for i in top_idx if imp[i] > 0]

            # 更新结果
            if theme_id in self.results:
                self.results[theme_id].model = model
                self.results[theme_id].top_features = top_features

            imp_data[theme_id] = [
                {"feature": name, "importance": imp_val}
                for name, imp_val in top_features
            ]

            # 获取该主题的OOS指标
            theme_result = self.results.get(theme_id)
            oos_ic = theme_result.oos_ic if theme_result else 0
            oos_icir = theme_result.oos_icir if theme_result else 0
            oos_r2 = theme_result.oos_r2 if theme_result else 0
            oos_hit = theme_result.oos_hit_rate if theme_result else 0
            theme_name = SubSignalBuilder.THEME_CONFIG.get(theme_id, {}).get("name", theme_id)
            n_samples = len(X)

            # Phase 8: 版本化保存 (带时间戳)
            path = registry.save_version(
                theme_id=theme_id,
                theme_name=theme_name,
                model=model,
                feature_cols=feat_cols,
                oos_ic=oos_ic,
                oos_icir=oos_icir,
                oos_r2=oos_r2,
                oos_hit_rate=oos_hit,
                n_samples=n_samples,
                fwd_window=self.fwd_window,
                data_start=data_start,
                data_end=data_end,
            )

            if verbose:
                top3_str = ", ".join([f"{n}({v:.3f})" for n, v in top_features[:3]])
                print(f"  💾 保存: {path.name} ({len(feat_cols)} 特征) | Top3: {top3_str}")

        # 清理旧版本 (每主题保留最佳5个)
        removed = registry.cleanup_old(keep_n=5)
        if verbose and removed > 0:
            print(f"  🧹 清理 {removed} 个旧版本")

        # 保存特征重要性JSON
        imp_path = Path(__file__).parent / "data" / "lgbm_feature_importance.json"
        with open(imp_path, "w") as f:
            json.dump(imp_data, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"  💾 特征重要性: {imp_path}")

    def retrain_if_needed(self, max_age_days: int = 30,
                          df: pd.DataFrame = None,
                          verbose: bool = True) -> bool:
        """
        检查模型年龄，超龄自动重训 (Phase 8)

        Args:
            max_age_days: 最大允许年龄 (默认30天)
            df: OHLCV数据 (若为None则自动拉取)
            verbose: 打印详细信息

        Returns:
            True if retrained, False if skipped
        """
        registry = ModelRegistry()
        summary = registry.get_summary()

        if not summary:
            if verbose:
                print("📦 无现有模型, 开始初始训练...")
            if df is not None:
                self.train_all(df, verbose=verbose)
                return True
            return False

        # 检查年龄
        max_age = max(s["age_days"] for s in summary) if summary else 0
        stale_themes = [s for s in summary if s["age_days"] > max_age_days]

        if not stale_themes:
            if verbose:
                print(f"✅ 所有模型年龄 < {max_age_days}天, 无需重训")
            return False

        if verbose:
            stale_names = ", ".join(s["theme_id"] for s in stale_themes)
            print(f"⚠️ {len(stale_themes)}/{len(summary)} 模型超龄 (> {max_age_days}天): {stale_names}")
            print(f"🔄 触发自动重训...")

        if df is None:
            if verbose:
                print("📊 拉取最新数据...")
            try:
                import ccxt
                exchange = ccxt.binance({"enableRateLimit": True, "timeout": 15000})
                ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1d", limit=400)
                df = pd.DataFrame(ohlcv, columns=["date", "open", "high", "low", "close", "volume"])
                df["date"] = pd.to_datetime(df["date"], unit="ms")
            except Exception as e:
                if verbose:
                    print(f"❌ 数据拉取失败: {e}")
                return False

        self.train_all(df, verbose=verbose)

        if verbose:
            print("✅ 自动重训完成!")
            print(compare_models())

        return True

    def report(self) -> str:
        """生成Markdown训练报告"""
        lines = [
            "## 🤖 LightGBM 训练报告",
            "",
            f"**训练时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**前向窗口**: {self.fwd_window}天",
            f"**CV**: PurgedKFold ({self.n_splits}折, purge={self.purge_pct:.0%})",
            f"**超参**: n_estimators={self.n_estimators}, max_depth={self.max_depth}, "
            f"num_leaves={self.num_leaves}, lr={self.learning_rate}",
            "",
            "| 主题 | 特征数 | 样本 | OOS R² | OOS IC | OOS ICIR | Hit Rate | Train R² |",
            "|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]

        for theme_id, r in self.results.items():
            icon = "✅" if abs(r.oos_icir) > 0.3 else "⚠️" if abs(r.oos_icir) > 0.1 else "❌"
            lines.append(
                f"| {icon} {r.theme_name} | {r.n_features} | {r.n_samples} | "
                f"{r.oos_r2:+.3f} | {r.oos_ic:+.3f} | {r.oos_icir:+.2f} | "
                f"{r.oos_hit_rate:.1%} | {r.train_r2:+.3f} |"
            )

        lines.append("")
        lines.append("### 🔝 各主题 Top-5 特征重要性")
        lines.append("")

        for theme_id, r in self.results.items():
            lines.append(f"**{r.theme_name}** ({theme_id}):")
            for name, imp in r.top_features[:5]:
                lines.append(f"  - `{name}`: {imp:.4f}")
            lines.append("")

        return "\n".join(lines)

    def save_report(self):
        """保存训练报告CSV"""
        out = Path(__file__).parent / "data" / "lgbm_train_report.csv"
        data = []
        for theme_id, r in self.results.items():
            data.append({
                "theme_id": theme_id,
                "theme_name": r.theme_name,
                "n_features": r.n_features,
                "n_samples": r.n_samples,
                "oos_r2": r.oos_r2,
                "oos_ic": r.oos_ic,
                "oos_icir": r.oos_icir,
                "oos_hit_rate": r.oos_hit_rate,
                "train_r2": r.train_r2,
            })
        pd.DataFrame(data).to_csv(out, index=False)
        print(f"📄 报告保存: {out}")


class LightGBMSignalPredictor:
    """加载LightGBM模型 → 对最新数据预测子信号收益 (Phase 8: 支持ModelRegistry)"""

    def __init__(self, model_dir: str = None, use_best: bool = True):
        """
        Args:
            model_dir: 模型目录
            use_best: True=加载每个主题ICIR最高的版本, False=加载默认路径版本
        """
        if model_dir is None:
            model_dir = Path(__file__).parent / "data" / "models"
        self.model_dir = Path(model_dir)
        self.models: Dict[str, dict] = {}  # theme_id → {model, feature_cols, ...}
        self.is_loaded = False
        self.use_best = use_best
        self._registry = ModelRegistry(model_dir)

    def load_models(self) -> bool:
        """加载所有模型 (优先从Registry加载最佳版本)"""
        if not self.model_dir.exists():
            return False

        loaded = 0

        if self.use_best:
            # Phase 8: 从Registry加载每个主题的最佳版本
            all_versions = self._registry.list_versions()
            themes = set(v.theme_id for v in all_versions)
            for theme_id in themes:
                best = self._registry.get_best(theme_id)
                if best and Path(best.file_path).exists():
                    try:
                        with open(best.file_path, "rb") as f:
                            self.models[theme_id] = pickle.load(f)
                        loaded += 1
                    except Exception:
                        continue

        # 回退: 加载默认路径模型 (兼容旧版)
        if loaded == 0:
            for path in self.model_dir.glob("lgbm_*.pkl"):
                # 跳过时间戳版本 (由上面处理)
                stem = path.stem
                if stem.count("_") >= 2 and len(stem.split("_")[-1]) == 13:
                    continue
                theme_id = stem.replace("lgbm_", "")
                if theme_id in self.models:
                    continue
                try:
                    with open(path, "rb") as f:
                        self.models[theme_id] = pickle.load(f)
                    loaded += 1
                except Exception:
                    continue

        self.is_loaded = loaded > 0
        return self.is_loaded

    def predict(self, feature_values: Dict[str, float]) -> Dict[str, float]:
        """
        对最新特征值预测每个主题的期望收益

        Args:
            feature_values: {feature_id: value} from compute_latest()

        Returns:
            {theme_id: predicted_forward_return}
        """
        predictions = {}

        for theme_id, data in self.models.items():
            model = data["model"]
            feat_cols = data["feature_cols"]

            # 构建特征向量
            X = np.array([[feature_values.get(c, 0.0) for c in feat_cols]])
            X = np.nan_to_num(X, nan=0.0)

            try:
                pred = float(model.predict(X)[0])
                predictions[theme_id] = pred
            except Exception:
                predictions[theme_id] = 0.0

        return predictions

    def available_themes(self) -> List[str]:
        return list(self.models.keys())


# ── CLI ──
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chase量化策略 · LightGBM训练器")
    parser.add_argument("--retrain", action="store_true",
                       help="自动重训 (检查模型年龄)")
    parser.add_argument("--compare", action="store_true",
                       help="对比最佳vs最新模型")
    parser.add_argument("--max-age", type=int, default=30,
                       help="最大模型年龄/天 (默认30)")
    parser.add_argument("--n-splits", type=int, default=5,
                       help="CV折数 (默认5)")
    parser.add_argument("--fwd-window", type=int, default=5,
                       help="前向窗口 (默认5)")
    args = parser.parse_args()

    print("=" * 60)
    print("🤖 LightGBM Trainer v2.0 — 非线性子信号建模 + 版本管理")
    print("=" * 60)

    if args.compare:
        # 仅对比模式
        print(compare_models())
        import sys
        sys.exit(0)

    if args.retrain:
        # 自动重训模式
        print(f"🔄 自动重训模式 (最大年龄: {args.max_age}天)")
        trainer = LightGBMTrainer(fwd_window=args.fwd_window, n_splits=args.n_splits)
        did_retrain = trainer.retrain_if_needed(max_age_days=args.max_age, verbose=True)
        if not did_retrain:
            print("💤 无需重训, 模型年龄正常")
        import sys
        sys.exit(0)

    # 默认: 全量训练
    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True, "timeout": 15000})
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1d", limit=400)
        df = pd.DataFrame(ohlcv, columns=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"], unit="ms")
        print(f"📊 BTC/USDT | {len(df)}天\n")

        trainer = LightGBMTrainer(fwd_window=args.fwd_window, n_splits=args.n_splits)
        results = trainer.train_all(df, verbose=True)

        print()
        print(trainer.report())
        trainer.save_report()

        # 测试预测器
        print("\n🧪 测试预测器...")
        predictor = LightGBMSignalPredictor()
        if predictor.load_models():
            latest = trainer.factory.compute_latest(df)
            preds = predictor.predict(latest)
            for theme_id, pred in preds.items():
                name = SubSignalBuilder.THEME_CONFIG.get(theme_id, {}).get("name", theme_id)
                icon = "🟢" if pred > 0 else "🔴" if pred < 0 else "⚪"
                print(f"  {icon} {name}: 预测收益={pred:+.4f}")

        # Phase 8: 显示模型版本摘要
        print("\n📦 模型版本摘要:")
        summary = ModelRegistry().get_summary()
        for s in summary:
            print(f"  {s['theme_name']:20s} | 最新: {s['latest_trained_at']} | "
                  f"ICIR: {s['latest_oos_icir']:.3f} | {s['age_label']} | "
                  f"{s['n_versions']}版本")

    except ImportError as e:
        print(f"⚠️ 缺少依赖: {e}")
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
