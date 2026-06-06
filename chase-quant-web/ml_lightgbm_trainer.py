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
                            verbose: bool):
        """全量训练最终模型并保存"""
        model_dir = Path(__file__).parent / "data" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

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

            # 保存
            path = model_dir / f"lgbm_{theme_id}.pkl"
            with open(path, "wb") as f:
                pickle.dump({
                    "model": model,
                    "theme_id": theme_id,
                    "feature_cols": feat_cols,
                    "feature_importance": top_features,
                    "fwd_window": self.fwd_window,
                    "trained_at": datetime.now().isoformat(),
                }, f)

            if verbose:
                top3_str = ", ".join([f"{n}({v:.3f})" for n, v in top_features[:3]])
                print(f"  💾 保存: {path} ({len(feat_cols)} 特征) | Top3: {top3_str}")

        # 保存特征重要性JSON
        imp_path = Path(__file__).parent / "data" / "lgbm_feature_importance.json"
        with open(imp_path, "w") as f:
            json.dump(imp_data, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"  💾 特征重要性: {imp_path}")

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
    """加载6个LightGBM模型 → 对最新数据预测子信号收益"""

    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = Path(__file__).parent / "data" / "models"
        self.model_dir = Path(model_dir)
        self.models: Dict[str, dict] = {}  # theme_id → {model, feature_cols, ...}
        self.is_loaded = False

    def load_models(self) -> bool:
        """加载所有已保存的模型"""
        if not self.model_dir.exists():
            return False

        loaded = 0
        for path in self.model_dir.glob("lgbm_*.pkl"):
            theme_id = path.stem.replace("lgbm_", "")
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
    print("=" * 60)
    print("🤖 LightGBM Trainer v1.0 — 非线性子信号建模")
    print("=" * 60)

    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True, "timeout": 15000})
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1d", limit=400)
        df = pd.DataFrame(ohlcv, columns=["date", "open", "high", "low", "close", "volume"])
        print(f"📊 BTC/USDT | {len(df)}天\n")

        trainer = LightGBMTrainer(fwd_window=5, n_splits=5)
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

    except ImportError as e:
        print(f"⚠️ 缺少依赖: {e}")
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
