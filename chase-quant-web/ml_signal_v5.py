"""
ML Signal Engine v5.0 — Qlib 深度学习 + LightGBM 融合
======================================================
Chase量化策略 Phase 9: 统一信号引擎, 同时支持传统 LightGBM 和 Qlib 模型

架构升级 (vs v4.0):
  v4.0: 286个特征 → 7个子信号 × LightGBM → 组合信号
  v5.0: 286个特征 → 7个子信号 × (LightGBM + ALSTM + Transformer + TabNet) → 模型融合 → 最终信号

模型融合策略:
  - 加权平均: weight by OOS ICIR
  - 投票机制: 多数模型方向一致 → 高置信度
  - 分歧检测: 模型间分歧过大 → 降置信度

使用:
  from ml_signal_v5 import MLSignalEngineV5
  engine = MLSignalEngineV5()
  signal = engine.generate_signal(df_btc, "BTC/USDT")
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings("ignore")

from ml_signal_v4 import (
    MLSignalEngineV4, SubSignalBuilder, SubSignal, EnsembleSignal,
)
from feature_ts import FeatureFactoryV4
from qlib_trainer import QlibTrainer, QlibPredictor

# Asset Graph (Phase 11)
try:
    from asset_graph import CrossAssetGATPredictor, AssetGraphBuilder
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False

# Alpha Mining (Phase 12)
try:
    from alpha_miner import (AlphaStore, AlphaExpressionParser, AlphaEvaluator,
                              evaluate_expression)
    ALPHA_AVAILABLE = True
except ImportError:
    ALPHA_AVAILABLE = False

DATA_DIR = Path(__file__).parent / "data"


@dataclass
class ModelPrediction:
    """单个模型的预测"""
    model_name: str            # lgbm / qlib_alstm / qlib_transformer / qlib_tabnet
    theme_id: str
    theme_name: str
    prediction: float          # 预测的前向收益
    direction: str             # LONG/SHORT/NEUTRAL
    confidence: float          # 基于OOS ICIR的置信度
    oos_icir: float
    weight: float              # 融合权重


@dataclass
class FusionSignal:
    """多模型融合信号"""
    timestamp: str
    symbol: str
    price: float

    # 各模型预测
    model_predictions: List[ModelPrediction]

    # 融合结果
    signal_weighted: float     # ICIR加权
    signal_equal: float        # 等权
    signal_consensus: float    # 共识投票

    # 分歧度
    divergence: float          # 模型间标准差, 越高越不确定
    consensus_ratio: float     # 方向一致比例

    # 最终决策
    action: str                # BUY / SELL / HOLD
    confidence: float          # 0-1
    suggested_size_pct: float

    # 元信息
    n_models_active: int
    n_themes_active: int
    models_available: List[str]

    # 兼容 v4
    signal_lgbm: float = 0.0
    lgbm_available: bool = False
    feature_count: int = 0


class MLSignalEngineV5:
    """
    统一信号引擎 v5.0 — LightGBM + Qlib 深度学习融合

    数据流:
      1. FeatureFactoryV4 → 全时序特征矩阵
      2. 对每个主题, 收集所有可用模型的预测
      3. 按 OOS ICIR 加权融合
      4. 评估模型间一致性和分歧
      5. 输出最终交易信号
    """

    # 信号 → 操作阈值 (动态调整，平衡信号频率与质量)
    # 🆕 v3: BUY/SELL对称阈值 + 极端恐惧时via auto_trade.py动态调整
    ACTION_THRESHOLDS = {
        "BUY": 0.06,    # 买入阈值 (F&G动态调整)
        "SELL": -0.06,  # 🆕 做空阈值与做多对称 (原-0.08太保守)
    }

    def __init__(self, use_qlib: bool = True, use_lgbm: bool = True,
                 qlib_models: Optional[List[str]] = None,
                 min_models_for_consensus: int = 2,
                 use_graph: bool = True,
                 use_alphas: bool = False):
        """
        Args:
            use_qlib: 是否加载 Qlib 深度学习模型
            use_lgbm: 是否使用 LightGBM (基础基线)
            qlib_models: Qlib 模型列表 (None = all available)
            min_models_for_consensus: 最少模型数才计算共识
            use_graph: 是否使用资产关系图增强 (Phase 11)
            use_alphas: 是否使用自动Alpha挖掘增强 (Phase 12)
        """
        self.use_qlib = use_qlib
        self.use_lgbm = use_lgbm
        self.min_models = min_models_for_consensus
        self.use_graph = use_graph and GRAPH_AVAILABLE
        self.use_alphas = use_alphas and ALPHA_AVAILABLE

        # 基础引擎 (v4)
        self.v4_engine = MLSignalEngineV4()
        self.feature_factory = FeatureFactoryV4()
        self.theme_config = SubSignalBuilder.THEME_CONFIG  # 类变量直接引用

        # Qlib 推理器
        self.qlib_predictor = None
        if use_qlib:
            try:
                self.qlib_predictor = QlibPredictor(models_to_use=qlib_models)
                print(f"🧠 Qlib 推理器已加载")
            except Exception as e:
                print(f"⚠️ Qlib 推理器加载失败: {e}")

        # 资产关系图 (Phase 11)
        self.graph_predictor = None
        self.graph_builder = None
        if self.use_graph:
            try:
                self.graph_predictor = CrossAssetGATPredictor()
                self.graph_builder = AssetGraphBuilder()
                print(f"🔗 资产关系图引擎已加载")
            except Exception as e:
                print(f"⚠️ 图引擎加载失败: {e}")
                self.use_graph = False

        # Alpha挖掘 (Phase 12)
        self.alpha_store = None
        self.alpha_parser = None
        if self.use_alphas:
            try:
                self.alpha_store = AlphaStore()
                self.alpha_parser = AlphaExpressionParser()
                n_alphas = len(self.alpha_store.load("latest"))
                print(f"🔬 Alpha挖掘引擎已加载 (已发现 {n_alphas} 个Alpha)")
            except Exception as e:
                print(f"⚠️ Alpha引擎加载失败: {e}")
                self.use_alphas = False

        # 加载模型权重 (基于 OOS ICIR)
        self._model_weights = self._load_model_weights()

    # ── Alpha Mining (Phase 12) ──

    def compute_alpha_features(self, df: pd.DataFrame,
                                n_top: int = 20) -> Optional[np.ndarray]:
        """
        计算top-N已发现Alpha的特征值矩阵。

        Args:
            df: OHLCV DataFrame
            n_top: 使用前N个Alpha

        Returns:
            (n_timesteps, n_alphas) feature matrix, or None
        """
        if not self.use_alphas or not self.alpha_store:
            return None

        alphas = self.alpha_store.get_top(n_top, min_icir=0.1)
        if not alphas:
            return None

        # Build data dictionary
        data = {}
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                data[col] = df[col].values.astype(float)

        features = []
        for a in alphas:
            try:
                ts = evaluate_expression(a.expression, data=data)
                features.append(ts)
            except Exception:
                features.append(np.full(len(df), np.nan))

        if not features:
            return None

        return np.column_stack(features)

    def generate_alpha_signal(self, df: pd.DataFrame, symbol: str,
                               n_top: int = 20) -> Optional[FusionSignal]:
        """
        纯Alpha驱动的信号生成 (不使用ML模型)。

        使用top-N已发现Alpha的加权组合产生交易信号。

        Args:
            df: OHLCV DataFrame
            symbol: 交易对名称
            n_top: 使用前N个Alpha

        Returns:
            FusionSignal or None
        """
        if not self.use_alphas or not self.alpha_store:
            return None

        alphas = self.alpha_store.get_top(n_top, min_icir=0.1)
        if not alphas:
            return None

        # Build data dictionary
        data = {}
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                data[col] = df[col].values.astype(float)

        # Compute each alpha's latest value and z-score
        alpha_values = []
        for a in alphas:
            try:
                ts = evaluate_expression(a.expression, data=data)
                valid = ts[~np.isnan(ts)]
                if len(valid) < 30:
                    continue
                # Z-score of latest value relative to history
                latest = valid[-1]
                mu = np.mean(valid)
                sigma = np.std(valid)
                z = (latest - mu) / (sigma + 1e-9)
                alpha_values.append((a, z, latest))
            except Exception:
                continue

        if not alpha_values:
            return None

        # Weight by |ICIR|, normalize
        total_weight = sum(abs(a.icir) for (a, _, _) in alpha_values)
        if total_weight <= 0:
            return None

        weighted_signal = sum(z * abs(a.icir) for (a, z, _) in alpha_values) / total_weight

        # Determine action
        if weighted_signal > 1.0:
            action = "BUY"
        elif weighted_signal < -1.0:
            action = "SELL"
        else:
            action = "HOLD"

        confidence = min(1.0, abs(weighted_signal) / 2.0)
        price = float(df["close"].iloc[-1])

        # Build simple model predictions list
        model_preds = []
        for a, z, raw in alpha_values[:5]:
            model_preds.append(ModelPrediction(
                model_name=f"alpha",
                theme_id=a.category,
                theme_name=a.name[:40],
                prediction=z / 3.0,
                direction="LONG" if z > 0 else "SHORT",
                confidence=min(1.0, abs(z) / 2.0),
                oos_icir=abs(a.icir),
                weight=abs(a.icir),
            ))

        return FusionSignal(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            price=price,
            model_predictions=model_preds,
            signal_weighted=weighted_signal / 3.0,  # scale to [-1, 1]
            signal_equal=weighted_signal / 3.0,
            signal_consensus=weighted_signal / 3.0,
            divergence=np.std([z for _, z, _ in alpha_values[:10]]) if len(alpha_values) > 1 else 0.0,
            consensus_ratio=1.0,
            action=action,
            confidence=confidence,
            suggested_size_pct=min(0.4, abs(weighted_signal) / 10.0),
            n_models_active=len(alpha_values),
            n_themes_active=len(set(a.category for a, _, _ in alpha_values)),
            models_available=[f"alpha_{a.name[:20]}" for a, _, _ in alpha_values[:5]],
            signal_lgbm=0.0,
            lgbm_available=False,
            feature_count=len(alpha_values),
        )

    def get_alpha_status(self) -> dict:
        """获取Alpha挖掘状态"""
        if not self.use_alphas or not self.alpha_store:
            return {"available": False, "reason": "Alpha mining not enabled"}
        alphas = self.alpha_store.get_top(50)
        passed = [a for a in alphas if a.passed]
        by_cat = {}
        for a in alphas:
            by_cat[a.category] = by_cat.get(a.category, 0) + 1
        return {
            "available": True,
            "n_total": len(self.alpha_store.load("latest")),
            "n_top": len(alphas),
            "n_passed": len(passed),
            "best_icir": max(abs(a.icir) for a in alphas) if alphas else 0.0,
            "best_expr": alphas[0].expression[:60] if alphas else "",
            "by_category": by_cat,
        }

    def generate_signal(self, df_btc: pd.DataFrame, symbol: str,
                        use_lgbm: bool = True) -> FusionSignal:
        """
        生成融合交易信号

        Args:
            df_btc: OHLCV DataFrame
            symbol: 交易对名称
            use_lgbm: 是否使用 LightGBM

        Returns:
            FusionSignal
        """
        # 1. 获取 v4 LightGBM 信号 (作为基线)
        lgbm_signal = None
        if self.use_lgbm and use_lgbm:
            try:
                lgbm_signal = self.v4_engine.generate_signal(df_btc, symbol, use_lgbm=True)
            except Exception as e:
                print(f"  ⚠️ LightGBM信号生成失败: {e}")

        # 2. Qlib 模型预测 (每个主题 × 每个模型)
        all_predictions: List[ModelPrediction] = []

        if self.use_qlib and self.qlib_predictor:
            for theme_id, theme_info in self.theme_config.items():
                theme_name = theme_info.get("name", theme_id)

                # 多模型集成预测
                ensemble_preds = self.qlib_predictor.ensemble_predict(df_btc, theme_id)
                if not ensemble_preds:
                    continue

                for model_name, pred_val in ensemble_preds.items():
                    if model_name == "ensemble_mean":
                        continue

                    direction = (
                        "LONG" if pred_val > 0.001 else
                        "SHORT" if pred_val < -0.001 else
                        "NEUTRAL"
                    )

                    # 权重 = OOS ICIR
                    oos_icir = self._model_weights.get(f"{model_name}_{theme_id}", 0.3)
                    weight = max(0.1, oos_icir)  # 最小权重0.1

                    all_predictions.append(ModelPrediction(
                        model_name=model_name,
                        theme_id=theme_id,
                        theme_name=theme_name,
                        prediction=pred_val,
                        direction=direction,
                        confidence=min(1.0, abs(pred_val) * 20),  # 缩放
                        oos_icir=oos_icir,
                        weight=weight,
                    ))

        # 3. 将 LightGBM 子信号转换为 ModelPrediction 格式
        if lgbm_signal and lgbm_signal.sub_signals:
            for sub in lgbm_signal.sub_signals:
                # 找到对应的 theme_id
                theme_id = self._find_theme_id(sub.name)
                pred_val = sub.value / 3.0  # 缩放到 [-1, +1]

                all_predictions.append(ModelPrediction(
                    model_name="lgbm",
                    theme_id=theme_id or sub.name.lower().replace(" ", "_"),
                    theme_name=sub.name,
                    prediction=pred_val,
                    direction=sub.direction,
                    confidence=sub.confidence,
                    oos_icir=0.3,  # LightGBM 基础权重
                    weight=0.3,
                ))

        # 4. 融合
        if not all_predictions:
            return self._empty_signal(symbol, df_btc)

        # ICIR 加权融合
        total_weight = sum(p.weight for p in all_predictions)
        if total_weight > 0:
            signal_weighted = sum(p.prediction * p.weight for p in all_predictions) / total_weight
        else:
            signal_weighted = np.mean([p.prediction for p in all_predictions])

        # 等权
        signal_equal = np.mean([p.prediction for p in all_predictions])

        # 共识: 多数方向
        directions = [p.direction for p in all_predictions]
        long_count = sum(1 for d in directions if d == "LONG")
        short_count = sum(1 for d in directions if d == "SHORT")
        neutral_count = sum(1 for d in directions if d == "NEUTRAL")

        if long_count > short_count and long_count > neutral_count:
            consensus_dir = "LONG"
            consensus_ratio = long_count / len(directions)
        elif short_count > long_count and short_count > neutral_count:
            consensus_dir = "SHORT"
            consensus_ratio = short_count / len(directions)
        else:
            consensus_dir = "NEUTRAL"
            consensus_ratio = neutral_count / max(1, len(directions))

        signal_consensus = signal_weighted * consensus_ratio  # 共识缩放

        # 分歧度: 模型预测的标准差
        pred_values = [p.prediction for p in all_predictions]
        divergence = np.std(pred_values) if len(pred_values) > 1 else 0.0

        # 最终信号 (保留分歧惩罚用于仓位大小)
        final_signal = signal_consensus * (1.0 - min(divergence * 2, 0.5))  # 分歧惩罚

        # 确定操作 — 用信号加权值 (不受分歧惩罚衰减)
        # 🆕 极端恐惧/贪婪动态阈值调整 (v2: 2026-06-24 修正方向)
        buy_threshold = self.ACTION_THRESHOLDS["BUY"]
        sell_threshold = self.ACTION_THRESHOLDS["SELL"]
        try:
            from feature_engine import _SENTIMENT_CTX
            fg = _SENTIMENT_CTX.get("fg_value", 50)
            if fg <= 25:
                # 极端恐惧 → 收紧买入 + 放宽做空（市场恐慌时优先做空）
                buy_threshold = buy_threshold * 2.0     # 0.06→0.12 更难触发买入
                sell_threshold = sell_threshold * 0.5   # -0.06→-0.03 更容易触发做空
            elif fg <= 35:
                # 恐惧 → 买入收紧 + 做空适度放宽
                buy_threshold = buy_threshold * 1.4
                sell_threshold = sell_threshold * 0.7
            elif fg >= 75:
                # 极端贪婪 → 做空收紧
                sell_threshold = sell_threshold * 1.5
            elif fg >= 65:
                # 偏贪婪 → 做空适度收紧
                sell_threshold = sell_threshold * 1.2
        except Exception:
            pass  # 如果 sentiment context 不可用，用默认阈值

        if signal_weighted > buy_threshold:
            action = "BUY"
        elif signal_weighted < sell_threshold:
            action = "SELL"
        else:
            action = "HOLD"

        # 置信度
        n_models = len(set(p.model_name for p in all_predictions))
        confidence = consensus_ratio * (1.0 - min(divergence * 1.5, 0.5))
        confidence = min(1.0, max(0.1, confidence * (n_models / max(1, self.min_models))))

        # 建议仓位
        suggested_size = abs(final_signal) * 0.4  # 最大40%
        suggested_size = min(0.4, max(0.05, suggested_size))

        # 收集活跃模型
        active_models = list(set(p.model_name for p in all_predictions))

        price = float(df_btc["close"].iloc[-1])

        return FusionSignal(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            price=price,
            model_predictions=all_predictions,
            signal_weighted=signal_weighted,
            signal_equal=signal_equal,
            signal_consensus=signal_consensus,
            divergence=divergence,
            consensus_ratio=consensus_ratio,
            action=action,
            confidence=confidence,
            suggested_size_pct=suggested_size,
            n_models_active=len(active_models),
            n_themes_active=len(set(p.theme_id for p in all_predictions)),
            models_available=active_models,
            signal_lgbm=lgbm_signal.signal_lgbm if lgbm_signal else 0.0,
            lgbm_available=lgbm_signal is not None and lgbm_signal.lgbm_available,
            feature_count=lgbm_signal.feature_count if lgbm_signal else 0,
        )

    def _find_theme_id(self, theme_name: str) -> Optional[str]:
        """通过中文名称找到 theme_id"""
        name_map = {
            "趋势跟踪": "trend",
            "均值回归": "reversal",
            "量价确认": "volume",
            "波动突破": "vol_breakout",
            "尾部风险": "tail_risk",
            "动量增强": "momentum_enhanced",
            "跨市场联动": "cross_market",
        }
        return name_map.get(theme_name)

    def _load_model_weights(self) -> Dict[str, float]:
        """从训练报告加载模型权重 (OOS ICIR)"""
        weights = {}

        # 从 Qlib 报告
        try:
            report_path = DATA_DIR / "qlib_reports" / "qlib_train_latest.json"
            if report_path.exists():
                with open(report_path) as f:
                    report = json.load(f)
                for m in report.get("models", []):
                    key = f"{m['model_name']}_{m['theme_id']}"
                    weights[key] = max(0.1, m['oos_icir'])
        except Exception:
            pass

        # 从 LightGBM 报告
        try:
            lgbm_path = DATA_DIR / "lgbm_train_report.csv"
            if lgbm_path.exists():
                df = pd.read_csv(lgbm_path)
                for _, row in df.iterrows():
                    key = f"lgbm_{row.get('theme_id', '')}"
                    weights[key] = max(0.1, row.get('oos_icir', 0.3))
        except Exception:
            pass

        return weights

    def _empty_signal(self, symbol: str, df_btc: pd.DataFrame) -> FusionSignal:
        """返回空信号"""
        price = float(df_btc["close"].iloc[-1])
        return FusionSignal(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            price=price,
            model_predictions=[],
            signal_weighted=0.0,
            signal_equal=0.0,
            signal_consensus=0.0,
            divergence=1.0,
            consensus_ratio=0.0,
            action="HOLD",
            confidence=0.0,
            suggested_size_pct=0.0,
            n_models_active=0,
            n_themes_active=0,
            models_available=[],
        )

    def generate_cross_asset_signals(self, symbols: List[str],
                                      force_build_graph: bool = False) -> Dict[str, FusionSignal]:
        """
        多资产联合预测 — 使用资产关系图增强 (Phase 11)

        Args:
            symbols: 资产列表 (如 ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
            force_build_graph: 强制重建资产关系图

        Returns:
            {symbol: FusionSignal, ...}
        """
        if not self.use_graph:
            print("⚠️ 资产关系图不可用")
            return {}

        results = {}
        try:
            import ccxt
            exchange = ccxt.binance({"enableRateLimit": True, "timeout": 15000})

            # 1. 获取所有资产的图增强预测
            graph_preds = self.graph_predictor.predict(symbols, force_build_graph)

            # 2. 为每个资产生成基础信号 (LightGBM + Qlib)
            for symbol in symbols:
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, "1d", limit=400)
                    df = pd.DataFrame(ohlcv, columns=["date", "open", "high", "low", "close", "volume"])
                    # 基础融合信号
                    base_signal = self.generate_signal(df, symbol)

                    # 图增强: 用邻居信息调整置信度
                    if symbol in graph_preds and graph_preds[symbol] is not None:
                        graph_pred = graph_preds[symbol]
                        # 图预测与自身信号方向一致 → 提升置信度
                        direction_agree = (graph_pred > 0) == (base_signal.signal_consensus > 0)
                        graph_bonus = 1.15 if direction_agree else 0.85

                        # 修改融合信号 (图增强版)
                        base_signal.signal_consensus *= graph_bonus
                        base_signal.confidence = min(1.0, base_signal.confidence * graph_bonus)
                        base_signal.divergence *= max(0.5, 2 - graph_bonus)

                        # 更新 action
                        if base_signal.signal_consensus > self.ACTION_THRESHOLDS["BUY"]:
                            base_signal.action = "BUY"
                        elif base_signal.signal_consensus < self.ACTION_THRESHOLDS["SELL"]:
                            base_signal.action = "SELL"
                        else:
                            base_signal.action = "HOLD"

                        # 标记图增强
                        base_signal.models_available.append("graph_enhanced")

                    results[symbol] = base_signal

                except Exception as e:
                    print(f"  ⚠️ {symbol} 信号生成失败: {e}")

        except ImportError:
            print("⚠️ ccxt 未安装, 跳过跨资产预测")

        return results

    def get_graph_status(self) -> dict:
        """获取资产关系图状态"""
        if not self.use_graph or not self.graph_predictor:
            return {"available": False, "reason": "图引擎未加载"}

        snapshot = self.graph_predictor.snapshot
        if snapshot is None:
            return {"available": False, "reason": "图快照未构建"}

        return {
            "available": True,
            "n_assets": snapshot.n_assets,
            "symbols": snapshot.symbols,
            "density": snapshot.graph_density,
            "avg_degree": snapshot.avg_degree,
            "n_communities": snapshot.n_communities,
            "top_edge": snapshot.top_edges[0] if snapshot.top_edges else None,
            "created_at": snapshot.created_at,
            "n_days": snapshot.n_days,
        }

    def get_available_models(self) -> List[str]:
        """检查哪些模型可用"""
        available = []
        if self.use_lgbm:
            available.append("lgbm")
        if self.qlib_predictor:
            for model_name in ["alstm", "transformer", "tabnet", "gat"]:
                try:
                    pred = self.qlib_predictor.predict_all_themes(
                        pd.DataFrame(), model_name
                    )
                    if pred:
                        available.append(f"qlib_{model_name}")
                except Exception:
                    pass
        return available


# ── CLI测试 ──
if __name__ == "__main__":
    print("🧬 ML Signal Engine v5.0 — Qlib + LightGBM 融合")
    print()

    engine = MLSignalEngineV5(use_qlib=True, use_lgbm=True)
    available = engine.get_available_models()
    print(f"📡 可用模型: {available}")
    print(f"📊 模型权重: {len(engine._model_weights)} 个条目")

    # 快速测试 (需要 ccxt)
    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True, "timeout": 15000})
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1d", limit=400)
        df_btc = pd.DataFrame(ohlcv, columns=["date", "open", "high", "low", "close", "volume"])

        signal = engine.generate_signal(df_btc, "BTC/USDT")
        print(f"\n🎯 最终信号: {signal.action}")
        print(f"   加权信号: {signal.signal_weighted:+.4f}")
        print(f"   共识比例: {signal.consensus_ratio:.0%}")
        print(f"   置信度:   {signal.confidence:.0%}")
        print(f"   分歧度:   {signal.divergence:.4f}")
        print(f"   活跃模型: {signal.models_available}")
        print(f"   建议仓位: {signal.suggested_size_pct:.0%}")

        # 各模型预测详情
        print(f"\n📋 各模型预测:")
        for p in sorted(signal.model_predictions, key=lambda x: -abs(x.prediction)):
            print(f"   {p.model_name:20s} | {p.theme_name:10s} | "
                  f"{p.prediction:+.4f} | {p.direction:6s} | "
                  f"w={p.weight:.3f}")

    except ImportError:
        print("⚠️ ccxt 未安装, 跳过实时测试")
