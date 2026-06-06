"""
ML Signal Engine v4.0 — 特征→子信号→组合→Streamlit集成
========================================================
Chase量化策略 Phase 3-4:
  N个独立子信号 × (等权/波动率加权/IC加权) → 最终交易信号

子信号架构 (西蒙斯风格 — 每个独立, 可单独debug):
  1. 趋势跟踪    — 正动量 + 趋势强度 + 均线方向
  2. 均值回归    — 超买超卖 + 波动率极端 + 反转形态
  3. 量价确认    — 放量 + 量价同向 + OBV确认
  4. 波动率突破  — 波动率锥 + 收敛 + 波动率回归
  5. 尾部风险    — 偏度/峰度 + 极端事件 + 下行风险
  6. 动量增强    — 排名分位 + 路径质量 + 动量比

组合方式:
  - 等权: signal = mean(s1..s6)
  - IC加权: signal = Σ(s_i × |ICIR_i|) / Σ|ICIR_i|
  - 波动率加权: signal = Σ(s_i / σ_i) / Σ(1/σ_i)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

from feature_ts import FeatureFactoryV4
from feature_backtest_v4 import FeatureBacktesterV4, FeatureTestResultV4


@dataclass
class SubSignal:
    """独立子信号"""
    id: str
    name: str
    value: float              # [-3, +3]
    confidence: float         # 0-1
    contributing_features: List[str]
    direction: str            # LONG/SHORT/NEUTRAL
    reasoning: str


@dataclass
class EnsembleSignal:
    """组合信号 — 最终输出"""
    timestamp: str
    symbol: str
    price: float

    sub_signals: List[SubSignal]

    # 三种组合
    signal_equal: float       # 等权
    signal_ic: float          # IC加权
    signal_vol: float         # 波动率加权

    consensus: float          # 子信号一致性
    confidence: float

    action: str               # BUY/SELL/HOLD
    suggested_size_pct: float
    risk_adjusted: float

    # 诊断
    feature_count: int
    n_sub_signals_active: int

    # LightGBM (Phase 5, 有默认值)
    signal_lgbm: float = 0.0
    lgbm_available: bool = False


class SubSignalBuilder:
    """按主题分组特征 → 构建独立子信号"""

    THEME_CONFIG = {
        "trend": {
            "name": "趋势跟踪",
            "categories": ["A"],  # 动量
            "feature_filter": lambda fid, r: (
                "mom" in fid and "accel" not in fid and "rank" not in fid
            ) or "sharpe" in fid or "ma_slope" in fid or "trend_strength" in fid,
        },
        "reversal": {
            "name": "均值回归",
            "categories": ["D", "E", "M", "N"],
            "feature_filter": lambda fid, r: (
                "rsi" in fid or "bb_" in fid or "cci" in fid or
                "ma_dist" in fid or "body_ratio" in fid or
                "wick" in fid or "near_" in fid or "co_ratio" in fid
            ),
        },
        "volume": {
            "name": "量价确认",
            "categories": ["F", "Q", "R"],
            "feature_filter": lambda fid, r: (
                "vol_ratio" in fid or "vol_trend" in fid or
                "obv" in fid or "vol_direction" in fid or
                "vol_concentration" in fid
            ),
        },
        "vol_breakout": {
            "name": "波动率突破",
            "categories": ["B", "O"],
            "feature_filter": lambda fid, r: (
                "vol_" in fid or "parkinson" in fid or "gk_vol" in fid or
                "yz_vol" in fid or "squeeze" in fid or "vol_cone" in fid
            ),
        },
        "tail_risk": {
            "name": "尾部风险",
            "categories": ["C"],
            "feature_filter": lambda fid, r: (
                "skew" in fid or "kurt" in fid or "tail" in fid or
                "asymmetry" in fid
            ),
        },
        "momentum_enhanced": {
            "name": "动量增强",
            "categories": ["S", "L"],
            "feature_filter": lambda fid, r: (
                "mom_rank" in fid or "path_smoothness" in fid or
                "mom_ratio" in fid or "fractal" in fid or
                "autocorr" in fid or "hurst" in fid or
                "ret_dispersion" in fid
            ),
        },
        "cross_market": {
            "name": "跨市场联动",
            "categories": ["T"],  # 跨资产 (Phase 6)
            "max_features": 12,   # 更多特征, 涵盖多维度
            "feature_filter": lambda fid, r: (
                "btc_eth_" in fid or
                (fid.startswith("corr_") and "autocorr" not in fid) or
                fid.startswith("beta_") or
                "funding" in fid or
                "fear_greed" in fid
            ),
        },
    }

    def __init__(self, backtest_results: List[FeatureTestResultV4],
                 factory_features: Optional[List] = None):
        # Build lookup: feature_id → result
        self.results = {r.feature_id: r for r in backtest_results}

        # Phase 6: 存储 factory feature specs (用于匹配无回测结果的T特征)
        self._factory_feat_ids: Dict[str, str] = {}  # feature_id → category
        if factory_features:
            self._factory_feat_ids = {
                f.id: f.category for f in factory_features
            }

        # Group features by theme
        self.theme_features: Dict[str, List[Tuple[str, float]]] = {}
        self._assign_features()

    def _assign_features(self):
        """分配特征到各主题, 用|ICIR|作为权重

        Phase 6: T类特征即使无回测结果也纳入 (跨市场数据刚激活)
        """
        for theme_id, config in self.THEME_CONFIG.items():
            theme_feats = []
            is_phase6_theme = (theme_id == "cross_market")

            # Pass 1: 已通过FDR的特征 (标准流程)
            for fid, r in self.results.items():
                if not r.passed:
                    continue
                if config["feature_filter"](fid, r):
                    weight = abs(r.icir)
                    theme_feats.append((fid, weight))

            # Pass 2: Phase 6宽松 — 匹配但未通过/未回测的特征也给默认权重
            if is_phase6_theme and len(theme_feats) < 3:
                used_fids = {f[0] for f in theme_feats}
                for fid, r in self.results.items():
                    if fid in used_fids:
                        continue
                    if r.passed:
                        continue  # 已处理
                    if config["feature_filter"](fid, r):
                        # 给一个保守默认ICIR (0.3 ≈ 中等有效性)
                        weight = max(abs(r.icir), 0.25) if r.icir != 0 else 0.25
                        theme_feats.append((fid, weight))
                        used_fids.add(fid)

            # Pass 3: Phase 6新增 — factory中有但backtest中没有的特征 (如funding_rate, fear_greed)
            if is_phase6_theme:
                used_fids = {f[0] for f in theme_feats}
                for fid, cat in self._factory_feat_ids.items():
                    if fid in used_fids:
                        continue
                    if cat != "T":
                        continue
                    # 新激活特征给稍高权重的默认值 (0.35), 确保进入Top-12
                    if config["feature_filter"](fid, None):
                        theme_feats.append((fid, 0.35))
                        used_fids.add(fid)

            # 排序取Top N
            theme_feats.sort(key=lambda x: x[1], reverse=True)
            max_n = config.get("max_features", 8)
            self.theme_features[theme_id] = theme_feats[:max_n]

    def build(self, feature_values: Dict[str, float]) -> List[SubSignal]:
        """从最新特征值构建子信号"""
        signals = []

        for theme_id, config in self.THEME_CONFIG.items():
            feat_weights = self.theme_features.get(theme_id, [])

            if not feat_weights:
                signals.append(SubSignal(
                    id=theme_id, name=config["name"],
                    value=0.0, confidence=0.0,
                    contributing_features=[],
                    direction="NEUTRAL",
                    reasoning=f"{config['name']}: 无有效特征",
                ))
                continue

            total_signal = 0.0
            total_weight = 0.0
            contributing = []

            for fid, weight in feat_weights:
                val = feature_values.get(fid, 0.0)
                r = self.results.get(fid)
                # Phase 6: 新特征可能无回测结果, 给默认IC (轻微正向)
                if r is None:
                    r = FeatureTestResultV4(
                        feature_id=fid, feature_name=fid, category="T",
                        ic=0.1, ic_std=0.5, icir=0.25, t_stat=0, p_value=1,
                        fdr_adjusted_p=1, hit_rate=0.5, sharpe=0,
                        long_sharpe=0, short_sharpe=0, stability=1,
                        ic_decay={}, nonlinear_r2=0, n_obs=100, passed=False
                    )

                # 方向: IC正→看涨信号, IC负→看跌信号
                direction = 1 if r.ic > 0 else -1
                weighted_val = val * direction * weight

                total_signal += weighted_val
                total_weight += weight
                contributing.append(fid)

            if total_weight > 0:
                raw_signal = total_signal / total_weight
                norm_signal = float(np.clip(raw_signal, -3, 3))

                confidence = min(0.9, len(contributing) / 8 * 0.85)

                if norm_signal > 0.3:
                    direction = "LONG"
                elif norm_signal < -0.3:
                    direction = "SHORT"
                else:
                    direction = "NEUTRAL"

                top3 = contributing[:3]
                reasoning = f"{config['name']}: {' | '.join(top3)} → {direction}"

                signals.append(SubSignal(
                    id=theme_id, name=config["name"],
                    value=round(norm_signal, 3),
                    confidence=round(confidence, 3),
                    contributing_features=contributing,
                    direction=direction,
                    reasoning=reasoning,
                ))
            else:
                signals.append(SubSignal(
                    id=theme_id, name=config["name"],
                    value=0.0, confidence=0.0,
                    contributing_features=[],
                    direction="NEUTRAL",
                    reasoning=f"{config['name']}: 权重不足",
                ))

        return signals


class MLSignalEngineV4:
    """ML增强信号引擎 v4.0 — Streamlit集成版 + LightGBM增强 (Phase 5)"""

    def __init__(self):
        self.factory = FeatureFactoryV4()

        # 加载回测结果
        self.backtest_results = self._load_results()
        self.builder = SubSignalBuilder(
            self.backtest_results,
            factory_features=self.factory.features,  # Phase 6: 让builder知道T类新特征
        )

        # 信号历史 (用于波动率加权)
        self._history: Dict[str, List[float]] = {
            "trend": [], "reversal": [], "volume": [],
            "vol_breakout": [], "tail_risk": [], "momentum_enhanced": [],
            "cross_market": [],
        }

        # 跨市场数据获取器 (Phase 6)
        self._cross_market_fetcher = None
        self._cross_market_status: Dict[str, str] = {}
        self._try_init_cross_market()

        # LightGBM 预测器 (Phase 5)
        self._lgbm_predictor = None
        self._lgbm_loaded = False
        self._lgbm_feature_importance: Dict[str, list] = {}
        self._try_load_lgbm()

    def _load_results(self) -> List[FeatureTestResultV4]:
        """加载回测结果"""
        cache = Path(__file__).parent / "data" / "feature_backtest_v4.csv"
        if not cache.exists():
            # 尝试旧版
            cache = Path(__file__).parent / "data" / "feature_backtest_results.csv"

        if not cache.exists():
            return []

        results = []
        df = pd.read_csv(cache)
        for _, row in df.iterrows():
            try:
                results.append(FeatureTestResultV4(
                    feature_id=row.get("feature_id", ""),
                    feature_name=row.get("name", ""),
                    category=row.get("category", ""),
                    ic=row.get("ic", 0),
                    ic_std=0,
                    icir=row.get("icir", 0),
                    t_stat=row.get("t_stat", 0),
                    p_value=row.get("p_value", 1),
                    fdr_adjusted_p=row.get("fdr_p", 1),
                    hit_rate=row.get("hit_rate", 0.5),
                    sharpe=row.get("sharpe", 0),
                    long_sharpe=0,
                    short_sharpe=0,
                    stability=row.get("stability", 1),
                    ic_decay={5: row.get("ic", 0)},
                    nonlinear_r2=row.get("nonlinear_r2", 0),
                    n_obs=row.get("n_obs", 0),
                    fwd_window=5,
                    passed=row.get("passed", False),
                ))
            except Exception:
                continue
        return results

    def _try_init_cross_market(self):
        """Phase 6: 初始化跨市场数据获取器"""
        try:
            from ml_cross_market import CrossMarketFetcher
            self._cross_market_fetcher = CrossMarketFetcher(cache_ttl_hours=4)
        except Exception:
            self._cross_market_fetcher = None

    def _enrich_cross_market(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Phase 6: 用跨市场数据增强 OHLCV DataFrame.
        在 compute_latest / compute_timeseries 之前调用.
        """
        if self._cross_market_fetcher is None:
            return df
        try:
            enriched = self._cross_market_fetcher.enrich_dataframe(df, use_cache=True)
            self._cross_market_status = self._cross_market_fetcher._status
            return enriched
        except Exception as e:
            self._cross_market_status["_last_error"] = str(e)[:60]
            return df

    @property
    def cross_market_available(self) -> bool:
        """跨市场数据是否可用"""
        if self._cross_market_fetcher is None:
            return False
        return self._cross_market_fetcher.is_healthy

    def get_cross_market_status(self) -> str:
        """获取跨市场数据状态报告"""
        if self._cross_market_fetcher is None:
            return "跨市场数据获取器未初始化"
        return self._cross_market_fetcher.status_report()

    def _try_load_lgbm(self):
        """尝试加载LightGBM模型"""
        try:
            from ml_lightgbm_trainer import LightGBMSignalPredictor
            self._lgbm_predictor = LightGBMSignalPredictor()
            self._lgbm_loaded = self._lgbm_predictor.load_models()

            # 加载特征重要性
            imp_path = Path(__file__).parent / "data" / "lgbm_feature_importance.json"
            if imp_path.exists():
                import json
                with open(imp_path) as f:
                    self._lgbm_feature_importance = json.load(f)
        except Exception:
            self._lgbm_loaded = False

    def generate_signal(self, df: pd.DataFrame, symbol: str = "BTC/USDT",
                        use_lgbm: bool = False) -> EnsembleSignal:
        """端到端信号生成

        Args:
            df: OHLCV DataFrame
            symbol: 交易对
            use_lgbm: 启用LightGBM增强 (Phase 5)
        """
        # Phase 6: 跨市场数据增强
        df = self._enrich_cross_market(df)
        price = float(df["close"].values[-1])

        # Step 1: 计算最新特征值
        latest_features = self.factory.compute_latest(df)

        # Step 2: 构建子信号 (线性IC加权)
        sub_signals = self.builder.build(latest_features)

        # Step 2b: LightGBM预测 (Phase 5)
        lgbm_preds = {}
        if use_lgbm and self._lgbm_loaded and self._lgbm_predictor:
            lgbm_preds = self._lgbm_predictor.predict(latest_features)

        # Step 3: 三种组合
        active_sigs = [s for s in sub_signals if s.confidence > 0.1]

        # 等权
        signal_equal = np.mean([s.value for s in active_sigs]) if active_sigs else 0.0

        # IC加权 (用子信号平均|ICIR|)
        ic_weights = {}
        for s in sub_signals:
            feat_icirs = [
                abs(self.builder.results.get(fid, FeatureTestResultV4(
                    feature_id=fid, feature_name="", category="",
                    ic=0, ic_std=0.1, icir=0.5, t_stat=0, p_value=1,
                    fdr_adjusted_p=1, hit_rate=0.5, sharpe=0,
                    long_sharpe=0, short_sharpe=0, stability=1,
                    ic_decay={}, nonlinear_r2=0, n_obs=0, passed=False
                )).icir)
                for fid in s.contributing_features
            ]
            ic_weights[s.id] = np.mean(feat_icirs) if feat_icirs else 0.5

        if active_sigs:
            ic_w_sum = sum(ic_weights.get(s.id, 0.5) * s.value for s in active_sigs)
            ic_w_total = sum(ic_weights.get(s.id, 0.5) for s in active_sigs)
            signal_ic = ic_w_sum / (ic_w_total + 1e-9)
        else:
            signal_ic = 0.0

        # 波动率加权
        for s in sub_signals:
            if s.id in self._history:
                self._history[s.id].append(s.value)
                if len(self._history[s.id]) > 100:
                    self._history[s.id] = self._history[s.id][-100:]

        if active_sigs:
            vol_w_sum = 0.0
            vol_w_total = 0.0
            for s in active_sigs:
                hist = self._history.get(s.id, [0])
                vol = np.std(hist[-20:]) + 0.1 if len(hist) > 3 else 0.5
                w = 1.0 / vol
                vol_w_sum += s.value * w
                vol_w_total += w
            signal_vol = vol_w_sum / (vol_w_total + 1e-9)
        else:
            signal_vol = 0.0

        # LightGBM 组合 (Phase 5)
        signal_lgbm = 0.0
        if use_lgbm and lgbm_preds:
            # 将LGBM预测收益转为信号: 用tanh归一化到[-1, 1]
            lgbm_values = []
            lgbm_weights = []
            for s in sub_signals:
                pred = lgbm_preds.get(s.id, 0.0)
                # 预测收益 → 信号值 (用tanh做非线性映射, 0.05=5%收益→≈+1)
                sig_val = float(np.tanh(pred * 20))  # scale factor 20
                ic_w = ic_weights.get(s.id, 0.5)
                lgbm_values.append(sig_val)
                lgbm_weights.append(ic_w)

                # 更新子信号reasoning
                icon = "📈" if pred > 0.005 else "📉" if pred < -0.005 else "➡️"
                s.reasoning += f" [LGBM: {pred:+.4f} {icon}]"

            if lgbm_weights and sum(lgbm_weights) > 0:
                signal_lgbm = sum(v * w for v, w in zip(lgbm_values, lgbm_weights)) / sum(lgbm_weights)

        # Step 4: 共识度
        long_count = sum(1 for s in sub_signals if s.direction == "LONG")
        short_count = sum(1 for s in sub_signals if s.direction == "SHORT")
        active_count = long_count + short_count
        consensus = max(long_count, short_count) / max(1, len(sub_signals))
        confidence = consensus * (active_count / max(1, len(sub_signals)))

        # Step 5: 确定操作 (LGBM模式优先使用LGBM信号)
        final_signal = signal_lgbm if (use_lgbm and self._lgbm_loaded) else signal_ic
        if final_signal > 0.5 and active_count >= 2:
            action = "BUY"
            size_pct = min(20, 5 + abs(final_signal) * 5)
        elif final_signal < -0.5 and active_count >= 2:
            action = "SELL"
            size_pct = min(20, 5 + abs(final_signal) * 5)
        else:
            action = "HOLD"
            size_pct = 0

        # Step 6: 风险调整
        from feature_ts import _ret
        vol_20d = float(np.std(_ret(df["close"].values, 1)[-20:]) * np.sqrt(365))
        risk_adj = final_signal / (vol_20d + 0.05)

        return EnsembleSignal(
            timestamp=pd.Timestamp.now().isoformat(),
            symbol=symbol,
            price=price,
            sub_signals=sub_signals,
            signal_equal=round(signal_equal, 3),
            signal_ic=round(signal_ic, 3),
            signal_vol=round(signal_vol, 3),
            signal_lgbm=round(signal_lgbm, 3),
            consensus=round(consensus, 2),
            confidence=round(confidence, 2),
            action=action,
            suggested_size_pct=round(size_pct, 1),
            risk_adjusted=round(risk_adj, 3),
            feature_count=len(latest_features),
            n_sub_signals_active=active_count,
            lgbm_available=self._lgbm_loaded,
        )

    def explain(self, signal: EnsembleSignal) -> str:
        """生成Markdown解释"""
        n_themes = len(SubSignalBuilder.THEME_CONFIG)
        lines = [
            f"## 🤖 ML信号引擎 v4.0 — {signal.symbol}",
            f"",
            f"**价格**: ¥{signal.price:,.2f} | **操作**: **{signal.action}**",
            f"**特征数**: {signal.feature_count} | **活跃子信号**: {signal.n_sub_signals_active}/{n_themes}",
        ]

        if signal.lgbm_available:
            lines.append(f"**LightGBM**: ✅ 已加载{n_themes}个模型 | LGBM信号: {signal.signal_lgbm:+.3f}")
        else:
            lines.append(f"**LightGBM**: ⚠️ 模型未加载 (运行 ml_lightgbm_trainer.py 训练)")

        lines += [
            f"",
            f"### 📊 组合信号",
            f"",
            f"| 组合方式 | 信号值 | 方向 |",
            f"|---------|:---:|:---:|",
        ]

        for name, val in [("等权", signal.signal_equal),
                          ("IC加权", signal.signal_ic),
                          ("波动率加权", signal.signal_vol)]:
            icon = "🟢" if val > 0.3 else "🔴" if val < -0.3 else "⚪"
            lines.append(f"| {name} | {val:+.3f} | {icon} |")

        if signal.lgbm_available:
            icon = "🟢" if signal.signal_lgbm > 0.3 else "🔴" if signal.signal_lgbm < -0.3 else "⚪"
            lines.append(f"| 🧠 LightGBM | {signal.signal_lgbm:+.3f} | {icon} |")

        lines.append(f"")
        lines.append(f"**共识度**: {signal.confidence:.0%} | **置信度**: {signal.confidence:.0%}")
        lines.append(f"**建议仓位**: {signal.suggested_size_pct:.0f}% | **风险调整**: {signal.risk_adjusted:+.3f}")
        lines.append(f"")
        lines.append(f"### 🧠 子信号明细")
        lines.append(f"")

        for s in signal.sub_signals:
            icon = "🟢" if s.direction == "LONG" else "🔴" if s.direction == "SHORT" else "⚪"
            lines.append(
                f"| {icon} **{s.name}** | {s.value:+.2f} | "
                f"置信:{s.confidence:.0%} | {len(s.contributing_features)}特征 |"
            )
            lines.append(f"| > {s.reasoning} |")
            lines.append("")

        return "\n".join(lines)


# ── CLI ──
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 ML Signal Engine v4.0 + LightGBM + 🌐跨市场")
    print("=" * 60)

    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True, "timeout": 15000})
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1d", limit=400)
        df = pd.DataFrame(ohlcv, columns=["date", "open", "high", "low", "close", "volume"])

        engine = MLSignalEngineV4()

        # Phase 6: 跨市场数据状态
        print(f"\n🌐 跨市场数据: {'✅ 已激活' if engine.cross_market_available else '⚠️ 不可用'}")
        print(engine.get_cross_market_status())

        # 线性模式
        signal_linear = engine.generate_signal(df, use_lgbm=False)
        print("\n" + engine.explain(signal_linear))

        # LGBM模式
        if engine._lgbm_loaded:
            signal_lgbm = engine.generate_signal(df, use_lgbm=True)
            print(f"\n🧠 LightGBM增强信号: {signal_lgbm.signal_lgbm:+.3f} → {signal_lgbm.action}")

            print(f"\n📊 线性 vs LightGBM 对比:")
            print(f"  线性IC加权: {signal_linear.signal_ic:+.3f} → {signal_linear.action}")
            print(f"  LightGBM:   {signal_lgbm.signal_lgbm:+.3f} → {signal_lgbm.action}")

            # LGBM预测值详情
            from ml_lightgbm_trainer import LightGBMSignalPredictor
            latest = engine.factory.compute_latest(df)
            preds = engine._lgbm_predictor.predict(latest)
            print(f"\n🔮 LGBM各主题预测收益:")
            for theme_id, pred in sorted(preds.items(), key=lambda x: abs(x[1]), reverse=True):
                name = SubSignalBuilder.THEME_CONFIG.get(theme_id, {}).get("name", theme_id)
                icon = "🟢" if pred > 0.005 else "🔴" if pred < -0.005 else "⚪"
                print(f"  {icon} {name:10s}: {pred:+.4f}")

        if engine.backtest_results:
            passed = sum(1 for r in engine.backtest_results if r.passed)
            n_themes = len(SubSignalBuilder.THEME_CONFIG)
            print(f"\n📊 特征: {len(engine.factory.features)}总 → {passed}通过FDR → {n_themes}子信号")
    except ImportError:
        print("⚠️ ccxt未安装")
