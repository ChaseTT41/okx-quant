"""
ML Signal Engine v3.0 — 特征组合 → 独立子信号 → 等权/波动率加权
==================================================================
进化路线:
  旧: 15个指标 → 人类阈值 → 1个综合分
  新: 72+存活特征 → 非线性组合 → N个独立子信号 → 等权/波动率加权

子信号 (每个独立):
  1. 趋势跟踪 (Trend Following)      — 正动量 + 低波动 + 趋势强度
  2. 均值回归 (Mean Reversion)       — 超卖 + 高波动 + 均值回归特征
  3. 量价确认 (Volume-Price)         — 放量 + 同向运动
  4. 波动率突破 (Vol Breakout)       — 波动率锥低位 + 收敛
  5. 跨市场联动 (Cross-Market)       — Beta/相关性异常
  6. 情绪极端 (Sentiment Extreme)    — 恐慌/贪婪极端值

组合方式:
  - 等权: signal = mean(s1, s2, ..., sN)
  - 波动率加权: signal = Σ(s_i / σ_i) / Σ(1/σ_i)
  - T-stat加权: signal = Σ(s_i * |t_i|) / Σ|t_i|

每个子信号:
  - 内部是一个加权组合 (权重来自t-stat)
  - 输出标准化到 [-3, +3]
  - 正值=做多, 负值=做空, 0=中性
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from feature_engine import FeatureFactory, Feature, _returns, _rsi, _latest
from feature_backtest import FeatureBacktester, FeatureTestResult


@dataclass
class SubSignal:
    """单个子信号"""
    id: str
    name: str
    theme: str             # 趋势/反转/量价/波动/跨市场/情绪
    value: float           # 标准化值 [-3, +3]
    confidence: float      # 0-1, 基于特征t-stat的置信度
    contributing_features: List[str]  # 贡献的特征ID列表
    direction: str         # LONG / SHORT / NEUTRAL
    reasoning: str         # 一句话解释


@dataclass
class EnsembleSignal:
    """组合信号"""
    timestamp: str
    symbol: str
    price: float

    # 子信号
    sub_signals: List[SubSignal]

    # 组合信号
    signal_equal_weight: float       # 等权组合 [-3, +3]
    signal_vol_weight: float         # 波动率加权 [-3, +3]
    signal_tstat_weight: float       # t-stat加权 [-3, +3]

    # 置信度
    consensus: float                 # 0-1, 子信号一致性
    confidence: float                # 0-1, 综合置信度

    # 交易建议
    action: str                      # BUY / SELL / HOLD
    suggested_size_pct: float        # 建议仓位 (占该市场%)

    # 风控
    stop_loss_pct: float
    take_profit_pct: float
    risk_adjusted_signal: float      # 信号强度 / 波动率


# ── 子信号构建器 ──
class SubSignalBuilder:
    """将特征分配给不同主题, 构建独立子信号"""

    def __init__(self, backtest_results: List[FeatureTestResult]):
        self.results = {r.feature_id: r for r in backtest_results}
        self.feature_map = self._build_feature_map()

    def _build_feature_map(self) -> Dict[str, List[Tuple[str, float]]]:
        """按主题分组: feature_id → List[(subsignal_theme, weight)]"""
        theme_map = {
            "trend": [],
            "reversal": [],
            "volume": [],
            "volatility": [],
            "cross_market": [],
            "sentiment": [],
        }

        for fid, r in self.results.items():
            if not r.passed:
                continue
            t = abs(r.t_stat)
            weight = t / (1 + r.stability)  # t-stat / 稳定性惩罚

            # 基于特征ID和类别分配主题
            cat = r.category
            name_lower = fid.lower()

            if cat == "A":  # 动量
                if "accel" in name_lower or "sharpe" in name_lower or "consecutive" in name_lower:
                    theme_map["trend"].append((fid, weight))
                else:
                    theme_map["reversal"].append((fid, weight))
            elif cat == "B":  # 波动率
                if "change" in name_lower or "cone" in name_lower:
                    theme_map["volatility"].append((fid, weight))
                else:
                    theme_map["reversal"].append((fid, weight))
            elif cat == "C":  # 高阶矩
                theme_map["reversal"].append((fid, weight))
            elif cat == "D":  # 均线
                if "slope" in name_lower or "cross" in name_lower:
                    theme_map["trend"].append((fid, weight))
                else:
                    theme_map["reversal"].append((fid, weight))
            elif cat == "E":  # 振荡器
                if "rsi" in name_lower and "change" in name_lower:
                    theme_map["trend"].append((fid, weight))
                else:
                    theme_map["reversal"].append((fid, weight))
            elif cat == "F":  # 成交量
                theme_map["volume"].append((fid, weight))
            elif cat == "G":  # 跨市场
                theme_map["cross_market"].append((fid, weight))
            elif cat == "H":  # 衍生品
                theme_map["sentiment"].append((fid, weight))
            elif cat == "I":  # 情绪
                theme_map["sentiment"].append((fid, weight))
            elif cat == "J":  # 链上proxy
                theme_map["volume"].append((fid, weight))
            elif cat == "K":  # 交互
                if "skew_kurt" in name_lower:
                    theme_map["reversal"].append((fid, weight))
                elif "vol_cone" in name_lower:
                    theme_map["volatility"].append((fid, weight))
                elif "obv" in name_lower:
                    theme_map["volume"].append((fid, weight))
                else:
                    theme_map["trend"].append((fid, weight))
            elif cat == "L":  # 时间序列
                theme_map["reversal"].append((fid, weight))
            elif cat == "M":  # 形态
                theme_map["reversal"].append((fid, weight))
            elif cat == "N":  # 比率
                theme_map["reversal"].append((fid, weight))
            else:
                theme_map["reversal"].append((fid, weight))

        # 每主题取Top 10特征
        for theme in theme_map:
            theme_map[theme].sort(key=lambda x: x[1], reverse=True)
            theme_map[theme] = theme_map[theme][:10]

        return theme_map

    def build(self, feature_values: Dict[str, float]) -> List[SubSignal]:
        """从特征值构建子信号"""
        signals = []

        theme_configs = [
            ("trend", "趋势跟踪", "LONG"),
            ("reversal", "均值回归", "LONG"),
            ("volume", "量价确认", "LONG"),
            ("volatility", "波动率突破", "LONG"),
            ("cross_market", "跨市场联动", "NEUTRAL"),
            ("sentiment", "情绪极端", "NEUTRAL"),
        ]

        for theme_id, theme_name, default_dir in theme_configs:
            feat_weights = self.feature_map.get(theme_id, [])
            if not feat_weights:
                signals.append(SubSignal(
                    id=theme_id, name=theme_name, theme=theme_id,
                    value=0, confidence=0, contributing_features=[],
                    direction="NEUTRAL", reasoning=f"{theme_name}: 无可用特征",
                ))
                continue

            total_signal = 0
            total_weight = 0
            contributing = []

            for fid, weight in feat_weights:
                val = feature_values.get(fid, 0)
                r = self.results.get(fid)

                if r is None:
                    continue

                # 特征值标准化 (Z-score relative to expected range)
                z_val = np.clip(val, -5, 5)  # clamp extremes

                # 方向: t-stat正 → 正信号, t-stat负 → 反信号
                direction = 1 if r.t_stat > 0 else -1
                weighted_val = z_val * direction * weight

                total_signal += weighted_val
                total_weight += weight
                contributing.append(fid)

            if total_weight > 0:
                raw_signal = total_signal / total_weight
                # 标准化到 [-3, +3]
                norm_signal = np.clip(raw_signal, -3, 3)

                # 置信度: 基于特征数量和质量
                confidence = min(0.95, len(contributing) / 10 * 0.8)

                # 确定方向
                if norm_signal > 0.3:
                    direction = "LONG"
                elif norm_signal < -0.3:
                    direction = "SHORT"
                else:
                    direction = "NEUTRAL"

                # 生成推理
                top_feat = contributing[:3]
                reasoning = f"{theme_name}: {' | '.join(top_feat)} → {direction}"

                signals.append(SubSignal(
                    id=theme_id, name=theme_name, theme=theme_id,
                    value=round(norm_signal, 3),
                    confidence=round(confidence, 3),
                    contributing_features=contributing,
                    direction=direction,
                    reasoning=reasoning,
                ))
            else:
                signals.append(SubSignal(
                    id=theme_id, name=theme_name, theme=theme_id,
                    value=0, confidence=0, contributing_features=[],
                    direction="NEUTRAL",
                    reasoning=f"{theme_name}: 特征不足",
                ))

        return signals


# ── 组合信号引擎 ──
class MLSignalEngine:
    """ML增强信号引擎 — 从500+特征到1个交易信号"""

    def __init__(self, backtest_results: Optional[List[FeatureTestResult]] = None):
        self.factory = FeatureFactory()

        # 加载回测结果
        if backtest_results is None:
            self.backtest_results = self._load_backtest_results()
        else:
            self.backtest_results = backtest_results

        self.builder = SubSignalBuilder(self.backtest_results)

        # 子信号波动率追踪 (用于波动率加权)
        self._signal_history: Dict[str, List[float]] = {s: [] for s in
            ["trend", "reversal", "volume", "volatility", "cross_market", "sentiment"]}

    def _load_backtest_results(self) -> List[FeatureTestResult]:
        """加载缓存的特征回测结果"""
        cache = Path(__file__).parent / "data" / "feature_backtest_results.csv"
        if not cache.exists():
            return []

        results = []
        df = pd.read_csv(cache)
        for _, row in df.iterrows():
            results.append(FeatureTestResult(
                feature_id=row["feature_id"],
                feature_name=row.get("name", ""),
                category=row.get("category", ""),
                t_stat=row["t_stat"],
                p_value=row.get("p_value", 0),
                ic=row.get("ic", 0),
                ic_std=0,
                hit_rate=row.get("hit_rate", 0.5),
                sharpe=row.get("sharpe", 0),
                icir=row.get("icir", 0),
                stability=row.get("stability", 1),
                nonlinear_r2=row.get("nonlinear_r2", 0),
                passed=row.get("passed", False),
            ))
        return results

    def generate_signal(self, df: pd.DataFrame, symbol: str = "BTC/USDT") -> EnsembleSignal:
        """
        端到端信号生成:
        1. 计算268个特征
        2. 只保留72个通过t-test的
        3. 分配到6个子信号
        4. 三种方式组合
        5. 输出最终交易建议
        """
        price = float(df["close"].values[-1])

        # ── Step 1: 计算所有特征值 ──
        all_features = self.factory.compute_active(df)

        # ── Step 2+3: 构建子信号 (内部自动过滤t<1.5的特征) ──
        sub_signals = self.builder.build(all_features)

        # ── Step 4: 组合 ──
        # 等权组合
        sig_values = [s.value for s in sub_signals if s.confidence > 0]
        signal_equal = np.mean(sig_values) if sig_values else 0

        # 波动率加权 (历史波动率越低的子信号权重越高)
        sig_vols = {}
        for s in sub_signals:
            if s.id in self._signal_history and len(self._signal_history[s.id]) > 5:
                sig_vols[s.id] = np.std(self._signal_history[s.id][-20:]) + 0.01
            else:
                sig_vols[s.id] = 0.5

        weighted_sum = 0
        weight_total = 0
        for s in sub_signals:
            if s.confidence > 0:
                w = 1.0 / sig_vols.get(s.id, 0.5)
                weighted_sum += s.value * w
                weight_total += w
        signal_vol = weighted_sum / weight_total if weight_total > 0 else 0

        # t-stat加权
        tstat_weights = {}
        for s in sub_signals:
            feat_tstats = [abs(self.builder.results.get(fid, FeatureTestResult(
                feature_id=fid, feature_name="", category="", t_stat=1.0,
                p_value=0, ic=0, ic_std=0, hit_rate=0.5, sharpe=0,
                icir=0, stability=1, nonlinear_r2=0, passed=True
            )).t_stat) for fid in s.contributing_features]
            tstat_weights[s.id] = np.mean(feat_tstats) if feat_tstats else 1.0

        t_weighted_sum = 0
        t_weight_total = 0
        for s in sub_signals:
            if s.confidence > 0:
                w = tstat_weights.get(s.id, 1.0)
                t_weighted_sum += s.value * w
                t_weight_total += w
        signal_tstat = t_weighted_sum / t_weight_total if t_weight_total > 0 else 0

        # 更新历史
        for s in sub_signals:
            if s.id in self._signal_history:
                self._signal_history[s.id].append(s.value)
                if len(self._signal_history[s.id]) > 100:
                    self._signal_history[s.id] = self._signal_history[s.id][-100:]

        # ── Step 5: 共识度 ──
        long_signals = sum(1 for s in sub_signals if s.direction == "LONG")
        short_signals = sum(1 for s in sub_signals if s.direction == "SHORT")
        active_signals = long_signals + short_signals
        consensus = max(long_signals, short_signals) / max(1, len(sub_signals))
        confidence = consensus * (active_signals / max(1, len(sub_signals)))

        # ── Step 6: 确定操作 ──
        final_signal = signal_tstat  # 用t-stat加权作为最终信号
        if final_signal > 0.5:
            action = "BUY"
            size_pct = min(20, 5 + abs(final_signal) * 5)
        elif final_signal < -0.5:
            action = "SELL"
            size_pct = min(20, 5 + abs(final_signal) * 5)
        else:
            action = "HOLD"
            size_pct = 0

        # ── 风险调整 ──
        vol_20d = float(np.std(_returns(df["close"].values, 1)[-20:]) * np.sqrt(365))
        risk_adjusted = final_signal / (vol_20d + 0.01)

        return EnsembleSignal(
            timestamp=pd.Timestamp.now().isoformat(),
            symbol=symbol,
            price=price,
            sub_signals=sub_signals,
            signal_equal_weight=round(signal_equal, 3),
            signal_vol_weight=round(signal_vol, 3),
            signal_tstat_weight=round(signal_tstat, 3),
            consensus=round(consensus, 2),
            confidence=round(confidence, 2),
            action=action,
            suggested_size_pct=round(size_pct, 1),
            stop_loss_pct=-8.0 if action == "BUY" else 0,
            take_profit_pct=15.0 if action == "BUY" else 0,
            risk_adjusted_signal=round(risk_adjusted, 3),
        )

    def explain_signal(self, signal: EnsembleSignal) -> str:
        """生成人类可读的决策解释"""
        lines = [
            f"## 🤖 ML信号引擎 v3.0 — {signal.symbol}",
            f"",
            f"**价格**: ¥{signal.price:,.2f} | **操作**: {signal.action}",
            f"",
            f"### 📊 组合信号",
            f"| 组合方式 | 信号值 | 方向 |",
            f"|---------|:---:|:---:|",
            f"| 等权组合 | {signal.signal_equal_weight:+.3f} | {'🟢 多' if signal.signal_equal_weight > 0.3 else '🔴 空' if signal.signal_equal_weight < -0.3 else '⚪ 中'} |",
            f"| 波动率加权 | {signal.signal_vol_weight:+.3f} | {'🟢 多' if signal.signal_vol_weight > 0.3 else '🔴 空' if signal.signal_vol_weight < -0.3 else '⚪ 中'} |",
            f"| t-stat加权 | {signal.signal_tstat_weight:+.3f} | {'🟢 多' if signal.signal_tstat_weight > 0.3 else '🔴 空' if signal.signal_tstat_weight < -0.3 else '⚪ 中'} |",
            f"",
            f"**共识度**: {signal.consensus:.0%} | **置信度**: {signal.confidence:.0%}",
            f"**建议仓位**: {signal.suggested_size_pct:.0f}% | **风险调整信号**: {signal.risk_adjusted_signal:+.3f}",
            f"",
            f"### 🧠 子信号明细",
        ]

        for s in signal.sub_signals:
            icon = "🟢" if s.direction == "LONG" else "🔴" if s.direction == "SHORT" else "⚪"
            lines.append(
                f"| {icon} **{s.name}** | {s.value:+.2f} | "
                f"置信:{s.confidence:.0%} | {len(s.contributing_features)}特征 |"
            )
            lines.append(f"| > {s.reasoning} |")

        lines.append("")
        lines.append(f"### ⚡ 风控")
        if signal.action == "BUY":
            lines.append(f"止损: {signal.stop_loss_pct:+.0f}% | 止盈: +{signal.take_profit_pct:.0f}%")
        else:
            lines.append(f"无交易 — 信号不足或方向不明")

        return "\n".join(lines)


# ── CLI ──
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("🤖 ML Signal Engine v3.0")
    print("=" * 60)

    # 用BTC数据测试完整流程
    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True, "timeout": 10000})
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1d", limit=400)
        df = pd.DataFrame(ohlcv, columns=["date", "open", "high", "low", "close", "volume"])

        engine = MLSignalEngine()
        signal = engine.generate_signal(df)

        print(engine.explain_signal(signal))

        # 简要统计
        n_passed = len(engine.backtest_results)
        n_total = len(engine.factory.features)
        print(f"\n📊 特征: {n_total}总 → {n_passed}通过t-test → 6个子信号")
        print(f"🎯 最终操作: {signal.action}")
        print(f"📈 信号强度: {signal.signal_tstat_weight:+.3f}")

    except ImportError:
        print("⚠️ ccxt 未安装, 使用模拟数据")
        # 用模拟数据演示流程
        np.random.seed(42)
        n = 400
        price = 60000 + np.cumsum(np.random.randn(n) * 500)
        df = pd.DataFrame({
            "open": price * (1 + np.random.randn(n) * 0.001),
            "high": price + abs(np.random.randn(n)) * 500,
            "low": price - abs(np.random.randn(n)) * 500,
            "close": price + np.random.randn(n) * 100,
            "volume": abs(np.random.randn(n)) * 1000 + 5000,
        })

        engine = MLSignalEngine()
        signal = engine.generate_signal(df, symbol="SIM/BTC")
        print(engine.explain_signal(signal))
