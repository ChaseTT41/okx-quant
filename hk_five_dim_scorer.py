"""
HK Stock Five-Dimensional Scorecard — 港股五维评分卡
Adapted from QuantDinger's multi-period objective consensus scoring for HK equities.

五维:
  1. 趋势强度 (25%): MACD + 均线排列 + ADX + K线位置
  2. 超买超卖 (15%): RSI + 布林带 + Williams %R
  3. 支撑阻力 (20%): 关键价位 + 距支撑/阻力距离
  4. 基本面    (25%): 成交量变化 + 价格动量 + 波动率结构
  5. 风险度    (15%): 历史波动率 + 最大回撤 + 流动性

输出: [买/卖/观] + ⭐(1-5) + 置信度% + 五维明细
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from hk_stock_data import get_data, HKStockData


@dataclass
class ScoreBreakdown:
    """单维度评分详情"""
    score: float          # 0-100
    weight: float         # 权重
    weighted: float       # score * weight
    signals: List[str]    # 触发信号列表


@dataclass
class FiveDimResult:
    """五维评分完整结果"""
    code: str
    name: str
    date: str
    close: float

    # 五维分数
    trend: ScoreBreakdown
    ob_os: ScoreBreakdown
    sr: ScoreBreakdown
    fundamental: ScoreBreakdown
    risk: ScoreBreakdown

    composite: float       # 加权综合分 0-100
    action: str            # BUY / WATCH / SELL
    stars: int             # 1-5
    confidence: float      # 0-1, 各维度一致性

    # 附加
    stop_loss: float = 0.0
    target: float = 0.0
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "代码": self.code, "名称": self.name, "日期": self.date,
            "收盘价": self.close, "综合分": round(self.composite, 1),
            "操作": self.action, "星级": self.stars,
            "置信度": f"{self.confidence:.0%}",
            "趋势": round(self.trend.score), "超买超卖": round(self.ob_os.score),
            "支撑阻力": round(self.sr.score), "基本面": round(self.fundamental.score),
            "风险": round(self.risk.score),
            "止损": round(self.stop_loss, 2), "目标": round(self.target, 2),
            "备注": self.note,
        }

    def __repr__(self) -> str:
        stars_str = "⭐" * self.stars + "☆" * (5 - self.stars)
        return (
            f"{self.code} {self.name} | {self.action} {stars_str} | "
            f"综合:{self.composite:.1f} | 置信:{self.confidence:.0%} | "
            f"¥{self.close:.2f}"
        )


# ── 权重配置 (与 Part 2 对齐) ──
WEIGHTS = {
    "trend": 0.25,
    "ob_os": 0.15,
    "sr": 0.20,
    "fundamental": 0.25,
    "risk": 0.15,
}

# 综合分 → 操作映射
ACTION_MAP = [
    (80, "强力买入", 5, 0.10, -0.05),
    (65, "买入", 4, 0.08, -0.08),
    (50, "轻仓试多", 3, 0.05, -0.10),
    (35, "观望", 2, 0.00, 0.00),
    (20, "轻仓对冲", 1, 0.03, -0.05),
    (0, "卖出/回避", 0, 0.00, 0.00),
]


def _resolve_action(composite: float) -> Tuple[str, int, float, float]:
    """综合分 → (操作, 星级, 仓位比, 止损比)"""
    for threshold, action, stars, position, stoploss in ACTION_MAP:
        if composite >= threshold:
            return action, stars, position, stoploss
    return "卖出/回避", 0, 0.0, 0.0


class FiveDimScorer:
    """港股五维评分器"""

    def __init__(self, data: Optional[HKStockData] = None):
        self.data = data or get_data()

    # ── 维度1: 趋势强度 (0-100) ──
    def _score_trend(self, df: pd.DataFrame) -> ScoreBreakdown:
        signals = []
        score = 0

        close = df["收盘"].values
        if len(close) < 60:
            return ScoreBreakdown(50, WEIGHTS["trend"], 50 * WEIGHTS["trend"], ["数据不足"])

        # MACD
        ema12 = pd.Series(close).ewm(span=12).mean()
        ema26 = pd.Series(close).ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line

        if histogram.iloc[-1] > 0:
            score += 20
            signals.append("MACD正柱")
        if histogram.iloc[-1] > histogram.iloc[-2] > histogram.iloc[-3]:
            score += 10
            signals.append("MACD柱放大")

        # 均线排列
        sma_5 = pd.Series(close).rolling(5).mean()
        sma_20 = pd.Series(close).rolling(20).mean()
        sma_60 = pd.Series(close).rolling(60).mean()

        ma_score = 0
        if sma_5.iloc[-1] > sma_20.iloc[-1]:
            ma_score += 15
        if sma_20.iloc[-1] > sma_60.iloc[-1]:
            ma_score += 15
        if close[-1] > sma_20.iloc[-1]:
            ma_score += 10
        score += ma_score
        if ma_score >= 30:
            signals.append("多头排列")
        elif ma_score >= 15:
            signals.append("均线偏多")

        # ADX
        adx_val = self._calc_adx(df, 14)
        if adx_val > 25:
            score += 20
            signals.append(f"ADX={adx_val:.0f}(趋势明确)")
        elif adx_val > 20:
            score += 10

        # 价格位置 vs 均线
        if close[-1] > sma_60.iloc[-1]:
            score += 10
            signals.append("站上60日均线")

        # Golden cross
        if len(sma_5) >= 7 and sma_5.iloc[-2] <= sma_20.iloc[-2] and sma_5.iloc[-1] > sma_20.iloc[-1]:
            score += 15
            signals.append("金叉(5×20)")

        final = min(score, 100)
        return ScoreBreakdown(final, WEIGHTS["trend"], final * WEIGHTS["trend"], signals)

    # ── 维度2: 超买超卖 (0-100, 中间最优) ──
    def _score_ob_os(self, df: pd.DataFrame) -> ScoreBreakdown:
        signals = []
        close = df["收盘"].values
        if len(close) < 14:
            return ScoreBreakdown(50, WEIGHTS["ob_os"], 50 * WEIGHTS["ob_os"], ["数据不足"])

        rsi = self._calc_rsi(close, 14)

        if 40 <= rsi <= 60:
            score = 80
            signals.append(f"RSI={rsi:.0f}(健康)")
        elif 30 <= rsi < 40:
            score = 60
            signals.append(f"RSI={rsi:.0f}(偏弱)")
        elif 60 < rsi <= 70:
            score = 50
            signals.append(f"RSI={rsi:.0f}(偏强)")
        elif rsi < 30:
            score = 30
            signals.append(f"RSI={rsi:.0f}(超卖)⚠️")
        elif rsi > 70:
            score = 20
            signals.append(f"RSI={rsi:.0f}(超买)⚠️")
        else:
            score = 10

        # 布林带位置
        sma20 = pd.Series(close).rolling(20).mean()
        std20 = pd.Series(close).rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        bb_position = (close[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) if bb_upper.iloc[-1] != bb_lower.iloc[-1] else 0.5

        if 0.3 <= bb_position <= 0.7:
            score = min(score + 10, 100)
        elif bb_position < 0.1:
            score -= 15
            signals.append("布林下轨(超卖)")
        elif bb_position > 0.9:
            score -= 15
            signals.append("布林上轨(超买)")

        final = max(0, min(score, 100))
        return ScoreBreakdown(final, WEIGHTS["ob_os"], final * WEIGHTS["ob_os"], signals)

    # ── 维度3: 支撑阻力 (0-100) ──
    def _score_sr(self, df: pd.DataFrame) -> ScoreBreakdown:
        signals = []
        close = df["收盘"].values
        high = df["最高"].values
        low = df["最低"].values

        if len(close) < 60:
            return ScoreBreakdown(50, WEIGHTS["sr"], 50 * WEIGHTS["sr"], ["数据不足"])

        price = close[-1]

        # 找最近60天的局部高/低点
        n = min(60, len(close) - 1)
        recent_high = max(high[-n:])
        recent_low = min(low[-n:])

        dist_support_pct = (price - recent_low) / price * 100
        dist_resist_pct = (recent_high - price) / price * 100

        score = 50
        if dist_support_pct < 2:
            score += 30
            signals.append(f"近支撑(距{dist_support_pct:.1f}%)")
        elif dist_support_pct < 5:
            score += 15
        elif dist_support_pct > 15:
            score -= 10
            signals.append("远离支撑")

        if dist_resist_pct > 10:
            score += 20
            signals.append(f"上方空间大({dist_resist_pct:.0f}%)")
        elif dist_resist_pct > 5:
            score += 10
        elif dist_resist_pct < 2:
            score -= 20
            signals.append("逼近阻力⚠️")

        # 历史高位比
        all_time_high = max(high)
        pct_from_ath = (all_time_high - price) / all_time_high * 100
        if pct_from_ath > 50:
            score += 10
            signals.append(f"距历史高{pct_from_ath:.0f}%(空间)")

        final = max(0, min(score, 100))
        return ScoreBreakdown(final, WEIGHTS["sr"], final * WEIGHTS["sr"], signals)

    # ── 维度4: 基本面 (0-100) ──
    def _score_fundamental(self, df: pd.DataFrame) -> ScoreBreakdown:
        signals = []
        close = df["收盘"].values
        volume = df["成交量"].values

        if len(close) < 20:
            return ScoreBreakdown(50, WEIGHTS["fundamental"], 50 * WEIGHTS["fundamental"], ["数据不足"])

        score = 50

        # 量价关系
        vol_5d = np.mean(volume[-5:])
        vol_20d = np.mean(volume[-20:])
        vol_ratio = vol_5d / vol_20d if vol_20d > 0 else 1

        if vol_ratio > 1.5:
            score += 10
            signals.append(f"放量({vol_ratio:.1f}x)")
        elif vol_ratio < 0.5:
            score -= 10
            signals.append("缩量")

        # 近期动量
        ret_5d = (close[-1] / close[-6] - 1) * 100 if len(close) > 5 else 0
        ret_20d = (close[-1] / close[-21] - 1) * 100 if len(close) > 20 else 0

        if ret_5d > 3:
            score += 10
            signals.append(f"5日动量+{ret_5d:.1f}%")
        elif ret_5d < -5:
            score -= 10
            signals.append(f"5日急跌{ret_5d:.1f}%")

        if ret_20d > 5:
            score += 10
            signals.append(f"20日趋势+{ret_20d:.1f}%")

        # 波动率适中
        returns = np.diff(close) / close[:-1]
        vol_20d_ret = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else 1
        if 0.15 < vol_20d_ret < 0.40:
            score += 10
            signals.append(f"波动适中({vol_20d_ret:.0%})")
        elif vol_20d_ret > 0.60:
            score -= 10
            signals.append(f"高波动({vol_20d_ret:.0%})⚠️")

        # 趋势持续性
        up_days = sum(1 for i in range(-20, 0) if i > -len(close) and close[i] > close[i - 1])
        if up_days >= 13:
            score += 10
            signals.append(f"强势({up_days}/20日涨)")
        elif up_days <= 7:
            score -= 10
            signals.append(f"弱势({up_days}/20日涨)")

        final = max(0, min(score, 100))
        return ScoreBreakdown(final, WEIGHTS["fundamental"], final * WEIGHTS["fundamental"], signals)

    # ── 维度5: 风险度 (0-100, 低风险=高分) ──
    def _score_risk(self, df: pd.DataFrame) -> ScoreBreakdown:
        signals = []
        close = df["收盘"].values

        if len(close) < 20:
            return ScoreBreakdown(70, WEIGHTS["risk"], 70 * WEIGHTS["risk"], ["数据不足"])

        score = 70

        # 历史波动率
        returns = np.diff(close) / close[:-1]
        hv_30d = np.std(returns[-min(30, len(returns)):]) * np.sqrt(252)

        if hv_30d > 0.80:
            score -= 30
            signals.append(f"极高波动({hv_30d:.0%})🔥")
        elif hv_30d > 0.60:
            score -= 20
            signals.append(f"高波动({hv_30d:.0%})")
        elif hv_30d < 0.25:
            score += 20
            signals.append(f"低波动({hv_30d:.0%})")
        elif hv_30d < 0.35:
            score += 10
            signals.append(f"波动可控({hv_30d:.0%})")

        # 最大回撤 (最近60天)
        if len(close) >= 60:
            peak = np.maximum.accumulate(close[-60:])
            drawdown = (close[-60:] - peak) / peak
            max_dd = abs(min(drawdown))
            if max_dd > 0.30:
                score -= 20
                signals.append(f"大回撤({max_dd:.0%})⚠️")
            elif max_dd > 0.15:
                score -= 10
                signals.append(f"回撤{max_dd:.0%}")

        # 流动性 (日均成交额)
        avg_volume = np.mean(df["成交量"].values[-20:])
        avg_price = np.mean(close[-20:])
        avg_turnover = avg_volume * avg_price
        if avg_turnover < 1e6:  # < 100万HKD
            score -= 15
            signals.append("低流动性⚠️")
        elif avg_turnover > 1e8:  # > 1亿HKD
            score += 10
            signals.append("高流动性")

        # VaR 95%
        if len(returns) >= 20:
            var_95 = np.percentile(returns[-60:], 5) if len(returns) >= 60 else np.percentile(returns, 5)
            if var_95 < -0.05:
                score -= 10
                signals.append(f"VaR95={var_95:.1%}")

        final = max(0, min(score, 100))
        return ScoreBreakdown(final, WEIGHTS["risk"], final * WEIGHTS["risk"], signals)

    # ── 综合评分 ──
    def score(self, code: str, date: Optional[str] = None) -> FiveDimResult:
        """对单只股票运行完整五维评分"""
        df = self.data.get_daily(code)
        if date:
            cutoff = pd.Timestamp(date)
            df = df[df["日期"] <= cutoff]

        if len(df) < 60:
            name = self.data.lookup(code)
            return FiveDimResult(
                code=code, name=name["名称"] if name else "?",
                date=str(df["日期"].iloc[-1].date()) if len(df) else "N/A",
                close=float(df["收盘"].iloc[-1]) if len(df) else 0,
                trend=ScoreBreakdown(50, 0.25, 12.5, ["<60天数据"]),
                ob_os=ScoreBreakdown(50, 0.15, 7.5, []),
                sr=ScoreBreakdown(50, 0.20, 10.0, []),
                fundamental=ScoreBreakdown(50, 0.25, 12.5, []),
                risk=ScoreBreakdown(50, 0.15, 7.5, []),
                composite=50, action="数据不足", stars=0, confidence=0,
            )

        trend = self._score_trend(df)
        ob_os = self._score_ob_os(df)
        sr = self._score_sr(df)
        fundamental = self._score_fundamental(df)
        risk = self._score_risk(df)

        composite = trend.weighted + ob_os.weighted + sr.weighted + fundamental.weighted + risk.weighted
        action, stars, position_pct, stoploss_pct = _resolve_action(composite)

        # 置信度 = 各维度得分方差的反比
        dim_scores = np.array([trend.score, ob_os.score, sr.score, fundamental.score, risk.score])
        score_std = np.std(dim_scores)
        if score_std == 0:
            confidence = 0.95
        elif score_std > 30:
            confidence = 0.30
        else:
            confidence = max(0.25, 1.0 - score_std / 60)

        name_info = self.data.lookup(code)
        name = name_info["名称"] if name_info else "?"
        close_price = float(df["收盘"].iloc[-1])
        stop_loss = close_price * (1 + stoploss_pct) if stoploss_pct else 0

        return FiveDimResult(
            code=code, name=name,
            date=str(df["日期"].iloc[-1].date()),
            close=close_price,
            trend=trend, ob_os=ob_os, sr=sr,
            fundamental=fundamental, risk=risk,
            composite=composite, action=action,
            stars=stars, confidence=confidence,
            stop_loss=stop_loss, target=close_price * 1.10 if action.startswith("买入") or action == "轻仓试多" else 0,
        )

    def score_batch(self, codes: List[str], min_data_days: int = 60) -> pd.DataFrame:
        """批量评分, 返回排序后的DataFrame"""
        results = []
        for code in codes:
            try:
                r = self.score(code)
                if r.stars > 0:
                    results.append(r.to_dict())
            except Exception as e:
                pass
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values("综合分", ascending=False).reset_index(drop=True)
        return df

    # ── 技术指标工具 ──
    @staticmethod
    def _calc_rsi(close: np.ndarray, period: int = 14) -> float:
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100 - 100 / (1 + rs))

    @staticmethod
    def _calc_adx(df: pd.DataFrame, period: int = 14) -> float:
        high = df["最高"].values
        low = df["最低"].values
        close = df["收盘"].values

        if len(close) < period + 1:
            return 20.0

        tr = np.zeros(len(close))
        plus_dm = np.zeros(len(close))
        minus_dm = np.zeros(len(close))

        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]
            plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0
            minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0

        atr = pd.Series(tr).ewm(alpha=1 / period).mean()
        plus_di = pd.Series(plus_dm).ewm(alpha=1 / period).mean() / atr * 100
        minus_di = pd.Series(minus_dm).ewm(alpha=1 / period).mean() / atr * 100

        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = pd.Series(dx).ewm(alpha=1 / period).mean()

        return float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 20.0


# ── CLI ──
if __name__ == "__main__":
    scorer = FiveDimScorer()

    # 测试几只知名港股
    test_codes = ["00700", "00005", "09988", "00388", "01299"]
    print("🎯 港股五维评分卡\n")

    for code in test_codes:
        r = scorer.score(code)
        print(r)
        print(f"  趋势:{r.trend.score:.0f}({', '.join(r.trend.signals[:2])})")
        print(f"  超买超卖:{r.ob_os.score:.0f}({', '.join(r.ob_os.signals[:2])})")
        print(f"  支撑阻力:{r.sr.score:.0f}({', '.join(r.sr.signals[:2])})")
        print(f"  基本面:{r.fundamental.score:.0f}({', '.join(r.fundamental.signals[:2])})")
        print(f"  风险:{r.risk.score:.0f}({', '.join(r.risk.signals[:2])})")
        print(f"  止损:¥{r.stop_loss:.2f} | 目标:¥{r.target:.2f}")
        print()
