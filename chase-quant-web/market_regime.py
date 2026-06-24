#!/usr/bin/env python3
"""
🏛️ Market Regime Classifier — 交易前安检门

在所有交易决策之前，先判断市场环境。
环境不对 → 直接锁门，信号再强也不开仓。

核心理念:
  1. BTC 是大盘锚 — BTC 在跌，山寨币做多胜率极低
  2. Fear & Greed 是情绪锚 — 极端恐惧不做多，极端贪婪不做空
  3. 波动率是杠杆锚 — 高波动降杠杆，低波动可适度加
  4. Alpha 信号是验证锚 — 多策略一致的方向才有执行力

规则引擎: 确定性规则，不做概率推断。这是安全门，不是预测器。
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# Enums & Data Classes
# ═══════════════════════════════════════════════════════════

class MarketRegime(str, Enum):
    EXTREME_BULLISH = "极度看多"
    MILD_BULLISH = "温和看多"
    NEUTRAL = "中性"
    MILD_BEARISH = "温和看空"
    EXTREME_BEARISH = "极度看空"


class ActionBias(str, Enum):
    LONG_ONLY = "long_only"
    SHORT_ONLY = "short_only"
    NEUTRAL = "neutral"
    ANY = "any"


class BTCTrend(str, Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"


class VolLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class RegimeResult:
    """安检门输出"""
    regime: str                    # "极度看多" / "温和看多" / "中性" / "温和看空" / "极度看空"
    action_bias: str               # "long_only" / "short_only" / "neutral" / "any"
    max_risk_percent: float        # 最大风险敞口 (占权益百分比)
    recommended_leverage: int       # 推荐杠杆倍数
    reason: str                    # 1-2句核心理由
    pass_: bool                    # 是否允许交易 (False = 全面禁止)
    # 附加诊断信息
    details: Dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════
# Classification Rules (Rule Table)
# ═══════════════════════════════════════════════════════════

# 规则格式: (F&G范围, BTC趋势, 波动率, Alpha方向) → (Regime, Bias, MaxRisk%, MaxLev, Pass)
# 规则按顺序匹配，命中即停。越上面的规则优先级越高。

REGIME_RULES: List[Tuple[dict, tuple]] = [
    # ── 极端恐惧 (F&G 0-20) ──
    # 市场恐慌 + BTC在跌 + Alpha做空 → 极端看空，只做空
    (
        {"fg_range": (0, 20), "btc_trend": "downtrend", "alpha_dir": "bearish"},
        ("极度看空", "short_only", 0.3, 3, True)
    ),
    # 市场恐慌 + BTC在跌 + Alpha中性 → 温和看空，只做空
    (
        {"fg_range": (0, 20), "btc_trend": "downtrend", "alpha_dir": "neutral"},
        ("温和看空", "short_only", 0.4, 5, True)
    ),
    # 🆕 市场恐慌 + BTC在跌 + Alpha任意 → 温和看空 (补漏: bullish/mixed alpha也挡不住BTC下跌趋势)
    (
        {"fg_range": (0, 20), "btc_trend": "downtrend"},
        ("温和看空", "short_only", 0.4, 5, True)
    ),
    # 市场恐慌 + BTC横盘 → 温和看空，only shorts
    (
        {"fg_range": (0, 20), "btc_trend": "sideways"},
        ("温和看空", "short_only", 0.4, 5, True)
    ),
    # 市场恐慌 + BTC在涨 → 中性 (可能反弹，但风险高)
    (
        {"fg_range": (0, 20), "btc_trend": "uptrend"},
        ("中性", "neutral", 0.5, 5, True)
    ),
    # 市场恐慌 + 高波动 → 中性观望 (波动太大不做)
    (
        {"fg_range": (0, 20), "vol_level": "high"},
        ("中性", "neutral", 0.3, 3, True)
    ),
    # 🆕 极端恐惧兜底规则: 不管BTC趋势/波动/Alpha如何，一律偏空
    # 防止因数据缺失或Alpha方向矛盾导致fallback→any
    (
        {"fg_range": (0, 20)},
        ("温和看空", "short_only", 0.3, 3, True)
    ),

    # ── 恐惧 (F&G 21-40) ──
    # 恐惧 + BTC跌 + Alpha空 → 温和看空
    (
        {"fg_range": (21, 40), "btc_trend": "downtrend", "alpha_dir": "bearish"},
        ("温和看空", "short_only", 0.4, 5, True)
    ),
    # 恐惧 + BTC跌 → 温和看空
    (
        {"fg_range": (21, 40), "btc_trend": "downtrend"},
        ("温和看空", "short_only", 0.5, 5, True)
    ),
    # 恐惧 + BTC涨 + Alpha多 → 温和看多 (可能是底部反弹)
    (
        {"fg_range": (21, 40), "btc_trend": "uptrend", "alpha_dir": "bullish"},
        ("温和看多", "any", 0.5, 8, True)
    ),
    # 恐惧 + 高波动 → 中性
    (
        {"fg_range": (21, 40), "vol_level": "high"},
        ("中性", "neutral", 0.4, 5, True)
    ),
    # 恐惧 + 其他 → 中性偏空
    (
        {"fg_range": (21, 40)},
        ("中性", "any", 0.5, 5, True)
    ),

    # ── 中性 (F&G 41-60) — 正常交易 ──
    # 中性 + BTC涨 + Alpha多 → 温和看多
    (
        {"fg_range": (41, 60), "btc_trend": "uptrend", "alpha_dir": "bullish"},
        ("温和看多", "any", 0.8, 10, True)
    ),
    # 中性 + BTC跌 + Alpha空 → 温和看空
    (
        {"fg_range": (41, 60), "btc_trend": "downtrend", "alpha_dir": "bearish"},
        ("温和看空", "any", 0.5, 8, True)
    ),
    # 中性 + 高波动 → 中性观望
    (
        {"fg_range": (41, 60), "vol_level": "high"},
        ("中性", "any", 0.5, 8, True)
    ),
    # 中性 + 其他 → 中性正常
    (
        {"fg_range": (41, 60)},
        ("中性", "any", 0.6, 10, True)
    ),

    # ── 贪婪 (F&G 61-80) ──
    # 贪婪 + BTC涨 + Alpha多 → 极度看多
    (
        {"fg_range": (61, 80), "btc_trend": "uptrend", "alpha_dir": "bullish"},
        ("极度看多", "long_only", 1.0, 12, True)
    ),
    # 贪婪 + BTC涨 → 温和看多
    (
        {"fg_range": (61, 80), "btc_trend": "uptrend"},
        ("温和看多", "long_only", 0.8, 10, True)
    ),
    # 贪婪 + BTC横 → 中性 (警惕回调)
    (
        {"fg_range": (61, 80), "btc_trend": "sideways"},
        ("中性", "any", 0.5, 8, True)
    ),
    # 贪婪 + 高波动 → 中性 (可能是顶部波动)
    (
        {"fg_range": (61, 80), "vol_level": "high"},
        ("中性", "any", 0.6, 8, True)
    ),
    # 贪婪 + 其他
    (
        {"fg_range": (61, 80)},
        ("温和看多", "any", 0.7, 10, True)
    ),

    # ── 极端贪婪 (F&G 81-100) ──
    # 极端贪婪 + BTC涨 → 温和看多 (牛市尾期，谨慎)
    (
        {"fg_range": (81, 100), "btc_trend": "uptrend"},
        ("温和看多", "long_only", 0.5, 5, True)
    ),
    # 极端贪婪 + BTC跌 → 温和看空 (顶部反转信号)
    (
        {"fg_range": (81, 100), "btc_trend": "downtrend"},
        ("温和看空", "neutral", 0.3, 3, True)
    ),
    # 极端贪婪 + 其他 → 中性 (太热了，观望)
    (
        {"fg_range": (81, 100)},
        ("中性", "neutral", 0.3, 3, True)
    ),
]

# 极端市场全面禁止交易规则
# 某些组合下，不管信号多强，一律不开仓
BLACKOUT_RULES = [
    # 极端恐惧 + BTC瀑布 + 高波动 = 混乱市，不做
    {"fg_range": (0, 15), "btc_trend": "downtrend", "vol_level": "high"},
    # 极端贪婪 + BTC滞涨 + 高波动 = 顶部出货，不做
    {"fg_range": (85, 100), "btc_trend": "sideways", "vol_level": "high"},
]


# ═══════════════════════════════════════════════════════════
# Market Regime Classifier
# ═══════════════════════════════════════════════════════════

class MarketRegimeClassifier:
    """
    市场环境分类器 — 交易前安检门

    用法:
        classifier = MarketRegimeClassifier()
        regime = classifier.classify(
            fg_value=23,
            fg_label="Fear",
            btc_trend_data=...,  # optional, auto-fetched
        )
        if not regime.pass_:
            print(f"🔒 安检不通过: {regime.reason}")
            return  # 跳过所有交易
    """

    def __init__(self, exchange=None):
        """
        Args:
            exchange: ccxt exchange 实例 (可选，用于获取BTC数据)
        """
        self._exchange = exchange
        self._cached_btc_data: Optional[Dict] = None

    # ═══════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════

    def classify(self,
                 fg_value: int,
                 fg_label: str = "",
                 btc_trend: Optional[str] = None,
                 vol_level: Optional[str] = None,
                 alpha_summary: Optional[str] = None,
                 news_impact: str = "neutral",
                 force_fetch: bool = False) -> RegimeResult:
        """
        主分类接口 — 输入市场数据，输出环境判决。

        Args:
            fg_value: Fear & Greed Index (0-100)
            fg_label:  F&G 标签 ("Extreme Fear" / "Fear" / "Neutral" / "Greed" / "Extreme Greed")
            btc_trend: BTC趋势 ("uptrend" / "downtrend" / "sideways")，None=自动获取
            vol_level:  波动率等级 ("high" / "medium" / "low")，None=自动获取
            alpha_summary: 跨市场Alpha摘要 ("bullish" / "bearish" / "neutral" / "mixed")
            news_impact: 新闻影响 ("positive" / "negative" / "neutral")
            force_fetch: 强制重新获取BTC数据 (忽略缓存)

        Returns:
            RegimeResult with full classification
        """
        # Step 1: 自动获取缺失数据
        if btc_trend is None or vol_level is None:
            btc_data = self._fetch_btc_data(force=force_fetch)
            if btc_trend is None:
                btc_trend = btc_data.get("trend", "sideways")
            if vol_level is None:
                vol_level = btc_data.get("vol_level", "medium")

        # Step 2: 标准化输入
        fg_value = max(0, min(100, int(fg_value)))
        btc_trend = self._normalize_trend(btc_trend)
        vol_level = self._normalize_vol(vol_level)
        alpha_dir = self._normalize_alpha(alpha_summary)

        # Step 3: 检查黑名单 (全面禁止)
        for rule in BLACKOUT_RULES:
            if self._match_rule(rule, fg_value, btc_trend, vol_level, alpha_dir):
                return RegimeResult(
                    regime="中性",
                    action_bias="neutral",
                    max_risk_percent=0.0,
                    recommended_leverage=0,
                    reason="🚫 市场混乱状态，全面禁止交易",
                    pass_=False,
                    details={
                        "fg_value": fg_value, "fg_label": fg_label,
                        "btc_trend": btc_trend, "vol_level": vol_level,
                        "alpha_dir": alpha_dir, "blackout": True,
                    }
                )

        # Step 4: 规则表匹配
        for conditions, outcome in REGIME_RULES:
            if self._match_rule(conditions, fg_value, btc_trend, vol_level, alpha_dir):
                regime, action_bias, max_risk, max_lev, do_pass = outcome
                reason = self._build_reason(
                    fg_value, fg_label, btc_trend, vol_level, alpha_dir,
                    regime, action_bias
                )
                return RegimeResult(
                    regime=regime,
                    action_bias=action_bias,
                    max_risk_percent=max_risk,
                    recommended_leverage=max_lev,
                    reason=reason,
                    pass_=do_pass,
                    details={
                        "fg_value": fg_value, "fg_label": fg_label,
                        "btc_trend": btc_trend, "vol_level": vol_level,
                        "alpha_dir": alpha_dir, "news_impact": news_impact,
                        "matched_conditions": conditions,
                    }
                )

        # Step 5: Fallback (理论上不会到这里，但做防御)
        return RegimeResult(
            regime="中性",
            action_bias="any",
            max_risk_percent=0.5,
            recommended_leverage=5,
            reason="未能匹配规则，使用保守默认值",
            pass_=True,
            details={"fg_value": fg_value, "btc_trend": btc_trend, "fallback": True}
        )

    def quick_check(self,
                    fg_value: int,
                    btc_trend: str = "sideways") -> Tuple[bool, str]:
        """
        快速检查 — 单行调用，返回 (是否允许交易, 原因)
        用于不需要完整 RegimeResult 的场景。
        """
        result = self.classify(fg_value=fg_value, btc_trend=btc_trend)
        return result.pass_, result.reason

    # ═══════════════════════════════════════════════════════
    # Data Fetching
    # ═══════════════════════════════════════════════════════

    def _fetch_btc_data(self, force: bool = False) -> Dict:
        """获取BTC趋势 + 波动率数据"""
        if self._cached_btc_data is not None and not force:
            return self._cached_btc_data

        try:
            # 尝试通过 ccxt 获取
            if self._exchange:
                return self._fetch_via_ccxt()

            # 尝试通过 okx.cab REST API
            return self._fetch_via_rest()
        except Exception as e:
            log.warning(f"获取BTC数据失败: {e}，使用默认值")
            return {"trend": "sideways", "vol_level": "medium",
                    "price": 0, "ma20": 0, "ma50": 0}

    def _fetch_via_ccxt(self) -> Dict:
        """通过 ccxt 获取 BTC 日线数据"""
        ohlcv = self._exchange.fetch_ohlcv("BTC/USDT", "1d", limit=60)
        if not ohlcv or len(ohlcv) < 50:
            return {"trend": "sideways", "vol_level": "medium"}

        closes = [c[4] for c in ohlcv]
        return self._analyze_btc(closes)

    def _fetch_via_rest(self) -> Dict:
        """通过 OKX REST API 获取 BTC 日线"""
        import requests
        try:
            url = "https://www.okx.cab/api/v5/market/candles"
            params = {"instId": "BTC-USDT", "bar": "1D", "limit": "60"}
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            if data.get("code") != "0":
                raise Exception(f"OKX API error: {data.get('msg')}")

            candles = data["data"]
            # OKX returns [ts, o, h, l, c, vol, volCcy] — newest first
            closes = [float(c[4]) for c in reversed(candles)]
            return self._analyze_btc(closes)
        except Exception:
            raise

    def _analyze_btc(self, closes: List[float]) -> Dict:
        """从收盘价序列计算趋势和波动率"""
        if len(closes) < 20:
            return {"trend": "sideways", "vol_level": "medium"}

        current_price = closes[-1]

        # MA 计算
        ma20 = sum(closes[-20:]) / 20
        ma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else ma20
        ma20_slope = (ma20 - sum(closes[-25:-5]) / 20) / ma20 if len(closes) >= 25 else 0

        # 趋势判定
        if current_price > ma20 > ma50 and ma20_slope > 0.002:
            trend = "uptrend"
        elif current_price < ma20 < ma50 and ma20_slope < -0.002:
            trend = "downtrend"
        elif abs(ma20_slope) < 0.001:
            trend = "sideways"
        elif current_price > ma20:
            trend = "uptrend"  # 弱上升
        elif current_price < ma20:
            trend = "downtrend"  # 弱下降
        else:
            trend = "sideways"

        # 波动率 (基于 14 日 ATR 百分比)
        atr_pct = self._calc_atr_pct(closes)
        if atr_pct > 0.05:        # >5% daily ATR
            vol = "high"
        elif atr_pct > 0.025:     # 2.5-5%
            vol = "medium"
        else:
            vol = "low"

        result = {
            "trend": trend,
            "vol_level": vol,
            "price": current_price,
            "ma20": ma20,
            "ma50": ma50,
            "ma20_slope": ma20_slope,
            "atr_pct": atr_pct,
        }
        self._cached_btc_data = result
        return result

    @staticmethod
    def _calc_atr_pct(closes: List[float], period: int = 14) -> float:
        """计算 ATR 占价格的百分比"""
        if len(closes) < period + 1:
            return 0.03  # 默认3%
        tr_sum = 0
        for i in range(len(closes) - period, len(closes)):
            tr_sum += abs(closes[i] - closes[i-1]) / closes[i-1]
        return tr_sum / period

    # ═══════════════════════════════════════════════════════
    # Rule Matching Engine
    # ═══════════════════════════════════════════════════════

    def _match_rule(self, conditions: dict, fg: int, btc_trend: str,
                    vol: str, alpha_dir: str) -> bool:
        """
        检查当前市场状态是否匹配一条规则。

        规则中每个 condition key 都是 AND 关系。
        规则中未出现的 key 不参与匹配（wildcard）。
        """
        if "fg_range" in conditions:
            lo, hi = conditions["fg_range"]
            if not (lo <= fg <= hi):
                return False
        if "btc_trend" in conditions:
            if conditions["btc_trend"] != btc_trend:
                return False
        if "vol_level" in conditions:
            if conditions["vol_level"] != vol:
                return False
        if "alpha_dir" in conditions:
            if conditions["alpha_dir"] != alpha_dir:
                return False
        # news_impact 暂不参与匹配（数据不稳定）
        return True

    # ═══════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _normalize_trend(raw: Optional[str]) -> str:
        if not raw:
            return "sideways"
        raw = raw.lower().strip()
        if raw in ("uptrend", "up", "bullish", "上升"):
            return "uptrend"
        if raw in ("downtrend", "down", "bearish", "下降"):
            return "downtrend"
        return "sideways"

    @staticmethod
    def _normalize_vol(raw: Optional[str]) -> str:
        if not raw:
            return "medium"
        raw = raw.lower().strip()
        if raw in ("high", "高", "high_volatility"):
            return "high"
        if raw in ("low", "低", "low_volatility"):
            return "low"
        return "medium"

    @staticmethod
    def _normalize_alpha(raw: Optional[str]) -> str:
        """将 alpha_summary 标准化为 bullish/bearish/neutral/mixed"""
        if not raw:
            return "neutral"
        raw = raw.lower().strip()
        if raw in ("bullish", "positive", "看多"):
            return "bullish"
        if raw in ("bearish", "negative", "看空"):
            return "bearish"
        if raw in ("mixed", "混合"):
            return "mixed"
        return "neutral"

    @staticmethod
    def _build_reason(fg: int, fg_label: str, btc_trend: str,
                      vol: str, alpha: str, regime: str,
                      bias: str) -> str:
        """生成人类可读的理由"""
        parts = []

        # F&G 描述
        fg_desc = fg_label or (
            "极度恐惧" if fg <= 20 else
            "恐惧" if fg <= 40 else
            "中性" if fg <= 60 else
            "贪婪" if fg <= 80 else "极度贪婪"
        )
        parts.append(f"F&G={fg}({fg_desc})")

        # BTC 趋势
        trend_cn = {"uptrend": "上涨", "downtrend": "下跌", "sideways": "横盘"}
        parts.append(f"BTC{trend_cn.get(btc_trend, btc_trend)}")

        # 波动
        vol_cn = {"high": "高波动⚠️", "medium": "中波动", "low": "低波动"}
        parts.append(vol_cn.get(vol, vol))

        # Alpha
        if alpha != "neutral":
            alpha_cn = {"bullish": "Alpha偏多", "bearish": "Alpha偏空", "mixed": "Alpha分歧"}
            parts.append(alpha_cn.get(alpha, alpha))

        base = "，".join(parts)

        # 操作建议
        bias_cn = {
            "long_only": "→ 仅做多",
            "short_only": "→ 仅做空",
            "neutral": "→ 观望为主",
            "any": "→ 多空皆可",
        }
        suggestion = bias_cn.get(bias, "")

        return f"{base}。判定: {regime}{suggestion}"

    # ═══════════════════════════════════════════════════════
    # Alpha Signal Summary (for integration with strategy_runner)
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def summarize_alpha_signals(signals: List[dict]) -> str:
        """
        从策略信号列表提取 Alpha 方向摘要。

        🆕 v2: 多策略加权投票，不再让跨市场Alpha套利一家独大。
        每个策略贡献一票（根据其净方向），多数决定整体Alpha方向。

        Args:
            signals: run_all_strategies() 的返回值

        Returns:
            "bullish" / "bearish" / "neutral" / "mixed"
        """
        if not signals:
            return "neutral"

        # 🆕 按策略分组，每个策略一票
        from collections import defaultdict
        strategy_votes = defaultdict(lambda: {"buy": 0, "sell": 0})
        for s in signals:
            strat = s.get("strategy_name", "unknown")
            action = s.get("action", "HOLD")
            if action == "BUY":
                strategy_votes[strat]["buy"] += 1
            elif action == "SELL":
                strategy_votes[strat]["sell"] += 1

        # 每个策略投票: bullish / bearish / neutral
        bullish_votes = 0
        bearish_votes = 0
        for strat, counts in strategy_votes.items():
            total = counts["buy"] + counts["sell"]
            if total == 0:
                continue
            buy_ratio = counts["buy"] / total
            if buy_ratio >= 0.67:
                bullish_votes += 1
            elif buy_ratio <= 0.33:
                bearish_votes += 1
            # else: mixed → abstain (不投票)

        total_votes = bullish_votes + bearish_votes
        if total_votes == 0:
            return "neutral"

        if bullish_votes > bearish_votes:
            return "bullish"
        elif bearish_votes > bullish_votes:
            return "bearish"
        else:
            return "mixed"


# ═══════════════════════════════════════════════════════════
# Convenience Function
# ═══════════════════════════════════════════════════════════

def check_regime(fg_value: int,
                 fg_label: str = "",
                 btc_trend: Optional[str] = None,
                 vol_level: Optional[str] = None,
                 alpha_signals: Optional[List[dict]] = None,
                 exchange=None) -> RegimeResult:
    """
    便捷函数 — 一行调用完成安检。

    Args:
        fg_value: Fear & Greed Index (0-100)
        fg_label: F&G 分类标签
        btc_trend: BTC趋势 (None=自动获取)
        vol_level: 波动率 (None=自动获取)
        alpha_signals: 策略信号列表 (用于提取Alpha方向)
        exchange: ccxt exchange 实例

    Returns:
        RegimeResult
    """
    classifier = MarketRegimeClassifier(exchange=exchange)

    alpha_summary = None
    if alpha_signals:
        alpha_summary = MarketRegimeClassifier.summarize_alpha_signals(alpha_signals)

    return classifier.classify(
        fg_value=fg_value,
        fg_label=fg_label,
        btc_trend=btc_trend,
        vol_level=vol_level,
        alpha_summary=alpha_summary,
    )


# ═══════════════════════════════════════════════════════════
# CLI Test
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("🏛️  Market Regime Classifier — 规则测试\n")

    classifier = MarketRegimeClassifier()

    # 测试各种场景
    test_cases = [
        # (fg, btc_trend, vol, alpha, label)
        (20, "downtrend", "high", "bearish", "当前市场: 恐慌+瀑布+高波动"),
        (23, "sideways", "medium", "neutral", "当前市场: 恐惧+横盘"),
        (50, "uptrend", "medium", "bullish", "中性+上涨+BTC涨"),
        (75, "uptrend", "low", "bullish", "贪婪+上涨+低波动"),
        (90, "downtrend", "high", "mixed", "极端贪婪+BTC跌"),
        (15, "downtrend", "high", "bearish", "极端恐惧+瀑布"),
    ]

    for fg, trend, vol, alpha, label in test_cases:
        result = classifier.classify(
            fg_value=fg,
            btc_trend=trend,
            vol_level=vol,
            alpha_summary=alpha,
        )
        icon = "✅" if result.pass_ else "🚫"
        print(f"{icon} [{label}]")
        print(f"   Regime: {result.regime} | Bias: {result.action_bias}")
        print(f"   风险: {result.max_risk_percent:.1%} | 杠杆: {result.recommended_leverage}x")
        print(f"   理由: {result.reason}")
        print()

    # 如果有命令行参数 --fg=N，做针对性测试
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--fg="):
                fg = int(arg.split("=")[1])
                result = classifier.classify(fg_value=fg)
                print(f"\n📊 F&G={fg} 模拟:")
                print(json.dumps({
                    "regime": result.regime,
                    "action_bias": result.action_bias,
                    "max_risk_percent": result.max_risk_percent,
                    "recommended_leverage": result.recommended_leverage,
                    "reason": result.reason,
                    "pass": result.pass_,
                }, ensure_ascii=False, indent=2))
