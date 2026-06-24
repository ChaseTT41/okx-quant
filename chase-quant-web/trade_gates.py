#!/usr/bin/env python3
"""
⚖️ Trade Gates — 策略共识 + 执行官双重审批

三层防火墙:
  L1: Market Regime Classifier (安检门) → 市场环境是否允许交易？
  L2: Strategy Consensus Engine (共识投票) → 多策略是否独立同意？
  L3: Execution Officer (执行官) → 最终检查清单通过了吗？

只有三层全部 PASS 的交易才能执行。
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# Strategy Definitions
# ═══════════════════════════════════════════════════════════

# 策略权重: 高权重策略即使单个也更有说服力
STRATEGY_WEIGHTS = {
    "ML融合动量": 0.35,
    "跨市场Alpha套利": 0.35,
    "均值回归网格": 0.15,
    "激进交易": 0.15,
    "裸K价格行为": 0.20,   # 裸K信号质量高但稀疏
}

# 策略质量等级
STRATEGY_QUALITY = {
    "ML融合动量": "high",
    "跨市场Alpha套利": "high",
    "均值回归网格": "medium",
    "激进交易": "low",       # 信号太多，质量低
    "裸K价格行为": "high",   # 信号少但精准
}

# ═══════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════

@dataclass
class ConsensusResult:
    """策略共识投票结果"""
    symbol: str
    consensus_action: str            # "BUY" / "SELL" / "HOLD"
    consensus_score: float           # 0-100 加权共识分
    consensus_level: str             # "strong" / "medium" / "weak" / "none"
    supporting_strategies: List[str] # 达成共识的策略名
    dissenting_strategies: List[str] # 反对的策略名
    neutral_strategies: List[str]    # 未表态的策略名
    pass_: bool                      # 是否通过共识

    # 每个策略的投票详情
    votes: Dict[str, dict] = field(default_factory=dict)

    # 信号汇总 (通过共识的)
    best_signal: Optional[dict] = None


@dataclass
class ExecutionDecision:
    """执行官的最终判决"""
    symbol: str
    execute: bool
    final_decision: str              # "执行" / "拒绝" / "HOLD"
    reject_reason: str               # 如果拒绝，详细理由
    suggested_position_size: float   # 建议仓位百分比 (0-1)
    final_confidence: float          # 最终置信度 (0-100)
    full_reasoning: str              # 完整分析总结
    checks_passed: List[str]         # 通过的检查项
    checks_failed: List[str]         # 未通过的检查项


# ═══════════════════════════════════════════════════════════
# Layer 2: Strategy Consensus Engine
# ═══════════════════════════════════════════════════════════

class StrategyConsensusEngine:
    """
    多策略共识投票引擎

    核心原则:
      1. ML动量 和 跨市场Alpha 权重更高 (各0.35)
      2. 必须至少2个独立策略方向一致
      3. 激进交易策略权重低 (0.15) 因为它太吵
      4. 裸K信号少但精准，出现时加分
    """

    MIN_STRATEGIES_CONSENSUS = 2     # 至少 N 个策略同意
    MIN_CONSENSUS_SCORE = 35         # 最低加权共识分 (0-100) — 🆕 方向受限时动态降低

    def evaluate(self, signals: List[dict],
                 regime_action_bias: str = "any") -> Dict[str, ConsensusResult]:
        """
        对所有信号按 symbol 分组，评估每个币种的多策略共识。

        Args:
            signals: run_all_strategies() 的输出，已过滤方向
            regime_action_bias: 安检门的行动偏好 ("long_only"/"short_only"/"neutral"/"any")

        Returns:
            {symbol: ConsensusResult} 每个币种的共识结果
        """
        if not signals:
            return {}

        # 1. 按 symbol 分组
        by_symbol: Dict[str, List[dict]] = {}
        for s in signals:
            sym = s.get("symbol", "")
            if sym not in by_symbol:
                by_symbol[sym] = []
            by_symbol[sym].append(s)

        # 2. 对每个币种评估共识
        results = {}
        for sym, sym_signals in by_symbol.items():
            result = self._evaluate_symbol(sym, sym_signals, regime_action_bias)
            results[sym] = result

        return results

    def _evaluate_symbol(self, symbol: str, signals: List[dict],
                         regime_bias: str) -> ConsensusResult:
        """评估单个币种的多策略共识"""

        # 按策略分组：每个策略只取最高置信度的信号
        by_strategy: Dict[str, dict] = {}
        for s in signals:
            strat = s.get("strategy_name", "unknown")
            if strat not in by_strategy or s.get("confidence", 0) > by_strategy[strat].get("confidence", 0):
                by_strategy[strat] = s

        # 统计投票
        buy_strategies = []
        sell_strategies = []
        neutral_strategies = []
        votes = {}

        for strat, sig in by_strategy.items():
            action = sig.get("action", "HOLD")
            conf = sig.get("confidence", 0.5)
            score = sig.get("score", 50)
            quality = STRATEGY_QUALITY.get(strat, "medium")
            weight = STRATEGY_WEIGHTS.get(strat, 0.15)

            vote = {
                "action": action,
                "confidence": conf,
                "score": score,
                "quality": quality,
                "weight": weight,
            }
            votes[strat] = vote

            if action == "BUY":
                buy_strategies.append(strat)
            elif action == "SELL":
                sell_strategies.append(strat)
            else:
                neutral_strategies.append(strat)

        # 判定方向
        buy_weight = sum(STRATEGY_WEIGHTS.get(s, 0.15) for s in buy_strategies)
        sell_weight = sum(STRATEGY_WEIGHTS.get(s, 0.15) for s in sell_strategies)
        total_active_weight = buy_weight + sell_weight

        if total_active_weight == 0:
            # 全中性 → HOLD
            return ConsensusResult(
                symbol=symbol,
                consensus_action="HOLD",
                consensus_score=0,
                consensus_level="none",
                supporting_strategies=[],
                dissenting_strategies=[],
                neutral_strategies=list(by_strategy.keys()),
                pass_=False,
                votes=votes,
            )

        # 多数方向 (平局 → HOLD，不勉强)
        if buy_weight > sell_weight:
            direction = "BUY"
            supporting = buy_strategies
            dissenting = sell_strategies
            dir_weight = buy_weight
        elif sell_weight > buy_weight:
            direction = "SELL"
            supporting = sell_strategies
            dissenting = buy_strategies
            dir_weight = sell_weight
        else:
            # 平局: 方向冲突无法裁决 → 强制HOLD
            return ConsensusResult(
                symbol=symbol,
                consensus_action="HOLD",
                consensus_score=0,
                consensus_level="none",
                supporting_strategies=buy_strategies + sell_strategies,  # 都有但方向冲突
                dissenting_strategies=[],
                neutral_strategies=neutral_strategies,
                pass_=False,
                votes=votes,
            )

        # 检查最少策略数 — 🆕 方向受限时1个策略即可
        min_strategies = 1 if regime_bias in ("short_only", "long_only") else self.MIN_STRATEGIES_CONSENSUS
        if len(supporting) < min_strategies:
            return ConsensusResult(
                symbol=symbol,
                consensus_action="HOLD",
                consensus_score=0,
                consensus_level="none",
                supporting_strategies=supporting,
                dissenting_strategies=dissenting,
                neutral_strategies=neutral_strategies,
                pass_=False,
                votes=votes,
            )

        # 计算加权共识分 (0-100)
        # 公式: (支持策略权重和 / 总活跃权重) × 平均(score × confidence × quality_factor)
        weight_ratio = dir_weight / total_active_weight if total_active_weight > 0 else 0

        quality_factor = {"high": 1.0, "medium": 0.8, "low": 0.5}
        avg_quality_score = 0
        for strat in supporting:
            v = votes[strat]
            qf = quality_factor.get(v["quality"], 0.8)
            avg_quality_score += v["score"] * v["confidence"] * qf
        avg_quality_score /= len(supporting)

        # Scale to 0-100: weight_ratio × avg_quality_score (already 0-100 range from score*confidence*quality)
        consensus_score = weight_ratio * avg_quality_score

        # 共识等级 (分数阈值在0-100范围)
        # 🆕 动态阈值: 方向受限时降低要求 (因为多空过滤后支持策略变少)
        if regime_bias in ("short_only", "long_only"):
            CONSENSUS_SCORE_STRONG = 25
            CONSENSUS_SCORE_MEDIUM = 15
            CONSENSUS_SCORE_WEAK = 8
        else:
            CONSENSUS_SCORE_STRONG = 45
            CONSENSUS_SCORE_MEDIUM = 30
            CONSENSUS_SCORE_WEAK = 20
        if consensus_score >= CONSENSUS_SCORE_STRONG:
            level = "strong"
        elif consensus_score >= CONSENSUS_SCORE_MEDIUM:
            level = "medium"
        elif consensus_score >= CONSENSUS_SCORE_WEAK:
            level = "weak"
        else:
            level = "none"

        # 是否通过 — 🆕 方向受限时降低绝对分门槛
        if regime_bias in ("short_only", "long_only"):
            min_consensus = 15  # 方向受限时，15分即可通过
        else:
            min_consensus = self.MIN_CONSENSUS_SCORE
        passed = level in ("strong", "medium") and consensus_score >= min_consensus

        # 如果安检门方向受限，冲突则拒绝
        if regime_bias == "long_only" and direction == "SELL":
            passed = False
        elif regime_bias == "short_only" and direction == "BUY":
            passed = False
        elif regime_bias == "neutral":
            # 中性模式：需要 strong 共识 + 2+ high-quality 策略
            high_quality_support = sum(1 for s in supporting
                                      if STRATEGY_QUALITY.get(s, "medium") == "high")
            if level != "strong" or high_quality_support < 2:
                passed = False

        # 寻找最佳信号 (支持策略中 confidence*score 最高)
        best_signal = None
        best_metric = 0
        for s in signals:
            if s.get("strategy_name") in supporting:
                metric = s.get("confidence", 0.5) * s.get("score", 50)
                if metric > best_metric:
                    best_metric = metric
                    best_signal = s

        return ConsensusResult(
            symbol=symbol,
            consensus_action=direction,
            consensus_score=round(consensus_score, 1),
            consensus_level=level,
            supporting_strategies=supporting,
            dissenting_strategies=dissenting,
            neutral_strategies=neutral_strategies,
            pass_=passed,
            votes=votes,
            best_signal=best_signal,
        )

    def get_qualified_signals(self, consensus_results: Dict[str, ConsensusResult]
                              ) -> List[dict]:
        """
        从共识结果中提取通过的高质量信号。
        只返回 pass_=True 且 consensus_level >= "medium" 的信号。
        """
        qualified = []
        for sym, result in consensus_results.items():
            if result.pass_ and result.consensus_level in ("strong", "medium"):
                if result.best_signal:
                    # 用共识信息增强信号
                    sig = dict(result.best_signal)
                    sig["consensus_score"] = result.consensus_score
                    sig["consensus_level"] = result.consensus_level
                    sig["consensus_strategies"] = result.supporting_strategies
                    qualified.append(sig)
        return qualified


# ═══════════════════════════════════════════════════════════
# Layer 3: Execution Officer
# ═══════════════════════════════════════════════════════════

class ExecutionOfficer:
    """
    极度保守的交易执行官 — 最终审批

    所有交易必须通过6项检查:
      1. 市场环境是否允许此方向？
      2. 是否有至少2个高质量策略达成共识？
      3. 单笔风险是否 ≤ max_risk%？
      4. 当前持仓集中度是否过高？
      5. 是否存在重大新闻/黑天鹅风险？
      6. 信号整体置信度是否足够？

    任何一项不通过 → 拒绝。
    """

    # 默认参数
    MAX_SINGLE_RISK_PCT = 0.05        # 单笔最大风险 5% (仓位×杠杆×止损)
    MAX_CONCENTRATION_PCT = 0.50      # 单币种最大集中度 50%
    MIN_OVERALL_CONFIDENCE = 45       # 最低综合置信度 (medium共识,多视角可过)
    MIN_CONFIDENCE_STRONG = 55        # strong共识
    MAX_SWAP_POSITIONS_SMALL = 2      # $0-100 权益: 最多2个仓位

    def review(self,
               symbol: str,
               consensus: ConsensusResult,
               regime: dict,          # 安检门输出
               position_size_pct: float,
               leverage: int,
               current_positions: List[str],
               total_equity: float,
               fg_value: int = 50,
               news_risk: str = "none"  # "none" / "low" / "medium" / "high"
               ) -> ExecutionDecision:
        """
        六项检查 → 最终判决

        Args:
            symbol: 交易标的
            consensus: 共识引擎的输出
            regime: 安检门输出 {"regime", "action_bias", "max_risk_percent", "recommended_leverage", "pass"}
            position_size_pct: 建议仓位 (0-1)
            leverage: 建议杠杆
            current_positions: 当前持仓列表
            total_equity: 总权益
            fg_value: F&G指数
            news_risk: 新闻风险评估

        Returns:
            ExecutionDecision
        """
        checks_passed = []
        checks_failed = []
        reasons = []

        # ── Check 1: 市场环境方向 ──
        action_bias = regime.get("action_bias", "any")
        direction = consensus.consensus_action

        if action_bias == "long_only" and direction == "SELL":
            checks_failed.append("市场环境")
            reasons.append("安检门仅允许做多，当前信号做空")
        elif action_bias == "short_only" and direction == "BUY":
            checks_failed.append("市场环境")
            reasons.append("安检门仅允许做空，当前信号做多")
        elif action_bias == "neutral" and consensus.consensus_level != "strong":
            checks_failed.append("市场环境")
            reasons.append(f"中性市场需要strong共识，当前为{consensus.consensus_level}")
        else:
            checks_passed.append("✅ 市场环境方向")

        # ── Check 2: 策略共识 ──
        # 🆕 方向受限时允许单策略信号通过
        min_supporting = 1 if regime_result.action_bias in ("short_only", "long_only") else 2
        if not consensus.pass_:
            checks_failed.append("策略共识")
            reasons.append("未达到多策略共识门槛")
        elif len(consensus.supporting_strategies) < min_supporting:
            checks_failed.append("策略共识")
            reasons.append(f"仅{len(consensus.supporting_strategies)}个策略支持，需要≥{min_supporting}个")
        else:
            checks_passed.append(f"✅ 策略共识 ({len(consensus.supporting_strategies)}个: {', '.join(consensus.supporting_strategies[:3])})")

        # ── Check 3: 单笔风险 ──
        # 实际风险 = 仓位% × 杠杆 × 止损%
        regime_max_risk = regime.get("max_risk_percent", 0.05)
        max_risk = min(self.MAX_SINGLE_RISK_PCT, regime_max_risk)
        # 止损距离: 从信号获取，或默认≥10% (v3.0 硬性规定)
        raw_stop_loss = consensus.best_signal.get("stop_loss", 0.10) or 0.10
        # 🔴 归一化: stop_loss 可能是价格(>1.0)也可能是百分比(0-1.0)
        if abs(raw_stop_loss) > 1.0:
            # 价格格式 → 转百分比
            entry_price = consensus.best_signal.get("price", 0)
            if entry_price > 0:
                est_stop_loss = abs(raw_stop_loss / entry_price - 1)
            else:
                est_stop_loss = 0.10  # 无价格信息时默认10%
        else:
            est_stop_loss = abs(raw_stop_loss)
        est_stop_loss = max(est_stop_loss, 0.10)  # 🔴 ≥10%, 不给噪音震出
        actual_risk = position_size_pct * leverage * est_stop_loss
        if actual_risk > max_risk:
            checks_failed.append("单笔风险")
            reasons.append(f"实际风险{actual_risk:.1%} > 最大{max_risk:.1%} "
                          f"(仓位{position_size_pct:.1%}×{leverage}x×SL{est_stop_loss:.0%})")
        else:
            checks_passed.append(f"✅ 单笔风险 ({actual_risk:.1%} ≤ {max_risk:.1%})")

        # ── Check 4: 持仓集中度 ──
        # 检查是否已有该币种或相关币种的仓位
        related_positions = [p for p in current_positions
                            if symbol.split("/")[0] in p or p.split("/")[0] in symbol]
        concentration_pct = len(related_positions) / max(len(current_positions), 1)
        max_positions = self._get_max_positions(total_equity)
        if len(current_positions) >= max_positions:
            checks_failed.append("持仓集中度")
            reasons.append(f"持仓已达上限 {max_positions}个 (权益${total_equity:.0f})")
        elif concentration_pct > self.MAX_CONCENTRATION_PCT and related_positions:
            checks_failed.append("持仓集中度")
            reasons.append(f"{symbol}相关仓位过多: {related_positions}")
        else:
            checks_passed.append(f"✅ 持仓集中度 ({len(current_positions)}/{max_positions}个)")

        # ── Check 5: 黑天鹅风险 ──
        if news_risk == "high":
            checks_failed.append("新闻风险")
            reasons.append("存在重大黑天鹅事件，暂停交易")
        elif fg_value <= 15:
            checks_failed.append("新闻风险")
            reasons.append(f"F&G={fg_value}，市场极度恐慌，暂停交易")
        else:
            checks_passed.append(f"✅ 新闻风险 (F&G={fg_value}, news={news_risk})")

        # ── Check 6: 综合置信度 ──
        final_confidence = self._compute_final_confidence(consensus, regime)
        # 根据共识等级动态调整最低置信度
        if consensus.consensus_level == "strong":
            min_conf = self.MIN_CONFIDENCE_STRONG
        else:
            min_conf = self.MIN_OVERALL_CONFIDENCE
        if final_confidence < min_conf:
            checks_failed.append("综合置信度")
            reasons.append(f"置信度{final_confidence:.0f} < 最低{min_conf} "
                          f"(共识:{consensus.consensus_level})")
        else:
            checks_passed.append(f"✅ 综合置信度 ({final_confidence:.0f}≥{min_conf})")

        # ── 最终判决 ──
        execute = len(checks_failed) == 0

        if execute:
            final_decision = "执行"
            reject = ""
            # 调整后仓位: 保守，取建议值和风险上限的较小值
            adj_size = min(position_size_pct, max_risk * 0.8)  # 留20%安全边际
        else:
            final_decision = "拒绝"
            reject = "；".join(reasons)
            adj_size = 0

        full_reasoning = self._build_reasoning(
            symbol, consensus, regime, checks_passed, checks_failed,
            final_decision, final_confidence, position_size_pct
        )

        return ExecutionDecision(
            symbol=symbol,
            execute=execute,
            final_decision=final_decision,
            reject_reason=reject,
            suggested_position_size=round(adj_size, 4),
            final_confidence=round(final_confidence, 1),
            full_reasoning=full_reasoning,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
        )

    # ── Helpers ──

    @staticmethod
    def _get_max_positions(equity: float) -> int:
        if equity <= 100:
            return 2
        elif equity <= 200:
            return 3
        elif equity <= 500:
            return 4
        else:
            return 5

    @staticmethod
    def _compute_final_confidence(consensus: ConsensusResult,
                                  regime: dict) -> float:
        """综合计算最终置信度"""
        base = consensus.consensus_score

        # 策略质量加权
        high_quality_count = sum(1 for s in consensus.supporting_strategies
                                if STRATEGY_QUALITY.get(s, "medium") == "high")
        quality_bonus = min(15, high_quality_count * 5)

        # 视角多样性加分: 技术(K线)/价值(权哥)/量化(西蒙斯) 各自独立
        # 多视角共识 > 同视角多策略共识
        value_perspectives = {"权哥价值投资"}
        quant_perspectives = {"西蒙斯量化统计"}
        tech_perspectives = {"ML融合动量", "跨市场Alpha套利", "均值回归网格",
                            "激进交易", "裸K价格行为", "stock_momentum_breakout",
                            "stock_fear_contrarian", "stock_oversold_bounce"}
        supporting_set = set(consensus.supporting_strategies)
        perspectives_represented = 0
        if supporting_set & value_perspectives:
            perspectives_represented += 1
        if supporting_set & quant_perspectives:
            perspectives_represented += 1
        if supporting_set & tech_perspectives:
            perspectives_represented += 1
        diversity_bonus = (perspectives_represented - 1) * 8  # 2视角=8分, 3视角=16分

        return min(100, max(0, base + quality_bonus + diversity_bonus))

    @staticmethod
    def _build_reasoning(symbol: str, consensus: ConsensusResult,
                         regime: dict, passed: list, failed: list,
                         decision: str, confidence: float,
                         size_pct: float) -> str:
        """生成完整分析总结"""
        lines = [
            f"📋 交易审批: {symbol}",
            f"   方向: {consensus.consensus_action} | "
            f"共识: {consensus.consensus_level} ({consensus.consensus_score:.0f}分)",
            f"   支持: {consensus.supporting_strategies}",
            f"   反对: {consensus.dissenting_strategies or '无'}",
            f"   中立: {consensus.neutral_strategies or '无'}",
            f"   安检门: {regime.get('regime', '?')} "
            f"(bias={regime.get('action_bias', '?')}, "
            f"risk={regime.get('max_risk_percent', 0):.0%}, "
            f"lev≤{regime.get('recommended_leverage', 0)}x)",
        ]
        for check in passed:
            lines.append(f"   {check}")
        for check in failed:
            lines.append(f"   ❌ {check}")
        lines.append(f"   → 判决: {decision} | 置信: {confidence:.0f} | 仓位: {size_pct:.1%}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# Convenience: Full Pipeline
# ═══════════════════════════════════════════════════════════

def run_trade_gates(signals: List[dict],
                    regime_result,     # RegimeResult from market_regime.py
                    current_positions: List[str],
                    total_equity: float,
                    fg_value: int = 50,
                    news_risk: str = "none") -> Dict[str, ExecutionDecision]:
    """
    完整审批流水线: L2(共识) → L3(执行官)

    Args:
        signals: 策略信号列表 (已过安检门方向过滤)
        regime_result: 安检门输出 (RegimeResult)
        current_positions: 当前持仓列表
        total_equity: 总权益
        fg_value: F&G值
        news_risk: 新闻风险

    Returns:
        {symbol: ExecutionDecision} 每个币种的执行判决
    """
    # L2: 策略共识
    consensus_engine = StrategyConsensusEngine()
    consensus_results = consensus_engine.evaluate(
        signals,
        regime_action_bias=regime_result.action_bias if hasattr(regime_result, 'action_bias') else "any"
    )

    # 获取通过共识的信号
    qualified = consensus_engine.get_qualified_signals(consensus_results)

    # L3: 执行官审批
    officer = ExecutionOfficer()
    regime_dict = {
        "regime": getattr(regime_result, 'regime', '中性'),
        "action_bias": getattr(regime_result, 'action_bias', 'any'),
        "max_risk_percent": getattr(regime_result, 'max_risk_percent', 0.03),
        "recommended_leverage": getattr(regime_result, 'recommended_leverage', 5),
        "pass": getattr(regime_result, 'pass_', True),
    }

    decisions = {}
    for sig in qualified:
        sym = sig["symbol"]
        # 从共识结果获取该币种的详情
        consensus = consensus_results.get(sym)
        if not consensus:
            continue

        position_size = sig.get("suggested_size", 0.05)
        leverage = sig.get("_max_leverage", 5)

        decision = officer.review(
            symbol=sym,
            consensus=consensus,
            regime=regime_dict,
            position_size_pct=position_size,
            leverage=leverage,
            current_positions=current_positions,
            total_equity=total_equity,
            fg_value=fg_value,
            news_risk=news_risk,
        )
        decisions[sym] = decision

    return decisions


# ═══════════════════════════════════════════════════════════
# CLI Test
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("⚖️  Trade Gates — 策略共识 + 执行官审批测试\n")

    # 模拟信号: 来自不同策略
    mock_signals = [
        # ML动量: BTC BUY, ETH SELL, SOL SELL
        {"symbol": "BTC/USDT", "action": "BUY", "score": 74, "confidence": 0.64,
         "strategy_name": "ML融合动量", "price": 88000, "reasons": ["趋势向好"],
         "risk_level": "medium", "suggested_size": 0.05, "stop_loss": 0, "take_profit": 0},
        {"symbol": "ETH/USDT", "action": "SELL", "score": 60, "confidence": 0.60,
         "strategy_name": "ML融合动量", "price": 3200, "reasons": ["趋势走弱"],
         "risk_level": "medium", "suggested_size": 0.05, "stop_loss": 0, "take_profit": 0},

        # 均值回归: ETH SELL
        {"symbol": "ETH/USDT", "action": "SELL", "score": 60, "confidence": 0.60,
         "strategy_name": "均值回归网格", "price": 3200, "reasons": ["超买回归"],
         "risk_level": "medium", "suggested_size": 0.05, "stop_loss": 0, "take_profit": 0},

        # 跨市场Alpha: BTC BUY, ETH BUY (与ML冲突!)
        {"symbol": "BTC/USDT", "action": "BUY", "score": 74, "confidence": 0.64,
         "strategy_name": "跨市场Alpha套利", "price": 88000, "reasons": ["跨市场溢价"],
         "risk_level": "medium", "suggested_size": 0.05, "stop_loss": 0, "take_profit": 0},
        {"symbol": "ETH/USDT", "action": "BUY", "score": 71, "confidence": 0.61,
         "strategy_name": "跨市场Alpha套利", "price": 3200, "reasons": ["ETH补涨"],
         "risk_level": "medium", "suggested_size": 0.05, "stop_loss": 0, "take_profit": 0},

        # 激进交易: 几乎所有都BUY
        {"symbol": "BTC/USDT", "action": "BUY", "score": 63, "confidence": 0.63,
         "strategy_name": "激进交易", "price": 88000, "reasons": ["短期强势"],
         "risk_level": "high", "suggested_size": 0.10, "stop_loss": 0, "take_profit": 0},
        {"symbol": "ETH/USDT", "action": "BUY", "score": 63, "confidence": 0.63,
         "strategy_name": "激进交易", "price": 3200, "reasons": ["短期强势"],
         "risk_level": "high", "suggested_size": 0.10, "stop_loss": 0, "take_profit": 0},
    ]

    # L2: 共识投票
    ce = StrategyConsensusEngine()
    results = ce.evaluate(mock_signals)

    print("=" * 60)
    print("L2: 策略共识投票")
    print("=" * 60)
    for sym, r in results.items():
        icon = "✅" if r.pass_ else "❌"
        print(f"\n{icon} {sym}: {r.consensus_action} | "
              f"{r.consensus_level} ({r.consensus_score:.0f}分)")
        print(f"   支持: {r.supporting_strategies}")
        print(f"   反对: {r.dissenting_strategies}")
        if r.neutral_strategies:
            print(f"   中立: {r.neutral_strategies}")
        for strat, v in r.votes.items():
            print(f"     [{strat}] {v['action']} "
                  f"score={v['score']} conf={v['confidence']:.0%} "
                  f"w={v['weight']:.2f} q={v['quality']}")

    # L3: 执行官
    print("\n" + "=" * 60)
    print("L3: 执行官审批")
    print("=" * 60)

    from dataclasses import dataclass as dc
    # 模拟安检门输出
    mock_regime = type('obj', (object,), {
        'regime': '中性',
        'action_bias': 'any',
        'max_risk_percent': 0.05,
        'recommended_leverage': 5,
        'pass_': True,
    })()

    decisions = run_trade_gates(
        signals=mock_signals,
        regime_result=mock_regime,
        current_positions=["MU/USDT"],
        total_equity=51.0,
        fg_value=23,
    )

    for sym, d in decisions.items():
        icon = "✅" if d.execute else "🚫"
        print(f"\n{icon} {sym}: {d.final_decision}")
        if d.reject_reason:
            print(f"   拒绝原因: {d.reject_reason}")
        print(f"   {d.full_reasoning}")

    # 汇总
    approved = [sym for sym, d in decisions.items() if d.execute]
    rejected = [sym for sym, d in decisions.items() if not d.execute]
    print(f"\n{'='*60}")
    print(f"📊 审批汇总: {len(approved)}执行, {len(rejected)}拒绝")
    if approved:
        print(f"   ✅ 批准: {approved}")
    if rejected:
        print(f"   🚫 拒绝: {rejected}")
