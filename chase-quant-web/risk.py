"""
Chase量化策略 — 风控控制器
五层风控防御: 事前→订单→持仓→组合→异常检测
"""
from __future__ import annotations
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


# ── 风控参数 ──
MAX_SINGLE_LOSS_PCT = 2.0     # 单笔最大亏损 ≤ 总资金2%
MAX_POSITIONS = 5             # 最大同时持仓数
HARD_STOP_LOSS_PCT = -8.0     # 硬止损线
DAILY_LOSS_LIMIT_PCT = -5.0   # 日亏损上限
MONTHLY_TARGET_PCT = 30.0     # 月盈利目标
MAX_POSITION_SIZE_PCT = 40.0  # 单仓位 ≤ 总资金40%
MIN_SCORE_FOR_ENTRY = 35      # 最低入场分 (v5 0-100评分体系)

# Phase 13: 执行层风控参数
MAX_PARTICIPATION_RATE = 0.10  # 最大成交量占比 (10%)
MAX_SPREAD_BPS = 100.0         # 最大可接受点差 (100bps)
MAX_IMPACT_BPS = 50.0          # 最大可接受冲击成本 (50bps)
MAX_SLIPPAGE_PER_SLICE_BPS = 25.0  # 单切片最大滑点


@dataclass
class RiskCheck:
    """风控检查结果"""
    passed: bool
    reason: str
    alert_level: str = ""     # info / warning / danger


class RiskController:
    """风控控制器"""

    def __init__(self, portfolio):
        self.pf = portfolio

    def pre_trade_check(self, market: str, amount: float, score: float,
                        total_value: float) -> RiskCheck:
        """事前风控: 交易前检查"""
        # 1. 最低入场分
        if score < MIN_SCORE_FOR_ENTRY:
            return RiskCheck(False, f"信号分{score:.0f}<{MIN_SCORE_FOR_ENTRY}最低门槛", "warning")

        # 2. 持仓上限
        open_count = len(self.pf.open_positions)
        if open_count >= MAX_POSITIONS:
            return RiskCheck(False, f"已持仓{open_count}只, 达上限{MAX_POSITIONS}", "warning")

        # 3. 单仓位上限
        pos_pct = amount / total_value * 100 if total_value > 0 else 100
        if pos_pct > MAX_POSITION_SIZE_PCT:
            return RiskCheck(False, f"仓位{pos_pct:.0f}% > {MAX_POSITION_SIZE_PCT}%上限", "warning")

        # 4. 单笔亏损上限
        max_loss_amount = total_value * MAX_SINGLE_LOSS_PCT / 100
        worst_case = amount * 0.08  # 假设-8%止损
        if worst_case > max_loss_amount:
            return RiskCheck(False, f"单笔潜在亏损¥{worst_case:.0f} > ¥{max_loss_amount:.0f}上限", "danger")

        # 5. 市场现金
        if not self.pf.can_buy(market, amount):
            return RiskCheck(False, f"{market} 现金不足", "warning")

        # 6. 日亏损检查
        if self.pf.total_pnl_pct < DAILY_LOSS_LIMIT_PCT:
            return RiskCheck(False, f"日亏损{self.pf.total_pnl_pct:.1f}%触发熔断", "danger")

        return RiskCheck(True, "通过", "info")

    def position_check(self, position_id: str) -> List[RiskCheck]:
        """持仓风控: 每笔持仓定期检查"""
        checks = []
        pos = self.pf.positions.get(position_id)
        if not pos or pos.status == "closed":
            return checks

        # 硬止损
        if pos.pnl_pct <= HARD_STOP_LOSS_PCT:
            checks.append(RiskCheck(
                False, f"{pos.symbol} 亏损{pos.pnl_pct:.1f}%触发硬止损!", "danger"))

        # 盈利回撤 (盈利>10%后回撤到5%)
        if pos.pnl_pct > 10:
            # 检查是否从高点回撤
            pass

        # 仓位占比 > 40%
        pos_pct = pos.value / self.pf.total_value * 100 if self.pf.total_value > 0 else 0
        if pos_pct > MAX_POSITION_SIZE_PCT:
            checks.append(RiskCheck(
                False, f"{pos.symbol} 仓位{pos_pct:.0f}%超限, 建议减仓", "danger"))

        return checks

    def execution_risk_check(self, order_size: float, avg_daily_volume: float,
                            spread: float = 0.001, volatility: float = 0.02,
                            n_slices: int = 1) -> RiskCheck:
        """
        Phase 13: 执行层风控检查

        在拆单执行前校验:
          1. 参与率 ≤ 10% ADV
          2. 点差 ≤ 100bps
          3. 预估冲击 ≤ 50bps
          4. 每片滑点 ≤ 25bps
        """
        # 1. 参与率
        participation = order_size / max(avg_daily_volume, 1)
        if participation > MAX_PARTICIPATION_RATE:
            return RiskCheck(
                False,
                f"参与率 {participation:.1%} > {MAX_PARTICIPATION_RATE:.0%}上限",
                "danger"
            )

        # 2. 点差
        spread_bps = spread * 10000
        if spread_bps > MAX_SPREAD_BPS:
            return RiskCheck(
                False,
                f"点差 {spread_bps:.0f}bps > {MAX_SPREAD_BPS:.0f}bps上限 (流动性不足)",
                "warning"
            )

        # 3. 预估冲击
        try:
            from execution import MarketImpactModel
            impact_model = MarketImpactModel()
            est = impact_model.estimate_impact(order_size, avg_daily_volume,
                                               volatility, spread)
            if est["total_bps"] > MAX_IMPACT_BPS:
                return RiskCheck(
                    False,
                    f"预估冲击 {est['total_bps']:.0f}bps > {MAX_IMPACT_BPS:.0f}bps上限",
                    "warning"
                )

            # 4. 每片滑点 (如果是大单拆片不足)
            slice_size = order_size / max(n_slices, 1)
            slice_est = impact_model.estimate_impact(slice_size, avg_daily_volume,
                                                     volatility, spread)
            if slice_est["total_bps"] > MAX_SLIPPAGE_PER_SLICE_BPS:
                return RiskCheck(
                    False,
                    f"单切片预估滑点 {slice_est['total_bps']:.0f}bps > {MAX_SLIPPAGE_PER_SLICE_BPS:.0f}bps, "
                    f"建议增加切片数 (当前{n_slices}片)",
                    "warning"
                )
        except ImportError:
            pass  # execution 模块不可用时跳过

        return RiskCheck(True, "执行层风控通过", "info")

    def daily_risk_report(self) -> dict:
        """每日风控报告"""
        pf = self.pf
        open_pos = pf.open_positions

        return {
            "total_value": pf.total_value,
            "total_pnl_pct": pf.total_pnl_pct,
            "open_positions": len(open_pos),
            "daily_loss_used": max(0, abs(pf.total_pnl_pct)) / abs(DAILY_LOSS_LIMIT_PCT) * 100,
            "monthly_progress": pf.total_pnl_pct / MONTHLY_TARGET_PCT * 100,
            "survival_score": max(0, 100 - max(0, pf.total_pnl_pct) * 2),  # 亏损越多分越低
            "alerts": pf.check_risk(),
        }
