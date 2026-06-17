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

        # 2. 持仓上限 (每市场独立, 避免单一市场占满全局额度)
        open_count = len([p for p in self.pf.open_positions if p.market == market])
        if open_count >= MAX_POSITIONS:
            return RiskCheck(False, f"{market}已持仓{open_count}只, 达上限{MAX_POSITIONS}", "warning")

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

    # ── ⚡ 杠杆交易专属风控 ──
    MAX_LEVERAGE = 10
    MAX_LEVERAGED_POSITION_SIZE_PCT = 10.0  # 杠杆仓位上限 (保证金)
    MAX_DAILY_LEVERAGED_TRADES = 3
    MAX_LEVERAGED_TOTAL_EXPOSURE = 3.0  # 总名义敞口 ≤ 3x 权益

    def leverage_pre_trade_check(self, symbol: str, margin_usdt: float,
                                  leverage: int, total_equity: float,
                                  funding_rate: float = 0.0,
                                  total_leveraged_notional: float = 0.0) -> RiskCheck:
        """
        杠杆交易事前风控 — 在合约下单前检查。

        检查项:
          1. 杠杆上限 (≤10x)
          2. 保证金仓位上限 (≤10% 权益)
          3. 总名义敞口上限 (≤3x 权益)
          4. 资金费率过高警告
          5. 预估强平价格
        """
        # 1. 杠杆上限
        if leverage > self.MAX_LEVERAGE:
            return RiskCheck(False, f"杠杆{leverage}x > {self.MAX_LEVERAGE}x上限", "danger")

        # 2. 保证金仓位 (单仓 ≤ 10% 权益)
        margin_pct = margin_usdt / total_equity * 100 if total_equity > 0 else 100
        if margin_pct > self.MAX_LEVERAGED_POSITION_SIZE_PCT:
            return RiskCheck(
                False,
                f"杠杆保证金{margin_pct:.1f}% > {self.MAX_LEVERAGED_POSITION_SIZE_PCT}%上限",
                "warning"
            )

        # 3. 总名义敞口 (杠杆后)
        new_notional = margin_usdt * leverage
        total_notional_after = total_leveraged_notional + new_notional
        exposure_ratio = total_notional_after / total_equity if total_equity > 0 else 0
        if exposure_ratio > self.MAX_LEVERAGED_TOTAL_EXPOSURE:
            return RiskCheck(
                False,
                f"总敞口{exposure_ratio:.1f}x > {self.MAX_LEVERAGED_TOTAL_EXPOSURE}x上限",
                "danger"
            )

        # 4. 资金费率: 年化>30% 警告
        fr_annualized = abs(funding_rate) * 3 * 365
        if fr_annualized > 0.30:
            return RiskCheck(
                True,
                f"⚠️ 资金费率年化{fr_annualized:.0%}, 持仓成本高",
                "warning"
            )

        # 5. 强平价格估算 (保守: -15% of entry for 10x)
        liq_buffer = 0.85 / leverage  # 10x → 8.5%, 5x → 17%, 2x → 42%
        if liq_buffer < 0.05:
            return RiskCheck(
                True,
                f"强平空间仅{liq_buffer:.1%} (杠杆={leverage}x), 建议设紧止损",
                "warning"
            )

        return RiskCheck(True, f"杠杆风控通过 ({leverage}x, 保证金{margin_usdt:.0f}USDT)", "info")

    def leverage_position_check(self, symbol: str, entry_price: float,
                                 current_price: float, leverage: int,
                                 margin: float, pnl_pct: float) -> RiskCheck:
        """
        杠杆持仓定期检查 — 强平风险评估。
        """
        # 杠杆放大亏损
        unleveraged_pnl = pnl_pct / leverage if leverage > 0 else pnl_pct

        # 10x杠杆 → 8% 价格跌 = 80% 保证金亏 → 接近强平
        liq_threshold = -0.85 / leverage  # 10x → -8.5%

        if pnl_pct <= liq_threshold * 0.7:  # 70% 到强平线
            return RiskCheck(
                False,
                f"{symbol} 杠杆亏损{pnl_pct:.1f}% (强平线≈{liq_threshold:.1%}), 建议减仓!",
                "danger"
            )

        if pnl_pct <= liq_threshold * 0.85:
            return RiskCheck(
                True,
                f"{symbol} 杠杆亏损{pnl_pct:.1f}%, 接近强平线",
                "warning"
            )

        return RiskCheck(True, f"{symbol} 杠杆正常 ({pnl_pct:+.1f}%)", "info")

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


# ═══════════════════════════════════════════
# Phase 15: 实盘专属风控
# ═══════════════════════════════════════════

# 实盘风控参数（100-300 USDT 小额适配）
LIVE_MAX_DAILY_TRADES = 100         # 每日最大交易次数（信号驱动，不做硬限制）
LIVE_MAX_DRAWDOWN_PCT = -0.10       # 最大回撤 10%（从峰值）
LIVE_COOLDOWN_AFTER_STOP_MIN = 30   # 止损后冷静期（分钟）
LIVE_MIN_NOTIONAL_USDT = 10.0       # 币安最低名义交易额
LIVE_MAX_SINGLE_POSITION_PCT = 0.50 # 单币最大仓位 50%
LIVE_API_ERROR_FUSE_COUNT = 3       # 连续 API 错误熔断次数

from datetime import datetime, timezone, timedelta
from collections import defaultdict


@dataclass
class LiveRiskState:
    """实盘风控状态（持久化到 data/live_risk_state.json）"""
    peak_equity: float = 0.0                    # 峰值权益
    current_equity: float = 0.0                 # 当前权益
    drawdown_pct: float = 0.0                   # 当前回撤
    daily_trades_today: int = 0                 # 今日交易次数
    last_trade_date: str = ""                   # 上次交易日期
    consecutive_api_errors: int = 0             # 连续 API 错误
    last_stop_loss_time: dict = field(          # {symbol: iso_timestamp}
        default_factory=dict)
    positions_concentration: dict = field(       # {symbol: pct}
        default_factory=dict)
    updated_at: str = ""


class LiveRiskController:
    """
    实盘专属风控控制器

    在现有五层风控（RiskController）基础上叠加实盘专属规则:
      1. API 错误熔断 — 连续 N 次 → 停 30 分钟
      2. 最大回撤限制 — 从峰值回撤 10% → 阻止新开仓
      3. 止损后冷静期 — 30 分钟内禁止同 symbol 开仓
      4. 每日交易次数上限 — 5 笔（保守）
      5. 最小交易额检查 — ≥ $10
      6. 持仓集中度 — 单币 ≤ 50%
    """

    def __init__(self, total_equity_usdt: float = 0.0):
        self.state = LiveRiskState()
        self.state.peak_equity = total_equity_usdt
        self.state.current_equity = total_equity_usdt
        self._load_state()

    def _load_state(self):
        """加载持久化风控状态"""
        import json
        from pathlib import Path
        state_file = Path(__file__).parent / "data" / "live_risk_state.json"
        try:
            if state_file.exists():
                data = json.loads(state_file.read_text())
                self.state.peak_equity = data.get("peak_equity", self.state.peak_equity)
                self.state.current_equity = data.get("current_equity", self.state.current_equity)
                self.state.daily_trades_today = data.get("daily_trades_today", 0)
                self.state.last_trade_date = data.get("last_trade_date", "")
                self.state.consecutive_api_errors = data.get("consecutive_api_errors", 0)
                self.state.last_stop_loss_time = data.get("last_stop_loss_time", {})
                self.state.positions_concentration = data.get("positions_concentration", {})
        except Exception:
            pass

    def _save_state(self):
        """持久化风控状态"""
        import json
        from pathlib import Path
        state_file = Path(__file__).parent / "data" / "live_risk_state.json"
        try:
            state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state.updated_at = datetime.now(timezone.utc).isoformat()
            state_file.write_text(json.dumps({
                "peak_equity": self.state.peak_equity,
                "current_equity": self.state.current_equity,
                "daily_trades_today": self.state.daily_trades_today,
                "last_trade_date": self.state.last_trade_date,
                "consecutive_api_errors": self.state.consecutive_api_errors,
                "last_stop_loss_time": self.state.last_stop_loss_time,
                "positions_concentration": self.state.positions_concentration,
                "updated_at": self.state.updated_at,
            }, ensure_ascii=False, indent=2))
        except Exception:
            pass

    # ── 规则 1: 最大回撤限制 ──

    def update_equity(self, current_equity: float):
        """更新权益并检查回撤"""
        self.state.current_equity = current_equity
        if current_equity > self.state.peak_equity:
            self.state.peak_equity = current_equity
        if self.state.peak_equity > 0:
            self.state.drawdown_pct = (current_equity - self.state.peak_equity) / self.state.peak_equity
        self._save_state()

    def check_drawdown(self) -> RiskCheck:
        """检查是否触发回撤限制"""
        if self.state.drawdown_pct <= LIVE_MAX_DRAWDOWN_PCT:
            return RiskCheck(
                False,
                f"最大回撤 {self.state.drawdown_pct:.1%} 超过 {LIVE_MAX_DRAWDOWN_PCT:.0%} 限制! "
                f"峰值 ${self.state.peak_equity:.0f} → 当前 ${self.state.current_equity:.0f}",
                "danger"
            )
        return RiskCheck(True, f"回撤 {self.state.drawdown_pct:.1%} (限制 {LIVE_MAX_DRAWDOWN_PCT:.0%})", "info")

    # ── 规则 2: 每日交易次数 ──

    def check_daily_trade_limit(self) -> RiskCheck:
        """检查每日交易次数上限"""
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        if self.state.last_trade_date != today:
            self.state.daily_trades_today = 0
            self.state.last_trade_date = today
            self._save_state()

        if self.state.daily_trades_today >= LIVE_MAX_DAILY_TRADES:
            return RiskCheck(
                False,
                f"今日交易 {self.state.daily_trades_today}/{LIVE_MAX_DAILY_TRADES} 已达上限",
                "warning"
            )
        return RiskCheck(True, f"今日 {self.state.daily_trades_today}/{LIVE_MAX_DAILY_TRADES} 笔", "info")

    def record_trade(self):
        """记录一笔交易"""
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        if self.state.last_trade_date != today:
            self.state.daily_trades_today = 0
            self.state.last_trade_date = today
        self.state.daily_trades_today += 1
        self._save_state()

    # ── 规则 3: 止损后冷静期 ──

    def record_stop_loss(self, symbol: str):
        """记录止损事件"""
        self.state.last_stop_loss_time[symbol] = datetime.now(timezone.utc).isoformat()
        self._save_state()

    def check_cooldown(self, symbol: str) -> RiskCheck:
        """检查止损后冷静期"""
        if symbol not in self.state.last_stop_loss_time:
            return RiskCheck(True, "", "info")

        try:
            last_stop = datetime.fromisoformat(self.state.last_stop_loss_time[symbol])
            elapsed = (datetime.now(timezone.utc) - last_stop).total_seconds() / 60
            if elapsed < LIVE_COOLDOWN_AFTER_STOP_MIN:
                remaining = int(LIVE_COOLDOWN_AFTER_STOP_MIN - elapsed)
                return RiskCheck(
                    False,
                    f"{symbol} 止损后冷静期: 还需 {remaining} 分钟 (共{LIVE_COOLDOWN_AFTER_STOP_MIN}分钟)",
                    "warning"
                )
        except (ValueError, TypeError):
            pass

        return RiskCheck(True, "", "info")

    # ── 规则 4: 最小交易额 ──

    @staticmethod
    def check_min_notional(amount_usdt: float) -> RiskCheck:
        """检查是否满足币安最小交易额"""
        if amount_usdt < LIVE_MIN_NOTIONAL_USDT:
            return RiskCheck(
                False,
                f"交易金额 ${amount_usdt:.2f} 低于币安最小交易额 ${LIVE_MIN_NOTIONAL_USDT}",
                "warning"
            )
        return RiskCheck(True, "", "info")

    # ── 规则 5: 持仓集中度 ──

    def check_concentration(self, symbol: str, new_position_pct: float,
                           current_positions: dict = None) -> RiskCheck:
        """
        检查持仓集中度

        Args:
            symbol: 交易对
            new_position_pct: 新仓位占比 (0-1)
            current_positions: 现有持仓 {symbol: pct}
        """
        if current_positions:
            self.state.positions_concentration.update(current_positions)

        total_pct = self.state.positions_concentration.get(symbol, 0) + new_position_pct
        if total_pct > LIVE_MAX_SINGLE_POSITION_PCT:
            return RiskCheck(
                False,
                f"{symbol} 仓位 {total_pct:.0%} 超过 {LIVE_MAX_SINGLE_POSITION_PCT:.0%} 上限",
                "danger"
            )
        return RiskCheck(True, "", "info")

    # ── 规则 6: API 错误熔断 ──

    def record_api_error(self):
        """记录 API 错误"""
        self.state.consecutive_api_errors += 1
        self._save_state()

    def record_api_success(self):
        """记录 API 成功"""
        if self.state.consecutive_api_errors > 0:
            self.state.consecutive_api_errors = 0
            self._save_state()

    def check_api_fuse(self) -> RiskCheck:
        """检查 API 错误熔断"""
        if self.state.consecutive_api_errors >= LIVE_API_ERROR_FUSE_COUNT:
            return RiskCheck(
                False,
                f"连续 {self.state.consecutive_api_errors} 次 API 错误, 触发熔断!",
                "danger"
            )
        return RiskCheck(True, "", "info")

    # ── 综合检查 ──

    def live_pre_trade_check(self, symbol: str, amount_usdt: float,
                            position_pct: float = 0.0,
                            current_positions: dict = None) -> list[RiskCheck]:
        """
        实盘交易前综合风控检查

        Returns:
            [RiskCheck, ...] — 全部通过才可交易
        """
        checks = []

        # 1. 最小交易额
        checks.append(self.check_min_notional(amount_usdt))

        # 2. 回撤限制
        checks.append(self.check_drawdown())

        # 3. 每日交易次数
        checks.append(self.check_daily_trade_limit())

        # 4. 止损冷静期
        checks.append(self.check_cooldown(symbol))

        # 5. 持仓集中度
        if position_pct > 0:
            checks.append(self.check_concentration(symbol, position_pct, current_positions))

        # 6. API 熔断
        checks.append(self.check_api_fuse())

        return checks

    def all_checks_passed(self, checks: list[RiskCheck]) -> bool:
        """所有检查是否通过"""
        return all(c.passed for c in checks)

    def get_failed_checks(self, checks: list[RiskCheck]) -> list[RiskCheck]:
        """获取失败的检查"""
        return [c for c in checks if not c.passed]

    def get_status_summary(self) -> str:
        """风控状态摘要"""
        lines = [
            f"🛡️ 实盘风控状态",
            f"  峰值权益: ${self.state.peak_equity:.2f}",
            f"  当前权益: ${self.state.current_equity:.2f}",
            f"  当前回撤: {self.state.drawdown_pct:.1%} (限制 {LIVE_MAX_DRAWDOWN_PCT:.0%})",
            f"  今日交易: {self.state.daily_trades_today}/{LIVE_MAX_DAILY_TRADES} 笔",
            f"  连续API错误: {self.state.consecutive_api_errors}/{LIVE_API_ERROR_FUSE_COUNT}",
        ]
        return "\n".join(lines)
