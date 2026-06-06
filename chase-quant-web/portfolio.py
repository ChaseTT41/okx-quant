"""
Chase量化策略 — 虚拟盘组合管理器
三个市场: A股 / 美股 / 加密货币 | 总本金 ¥10,000
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict
from dataclasses import dataclass, field, asdict

DATA_DIR = Path(__file__).parent / "data"
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
TRADES_FILE = DATA_DIR / "trades.json"
SNAPSHOT_FILE = DATA_DIR / "snapshots.json"

# 初始资金分配
INITIAL_CAPITAL = 10000.0  # RMB
ALLOCATION = {
    "crypto":  4000.0,   # 加密货币 (Binance/OKX)
    "a_stock": 3500.0,   # A股 (同花顺标准)
    "us_stock": 2500.0,   # 美股 (VSTrader)
}

# 手续费
FEES = {
    "crypto":  0.001,    # 0.1%
    "a_stock": 0.0003,   # 万三佣金 + 千一印花税(卖) ≈ 综合0.03%
    "us_stock": 0.005,   # 0.5% 含汇损
}


@dataclass
class Position:
    """单笔持仓"""
    id: str
    market: str           # crypto / a_stock / us_stock
    symbol: str           # 代码 e.g. BTC/USDT, 000001, AAPL
    name: str             # 名称
    entry_price: float
    current_price: float
    quantity: float
    entry_time: str       # ISO timestamp
    entry_reason: str     # 为什么买入
    stop_loss: float      # 止损价
    take_profit: float    # 止盈价
    status: str = "open"  # open / closed

    @property
    def cost(self) -> float:
        return self.entry_price * self.quantity

    @property
    def value(self) -> float:
        return self.current_price * self.quantity

    @property
    def pnl(self) -> float:
        return self.value - self.cost

    @property
    def pnl_pct(self) -> float:
        return (self.current_price / self.entry_price - 1) * 100 if self.entry_price > 0 else 0


@dataclass
class Trade:
    """已完成交易记录"""
    id: str
    market: str
    symbol: str
    name: str
    side: str            # buy / sell
    price: float
    quantity: float
    amount: float        # price * quantity
    fee: float
    time: str
    reason: str
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class PortfolioSnapshot:
    """组合快照 (用于画净值曲线)"""
    time: str
    total_value: float
    cash: dict           # {market: cash_amount}
    positions_value: float
    pnl_total: float
    pnl_pct: float


class PortfolioManager:
    """虚拟盘组合管理器"""

    def __init__(self):
        self._ensure_data()
        self.cash: Dict[str, float] = {}
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.snapshots: List[PortfolioSnapshot] = []
        self._load()

    def _ensure_data(self):
        DATA_DIR.mkdir(exist_ok=True)
        # 初始化 portfolio
        if not PORTFOLIO_FILE.exists():
            init_data = {
                "cash": dict(ALLOCATION),
                "positions": {},
                "created_at": datetime.now(timezone.utc).isoformat(),
                "initial_capital": INITIAL_CAPITAL,
            }
            PORTFOLIO_FILE.write_text(json.dumps(init_data, indent=2, ensure_ascii=False))
        # 初始化 trades
        if not TRADES_FILE.exists():
            TRADES_FILE.write_text("[]")
        # 初始化 snapshots
        if not SNAPSHOT_FILE.exists():
            SNAPSHOT_FILE.write_text("[]")

    def _load(self):
        pf = json.loads(PORTFOLIO_FILE.read_text())
        self.cash = pf.get("cash", dict(ALLOCATION))
        self.positions = {}
        for pid, pdata in pf.get("positions", {}).items():
            self.positions[pid] = Position(**pdata)

        self.trades = [Trade(**t) for t in json.loads(TRADES_FILE.read_text())]
        self.snapshots = [PortfolioSnapshot(**s) for s in json.loads(SNAPSHOT_FILE.read_text())]

    def _save(self):
        pf = {
            "cash": self.cash,
            "positions": {pid: asdict(p) for pid, p in self.positions.items()},
            "initial_capital": INITIAL_CAPITAL,
            "created_at": json.loads(PORTFOLIO_FILE.read_text()).get("created_at",
                                datetime.now(timezone.utc).isoformat()),
        }
        PORTFOLIO_FILE.write_text(json.dumps(pf, indent=2, ensure_ascii=False))
        TRADES_FILE.write_text(json.dumps([asdict(t) for t in self.trades], indent=2, ensure_ascii=False))
        SNAPSHOT_FILE.write_text(json.dumps([asdict(s) for s in self.snapshots[-200:]], indent=2, ensure_ascii=False))

    # ── 属性 ──
    @property
    def total_cash(self) -> float:
        return sum(self.cash.values())

    @property
    def positions_value(self) -> float:
        return sum(p.value for p in self.positions.values())

    @property
    def total_value(self) -> float:
        return self.total_cash + self.positions_value

    @property
    def total_pnl(self) -> float:
        return self.total_value - INITIAL_CAPITAL

    @property
    def total_pnl_pct(self) -> float:
        return (self.total_value / INITIAL_CAPITAL - 1) * 100

    @property
    def open_positions(self) -> List[Position]:
        return [p for p in self.positions.values() if p.status == "open"]

    def cash_for(self, market: str) -> float:
        return self.cash.get(market, 0)

    # ── 交易执行 ──
    def can_buy(self, market: str, amount: float) -> bool:
        """检查是否有足够现金"""
        fee = amount * FEES.get(market, 0.001)
        total = amount + fee
        return self.cash.get(market, 0) >= total

    def buy(self, market: str, symbol: str, name: str, price: float,
            quantity: float, reason: str) -> Optional[Trade]:
        """执行买入"""
        amount = price * quantity
        fee = amount * FEES.get(market, 0.001)

        if not self.can_buy(market, amount + fee):
            return None

        # 扣钱
        self.cash[market] -= (amount + fee)

        # 建仓
        pid = f"{market}_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        pos = Position(
            id=pid, market=market, symbol=symbol, name=name,
            entry_price=price, current_price=price, quantity=quantity,
            entry_time=datetime.now(timezone.utc).isoformat(),
            entry_reason=reason,
            stop_loss=price * 0.92,   # -8% 硬止损
            take_profit=price * 1.15,  # +15% 止盈
        )
        self.positions[pid] = pos

        trade = Trade(
            id=pid, market=market, symbol=symbol, name=name,
            side="buy", price=price, quantity=quantity,
            amount=amount, fee=fee,
            time=datetime.now(timezone.utc).isoformat(),
            reason=reason,
        )
        self.trades.append(trade)
        self._save()
        return trade

    def sell(self, position_id: str, price: float, reason: str) -> Optional[Trade]:
        """执行卖出"""
        pos = self.positions.get(position_id)
        if not pos or pos.status == "closed":
            return None

        amount = price * pos.quantity
        fee = amount * FEES.get(pos.market, 0.001)

        # 加钱
        self.cash[pos.market] += (amount - fee)

        pnl = (price - pos.entry_price) * pos.quantity
        pnl_pct = (price / pos.entry_price - 1) * 100

        pos.status = "closed"
        pos.current_price = price

        trade = Trade(
            id=position_id, market=pos.market, symbol=pos.symbol,
            name=pos.name, side="sell", price=price,
            quantity=pos.quantity, amount=amount, fee=fee,
            time=datetime.now(timezone.utc).isoformat(),
            reason=reason, pnl=pnl, pnl_pct=pnl_pct,
        )
        self.trades.append(trade)
        self._save()
        return trade

    def update_price(self, position_id: str, price: float):
        """更新持仓市价"""
        if position_id in self.positions:
            self.positions[position_id].current_price = price

    # ── 快照 ──
    def take_snapshot(self):
        snap = PortfolioSnapshot(
            time=datetime.now(timezone.utc).isoformat(),
            total_value=self.total_value,
            cash=dict(self.cash),
            positions_value=self.positions_value,
            pnl_total=self.total_pnl,
            pnl_pct=self.total_pnl_pct,
        )
        self.snapshots.append(snap)
        self._save()

    # ── 风险检查 ──
    def check_risk(self) -> List[str]:
        """返回风险警报列表"""
        alerts = []

        # 总亏损 > 5%
        if self.total_pnl_pct < -5:
            alerts.append(f"🔴 总亏损 {self.total_pnl_pct:.1f}%, 已触发日止损线")

        # 总亏损 > 10%
        if self.total_pnl_pct < -10:
            alerts.append(f"💀 总亏损 {self.total_pnl_pct:.1f}%, 接近淘汰线")

        # 单仓位 > 40%
        for p in self.open_positions:
            pct = p.value / self.total_value * 100 if self.total_value > 0 else 0
            if pct > 40:
                alerts.append(f"⚠️ {p.symbol} 仓位 {pct:.0f}% 超过40%上限")
            if p.pnl_pct < -8:
                alerts.append(f"🔴 {p.symbol} 亏损 {p.pnl_pct:.1f}%, 触发硬止损!")

        # 持仓数
        if len(self.open_positions) > 5:
            alerts.append(f"⚠️ 同时持仓 {len(self.open_positions)} > 5 只上限")

        return alerts

    def get_allocation_summary(self) -> dict:
        """各市场资金分配概览"""
        summary = {}
        for market, label, color in [
            ("crypto", "₿ 加密货币", "#F7931A"),
            ("a_stock", "🇨🇳 A股", "#FF0000"),
            ("us_stock", "🇺🇸 美股", "#1E90FF"),
        ]:
            positions_val = sum(p.value for p in self.open_positions if p.market == market)
            cash_val = self.cash.get(market, 0)
            total = positions_val + cash_val
            pnl = total - ALLOCATION.get(market, 0)
            summary[market] = {
                "label": label, "color": color,
                "allocated": ALLOCATION.get(market, 0),
                "cash": cash_val, "positions": positions_val,
                "total": total, "pnl": pnl, "pnl_pct": (pnl / ALLOCATION.get(market, 1)) * 100,
            }
        return summary


# 全局单例
_pf: Optional[PortfolioManager] = None


def get_portfolio() -> PortfolioManager:
    global _pf
    if _pf is None:
        _pf = PortfolioManager()
    return _pf
