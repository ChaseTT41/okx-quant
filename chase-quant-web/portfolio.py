"""
Chase量化策略 — 虚拟盘组合管理器
五个市场: A股 / 美股 / 加密货币 / 港股 / bStocks(币安代币化美股) | 总本金 ¥10,000
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
    "crypto":  3500.0,   # 加密货币 (Binance/OKX)
    "a_stock": 3000.0,   # A股 (同花顺标准)
    "us_stock": 2000.0,   # 美股 (VSTrader)
    "hk_stock": 1000.0,   # 港股 (港交所)
    "b_stock": 1500.0,   # bStocks (币安代币化美股, 24/7, USDT计价)
}

# 手续费
FEES = {
    "crypto":  0.001,    # 0.1%
    "a_stock": 0.0003,   # 万三佣金 + 千一印花税(卖) ≈ 综合0.03%
    "us_stock": 0.005,   # 0.5% 含汇损
    "hk_stock": 0.003,   # 0.1%印花税 + 0.1%佣金 + 0.1%杂费 ≈ 综合0.3%
    "b_stock": 0.000,    # bStocks挂单费豁免至2026-09-01, 吃单0.01%暂忽略
}

# 币种全名映射 (ticker → full name)
COIN_FULL_NAMES = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "BNB": "BNB Chain",
    "SOL": "Solana", "XRP": "XRP", "ADA": "Cardano", "DOGE": "Dogecoin",
    "AVAX": "Avalanche", "DOT": "Polkadot", "LINK": "Chainlink",
    "MATIC": "Polygon", "ATOM": "Cosmos", "LTC": "Litecoin",
    "UNI": "Uniswap", "APT": "Aptos", "NEAR": "NEAR Protocol",
    "OP": "Optimism", "ARB": "Arbitrum", "SUI": "Sui", "TON": "Toncoin",
    "FIL": "Filecoin", "TRX": "TRON", "ETC": "Ethereum Classic",
    "ICP": "Internet Computer", "RENDER": "Render",
    "WIF": "dogwifhat", "PEPE": "Pepe", "AAVE": "Aave", "ORDI": "ORDI",
}


def _coin_full_name(ticker: str) -> str:
    """从 ticker 获取全名，找不到则返回 ticker 本身"""
    return COIN_FULL_NAMES.get(ticker.upper(), ticker)


@dataclass
class Position:
    """单笔持仓"""
    id: str
    market: str           # crypto / a_stock / us_stock / hk_stock
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
    avg_entry_price: float = 0.0     # Phase 13: 分笔成交后的加权均价
    entry_prices: list = field(default_factory=list)  # Phase 13: 各切片成交价记录
    execution_strategy: str = ""     # Phase 13: 使用的执行策略

    @property
    def cost(self) -> float:
        px = self.avg_entry_price if self.avg_entry_price > 0 else self.entry_price
        return px * self.quantity

    @property
    def value(self) -> float:
        return self.current_price * self.quantity

    @property
    def pnl(self) -> float:
        return self.value - self.cost

    @property
    def pnl_pct(self) -> float:
        px = self.avg_entry_price if self.avg_entry_price > 0 else self.entry_price
        return (self.current_price / px - 1) * 100 if px > 0 else 0


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
        # 迁移: 自动添加新市场
        for mkt, amt in ALLOCATION.items():
            if mkt not in self.cash:
                self.cash[mkt] = amt
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
        return sum(p.value for p in self.open_positions)

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
        """执行买入 (单笔市价单)"""
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
            avg_entry_price=price,
            entry_prices=[price],
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

    def buy_partial(self, position_id: str, price: float, quantity: float,
                    slice_num: int = 0, total_slices: int = 0,
                    strategy: str = "") -> Optional[bool]:
        """
        Phase 13: 分笔成交 — 向已有持仓追加数量

        用于拆单算法中的每个子订单成交后累加仓位。
        第一次调用创建持仓, 后续调用累加数量并更新加权均价。

        Args:
            position_id: 持仓 ID (同一个 parent order 共用)
            price: 本笔成交价
            quantity: 本笔数量
            slice_num: 当前切片编号 (0-based)
            total_slices: 总切片数
            strategy: 执行策略名

        Returns:
            True if success, None if insufficient cash
        """
        amount = price * quantity
        fee = amount * FEES.get("crypto", 0.001)

        if not self.can_buy("crypto", amount + fee):
            return None

        # 扣钱
        self.cash["crypto"] -= (amount + fee)

        if position_id in self.positions:
            # 追加到已有持仓
            pos = self.positions[position_id]
            total_qty = pos.quantity + quantity

            # 更新加权均价
            old_notional = pos.avg_entry_price * pos.quantity if pos.avg_entry_price > 0 else pos.entry_price * pos.quantity
            new_notional = price * quantity
            pos.avg_entry_price = (old_notional + new_notional) / total_qty if total_qty > 0 else price
            pos.quantity = total_qty
            pos.current_price = price
            pos.entry_prices.append(price)
        else:
            # 首次建仓
            symbol_clean = position_id.split("_", 2)[-1] if "_" in position_id else position_id
            ticker = symbol_clean.replace("/USDT", "")
            name = _coin_full_name(ticker)
            pos = Position(
                id=position_id,
                market="crypto",
                symbol=f"crypto_{symbol_clean}" if "/" in symbol_clean else symbol_clean,
                name=name,
                entry_price=price,
                current_price=price,
                quantity=quantity,
                entry_time=datetime.now(timezone.utc).isoformat(),
                entry_reason=f"分笔成交 [{strategy}] ({slice_num+1}/{total_slices})" if total_slices > 0 else f"分笔成交 [{strategy}]",
                stop_loss=price * 0.92,
                take_profit=price * 1.15,
                avg_entry_price=price,
                entry_prices=[price],
                execution_strategy=strategy,
            )
            self.positions[position_id] = pos

        # 记录子交易
        slice_trade = Trade(
            id=f"{position_id}_s{slice_num}",
            market="crypto",
            symbol=self.positions[position_id].symbol,
            name=self.positions[position_id].name,
            side="buy",
            price=price,
            quantity=quantity,
            amount=amount,
            fee=fee,
            time=datetime.now(timezone.utc).isoformat(),
            reason=f"Slice {slice_num+1}/{total_slices} [{strategy}]",
        )
        self.trades.append(slice_trade)
        self._save()
        return True

    def buy_sliced(self, symbol: str, name: str, total_quantity: float,
                   slice_prices: List[float], slice_quantities: List[float],
                   strategy: str = "", reason: str = "") -> Optional[str]:
        """
        Phase 13: 一次执行完整拆单计划

        Args:
            symbol: 交易对
            name: 名称
            total_quantity: 总数量
            slice_prices: 每片成交价列表
            slice_quantities: 每片数量列表
            strategy: 执行策略名
            reason: 交易理由

        Returns:
            position_id if all slices filled, None if failed
        """
        market = "crypto"
        pid = f"{market}_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        total_filled = 0.0
        n = len(slice_prices)

        for i, (px, qty) in enumerate(zip(slice_prices, slice_quantities)):
            result = self.buy_partial(pid, px, qty, slice_num=i,
                                      total_slices=n, strategy=strategy)
            if result is None:
                # 现金不足, 停止后续切片
                break
            total_filled += qty

        if total_filled <= 0:
            return None

        # 更新最终持仓信息
        if pid in self.positions:
            pos = self.positions[pid]
            pos.entry_reason = f"[{strategy.upper()}] {reason}" if reason else f"[{strategy.upper()}] 分{N}笔执行"
            pos.execution_strategy = strategy
            self._save()

        return pid

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
            self._save()  # 立即持久化, 确保API服务器读到最新价

    # ── 快照 ──
    def take_snapshot(self):
        """记录快照 — 只写 snapshots.json，不覆盖 portfolio.json (避免和 auto_trade 冲突)"""
        snap = PortfolioSnapshot(
            time=datetime.now(timezone.utc).isoformat(),
            total_value=self.total_value,
            cash=dict(self.cash),
            positions_value=self.positions_value,
            pnl_total=self.total_pnl,
            pnl_pct=self.total_pnl_pct,
        )
        self.snapshots.append(snap)
        # 只保存快照，不覆盖 portfolio.json 和 trades.json
        SNAPSHOT_FILE.write_text(json.dumps(
            [asdict(s) for s in self.snapshots[-200:]], indent=2, ensure_ascii=False))

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
            ("hk_stock", "🇭🇰 港股", "#DC143C"),
            ("b_stock", "🏦 bStocks", "#00D4AA"),
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
